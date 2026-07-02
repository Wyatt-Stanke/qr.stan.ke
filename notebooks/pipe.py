"""
qr_streetview_pipeline.py
=========================

Recall-maximising pipeline for finding QR codes in Google Street View
equirectangular panoramas (16384 x 8192) and turning them into geolocated
map points.

Pipeline stages
---------------
1. Reproject the equirectangular pano into a set of overlapping rectilinear
   (gnomonic) perspective views.  De-warps codes so they look planar, which is
   what every decoder expects.  (Implemented by the caller; see PanoMeta.)
2. Scan each view at TWO scales:
     - the whole view (owns large / billboard-sized codes), and
     - overlapping tiles (owns small / distant codes).
   Multi-scale is what makes the tiling seam-proof in both directions: a code
   too big to fit in a tile is caught by the full-view pass, a code too small
   to survive the full-view downsample is caught by a tile.
3. Localise with OpenCV's WeChat detector (CNN detector + QRSR super-resolution)
   and read with an ENSEMBLE of decoders (WeChat / zxing-cpp / pyzbar).  A
   detector box that fails to decode in a tile is re-cropped from the FULL-RES
   view (ignoring tile seams) and retried over several preprocessing variants:
   "detect locally, decode globally".
4. Take the UNION of every decode.  A valid decode is self-validating
   (Reed-Solomon), so every stage runs at aggressive settings -- recall is
   essentially free of precision cost.
5. Map each detection's pixel back to an azimuth/elevation in the pano frame,
   convert to a compass bearing, and triangulate a payload seen from multiple
   panos into a lat/lng point.
6. Emit GeoJSON for the map.

Dependencies
------------
    pip install opencv-contrib-python numpy zxing-cpp pyzbar
    # pyzbar also needs the native zbar lib: apt-get install libzbar0

WeChat model files (4 files) are NOT shipped with opencv-contrib-python.
Download them once into Config.model_dir from the opencv_3rdparty
"wechat_qrcode" branch (detect.prototxt, detect.caffemodel, sr.prototxt,
sr.caffemodel):
    https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode

Swap-in note: to use a YOLO QR detector instead, replace
QRScanner._wechat_detect() with your model's inference (return boxes in the
image's local pixel coords) and keep the rest of the pipeline unchanged.
"""

from __future__ import annotations

import os
import math
import json
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np
import cv2

# Optional decoders -- degrade gracefully if not installed.
try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:
    _HAS_ZXING = False

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- Stage 1: reprojection ------------------------------------------------
    fov_deg: float = 90.0                 # horizontal FOV of each perspective view
    yaw_step_deg: float = 60.0            # 60 deg step w/ 90 deg FOV => ~33% overlap
    pitch_deg: tuple[float, ...] = (-20.0, 10.0, 35.0)  # codes cluster near/above horizon
    view_size: int = 2560                 # output view is view_size x view_size px
    # 16384px / 360deg = ~45.5 px/deg native.  90deg view at 2560px = ~28 px/deg,
    # a mild downsample that tiling recovers.  Raise for more recall on tiny codes.

    # --- Stage 2: tiling ------------------------------------------------------
    tile: int = 640                       # tile edge fed to the detector (0 disables tiling)
    tile_overlap: float = 0.25            # fractional overlap between tiles

    # --- Stage 3/4: detection + decoding -------------------------------------
    model_dir: str = "./wechat_models"    # holds the 4 WeChat caffe files
    upscale_factors: tuple[float, ...] = (1.0, 2.0, 4.0)
    rotations_deg: tuple[int, ...] = (0, 90, 180, 270)
    min_crop_px: int = 12                 # ignore candidate crops smaller than this
    box_merge_iou: float = 0.2            # merge overlapping detector boxes before decode
    decode_pad_px: int = 8                # min padding when cropping a box for decoding
    decode_pad_frac: float = 0.15         # extra padding as a fraction of box size

    # --- Stage 5: geo ---------------------------------------------------------
    # Triangulated intersection points of one payload that fall within this
    # radius are treated as the SAME physical code; separate clusters mean the
    # same payload was reused at different physical locations (e.g. a chain).
    cluster_radius_m: float = 25.0
    triangulate: bool = True


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ViewParams:
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    size: int


@dataclass
class PanoMeta:
    pano_id: str
    lat: float
    lng: float
    heading_deg: float = 0.0   # compass heading of the pano's centre column (azimuth 0)


@dataclass
class Detection:
    payload: str
    pano_id: str
    pano_lat: float
    pano_lng: float
    azimuth_deg: float         # relative to pano centre (-180..180)
    elevation_deg: float       # +up
    compass_bearing_deg: float # absolute, derived from pano heading
    view: ViewParams


@dataclass
class MappedCode:
    payload: str
    lat: Optional[float]       # triangulated position, if available
    lng: Optional[float]
    observations: list[Detection] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage 1 -- equirectangular -> rectilinear perspective views
# --------------------------------------------------------------------------- #
def generate_views(cfg: Config) -> list[ViewParams]:
    yaws = np.arange(0.0, 360.0, cfg.yaw_step_deg)
    return [
        ViewParams(float(y), float(p), cfg.fov_deg, cfg.view_size)
        for y in yaws for p in cfg.pitch_deg
    ]


def equirect_to_perspective(equ: np.ndarray, view: ViewParams):
    """
    Render one rectilinear view from an equirectangular pano.

    Returns (image, azimuth_map, elevation_map) where the two maps give, for
    every output pixel, the direction it samples expressed in the pano frame:
        azimuth   = degrees left/right of pano centre, range (-180, 180]
        elevation = degrees above the horizon (+up)
    These let us recover a code's bearing from its pixel location later.
    """
    H, W = equ.shape[:2]
    out = view.size
    wFOV = view.fov_deg
    hFOV = wFOV  # square view => equal h/v FOV
    w_len = math.tan(math.radians(wFOV / 2.0))
    h_len = math.tan(math.radians(hFOV / 2.0))

    # Camera looks along +x; y is to the left, z is up (Equirec2Perspec convention).
    x_map = np.ones((out, out), np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, out), (out, 1)).astype(np.float32)
    z_map = -np.tile(np.linspace(-h_len, h_len, out), (out, 1)).T.astype(np.float32)

    norm = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / norm[:, :, None]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R_yaw, _ = cv2.Rodrigues(z_axis * math.radians(view.yaw_deg))
    R_pitch, _ = cv2.Rodrigues(np.dot(R_yaw, y_axis) * math.radians(-view.pitch_deg))

    xyz = xyz.reshape((-1, 3)).T
    xyz = np.dot(R_pitch, np.dot(R_yaw, xyz)).T

    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lon_deg = np.degrees(lon).reshape((out, out))
    lat_deg = -np.degrees(lat).reshape((out, out))  # +down, matches image y

    map_x = (lon_deg / 180.0 * (W / 2.0) + W / 2.0).astype(np.float32)
    map_y = (lat_deg / 90.0 * (H / 2.0) + H / 2.0).astype(np.float32)
    img = cv2.remap(equ, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    azimuth_map = lon_deg                 # already pano-relative
    elevation_map = -lat_deg              # +up
    return img, azimuth_map, elevation_map


# --------------------------------------------------------------------------- #
# Stage 2 -- tiling
# --------------------------------------------------------------------------- #
def tile_positions(size: int, tile: int, overlap: float) -> list[int]:
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, size - tile + 1), step))
    if xs[-1] != size - tile:           # ensure the far edge is covered
        xs.append(size - tile)
    return xs


def iter_tiles(img: np.ndarray, cfg: Config) -> Iterator[tuple[int, int, np.ndarray]]:
    h, w = img.shape[:2]
    tile = min(cfg.tile, h, w)
    for y0 in tile_positions(h, tile, cfg.tile_overlap):
        for x0 in tile_positions(w, tile, cfg.tile_overlap):
            yield x0, y0, img[y0:y0 + tile, x0:x0 + tile]


# --------------------------------------------------------------------------- #
# Box helpers (view-coordinate axis-aligned boxes)
# --------------------------------------------------------------------------- #
def _aabb(box: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = box.min(0)
    x1, y1 = box.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def _center(box: np.ndarray) -> tuple[float, float]:
    c = box.mean(0)
    return float(c[0]), float(c[1])


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(aabbs: list[tuple], iou_thresh: float) -> list[tuple]:
    """Greedily union overlapping axis-aligned boxes, iterated to a fixed point."""
    merged = list(aabbs)
    changed = True
    while changed:
        changed = False
        out: list[tuple] = []
        for box in merged:
            for i, m in enumerate(out):
                if _iou(box, m) > iou_thresh:
                    out[i] = (min(box[0], m[0]), min(box[1], m[1]),
                              max(box[2], m[2]), max(box[3], m[3]))
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


# --------------------------------------------------------------------------- #
# Stages 3 & 4 -- multi-scale detection + ensemble decoding
# --------------------------------------------------------------------------- #
class QRScanner:
    """Localises and decodes QR codes in a single perspective view."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.wechat = self._load_wechat(cfg.model_dir)

    @staticmethod
    def _load_wechat(model_dir: str):
        f = lambda n: os.path.join(model_dir, n)
        files = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
        if not all(os.path.exists(f(n)) for n in files):
            print(f"[warn] WeChat model files missing in {model_dir!r}; "
                  f"falling back to ensemble decoders only.")
            return None
        return cv2.wechat_qrcode.WeChatQRCode(
            f("detect.prototxt"), f("detect.caffemodel"),
            f("sr.prototxt"), f("sr.caffemodel"),
        )

    # ---- decoders: each returns list[(text_or_None, box Nx2 float | None)] --- #
    def _wechat_detect(self, img: np.ndarray):
        """WeChat detector: returns boxes, with decoded text where available."""
        if self.wechat is None:
            return []
        try:
            texts, pts = self.wechat.detectAndDecode(img)
        except Exception:
            return []
        out = []
        for txt, box in itertools.zip_longest(list(texts), list(pts), fillvalue=None):
            b = np.asarray(box, np.float32).reshape(-1, 2) if box is not None else None
            out.append((txt if txt else None, b))
        return out

    @staticmethod
    def _zxing(img: np.ndarray):
        if not _HAS_ZXING:
            return []
        out = []
        try:
            for r in zxingcpp.read_barcodes(img):
                if not r.text:
                    continue
                box = None
                try:
                    p = r.position
                    box = np.array([[p.top_left.x, p.top_left.y],
                                    [p.top_right.x, p.top_right.y],
                                    [p.bottom_right.x, p.bottom_right.y],
                                    [p.bottom_left.x, p.bottom_left.y]], np.float32)
                except Exception:
                    box = None
                out.append((r.text, box))
        except Exception:
            pass
        return out

    @staticmethod
    def _pyzbar(img: np.ndarray):
        if not _HAS_PYZBAR:
            return []
        out = []
        try:
            for s in _pyzbar_decode(img):
                if s.type != "QRCODE" or not s.data:
                    continue
                text = s.data.decode("utf-8", "replace")
                if getattr(s, "polygon", None):
                    box = np.array([[pt.x, pt.y] for pt in s.polygon], np.float32)
                else:
                    r = s.rect
                    box = np.array([[r.left, r.top], [r.left + r.width, r.top],
                                    [r.left + r.width, r.top + r.height],
                                    [r.left, r.top + r.height]], np.float32)
                out.append((text, box))
        except Exception:
            pass
        return out

    def _software_decode(self, img: np.ndarray):
        """Pure-software decoders scan real pixels (no CNN input resize)."""
        return self._zxing(img) + self._pyzbar(img)

    # ---- preprocessing variants for the "decode globally" retry ------------- #
    def _variants(self, crop: np.ndarray) -> Iterator[np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        for scale in self.cfg.upscale_factors:
            g = (cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC) if scale != 1.0 else gray)
            variants = [g]
            try:
                variants.append(cv2.threshold(
                    g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
                m = min(g.shape[:2])
                bs = max(3, min(31, m if m % 2 == 1 else m - 1))
                variants.append(cv2.adaptiveThreshold(
                    g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 5))
            except Exception:
                pass
            for base in variants:
                for rot in self.cfg.rotations_deg:
                    if rot == 0:
                        yield base
                    else:
                        k = {90: cv2.ROTATE_90_CLOCKWISE,
                             180: cv2.ROTATE_180,
                             270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
                        yield cv2.rotate(base, k)

    def _ensemble_decode(self, crop: np.ndarray) -> set[str]:
        results: set[str] = set()
        for v in self._variants(crop):
            for text, _ in self._software_decode(v):
                results.add(text)
            if results:                  # one good decode is enough for this crop
                break
        return results

    # ---- view entry point --------------------------------------------------- #
    def scan_view(self, view_img: np.ndarray) -> list[tuple[str, tuple[float, float]]]:
        """
        Scan one perspective view at two scales.
        Returns list of (payload, (cx, cy)) in VIEW pixel coordinates.
        """
        H, W = view_img.shape[:2]
        results: dict[str, tuple[float, float]] = {}
        undecoded: list[tuple] = []      # aabb boxes (view coords) needing a decode

        def collect(img, ox, oy):
            # WeChat: boxes (+ free decodes), plus boxes it couldn't read.
            for text, box in self._wechat_detect(img):
                b = None
                if box is not None:
                    b = box.copy()
                    b[:, 0] += ox
                    b[:, 1] += oy
                if text:
                    c = _center(b) if b is not None else (ox + img.shape[1] / 2,
                                                          oy + img.shape[0] / 2)
                    results.setdefault(text, c)
                elif b is not None:
                    undecoded.append(_aabb(b))
            # Software decoders on this scale (local binarisation can win on tiles).
            for text, box in self._software_decode(img):
                if box is not None:
                    b = box.copy()
                    b[:, 0] += ox
                    b[:, 1] += oy
                    c = _center(b)
                else:
                    c = (ox + img.shape[1] / 2, oy + img.shape[0] / 2)
                results.setdefault(text, c)

        # Scale 0: whole view -> owns big / billboard-sized codes.
        collect(view_img, 0, 0)

        # Scale 1: tiles -> owns small / distant codes.
        if self.cfg.tile:
            for x0, y0, tile in iter_tiles(view_img, self.cfg):
                collect(tile, x0, y0)

        # Decode globally: crop each merged undecoded box from the FULL-RES view
        # (seams ignored, since boxes are already in view coords) and retry.
        for (x0, y0, x1, y1) in merge_boxes(undecoded, self.cfg.box_merge_iou):
            pad = max(self.cfg.decode_pad_px, self.cfg.decode_pad_frac * max(x1 - x0, y1 - y0))
            cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
            if cx1 - cx0 < self.cfg.min_crop_px or cy1 - cy0 < self.cfg.min_crop_px:
                continue
            crop = view_img[cy0:cy1, cx0:cx1]
            for payload in self._ensemble_decode(crop):
                results.setdefault(payload, ((cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0))

        return list(results.items())


# --------------------------------------------------------------------------- #
# Stage 5 -- geometry: pixel -> bearing, triangulate
# --------------------------------------------------------------------------- #
def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def detections_from_view(
    hits: list[tuple[str, tuple[float, float]]],
    az_map: np.ndarray, el_map: np.ndarray,
    view: ViewParams, pano: PanoMeta,
) -> list[Detection]:
    out = []
    H, W = az_map.shape
    for payload, (cx, cy) in hits:
        px = int(np.clip(round(cx), 0, W - 1))
        py = int(np.clip(round(cy), 0, H - 1))
        az = float(az_map[py, px])
        el = float(el_map[py, px])
        bearing = (pano.heading_deg + az) % 360.0
        out.append(Detection(
            payload=payload, pano_id=pano.pano_id,
            pano_lat=pano.lat, pano_lng=pano.lng,
            azimuth_deg=az, elevation_deg=el,
            compass_bearing_deg=bearing, view=view,
        ))
    return out


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_intersection(lat1, lon1, brng1, lat2, lon2, brng2):
    """Great-circle intersection of two (point, bearing) rays. (lat, lon) or None."""
    f1, l1, f2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    t13, t23 = math.radians(brng1), math.radians(brng2)
    df, dl = f2 - f1, l2 - l1
    d12 = 2 * math.asin(math.sqrt(math.sin(df / 2) ** 2 +
                                  math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2))
    if d12 == 0:
        return None
    cosTa = (math.sin(f2) - math.sin(f1) * math.cos(d12)) / (math.sin(d12) * math.cos(f1))
    cosTb = (math.sin(f1) - math.sin(f2) * math.cos(d12)) / (math.sin(d12) * math.cos(f2))
    tA = math.acos(max(-1.0, min(1.0, cosTa)))
    tB = math.acos(max(-1.0, min(1.0, cosTb)))
    if math.sin(l2 - l1) > 0:
        t12, t21 = tA, 2 * math.pi - tB
    else:
        t12, t21 = 2 * math.pi - tA, tB
    a1, a2 = t13 - t12, t21 - t23
    if math.sin(a1) == 0 and math.sin(a2) == 0:
        return None
    if math.sin(a1) * math.sin(a2) < 0:
        return None
    a3 = math.acos(-math.cos(a1) * math.cos(a2) +
                   math.sin(a1) * math.sin(a2) * math.cos(d12))
    d13 = math.atan2(math.sin(d12) * math.sin(a1) * math.sin(a2),
                     math.cos(a2) + math.cos(a1) * math.cos(a3))
    f3 = math.asin(max(-1.0, min(1.0,
                   math.sin(f1) * math.cos(d13) + math.cos(f1) * math.sin(d13) * math.cos(t13))))
    dl13 = math.atan2(math.sin(t13) * math.sin(d13) * math.cos(f1),
                      math.cos(d13) - math.sin(f1) * math.sin(f3))
    l3 = l1 + dl13
    return math.degrees(f3), _wrap180(math.degrees(l3))


def _circular_mean_deg(angles: list[float]) -> float:
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c)) % 360.0


def consolidate(detections: Iterable[Detection], cfg: Config) -> list[MappedCode]:
    """
    Group detections by payload and resolve physical positions.

    The same payload may be (a) the same physical code seen from several panos,
    or (b) the same string reused on different signs.  We don't use pano
    proximity to decide -- we triangulate every pair of viewpoints and cluster
    the resulting intersection POINTS.  Each tight cluster of intersections is
    one physical code; well-separated clusters are distinct codes that happen to
    share a payload.
    """
    by_payload: dict[str, list[Detection]] = {}
    for d in detections:
        by_payload.setdefault(d.payload, []).append(d)

    codes: list[MappedCode] = []
    for payload, obs in by_payload.items():
        # One representative viewpoint per pano (overlapping views/tiles of the
        # same pano give near-identical bearings -> collapse them).
        by_pano: dict[str, list[Detection]] = {}
        for d in obs:
            by_pano.setdefault(d.pano_id, []).append(d)
        viewpoints = []
        for pid, group in by_pano.items():
            brg = _circular_mean_deg([d.compass_bearing_deg for d in group])
            viewpoints.append((group[0].pano_lat, group[0].pano_lng, brg, group))

        if not cfg.triangulate or len(viewpoints) < 2:
            codes.append(MappedCode(payload, None, None, observations=list(obs)))
            continue

        # All pairwise bearing intersections.
        inter: list[tuple[float, float, int, int]] = []
        for (i, a), (j, b) in itertools.combinations(enumerate(viewpoints), 2):
            r = bearing_intersection(a[0], a[1], a[2], b[0], b[1], b[2])
            if r is not None:
                inter.append((r[0], r[1], i, j))

        if not inter:                       # parallel / divergent rays
            codes.append(MappedCode(payload, None, None, observations=list(obs)))
            continue

        # Cluster the intersection points by proximity.
        clusters: list[list[tuple[float, float, int, int]]] = []
        for pt in inter:
            for cl in clusters:
                if _haversine_m(cl[0][0], cl[0][1], pt[0], pt[1]) <= cfg.cluster_radius_m:
                    cl.append(pt)
                    break
            else:
                clusters.append([pt])

        for cl in clusters:
            lat = float(np.median([p[0] for p in cl]))
            lng = float(np.median([p[1] for p in cl]))
            idxs = {p[2] for p in cl} | {p[3] for p in cl}
            cl_obs = [d for k in idxs for d in viewpoints[k][3]]
            codes.append(MappedCode(payload, lat, lng, observations=cl_obs))
    return codes


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_panorama(equ: np.ndarray, pano: PanoMeta,
                     scanner: QRScanner, cfg: Config) -> list[Detection]:
    detections: list[Detection] = []
    for view in generate_views(cfg):
        img, az_map, el_map = equirect_to_perspective(equ, view)
        hits = scanner.scan_view(img)
        if hits:
            detections += detections_from_view(hits, az_map, el_map, view, pano)
    return detections


def process_dataset(panos: Iterable[tuple[np.ndarray, PanoMeta]],
                    cfg: Config) -> list[MappedCode]:
    scanner = QRScanner(cfg)
    all_dets: list[Detection] = []
    for equ, pano in panos:
        all_dets += process_panorama(equ, pano, scanner, cfg)
    return consolidate(all_dets, cfg)


def to_geojson(codes: list[MappedCode]) -> dict:
    feats = []
    for c in codes:
        # Prefer the triangulated point; otherwise drop a marker at the first
        # observing pano so nothing is lost from the map.
        if c.lat is not None:
            lat, lng, kind = c.lat, c.lng, "triangulated"
        else:
            o = c.observations[0]
            lat, lng, kind = o.pano_lat, o.pano_lng, "pano_location"
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {
                "payload": c.payload,
                "position_kind": kind,
                "n_observations": len(c.observations),
                "pano_ids": sorted({o.pano_id for o in c.observations}),
                "bearings": [round(o.compass_bearing_deg, 1) for o in c.observations],
            },
        })
    return {"type": "FeatureCollection", "features": feats}


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config()

    # Replace this loader with your Street View fetch + stitch step.  The Street
    # View Image API tops out around 640x640 tiles; for full 16384x8192 you must
    # request the raw pano tiles (zoom 5) and stitch them, or use a library that
    # does.  `equ` must be a BGR uint8 array of shape (8192, 16384, 3).
    def load_panos() -> Iterator[tuple[np.ndarray, PanoMeta]]:
        for path, meta in [
            # ("pano_AAA.jpg", PanoMeta("AAA", 40.7128, -74.0060, heading_deg=12.0)),
            # ("pano_BBB.jpg", PanoMeta("BBB", 40.7129, -74.0058, heading_deg=95.0)),
        ]:
            img = cv2.imread(path)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (16384, 8192):
                img = cv2.resize(img, (16384, 8192), interpolation=cv2.INTER_AREA)
            yield img, meta

    codes = process_dataset(load_panos(), cfg)
    print(f"Found {len(codes)} distinct QR payload(s).")
    with open("qr_codes.geojson", "w") as fh:
        json.dump(to_geojson(codes), fh, indent=2)
    print("Wrote qr_codes.geojson")