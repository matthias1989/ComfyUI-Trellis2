"""
HairDetectionUtils — debug-region painting + hairline detection.  VERSION 2

Changes from v1:

  1. Post-Canny filters on cyan (_detect_hairline_canny):
       Y-band floor  — generous, only trims absolute bottom of nape.
       Z upper-bound — removes face-interior UV-seam noise at high Z.
     Together they replace the blunt Y-only filter that clipped temple edges.

  2. Neck-region filter (_neck_filter / _neck_filter_idx):
     Applied to ALL detection layers (cyan, yellow, green) after each step.
     Suppresses any pixel that lies in the neck zone (world space):
       Y  ∈ [T_seam_floor − NECK_Y_BELOW,  T_seam_floor + NECK_Y_ABOVE]
       Z  > NECK_Z_FLOOR   (forward-facing only; Z≤0 back-hair is safe)
       |X| < x_half × NECK_X_HALF_FRAC  (central, not temples/ears)
     The neck is never hair — anything detected there is UV-seam noise.

Versioning:
  hair_detection_utils.py    — v1: original, no post-Canny filter
  hair_detection_utils_v2.py — v2: Y+Z filter + neck-region filter (this file)

Usage (swap import in texture_projection_multiview.py):
  from .hair_detection_utils_v2 import HairDetectionUtils
"""

import numpy as np
import cv2
import time as _time
import scipy.ndimage  as _spndimage
import scipy.spatial  as _spspatial

# ── v2 tuning ────────────────────────────────────────────────────────────────
# Y-band filter: floor = T_hair - CYAN_Y_MARGIN * (T_hair - T_seam_floor)
# 0.95 = floor at 0.377, only 0.005 above T_seam_floor (nape absolute bottom).
# Essentially the full head zone — lets the temple/ear hairline go as low as
# needed.  The Z filter is the real discriminator against neck noise.
# Lower = tighter (clips temples); raise to 1.0 to disable Y filter entirely.
CYAN_Y_MARGIN = 0.45

# Z upper-bound filter: keep edges where Z < Z_max * CYAN_Z_POS_FRAC.
# The real hairline is near Z=0; face-interior UV-seam noise is at high Z.
# 0.42 = keep only edges in the lower 42% of the forward Z range (~Z < 0.029).
# Removes more face-cap interior fingerprint dots and some neck UV-seam edges
# that sit in the 0.029–0.035 band.
# Lower = stricter; higher = more permissive (may reintroduce face noise).
CYAN_Z_POS_FRAC = 0.72

# ── Neck-region filter ────────────────────────────────────────────────────────
# Applied to ALL detection layers (cyan, yellow, green) after each step.
# The neck is never hair — any edge detected there is UV-seam noise.
#
# Neck zone (world space):
#   Y  ∈ [T_seam_floor − NECK_Y_BELOW,  T_seam_floor + NECK_Y_ABOVE]
#   Z  > NECK_Z_FLOOR   (forward-facing; back of head is Z≤0 and is safe)
#   |X| < x_half × NECK_X_HALF_FRAC  (central strip, not temples/ears)
#
# NECK_Y_BELOW : world-Y to extend below T_seam_floor (upper neck / nape).
#   Increase if neck UV-seam noise appears in the torso zone below the head.
# NECK_Y_ABOVE : world-Y above T_seam_floor (into head zone) to suppress.
#   Increase if forward-facing strays just above the head-body boundary survive.
# NECK_Z_FLOOR : minimum Z to suppress.  0.0 = forward-facing only.
#   Raise (e.g. 0.02) to be stricter (only very front-facing); keep at 0.0
#   to catch everything in front of the Z=0 plane.
# NECK_X_HALF_FRAC : neck half-width as a fraction of the head's X half-extent.
#   The neck is narrower than the head; 0.60 leaves temples/ears untouched.
#   Lower to narrow the suppression zone; raise if side-neck strays survive.
NECK_Y_BELOW     = 0.06
NECK_Y_ABOVE     = 0.04
NECK_Z_FLOOR     = 0.0
NECK_X_HALF_FRAC = 0.60

# ── Region catalogue ─────────────────────────────────────────────────────────
_REGION_NAMES  = ["foot", "lower_leg", "upper_leg", "torso_back",
                   "torso_front", "arms", "head", "hair"]
_REGION_COLORS = [
    (255,   0, 150),   # 0 foot        — pink
    (  0, 220, 220),   # 1 lower_leg   — cyan
    (200,   0, 255),   # 2 upper_leg   — magenta
    (180, 100,   0),   # 3 torso_back  — brown
    (255,   0,   0),   # 4 torso_front — red
    (  0, 200,   0),   # 5 arms        — green
    (255, 120,   0),   # 6 head/face   — orange
    (  0, 160, 255),   # 7 hair        — sky-blue
]


class HairDetectionUtils:
    """
    Coordinate-based region labelling + multi-layer hairline detection.

    Key insight:
      Face/neck skin normals point forward (+Z) → normal.Z ≈ +0.5 to +0.9 (HIGH).
      Hair geometry on crown / sides / back faces upward or sideways / backward →
        normal.Z ≈ 0 to –0.5 (LOW).
      Nose and eye-socket normals are also mostly forward-pointing → they stay
      as "face", not mis-classified as hair.

    Detection layers:
      Cyan    — Z-Canny on UV-space Z-grayscale (front hairline, best known
                params — do NOT change: blur=(5,5) σ2, Canny 5/15, erosion 12px)
      Yellow  — Normal-Y Canny filtered to back/sides (geometry edges, no UV
                seam fingerprint noise)
      Green   — hanging / floating hair (body zone + head zone dual-Z + back cap
                rim)
      Magenta — outer contour of the combined hair region (hairline border,
                21-px MORPH_CLOSE to bridge the fragmentary UV-space gaps)

    See NOTES.md for full research history (10 approaches, what failed and why).
    """

    def __init__(self, texture_size: int, uv_mask: np.ndarray,
                 tex_pos_np: np.ndarray, tex_normals_np: np.ndarray,
                 neck_y_min: float = 0.30,
                 neck_y_max: float = 0.42,
                 cyan_y_margin: float = CYAN_Y_MARGIN,
                 cyan_z_frac:   float = CYAN_Z_POS_FRAC):
        """
        Parameters
        ----------
        texture_size   : atlas resolution (e.g. 4096)
        uv_mask        : bool (H, W)  — True inside UV islands
        tex_pos_np     : float32 (H, W, 3) — world-space position per texel
        tex_normals_np : float32 (H, W, 3) — world-space smooth normal per texel
        neck_y_min     : world-Y bottom of neck suppression zone (absolute, norm ±0.575)
        neck_y_max     : world-Y top of neck suppression zone (absolute)
                         Forward-facing pixels (Z > 0) in [neck_y_min, neck_y_max] are suppressed.
        cyan_y_margin  : Y-band floor for cyan Canny edges
                         floor = T_hair − margin × (T_hair − T_seam_floor)
        cyan_z_frac    : Z upper-bound for cyan edges (keep where Z < Z_max × frac)
        """
        self.texture_size   = texture_size
        self.uv_mask        = uv_mask
        self.tex_pos_np     = tex_pos_np
        self.tex_normals_np = tex_normals_np

        # Tunable filter params (override module-level defaults via ComfyUI node)
        self._neck_y_min    = float(neck_y_min)
        self._neck_y_max    = float(neck_y_max)
        self._cyan_y_margin = float(cyan_y_margin)
        self._cyan_z_frac   = float(cyan_z_frac)

        # Gather all UV-island pixels once
        self._uv_y, self._uv_x = np.where(uv_mask)
        pos3d = tex_pos_np[self._uv_y, self._uv_x]
        self.p_y = pos3d[:, 1]
        self.p_x = pos3d[:, 0]
        self.p_z = pos3d[:, 2]

        # Mesh extents
        self._y_max  = float(self.p_y.max())
        self._y_min  = float(self.p_y.min())
        self._y_rng  = self._y_max - self._y_min
        self._x_half = max(float(np.abs(self.p_x).max()), 1e-6)

    # =========================================================================
    # Public entry point
    # =========================================================================
    def run(self, color_np: np.ndarray = None) -> np.ndarray:
        """
        Build and return the debug canvas (texture_size × texture_size × 3, uint8).

        color_np — optional projected colour array used only for per-region
                   average-colour logging.  Pass None to skip that log.
        """
        tex  = self.texture_size
        uv_y = self._uv_y;  uv_x = self._uv_x

        # ── Thresholds ───────────────────────────────────────────────────────
        T = self._compute_thresholds()

        # ── Base labels ──────────────────────────────────────────────────────
        labels = self._compute_labels(T)

        # ── Seam detection (coord-based face/hair split) ─────────────────────
        _hi, seam_pxmap, T = self._detect_hair_seam(labels, T)
        labels[_hi[self._is_hair[:]]]  = 7
        labels[_hi[~self._is_hair[:]]] = 6
        # Re-apply arms override
        labels[np.abs(self.p_x) > T['x_arm']] = 5

        # ── Paint base regions (log only — no region colors, hair detection only) ──
        debug_canvas = np.zeros((tex, tex, 3), dtype=np.uint8)
        _uv_pixels = color_np[uv_y, uv_x].astype(np.float32) if color_np is not None \
                     else np.zeros((len(uv_y), 3), dtype=np.float32)
        print("[DebugRegions] Region summary:")
        for ri, (rname, rcolor) in enumerate(zip(_REGION_NAMES, _REGION_COLORS)):
            _mask_r = (labels == ri)
            _avg = _uv_pixels[_mask_r].mean(axis=0) if _mask_r.any() else np.zeros(3)
            print(f"  [{ri}] {rname:12s}: {rcolor}  "
                  f"avg_RGB=({_avg[0]:.0f},{_avg[1]:.0f},{_avg[2]:.0f})  "
                  f"px={int(_mask_r.sum()):,}")

        # ── Hairline detection (requires enough head-zone pixels) ────────────
        if len(_hi) >= 100:
            cy_edges, yl_edges, _hl_safe_hi, _hl_excl_hi = \
                self._detect_hairline_canny(_hi, T, debug_canvas)

            self._detect_hanging_hair(debug_canvas, T)
            self._detect_back_hair_rim(debug_canvas, T)
            self._detect_back_braid(debug_canvas, T)

            self._paint_seed(debug_canvas, _hi, T)
            self._draw_hair_border(debug_canvas, cy_edges, _hl_safe_hi, T)
            self._draw_back_rim_border(debug_canvas, T)

        else:
            print("  [SeamWalk] Too few head-zone texels — keeping Y-threshold only")

        return debug_canvas

    # =========================================================================
    # Hair mask — silent detection for downstream fill passes
    # =========================================================================
    def get_hair_mask(self) -> np.ndarray:
        """
        Returns bool (H×W) mask of all hair UV pixels. Silent — no canvas painting.

        Cap boundary: hair_cap_mask — the FILLED CONTOUR from _draw_hair_border,
        derived from the same Canny hairline edges that produce the magenta debug
        border.  Using this directly (not intersected with seam-walk pixels) ensures:
          1. The hairline matches the magenta border exactly — no over-extension
             onto the forehead from the seam-walk's fuzzy classification.
          2. The full cap interior is filled — no gaps from missing forward-facing
             pixels that occurred when the nz<0.3 filter was active.
        Braid pixels are appended as before (3D-classified, body zone).
        """
        import io, contextlib
        tex = self.texture_size
        dummy = np.zeros((tex, tex, 3), dtype=np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            T      = self._compute_thresholds()
            labels = self._compute_labels(T)
            _hi, _, T = self._detect_hair_seam(labels, T)
            if len(_hi) >= 100:
                cy_edges, yl_edges, _hl_safe_hi, _hl_excl_hi = \
                    self._detect_hairline_canny(_hi, T, dummy)
                self._detect_back_braid(dummy, T)
                self._draw_hair_border(dummy, cy_edges, _hl_safe_hi, T)

        mask = np.zeros((tex, tex), dtype=bool)

        # Primary: Canny contour fill — exact same boundary as magenta debug border
        hair_cap_mask = T.get('hair_cap_mask')
        if hair_cap_mask is not None:
            mask = hair_cap_mask.copy()
        else:
            # Fallback if Canny chain failed: seam-walk result
            if len(_hi) > 0:
                hair_hi = _hi[self._is_hair]
                mask[self._uv_y[hair_hi], self._uv_x[hair_hi]] = True

        # Braid / ponytail (3D-classified, hangs in body zone)
        braid_idx = T.get('braid_idx')
        if braid_idx is not None and len(braid_idx) > 0:
            mask[self._uv_y[braid_idx], self._uv_x[braid_idx]] = True

        return mask

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _compute_thresholds(self) -> dict:
        y_max = self._y_max;  y_min = self._y_min;  y_rng = self._y_rng
        T_head    = y_max - 0.28 * y_rng
        T_hair    = y_max - 0.20 * (y_max - T_head)
        T_torso_b = y_min + 0.42 * y_rng
        T_knee    = y_min + 0.22 * y_rng
        T_foot    = y_min + 0.07 * y_rng
        T_x_arm   = self._x_half * 0.38
        T_seam_floor = T_head + 0.60 * (T_hair - T_head)
        print(f"  [Thresholds] Y: hair>{T_hair:.3f}  head>{T_head:.3f}  "
              f"torso>{T_torso_b:.3f}  knee>{T_knee:.3f}  foot>{T_foot:.3f}  "
              f"|X|arm>{T_x_arm:.3f}  seam_floor>{T_seam_floor:.3f}")
        return dict(head=T_head, hair=T_hair, torso_b=T_torso_b,
                    knee=T_knee, foot=T_foot, x_arm=T_x_arm,
                    seam_floor=T_seam_floor)

    def _compute_labels(self, T: dict) -> np.ndarray:
        p_y = self.p_y;  p_z = self.p_z;  p_x = self.p_x
        labels = np.zeros(len(p_y), dtype=np.uint8)
        labels[p_y > T['foot']]                      = 1
        labels[p_y > T['knee']]                      = 2
        labels[p_y > T['torso_b']]                   = 3
        labels[(p_y > T['torso_b']) & (p_z > 0.01)] = 4
        labels[p_y > T['head']]                      = 6
        labels[np.abs(p_x) > T['x_arm']]             = 5
        return labels

    def _detect_hair_seam(self, labels, T: dict):
        """
        Coordinate-based face/hair split inside the head zone.
        Sets self._is_hair (bool array over _hi indices).
        Returns (_hi, seam_pxmap, T) — T updated with head_seam thresholds.
        """
        p_y = self.p_y;  p_x = self.p_x;  p_z = self.p_z
        uv_y = self._uv_y;  uv_x = self._uv_x
        tex  = self.texture_size
        cKDTree = _spspatial.cKDTree
        t0 = _time.perf_counter()

        T_seam_floor = T['seam_floor']
        T_hair       = T['hair']
        head_mask    = (p_y > T_seam_floor) & (np.abs(p_x) <= T['x_arm'])
        _hi          = np.where(head_mask)[0]

        seam_pxmap   = None
        self._is_hair = np.zeros(len(_hi), dtype=bool)   # populated below

        if len(_hi) < 100:
            return _hi, seam_pxmap, T

        _hy = p_y[_hi];  _hx = p_x[_hi];  _hz = p_z[_hi]

        # Diagnostic: frontmost / topmost / chin-area normals
        _hn = self.tex_normals_np[uv_y[_hi], uv_x[_hi]]
        _f50 = np.argsort(_hz)[-50:]
        _t50 = np.argsort(_hy)[-50:]
        print(f"  [SeamDiag] Frontmost-50 (face?) avg:"
              f"  p_z={float(_hz[_f50].mean()):.3f}"
              f"  nz={float(_hn[_f50, 2].mean()):.3f}")
        print(f"  [SeamDiag] Topmost-50   (crown?) avg:"
              f"  p_y={float(_hy[_t50].mean()):.3f}"
              f"  nz={float(_hn[_t50, 2].mean()):.3f}")

        # Coord-based split: Z≤0 → hair; Y>T_hair → hair; top 4% Y → crown
        T_crown = self._y_max - 0.04 * self._y_rng
        is_hair = (_hz <= 0.0) | (_hy > T_hair)
        is_hair[_hy > T_crown] = True
        self._is_hair = is_hair

        n_h = int(is_hair.sum())
        print(f"  [SeamWalk] coord-based  T_hair={T_hair:.3f}  T_crown={T_crown:.3f}")
        print(f"  [SeamWalk] {len(_hi)-n_h:,} face px  {n_h:,} hair px "
              f"({_time.perf_counter()-t0:.1f}s)")

        # 3-D seam: face pixels adjacent to hair pixels
        seam_pxmap = np.zeros((tex, tex), dtype=bool)
        face_local = np.where(~is_hair)[0]
        hair_local = np.where( is_hair)[0]
        if len(face_local) > 0 and len(hair_local) > 0:
            pts3d        = np.column_stack([_hx, _hy, _hz]).astype(np.float64)
            seam_bw      = (self._y_max - T_seam_floor) * 0.015
            hair_tree    = cKDTree(pts3d[hair_local])
            face_pts3d   = pts3d[face_local]
            dist, _      = hair_tree.query(face_pts3d, k=1,
                                           distance_upper_bound=seam_bw)
            is_hairline  = dist < seam_bw

            # Exclude central face interior (nose, eyes, mouth)
            _hx_rng       = float(_hx.max() - _hx.min())
            _hz_min       = float(_hz.min())
            _hz_rng       = max(float(_hz.max()) - _hz_min, 1e-6)
            _head_y_span  = T_hair - T_seam_floor
            excl_x_half   = 0.18 * _hx_rng
            excl_y_ceil   = T_seam_floor + 0.85 * _head_y_span
            excl_z_thresh = _hz_min + 0.70 * _hz_rng
            face_central  = (
                (np.abs(face_pts3d[:, 0]) < excl_x_half) &
                (face_pts3d[:, 1] < excl_y_ceil) &
                (face_pts3d[:, 2] > excl_z_thresh)
            )
            is_hairline &= ~face_central
            seam_hi_idx  = _hi[face_local[is_hairline]]
            seam_pxmap[uv_y[seam_hi_idx], uv_x[seam_hi_idx]] = True
            print(f"  [SeamWalk] 3-D seam: {int(seam_pxmap.sum()):,} px"
                  f"  bw={seam_bw:.4f}")

        T['crown']    = T_crown
        T['_hi']      = _hi
        T['_hx']      = _hx
        T['_hy']      = _hy
        T['_hz']      = _hz
        return _hi, seam_pxmap, T

    def _detect_hairline_canny(self, _hi, T: dict, debug_canvas: np.ndarray):
        """
        Build Z-grayscale + normal-Y images and run Canny to get:
          cyan  — Z-Canny (front hairline, UV-seam noise present but best signal)
          yellow — Normal-Y Canny filtered to back/sides

        Returns (cyan_edges, yellow_edges, safe_hi_indices, excl_hi_indices)
        """
        tex  = self.texture_size
        uv_y = self._uv_y;  uv_x = self._uv_x
        p_y  = self.p_y;    p_x  = self.p_x;  p_z = self.p_z
        T_seam_floor = T['seam_floor']
        T_hair       = T['hair']
        _hx = T['_hx'];  _hy = T['_hy'];  _hz = T['_hz']

        # ── Safe mask = head zone minus face exclusion zone ───────────────────
        _hl_hx_rng = float(_hx.max() - _hx.min())
        _hl_hz_min = float(_hz.min())
        _hl_hz_rng = max(float(_hz.max()) - _hl_hz_min, 1e-6)
        _hl_y_span = T_hair - T_seam_floor

        _hl_excl_x = 0.18 * _hl_hx_rng
        _hl_excl_y = T_seam_floor + 0.85 * _hl_y_span
        _hl_excl_z = _hl_hz_min + 0.70 * _hl_hz_rng

        _hl_excl = (
            (np.abs(_hx) < _hl_excl_x) &
            (_hy < _hl_excl_y) &
            (_hz > _hl_excl_z)
        )
        _hl_safe     = ~_hl_excl
        _hl_safe_hi  = _hi[_hl_safe]
        _hl_excl_hi  = _hi[_hl_excl]
        _hz_s        = _hz[_hl_safe]
        print(f"  [Hairline-S1] head={len(_hi):,}  "
              f"excl={int(_hl_excl.sum()):,}  "
              f"safe={int(_hl_safe.sum()):,}")

        # ── Z-grayscale (positive-Z only range) ──────────────────────────────
        _z_lo  = 0.0
        _z_hi  = float(_hz_s.max())
        _z_rng = max(_z_hi - _z_lo, 1e-6)
        _gray  = np.clip((_hz_s - _z_lo) / _z_rng * 255, 0, 255).astype(np.uint8)

        debug_canvas[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = \
            np.column_stack([_gray, _gray, _gray])
        debug_canvas[uv_y[_hl_excl_hi], uv_x[_hl_excl_hi]] = (0, 0, 0)

        # ── Interior erosion mask ─────────────────────────────────────────────
        _hl_gray_img = np.zeros((tex, tex), dtype=np.uint8)
        _hl_gray_img[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _gray

        _hl_safe_uv = np.zeros((tex, tex), dtype=bool)
        _hl_safe_uv[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = True
        _hl_interior = _spndimage.binary_erosion(_hl_safe_uv, iterations=12)

        # ── CYAN — Z-grayscale Canny (params unchanged from v1) ─────────────
        # Do not change: blur=(5,5) σ=2.0  |  Canny low=5, high=15  |  erosion=12px
        _hl_blurA  = cv2.GaussianBlur(_hl_gray_img, (5, 5), 2.0)
        _hl_edgesA = (cv2.Canny(_hl_blurA, 5, 15) > 0) & _hl_interior
        print(f"  [Hairline-S3] Canny cyan raw (blur=5, 5/15): "
              f"edge_px={int(_hl_edgesA.sum()):,}")

        # ── v2: Y-floor (generous) + Z upper-bound ───────────────────────────
        #
        # WHY two filters instead of one:
        #   A pure Y floor clips the temple hairline (which dips below T_hair)
        #   while leaving face-interior UV-seam false edges untouched.
        #   A Z upper-bound removes those face-interior edges (they live at
        #   high Z, far from the Z=0 face→hair boundary) without caring about Y.
        #   Together they cover both failure modes.
        #
        # Y floor — generous, only trims the very bottom of the nape
        _hl_cyan_y_floor = T_hair - self._cyan_y_margin * (T_hair - T_seam_floor)
        _hl_cyan_y_map   = np.zeros((tex, tex), dtype=np.float32)
        _hl_cyan_y_map[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _hy[_hl_safe]
        _hl_cyan_y_mask  = _hl_cyan_y_map >= _hl_cyan_y_floor

        # Z upper-bound — removes edges deep in the face cap (Z >> 0).
        # The real hairline is near Z=0; UV-seam noise inside the cap is at
        # Z close to Z_max (~0.07).  Threshold = Z_max * cyan_z_frac.
        _hl_cyan_z_map  = np.zeros((tex, tex), dtype=np.float32)
        _hl_cyan_z_map[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _hz[_hl_safe]
        _z_thresh       = _z_hi * self._cyan_z_frac
        _hl_cyan_z_mask = _hl_cyan_z_map < _z_thresh

        _n_raw = int(_hl_edgesA.sum())
        _hl_edgesA = _hl_edgesA & _hl_cyan_y_mask & _hl_cyan_z_mask
        print(f"  [Hairline-S3-v2] Y>={_hl_cyan_y_floor:.3f} (margin={self._cyan_y_margin}) "
              f"& Z<{_z_thresh:.3f} (frac={self._cyan_z_frac}, Z_max={_z_hi:.3f}): "
              f"{_n_raw}→{int(_hl_edgesA.sum()):,} px")

        # ── YELLOW — Normal-Y Canny filtered to back/sides ───────────────────
        _hl_norm_np = self.tex_normals_np
        _hl_ny_img  = np.zeros((tex, tex), dtype=np.float32)
        _hl_ny_img[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = \
            _hl_norm_np[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi], 1]
        _hl_ny_u8   = np.clip((_hl_ny_img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        _hl_blurY   = cv2.GaussianBlur(_hl_ny_u8, (5, 5), 2.0)
        _hl_edgesY  = (cv2.Canny(_hl_blurY, 5, 15) > 0) & _hl_interior

        _hl_wy_img2 = np.zeros((tex, tex), dtype=np.float32)
        _hl_wy_img2[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _hy[_hl_safe]
        _hl_wz_img2 = np.zeros((tex, tex), dtype=np.float32)
        _hl_wz_img2[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _hz[_hl_safe]
        _hl_wx_img2 = np.zeros((tex, tex), dtype=np.float32)
        _hl_wx_img2[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = _hx[_hl_safe]

        _hl_x_head_half = float(np.abs(_hx).max())
        _hl_ear_thresh  = _hl_x_head_half * 0.80
        T['ear_thresh'] = _hl_ear_thresh

        _hl_y_mask = (
            (_hl_wy_img2 >= float(T_seam_floor)) &
            (_hl_wz_img2 <= 0.0) &
            (np.abs(_hl_wx_img2) < _hl_ear_thresh)
        )
        _hl_edgesY = _hl_edgesY & _hl_y_mask

        # ── Paint cyan + yellow ───────────────────────────────────────────────
        debug_canvas[_hl_edgesA] = (0, 255, 255)
        debug_canvas[_hl_edgesY & ~_hl_edgesA] = (255, 255, 0)
        debug_canvas[uv_y[_hl_excl_hi], uv_x[_hl_excl_hi]] = (0, 0, 0)

        print(f"  [Hairline-S3] Normal-Y Canny yellow "
              f"(Y>={float(T_seam_floor):.3f} & Z<=0 & |X|<{_hl_ear_thresh:.3f}): "
              f"edge_px={int(_hl_edgesY.sum()):,}  "
              f"yellow_only={int((_hl_edgesY & ~_hl_edgesA).sum()):,}")

        return _hl_edgesA, _hl_edgesY, _hl_safe_hi, _hl_excl_hi

    def _detect_hanging_hair(self, debug_canvas: np.ndarray, T: dict) -> None:
        """Step 4 — floating hair in body zone (braids, ponytails)."""
        uv_y = self._uv_y;  uv_x = self._uv_x
        p_y  = self.p_y;    p_x  = self.p_x;  p_z = self.p_z
        T_seam_floor = T['seam_floor']

        _hh_Nx = 80;  _hh_Ny = 100
        _hh_y_lo = float(self._y_min)
        _hh_y_hi = float(T_seam_floor)

        # Z > 0: forward-facing only — hanging hair would be in front of the body.
        # Excluding Z ≤ 0 removes back-of-legs, back-of-feet etc. (body geometry noise).
        body_mask = (p_y >= _hh_y_lo) & (p_y < _hh_y_hi) & (p_z > 0.0)
        _hh_bi    = np.where(body_mask)[0]
        if len(_hh_bi) < 100:
            return

        _hh_by = p_y[_hh_bi];  _hh_bx = p_x[_hh_bi];  _hh_bz = p_z[_hh_bi]
        _hh_xb = np.linspace(float(p_x.min()), float(p_x.max()), _hh_Nx + 1)
        _hh_yb = np.linspace(_hh_y_lo, _hh_y_hi, _hh_Ny + 1)
        _hh_bxi = np.clip(np.searchsorted(_hh_xb[1:], _hh_bx), 0, _hh_Nx - 1)
        _hh_byi = np.clip(np.searchsorted(_hh_yb[1:], _hh_by), 0, _hh_Ny - 1)

        _hh_zs = np.zeros((_hh_Ny, _hh_Nx), dtype=np.float64)
        _hh_zn = np.zeros((_hh_Ny, _hh_Nx), dtype=np.float64)
        np.add.at(_hh_zs, (_hh_byi, _hh_bxi), _hh_bz)
        np.add.at(_hh_zn, (_hh_byi, _hh_bxi), 1.0)
        _hh_zgrid = np.where(_hh_zn > 0, _hh_zs / _hh_zn, np.nan)

        _hh_row_med = np.full(_hh_Ny, np.nan)
        _hh_row_std = np.full(_hh_Ny, np.nan)
        for _ry in range(_hh_Ny):
            _rv = _hh_zgrid[_ry, _hh_zn[_ry, :] > 0]
            if len(_rv) >= 3:
                _hh_row_med[_ry] = float(np.median(_rv))
                _hh_row_std[_ry] = float(np.std(_rv))

        _hh_med_g = _hh_row_med[:, np.newaxis]
        _hh_std_g = _hh_row_std[:, np.newaxis]
        _hh_elevated = (
            (~np.isnan(_hh_med_g)) & (_hh_zn > 0) &
            (_hh_zgrid > _hh_med_g + 1.5 * _hh_std_g)
        )
        _hh_elev_px  = _hh_elevated[_hh_byi, _hh_bxi]
        _hh_elev_idx = _hh_bi[_hh_elev_px]
        _hh_elev_idx = self._neck_filter_idx(_hh_elev_idx, T, "yellow-4")
        debug_canvas[uv_y[_hh_elev_idx], uv_x[_hh_elev_idx]] = (255, 255, 0)

        _hh_anchor_row = np.full(_hh_Nx, -1, dtype=int)
        for _cx in range(_hh_Nx):
            _elev_rows = np.where(_hh_elevated[:, _cx])[0]
            if len(_elev_rows) > 0:
                _hh_anchor_row[_cx] = int(_elev_rows[0])

        _hh_valid = _hh_anchor_row >= 0
        if int(_hh_valid.sum()) >= 3:
            _med = int(np.median(_hh_anchor_row[_hh_valid]))
            _hh_valid &= (np.abs(_hh_anchor_row - _med) <= 20)

        _n_cyan = 0
        for _cx in range(_hh_Nx):
            if not _hh_valid[_cx]:
                continue
            _cell_px  = (_hh_byi == _hh_anchor_row[_cx]) & (_hh_bxi == _cx) & _hh_elev_px
            _cell_idx = _hh_bi[_cell_px]
            _cell_idx = self._neck_filter_idx(_cell_idx, T)
            if len(_cell_idx) > 0:
                debug_canvas[uv_y[_cell_idx], uv_x[_cell_idx]] = (0, 255, 255)
                _n_cyan += len(_cell_idx)
        print(f"  [HangHair-S4] elevated_cells={int(_hh_elevated.sum())}  cyan_anchors={_n_cyan}")

        # ── Step 4b: dual-Z in head zone (hanging hair side boundary) ────────
        T_hair = T['hair'];  ear_thresh = T.get('ear_thresh', self._x_half * 0.8)
        _hf_Nx = 160;  _hf_Ny = 160
        _hf_y_lo = float(T['seam_floor']); _hf_y_hi = float(T_hair)

        _hf_face_mask = (
            (p_y >= _hf_y_lo) & (p_y < _hf_y_hi) &
            (np.abs(p_x) < ear_thresh) &
            (p_z > 0.0)
        )
        _hf_fi = np.where(_hf_face_mask)[0]
        print(f"  [HangHair-S4b] head_zone px={len(_hf_fi):,}")

        if len(_hf_fi) >= 100:
            _hf_fy = p_y[_hf_fi]; _hf_fx = p_x[_hf_fi]; _hf_fz = p_z[_hf_fi]
            _hf_xb = np.linspace(-ear_thresh, ear_thresh, _hf_Nx + 1)
            _hf_yb = np.linspace(_hf_y_lo, _hf_y_hi, _hf_Ny + 1)
            _hf_fxi = np.clip(np.searchsorted(_hf_xb[1:], _hf_fx), 0, _hf_Nx - 1)
            _hf_fyi = np.clip(np.searchsorted(_hf_yb[1:], _hf_fy), 0, _hf_Ny - 1)

            _INF = float('inf')
            _hf_zmin = np.full((_hf_Ny, _hf_Nx),  _INF, dtype=np.float64)
            _hf_zmax = np.full((_hf_Ny, _hf_Nx), -_INF, dtype=np.float64)
            np.minimum.at(_hf_zmin, (_hf_fyi, _hf_fxi), _hf_fz)
            np.maximum.at(_hf_zmax, (_hf_fyi, _hf_fxi), _hf_fz)

            _hf_n_bot   = max(4, _hf_Ny // 40)
            _hf_bot_v   = _hf_zmax[:_hf_n_bot][_hf_zmax[:_hf_n_bot] < _INF]
            _hf_skin_c  = min(float(_hf_bot_v.max()) * 2.5, 0.018) if len(_hf_bot_v) else 0.012
            _hf_hair_fl = _hf_skin_c * 2.5

            _hf_has_px  = _hf_zmax > -_INF
            _hf_is_hair = _hf_has_px & (_hf_zmax > _hf_hair_fl)
            _hf_not_h   = ~_hf_is_hair
            _cx_arr     = np.arange(_hf_Nx, dtype=int)[np.newaxis, :]
            _cx_mid     = _hf_Nx // 2

            _hf_right_nh = np.zeros_like(_hf_not_h); _hf_right_nh[:, :-1] = _hf_not_h[:, 1:]
            _hf_left_nh  = np.zeros_like(_hf_not_h); _hf_left_nh[:,  1:]  = _hf_not_h[:, :-1]
            _hf_boundary = (
                (_hf_is_hair & (_cx_arr <  _cx_mid) & _hf_right_nh) |
                (_hf_is_hair & (_cx_arr >= _cx_mid) & _hf_left_nh)
            )
            _hf_bound_px = _hf_boundary[_hf_fyi, _hf_fxi] & (_hf_fz > _hf_hair_fl)
            _hf_elev_idx = _hf_fi[_hf_bound_px]
            print(f"  [HangHair-S4b] hair_floor={_hf_hair_fl:.4f}  "
                  f"elevated_px={len(_hf_elev_idx):,}")

            # Exclude nose/chin exclusion zone
            _hl_safe_hi = T.get('_safe_hi')
            if _hl_safe_hi is not None:
                _hf_excl_flag = np.zeros(len(p_y), dtype=bool)
                _hf_excl_flag[_hl_safe_hi] = False   # safe_hi are NOT excluded
                # re-derive excl flags from thresholds
                _hi  = T['_hi']; _hx = T['_hx']; _hy = T['_hy']; _hz = T['_hz']
                _hx_rng = float(_hx.max() - _hx.min())
                _hz_min = float(_hz.min()); _hz_rng = max(float(_hz.max()) - _hz_min, 1e-6)
                _hl_y_span = T_hair - T['seam_floor']
                _excl_flag_hi = np.zeros(len(_hi), dtype=bool)
                _excl_flag_hi[
                    (np.abs(_hx) < 0.18 * _hx_rng) &
                    (_hy < T['seam_floor'] + 0.85 * _hl_y_span) &
                    (_hz > _hz_min + 0.70 * _hz_rng)
                ] = True
                _hf_excl_flag_full = np.zeros(len(p_y), dtype=bool)
                _hf_excl_flag_full[_hi[_excl_flag_hi]] = True
                _hf_final_idx = _hf_elev_idx[~_hf_excl_flag_full[_hf_elev_idx]]
            else:
                _hf_final_idx = _hf_elev_idx

            _hf_final_idx = self._neck_filter_idx(_hf_final_idx, T, "green-4b")
            debug_canvas[uv_y[_hf_final_idx], uv_x[_hf_final_idx]] = (0, 220, 0)

    def _detect_back_hair_rim(self, debug_canvas: np.ndarray, T: dict) -> None:
        """Step 4c — bottom rim of back hair cap."""
        uv_y = self._uv_y;  uv_x = self._uv_x
        p_y  = self.p_y;    p_x  = self.p_x;  p_z = self.p_z
        ear_thresh = T.get('ear_thresh', self._x_half * 0.8)
        T_seam_floor = T['seam_floor'];  T_hair = T['hair']

        _hb_Nx = 160;  _hb_Ny = 160
        _hb_back_mask = (
            (p_y >= T_seam_floor) & (p_y < T_hair) &
            (np.abs(p_x) < ear_thresh) &
            (p_z < 0.0)
        )
        _hb_bi = np.where(_hb_back_mask)[0]
        print(f"  [BackHair-S4c] back_zone px={len(_hb_bi):,}")
        if len(_hb_bi) < 100:
            return

        _hb_by = p_y[_hb_bi]; _hb_bx = p_x[_hb_bi]
        _hb_xb = np.linspace(-ear_thresh, ear_thresh, _hb_Nx + 1)
        _hb_yb = np.linspace(float(T_seam_floor), float(T_hair), _hb_Ny + 1)
        _hb_bxi = np.clip(np.searchsorted(_hb_xb[1:], _hb_bx), 0, _hb_Nx - 1)
        _hb_byi = np.clip(np.searchsorted(_hb_yb[1:], _hb_by), 0, _hb_Ny - 1)

        _hb_count = np.zeros((_hb_Ny, _hb_Nx), dtype=np.int32)
        np.add.at(_hb_count, (_hb_byi, _hb_bxi), 1)
        _hb_has_px   = _hb_count >= 3
        _hb_has_col  = _hb_has_px.any(axis=0)
        _hb_col_bot  = np.where(_hb_has_col, np.argmax(_hb_has_px, axis=0), -1).astype(int)
        _hb_boundary = np.zeros((_hb_Ny, _hb_Nx), dtype=bool)
        _hb_vcols    = np.where(_hb_col_bot >= 0)[0]
        _hb_boundary[_hb_col_bot[_hb_vcols], _hb_vcols] = True
        _hb_bound_px = _hb_boundary[_hb_byi, _hb_bxi]
        _hb_elev_idx = _hb_bi[_hb_bound_px]
        print(f"  [BackHair-S4c] boundary_cols={int(_hb_has_col.sum())}  "
              f"elevated_px={len(_hb_elev_idx):,}")
        debug_canvas[uv_y[_hb_elev_idx], uv_x[_hb_elev_idx]] = (0, 220, 0)
        T['back_rim_idx'] = _hb_elev_idx

    def _detect_back_braid(self, debug_canvas: np.ndarray, T: dict = None) -> None:
        """Step 4d — braid/ponytail: grid detection + per-row X-extreme silhouette.

        Hybrid approach — two separate concerns handled by the right tool:

        DETECTION  (why Z_min grid, not pixel-level median):
          The pixel-level row median is too shallow: each body row is dominated
          by near-surface texels (leg backs, side torso, Z≈-0.01 to -0.03), so
          the median Z is shallow and anything deeper qualifies as "braid" —
          including the spine and leg concavities.
          The Z_min-per-cell grid is naturally self-compensating: if braid cells
          exist in a row, their Z_min is very negative and pulls the row median
          deeper, automatically raising the threshold so only pixels genuinely
          MORE negative than the braid's own depth still qualify.
          Result: stable, conservative detection — confirmed working previously.

        BOUNDARY  (why per-row X-extremes, not "all boundary-cell pixels"):
          A 3D grid boundary cell maps to hundreds of UV pixels (scattered UV
          islands), so painting all boundary-cell pixels fills the braid interior.
          Instead: take all pixels from ALL detected braid cells, then for each
          fine Y-row pick only the leftmost-X and rightmost-X pixel.  This gives
          exactly 2 UV pixels per row — a genuinely thin outline.
        """
        uv_y = self._uv_y;  uv_x = self._uv_x
        p_y  = self.p_y;    p_x  = self.p_x;  p_z = self.p_z
        T_seam_floor = T['seam_floor']

        # ── Step 1: Z<0 body-zone pixels ─────────────────────────────────────
        _hbd_back_mask = (
            (p_y >= float(self._y_min)) & (p_y < T_seam_floor) &
            (np.abs(p_x) < 0.45) & (p_z < 0.0)
        )
        _hbd_bi = np.where(_hbd_back_mask)[0]
        print(f"  [BackHair-S4d] Z<0 body px={len(_hbd_bi):,}")
        if len(_hbd_bi) < 100:
            if T is not None:
                T['braid_idx'] = np.empty(0, dtype=np.int64)
            return

        _hbd_by = p_y[_hbd_bi]; _hbd_bx = p_x[_hbd_bi]; _hbd_bz = p_z[_hbd_bi]

        # ── Step 2: Z_min grid + per-row median of Z_min ─────────────────────
        # Grid Z_min = most-negative Z per cell (0.0 = empty cell).
        # Row reference = median(Z_min across occupied X cells in that row).
        # This is more conservative than pixel-level median: braid cells have
        # very negative Z_min, which self-raises the row threshold.
        _hbd_Nx = 240;  _hbd_Ny = 240
        _hbd_xb = np.linspace(-0.45, 0.45, _hbd_Nx + 1)
        _hbd_yb = np.linspace(float(self._y_min), T_seam_floor, _hbd_Ny + 1)
        _hbd_bxi = np.clip(np.searchsorted(_hbd_xb[1:], _hbd_bx), 0, _hbd_Nx - 1)
        _hbd_byi = np.clip(np.searchsorted(_hbd_yb[1:], _hbd_by), 0, _hbd_Ny - 1)

        _hbd_zmin = np.full((_hbd_Ny, _hbd_Nx), 0.0, dtype=np.float32)
        np.minimum.at(_hbd_zmin, (_hbd_byi, _hbd_bxi), _hbd_bz.astype(np.float32))
        _hbd_has_px = _hbd_zmin < 0.0

        _hbd_row_ref = np.zeros(_hbd_Ny, dtype=np.float32)
        for _ry in range(_hbd_Ny):
            _rv = _hbd_zmin[_ry, _hbd_has_px[_ry]]
            if len(_rv) >= 4:
                _hbd_row_ref[_ry] = float(np.median(_rv))

        _hbd_row_valid = _hbd_row_ref < -0.005
        if _hbd_row_valid.any():
            print(f"  [BackHair-S4d] row_ref range "
                  f"[{float(_hbd_row_ref[_hbd_row_valid].min()):.3f}, "
                  f"{float(_hbd_row_ref[_hbd_row_valid].max()):.3f}]")

        # ── Step 3: braid CELLS ───────────────────────────────────────────────
        # Two-pass detection — keeps anchor detection strict, fills gaps safely:
        #
        # Pass 1 (ANCHOR): strict threshold → confirmed braid cells only.
        #   BRAID_EXCESS=0.02 gives ~54 clean cells with no arm/torso bleeding.
        #   These cells define WHERE the braid actually is in 3D world space.
        #
        # Pass 2 (BBOX FILL): within the padded bounding box of the anchor cells,
        #   include ALL Z<0 cells (no depth threshold inside a confirmed braid box).
        #   This fills UV-island gaps whose geometry is shallower than the anchor
        #   threshold (braid attachment / top-of-braid region).
        #   Arms and torso can't enter because they're at completely different X
        #   grid columns from the braid's tight XY bounding box.
        BRAID_EXCESS = 0.02   # anchor threshold — raise if torso bleeds in
        _hbd_braid_cells = (
            _hbd_has_px &
            _hbd_row_valid[:, np.newaxis] &
            (_hbd_zmin < _hbd_row_ref[:, np.newaxis] - BRAID_EXCESS)
        )
        n_cells = int(_hbd_braid_cells.sum())
        print(f"  [BackHair-S4d] anchor cells (excess>{BRAID_EXCESS:.3f}): {n_cells}")
        if n_cells == 0:
            if T is not None:
                T['braid_idx'] = np.empty(0, dtype=np.int64)
            return

        # ── Step 4: fill within anchor bounding box ──────────────────────────
        _anchor_rows = np.where(_hbd_braid_cells.any(axis=1))[0]
        _anchor_cols = np.where(_hbd_braid_cells.any(axis=0))[0]
        PAD_Y = 5;  PAD_X = 3
        row_lo = max(0,       int(_anchor_rows.min()) - PAD_Y)
        row_hi = min(_hbd_Ny, int(_anchor_rows.max()) + PAD_Y + 1)
        col_lo = max(0,       int(_anchor_cols.min()) - PAD_X)
        col_hi = min(_hbd_Nx, int(_anchor_cols.max()) + PAD_X + 1)

        _bbox = np.zeros((_hbd_Ny, _hbd_Nx), dtype=bool)
        _bbox[row_lo:row_hi, col_lo:col_hi] = True

        BRAID_EXCESS_FILL = 0.01
        _braid_fill = (
            _bbox &
            _hbd_has_px &
            _hbd_row_valid[:, np.newaxis] &
            (_hbd_zmin < _hbd_row_ref[:, np.newaxis] - BRAID_EXCESS_FILL)
        )

        n_fill_cells = int(_braid_fill.sum())
        print(f"  [BackHair-S4d] bbox [{row_lo},{row_hi}) × [{col_lo},{col_hi}) "
              f"excess>{BRAID_EXCESS_FILL:.3f}: {n_fill_cells} fill cells")

        _hbd_in_braid  = _braid_fill[_hbd_byi, _hbd_bxi]
        _hbd_braid_idx = _hbd_bi[_hbd_in_braid]
        n_braid_px     = len(_hbd_braid_idx)
        print(f"  [BackHair-S4d] braid region: {n_braid_px:,} UV px")

        # Paint green — same convention as other detection steps.
        # _draw_back_rim_border picks up T['braid_idx'] for the magenta border.
        if n_braid_px > 0:
            debug_canvas[uv_y[_hbd_braid_idx], uv_x[_hbd_braid_idx]] = (0, 200, 0)

        if T is not None:
            T['braid_idx'] = _hbd_braid_idx

    def _paint_seed(self, debug_canvas: np.ndarray, _hi, T: dict) -> None:
        """Purple cross at the pixel closest to (x=0, y=T_head)."""
        uv_y = self._uv_y;  uv_x = self._uv_x
        T_head = T['head']
        _sp_hx   = self.p_x[_hi];  _sp_hy = self.p_y[_hi]
        _sp_x_sc = float(_sp_hx.std()) + 1e-6
        _sp_y_sc = float(_sp_hy.std()) + 1e-6
        _sp_dist = (_sp_hx / _sp_x_sc) ** 2 + ((_sp_hy - T_head) / _sp_y_sc) ** 2
        _best    = int(np.argmin(_sp_dist))
        _seed_px = _hi[_best]
        _sy0, _sx0 = int(uv_y[_seed_px]), int(uv_x[_seed_px])
        _r   = 4
        tex  = self.texture_size
        _cy  = np.clip(np.arange(_sy0 - _r, _sy0 + _r + 1), 0, tex - 1)
        _cx  = np.clip(np.arange(_sx0 - _r, _sx0 + _r + 1), 0, tex - 1)
        debug_canvas[_cy, _sx0] = (180, 0, 255)
        debug_canvas[_sy0, _cx] = (180, 0, 255)
        print(f"  [Seed] Purple seed at UV=({_sx0},{_sy0})  "
              f"3D=({float(self.p_x[_seed_px]):.3f}, "
              f"{float(self.p_y[_seed_px]):.3f}, "
              f"{float(self.p_z[_seed_px]):.3f})")

    def _neck_filter(self, pixel_mask: np.ndarray, T: dict,
                     label: str = "") -> np.ndarray:
        """
        Suppress detected pixels that lie in the neck zone (bool UV-space mask).

        Works on a full (tex × tex) boolean array.  Pixels whose world-space
        position satisfies the neck-zone conditions are set to False.

        The neck zone never contains hair — anything firing there is UV-seam
        noise from the head-body boundary.  Z≤0 (back of head) is NOT touched,
        so the border code's back-hair anchor is unaffected.
        """
        pos_y = self.tex_pos_np[:, :, 1]
        pos_z = self.tex_pos_np[:, :, 2]
        neck_zone = (
            (pos_y >= self._neck_y_min) &
            (pos_y <= self._neck_y_max) &
            (pos_z  > 0.0)
        )
        n_before = int(pixel_mask.sum())
        filtered  = pixel_mask & ~neck_zone
        n_after   = int(filtered.sum())
        if n_before != n_after:
            print(f"  [NeckFilter] {label}: {n_before}→{n_after} px "
                  f"(-{n_before - n_after} in Y=[{self._neck_y_min:.3f},{self._neck_y_max:.3f}] Z>0)")
        return filtered

    def _neck_filter_idx(self, idx: np.ndarray, T: dict,
                         label: str = "") -> np.ndarray:
        """
        Neck filter for an index array (indices into self.p_y / p_z / p_x).

        Equivalent to _neck_filter but operates on the 1-D pixel-index arrays
        used inside _detect_hanging_hair and _detect_back_hair_rim.
        """
        if len(idx) == 0:
            return idx
        in_neck = (
            (self.p_y[idx] >= self._neck_y_min) &
            (self.p_y[idx] <= self._neck_y_max) &
            (self.p_z[idx]  > 0.0)
        )
        filtered  = idx[~in_neck]
        n_removed = len(idx) - len(filtered)
        if n_removed:
            print(f"  [NeckFilter] {label}: {len(idx)}→{len(filtered)} idx "
                  f"(-{n_removed} neck px)")
        return filtered

    def _draw_back_rim_border(self, debug_canvas: np.ndarray, T: dict) -> None:
        """
        Magenta border around the back/hanging hair in the body zone.

        braid_idx already contains 3D-boundary pixels (computed in _detect_back_braid
        in 3D grid space, not UV space).  UV-space contouring is avoided because UV
        islands can be scattered across the atlas — MORPH_CLOSE in UV space cannot
        bridge those gaps reliably.

        Instead: paint the 3D-boundary pixels directly, then dilate a few pixels in
        UV space to give the border a visible line thickness.
        """
        tex  = self.texture_size
        uv_y = self._uv_y;  uv_x = self._uv_x

        rim_idx   = T.get('back_rim_idx')
        braid_idx = T.get('braid_idx')
        n_rim     = len(rim_idx)   if rim_idx   is not None else 0
        n_braid   = len(braid_idx) if braid_idx is not None else 0
        print(f"  [BackRimBorder] rim={n_rim:,}  braid={n_braid:,}")
        if n_rim == 0 and n_braid == 0:
            return

        # Paint 3D-boundary pixels with minimal UV dilation for visibility.
        border_img = np.zeros((tex, tex), dtype=np.uint8)
        if n_rim   > 0: border_img[uv_y[rim_idx],   uv_x[rim_idx]]   = 255
        if n_braid > 0: border_img[uv_y[braid_idx], uv_x[braid_idx]] = 255

        kern     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(border_img, cv2.MORPH_GRADIENT, kern)
        debug_canvas[gradient > 0] = np.array([220, 0, 220], dtype=np.uint8)
        print(f"  [BackRimBorder] → {int((gradient > 0).sum()):,} magenta px (outline only)")

    def _draw_hair_border(self, debug_canvas: np.ndarray,
                          cyan_edges: np.ndarray, _hl_safe_hi,
                          T: dict = None) -> None:
        """
        Magenta hairline border — follows the cyan edges.

        Strategy:
          1. MORPH_CLOSE the cyan mask with a moderate kernel to bridge
             inter-fragment gaps (typical gap 5–25 px in UV space).
          2. Also close the Z≤0 back/side pixels to anchor the rear portion.
             A Y margin above T_seam_floor is applied so that nape/neck pixels
             at the very bottom of the head zone don't extend the border into
             the neck area — those pixels are neck skin, not hair cap.
          3. Merge both, find outer contours, draw as magenta.
        """
        tex  = self.texture_size
        uv_y = self._uv_y;  uv_x = self._uv_x

        # ── Head-zone UV mask — constrains CLOSE expansion to head UV islands ──
        # Without this, MORPH_CLOSE(21px) bridges into adjacent neck/body/shoulder
        # UV islands that happen to sit close in atlas space.
        _head_uv = np.zeros((tex, tex), dtype=np.uint8)
        if _hl_safe_hi is not None and len(_hl_safe_hi) > 0:
            _head_uv[uv_y[_hl_safe_hi], uv_x[_hl_safe_hi]] = 255

        # ── Bridge cyan fragments ─────────────────────────────────────────────
        canny_u8     = (cyan_edges.astype(np.uint8) * 255) & _head_uv
        bridge_kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        canny_closed = cv2.morphologyEx(canny_u8, cv2.MORPH_CLOSE, bridge_kern)
        canny_closed = canny_closed & _head_uv   # clip expansion back to head zone

        # ── Back hair (Z≤0) as anchor for rear portion ───────────────────────
        # Exclude pixels within NAPE_MARGIN above T_seam_floor: those are the
        # nape/neck boundary and pulling them in causes the border to dip into
        # the neck after MORPH_CLOSE.
        NAPE_MARGIN = 0.04
        hair_raw = np.zeros((tex, tex), dtype=np.uint8)
        if _hl_safe_hi is not None:
            pz_safe  = self.p_z[_hl_safe_hi]
            hair_idx = _hl_safe_hi[pz_safe <= 0.0]
            if T is not None:
                T_sf     = T['seam_floor']
                hair_y   = self.p_y[hair_idx]
                n_before = len(hair_idx)
                hair_idx = hair_idx[hair_y > T_sf + NAPE_MARGIN]
                print(f"  [HairBorder] nape trim (Y>{T_sf+NAPE_MARGIN:.3f}): "
                      f"{n_before:,}→{len(hair_idx):,} back-hair px (Z≤0)")
            else:
                print(f"  [HairBorder] back-hair px (Z≤0): {len(hair_idx):,}")
            hair_raw[uv_y[hair_idx], uv_x[hair_idx]] = 255

        back_closed  = cv2.morphologyEx(hair_raw, cv2.MORPH_CLOSE, bridge_kern)
        back_closed  = back_closed & _head_uv    # clip expansion back to head zone

        # ── Merge and find outer contour ─────────────────────────────────────
        combined = np.clip(
            canny_closed.astype(np.uint16) + back_closed.astype(np.uint16),
            0, 255
        ).astype(np.uint8)

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        border_img = np.zeros((tex, tex), dtype=np.uint8)
        # Also build a solid fill for the hair cap mask (used by get_hair_mask).
        # Same arc-length filter as the border — same contours, filled instead of outlined.
        hair_fill  = np.zeros((tex, tex), dtype=np.uint8)
        for cnt in cnts:
            if cv2.arcLength(cnt, closed=False) > 80:
                cv2.drawContours(border_img, [cnt], -1, 255, thickness=6)
                cv2.drawContours(hair_fill,  [cnt], -1, 255, thickness=cv2.FILLED)

        debug_canvas[border_img > 0] = np.array([220, 0, 220], dtype=np.uint8)
        print(f"  [HairBorder] {len(cnts)} contours → "
              f"{int((border_img > 0).sum()):,} magenta border px  "
              f"{int((hair_fill > 0).sum()):,} cap fill px "
              f"(cyan-close 21px + Z≤0 anchor)")

        # Store the filled cap region so get_hair_mask() can use it instead of raw Z≤0
        if T is not None:
            T['hair_cap_mask'] = (hair_fill > 0) & self.uv_mask
