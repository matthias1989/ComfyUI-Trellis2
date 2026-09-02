"""
FaceDetectionUtils — front-face region detection for KAN-9 (skin/face texture
projection misalignment).

STATUS: scaffold only. Not currently wired into texture_projection_multiview.py.
Restore the import + call sites in multiview to re-enable.

Why this exists
---------------
The multi-view projection blends N views per UV texel with weights derived from
camera-to-normal dot product. On the face, per-feature alignment differs
between views (mouth wants one offset, eyes another). Any blend that includes
side / SV3D views smears features into ghost overlays. Uniform u/v offsets
cannot reconcile this — the disagreement is per-feature, not global.

Strategy
--------
Detect face-front texels in 3D world space and emit a SOFT WEIGHT in [0, 1]
that the projection loop can use to suppress non-front view contributions in
the face region.

KNOWN ISSUE (revealed by first attempt at wiring this up)
---------------------------------------------------------
The naive (n_z > 0.7) + (top Y) + (central X) filter ALSO catches hair texels
on a forward-swept fringe — they sit high on the bbox, are central, and a
forward-swept hair strand has forward-pointing normals. The face mask must
therefore EXCLUDE hair texels to avoid stripping hair colour. Pass a
pre-computed hair_mask to the constructor for that.
"""

import numpy as np


# ── Tuning constants ─────────────────────────────────────────────────────────
# All thresholds use NORMALIZED world-space coordinates (norm_size=1.15, so
# the character spans roughly Y ∈ [-0.575, +0.575], X half-extent depends on
# pose). Keep parameters bbox-RELATIVE where possible so the same values work
# across different character heights and silhouettes.
#
# All ramps use smoothstep — hard boolean masks at the face boundary leave a
# visible seam where blend weights jump discontinuously.

# Forward-facing test: keep texels whose normal points toward the front camera.
FACE_NZ_FULL = 0.70   # n_z ≥ this → 1.0 face weight
FACE_NZ_NONE = 0.30   # n_z ≤ this → 0.0 face weight

# Vertical band: face sits in the upper portion of the mesh bbox.
# y_from_top_frac = (y_max - p_y) / y_rng.
# For norm_size=1.15: chin ≈ 0.135, top of head = 0.0.
FACE_Y_FULL_FRAC   = 0.14   # below this (from top) → 1.0 face weight
FACE_Y_FADE_FRAC   = 0.18   # by this fraction → 0.0 face weight

# Horizontal band: central X. For T-pose, x_half is ≈ arm-span/2 so face is
# only a small fraction of that. Y+Z filters do most of the work; X is a guard
# against ears creeping in.
FACE_X_FULL_FRAC = 0.30
FACE_X_FADE_FRAC = 0.50

# Azimuth window that counts as "front view" when modulating per-view weights.
FRONT_VIEW_AZ_HALF_WIDTH_DEG = 30.0


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep — returns 0 below a, 1 above b, smooth ramp between."""
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class FaceDetectionUtils:
    """
    Per-texel face-front weight builder.

    Coordinate system (after normalization, norm_size=1.15):
      +Y = up
      +Z = front (toward camera at azimuth 0)
      +X = character's LEFT (from viewer's perspective)
    """

    def __init__(self,
                 texture_size: int,
                 uv_mask: np.ndarray,
                 tex_pos_np: np.ndarray,
                 tex_normals_np: np.ndarray,
                 neck_y_min: float = 0.30,
                 neck_y_max: float = 0.42,
                 hair_mask: np.ndarray = None):
        """
        Parameters
        ----------
        texture_size   : atlas resolution (e.g. 4096)
        uv_mask        : bool (H, W)  — True inside UV islands
        tex_pos_np     : float32 (H, W, 3) — world-space position per texel
        tex_normals_np : float32 (H, W, 3) — world-space smooth normal per texel
        neck_y_min     : world-Y bottom of neck zone (for parity with hair API)
        neck_y_max     : world-Y top of neck zone — face mask is hard-floored here
        hair_mask      : OPTIONAL bool (H, W) — texels flagged as hair.
                         Face mask is zero'd at these texels so hair colour is
                         not over-painted with skin from the front view. Get
                         this from HairDetectionUtils.get_hair_mask().
        """
        self.texture_size   = texture_size
        self.uv_mask        = uv_mask
        self.tex_pos_np     = tex_pos_np
        self.tex_normals_np = tex_normals_np
        self._neck_y_max    = float(neck_y_max)
        self._hair_mask     = hair_mask

        self._uv_y, self._uv_x = np.where(uv_mask)
        pos3d = tex_pos_np[self._uv_y, self._uv_x]
        self.p_x = pos3d[:, 0]
        self.p_y = pos3d[:, 1]
        self.p_z = pos3d[:, 2]
        nor3d = tex_normals_np[self._uv_y, self._uv_x]
        self.n_z = nor3d[:, 2]

        self._y_max  = float(self.p_y.max())
        self._y_min  = float(self.p_y.min())
        self._y_rng  = max(self._y_max - self._y_min, 1e-6)
        self._x_half = max(float(np.abs(self.p_x).max()), 1e-6)

        self._face_front_weight_cache = None

    # =========================================================================
    # Public API
    # =========================================================================
    def get_face_front_weight(self) -> np.ndarray:
        """
        Soft weight in [0, 1] per UV texel, shape (H, W).
          1.0 = "this is a face-front texel; use front view ONLY"
          0.0 = "not face; use default multi-view blend"
        """
        if self._face_front_weight_cache is None:
            self._face_front_weight_cache = self._build_face_region_mask()
        return self._face_front_weight_cache

    # =========================================================================
    # Internal
    # =========================================================================
    def _build_face_region_mask(self) -> np.ndarray:
        """
        Soft mask = n_z ramp × Y ramp (with hard floor at neck) × X ramp,
        then MINUS any texel flagged as hair (so hair colour wins there).
        """
        H = W = self.texture_size
        out = np.zeros((H, W), dtype=np.float32)

        w_nz = _smoothstep(FACE_NZ_NONE, FACE_NZ_FULL, self.n_z)

        y_from_top_frac = (self._y_max - self.p_y) / self._y_rng
        w_y_top = 1.0 - _smoothstep(FACE_Y_FULL_FRAC,
                                    FACE_Y_FADE_FRAC,
                                    y_from_top_frac)
        w_y_floor = (self.p_y > self._neck_y_max).astype(np.float32)
        w_y = w_y_top * w_y_floor

        x_frac = np.abs(self.p_x) / self._x_half
        w_x = 1.0 - _smoothstep(FACE_X_FULL_FRAC, FACE_X_FADE_FRAC, x_frac)

        weight = (w_nz * w_y * w_x).astype(np.float32)
        out[self._uv_y, self._uv_x] = weight

        # ── Hair exclusion: face mask must NOT cover hair texels ─────────────
        if self._hair_mask is not None:
            _hair_count_before = int((out > 0.05).sum())
            out[self._hair_mask] = 0.0
            _hair_count_after = int((out > 0.05).sum())
            print(f"[FaceDetect] hair exclusion: "
                  f"{_hair_count_before - _hair_count_after:,} texels removed "
                  f"from face mask (overlap with hair)")

        _n_full    = int((out >= 0.95).sum())
        _n_partial = int(((out > 0.05) & (out < 0.95)).sum())
        print(f"[FaceDetect] face mask: {_n_full:,} full / {_n_partial:,} partial   "
              f"x_half={self._x_half:.3f}  y_range=[{self._y_min:+.3f},{self._y_max:+.3f}]  "
              f"neck_y_max={self._neck_y_max:.3f}  "
              f"hair_excl={'yes' if self._hair_mask is not None else 'no'}")
        return out

    def save_debug_mask(self, path: str) -> None:
        """Write the face-front weight as a grayscale PNG."""
        from PIL import Image as _PIL
        mask = self.get_face_front_weight()
        _PIL.fromarray((mask * 255).astype(np.uint8), mode='L').save(path)
        print(f"[FaceDetect] debug mask saved to {path}")


def is_front_view(azimuth_deg: float,
                  half_width_deg: float = FRONT_VIEW_AZ_HALF_WIDTH_DEG) -> bool:
    """True if camera azimuth counts as the front view for KAN-9 purposes."""
    az = ((azimuth_deg + 180.0) % 360.0) - 180.0
    return abs(az) <= half_width_deg
