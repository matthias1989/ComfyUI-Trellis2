# Projection Pipeline — Research & Design Notes

These notes were preserved from inline comments in `texture_projection_multiview.py`
during the refactor into utility classes.  They document *why* things work the way
they do, approaches that were tried and abandoned, and the current state of ongoing
work.

---

## Coordinate System

Y-up world space, consistent with Blender's auto-centre-and-scale (norm_size = 1.15):

```
+X = character's LEFT  (from viewer's perspective)
+Y = UP
+Z = toward viewer (front of character)

Azimuth   0 → camera at +Z, looking toward –Z  (front view)
Azimuth  90 → camera at +X, looking toward –X
Azimuth 180 → camera at –Z, looking toward +Z  (back view)
Elevation  0 → horizontal
Elevation 90 → camera above, looking down
```

Typical normalised character extents (norm_size = 1.15):

```
Y: –0.575 (feet) to +0.575 (top of head)
T_seam_floor ≈ 0.372  — nape of neck / bottom of head zone
T_hair       ≈ 0.467  — Y level of the front forehead hairline
y_max        ≈ 0.525  — top of head
```

---

## Fill Passes — Design Rationale

### Why 3-D KD-tree fill instead of UV-space distance_transform_edt

UV-space distance is geometrically misleading: the back-of-head UV island can be
2-D-adjacent to the face-centre UV island even though they are far apart in 3-D.
The old 2-D fill therefore spread eye/nose texture onto the back of the head, and
torso-background colours onto the arms.

The 3-D fill builds a KD-tree of *covered* texels using their world-space positions
and finds each uncovered texel's nearest covered 3-D neighbour.  The back of the
head gets colour from the 3-D-nearest covered point (side of the head → hair / skin
tone), not from a UV-adjacent face texel.

### Pass 1 brightness filter

Source points are brightness-filtered before the KD-tree is built.  Arm/leg edges
projected at a grazing angle come in dark even though the surface itself is
skin-coloured; without this filter the fill propagates those dark edge values to
uncovered areas (arm tops, inner thighs, etc.) instead of the correct bright skin.
Threshold: mean RGB > 70 (≈ 0.27 normalised).

### Pass 1b relative vs absolute mode

**Relative mode** (default, `p1b_relative_mode=True`):
Compute the median brightness of all saturated UV pixels and use
`median × p1b_bright_fraction` as the source brightness floor.  Adapts
automatically to dark or light skin tones — no manual threshold tuning per
character.

**Absolute mode** (`p1b_relative_mode=False`):
Fixed `p1b_bright_thresh`, same as the original behaviour.

### Pass 1c colour-consistency correction

The earlier passes fix uncovered texels and rescue dark ones, but leave alone any
texel that is (a) covered by some view and (b) not dark.  That includes
"chalk-white" / desaturated grey patches — arm tops, shoulder crowns — that
received a grazing-angle projection and ended up washed-out even though they are
technically "covered" and "bright enough".

For each UV texel: find its K nearest 3-D neighbours.  If this texel's colour
saturation is far below the neighbourhood median AND the mean per-channel colour
distance is large enough, replace it with the neighbourhood median.  The saturation
comparison focuses specifically on the grey-vs-skin outlier pattern while leaving
legitimate variation (hair, lips, face features) untouched.

In relative mode (adaptive floor more permissive) the grey patches tend to be
larger, so the full K + multi-iteration flood-fill is used (K=96, 5 iterations).
In absolute mode the old fast settings (K=24, 1 iteration) are sufficient.

### Pass 2 seam bleed — alpha=0

Colour is bled 8 px outside UV islands so bilinear sampling at seam edges gets the
correct hue, but alpha is set to 0 so the Blender material (alpha-clip mode) does
NOT render those pixels.  This eliminates the silhouette "glow/halo" that appeared
at the mesh perimeter.  The colour is still available for the GPU's bilinear filter
to blend at the island boundary; only alpha=0 prevents those texels from being
shaded independently.

### Colour correction — SV3D compensation

SV3D drains warmth and saturation; boosting S and slightly darkening V makes skin
look like proper golden/warm tones instead of chalk-white.  Default is no change
(1.0, 1.0) so the correction is opt-in per run.

---

## Hairline Detection — Research Notes

### Goal

Detect the hair region on the Trellis mesh so it can be filled with hair colour and
the rest of the head filled with skin colour in the final texture.  We do NOT need
a perfect 1-px hairline — we need a reliable hair-region *mask*.  The boundary of
that mask is the hairline.

### Key insight

Face/neck skin normals point forward (+Z) → `normal.Z ≈ +0.5 to +0.9` (HIGH).
Hair geometry on the crown / sides / back faces upward or sideways / backward →
`normal.Z ≈ 0 to –0.5` (LOW).  Nose and eye-socket normals are also mostly
forward-pointing → they stay as "face", not mis-classified as hair.

### Z-value distribution (reference character)

75% of safe head pixels have Z < 0 (sides/back).  Only the front-facing cap and
face have Z > 0.  Z_max ≈ 0.070.  Hair cap and forehead Z values are very close
together (Δ ≈ 0.01–0.02).

### Approaches tried and abandoned

| # | Approach | Why abandoned |
|---|----------|---------------|
| 1 | Per-column largest-Z-jump scan | Fixed T_hair window missed side hairline (hairline drops at temples). Full-column scan was noisy. |
| 2 | Binary Z-threshold + grid boundary (80×80 world-space cells) | Using 0.5×Z_pixel_max → only 1 cyan dot (cell averages much lower than pixel max). Switching to p60 of cell averages → fired all over the hair cap, not just the boundary, because Canny detects ALL Z gradients including internal hair ridges. |
| 3 | UV-space Canny on Z-grayscale | Fires at UV island seam boundaries (fingerprint noise all over hair cap). BUT: the **front hairline IS detectable** (cyan blur=5 σ2, Canny 5/15). Kept as cyan baseline — do not change these params. |
| 4 | Grid-space Canny (80×80 Z-grid scaled 8×) | Avoided UV seam artifacts but detected internal hair ridges/braids instead of the outer hairline. |
| 5 | Connected-component filter on cyan edges | 2979 components, ALL < 100 px. UV mapping fragments the hairline across islands so even the real hairline has tiny components. CC filtering in UV space fundamentally doesn't work. |
| 6 | World-Y range filter on cyan edges | Kept edges in [T_seam_floor, T_hair+0.05]. Most UV-seam noise is in the same Y band as the hairline → didn't filter enough. |
| 7 | Local Z-contrast filter on cyan edges | 9×9 window max-min. Contrast at all edge pixels: p50=20, max=94. Hair cap and face have nearly identical Z → contrast can't separate hairline from UV-seam noise. |
| 8 | Binary-threshold blob → fill_holes → outer boundary | fill_holes leaked through the face exclusion-zone hole and flooded the entire face area. Large dilation also bled onto ears/neck. |
| 9 | World-Y threshold (Y > T_hair) as direct hair mask | Produced straight horizontal lines in UV space — the UV layout maps world-Y to horizontal strips so the boundary was a flat stripe. Not useful as a hairline. |
| 10 | **Normal-Y Canny (geometry edges)** ← **CURRENT** | tex_normals are purely geometric → no fingerprint noise. Fires on hair geometry edges (ridges, cap rim, braids). Covers areas cyan misses (sides, back of cap). Filter: Y ≥ T_seam_floor AND NOT (Y < T_hair AND Z > 0) → removes front-face below hairline while keeping the lower back of the hair cap. |

### Current state

- **Cyan** = Z-based Canny, UV texture space (UV seam fingerprint noise present, best params locked: blur=5 σ2, Canny 5/15, interior erosion 12 px)
- **Yellow** = Normal-Y Canny filtered to Y ≥ T_seam_floor (geometry edges, no UV noise)
- **Green** = hanging / floating hair (body zone + head zone dual-Z + back cap rim)
- **Magenta** = outer contour of the combined hair region (hairline border, 21 px MORPH_CLOSE)

Together they aim to cover the full hair cap region.
Body-zone hanging hair (below T_seam_floor) is handled by a separate detection pass.

### Next steps

- Combine cyan + yellow into a single hair-region mask (dilate + close)
- Filter ear false-positives by `|world-X| < threshold`
- Handle body-zone hanging hair via the elevation detection step
- Once mask is solid: use it to assign hair colour vs skin colour in the final
  texture fill pass

### Cyan baseline — do NOT change

```
GaussianBlur (5, 5) σ=2.0  →  Canny (low=5, high=15)  →  interior erosion 12 px
```

Catches front hairline well.  UV-seam fingerprint noise is present but the hairline
signal is clearly there.  Every other parameter combination was tested and either
lost the hairline or made the noise worse.

---

## Seam-Walk 3-D Face/Hair Classification

### Pipeline

1. **Phase 1** — XZ cell grid (80×80), mean_nz per cell, Otsu threshold.  Otsu is
   computed from the *upper skull zone only* (Y > T_head_seam) where the
   distribution is cleanly bimodal (face skin vs scalp/hair).  Neck pixels have
   very high nz and would push the threshold too high if included.
2. **Phase 2** — per-pixel initial label from each pixel's own normal.Z.
3. **Phase 3** — iterative KNN smoothing (5 passes, K=25) using a 4-D KD-tree
   (x, y, z, nz_weight) so the boundary converges to a geometrically precise,
   smooth seam.
4. **Phase 4** — coordinate-based face/hair split:
   - Face = forward-facing (p_z > 0) AND below hairline (p_y ≤ T_hair)
   - Hair = backward/sideward (p_z ≤ 0) OR above hairline (p_y > T_hair)
   - Produces forehead seam at y ≈ T_hair, temple seam at z ≈ 0, no false lines
     across nose/eyes/lips.

### Central-face exclusion zone

Three conditions (ALL must be true to suppress a seam candidate):
- `|p_x| < 0.18 × head_x_range` — central 36% of head width
- `p_y < T_seam_floor + 0.85 × (T_hair – T_seam_floor)` — below 85% of face-zone height
- `p_z > hz_min + 0.70 × hz_range` — front 30% of head depth (nose/eyes area)

Leaves the top 15% (forehead near hairline) as valid seam candidates.

---

## Dropped Gemini Code Block

A block titled "NEW INSERTION: Filter the Cyan Border & Create Brown Duplicate"
was inserted by Gemini (May 2025) and removed during the refactor.  It was a
complete no-op: `canvas_tensor` was retrieved via `locals().get('img_out', ...)` —
a variable that never exists in the running scope — so `canvas_tensor` was always
`None` and the entire body of the `if canvas_tensor is not None:` branch never ran.
The try/except swallowed the silent failure every run.  The approach of keeping
only `max_z` / `min_y` extreme pixels would also only have kept a handful of
forehead/bottom pixels rather than the full hairline contour.
