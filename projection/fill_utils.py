"""
FillUtils — texture fill passes extracted from texture_projection_multiview.py.

Versioning:
  fill_utils.py    — v1: 3-D KD-tree fill, dark rescue, consistency correction, seam bleed
  fill_utils_v2.py — create alongside this file when experimenting; swap import in main.

Usage:
  from .fill_utils import FillUtils
  fu = FillUtils(uv_mask, tex_pos_np)
  fu.fill_uncovered_3d(color_np, alpha_np, covered, uncovered_uv)
  fu.fill_dark_rescue(color_np, alpha_np, uv_mask, ...)
  fu.correct_color_consistency(color_np, uv_mask, ...)
  fu.seam_bleed(color_np, alpha_np)
  fu.colour_correct(color_np, uv_mask, saturation_boost, brightness_correction)
"""

import numpy as np
import cv2
import os
from scipy.ndimage import distance_transform_edt, binary_closing, binary_dilation, binary_erosion
from scipy.spatial import cKDTree


class FillUtils:
    """
    Stateless helper — tex_pos_np is stored once and reused across pass calls.
    All passes mutate color_np / alpha_np in-place.
    """

    def __init__(self, uv_mask: np.ndarray, tex_pos_np: np.ndarray):
        """
        Parameters
        ----------
        uv_mask   : bool (H, W) — True for every texel inside a UV island
        tex_pos_np: float32 (H, W, 3) — world-space position per texel
        """
        self.uv_mask    = uv_mask
        self.tex_pos_np = tex_pos_np

    # -------------------------------------------------------------------------
    # Cylindrical fill — photo-derived, height-varying skin for back/sides
    # -------------------------------------------------------------------------
    def fill_cylindrical(self, color_np: np.ndarray, alpha_np: np.ndarray,
                         covered: np.ndarray, n_bands: int = 200,
                         edge_frac: float = 0.18, region_mask: np.ndarray = None,
                         fill_hair: bool = True) -> None:
        """
        Fill every uncovered (back/side) texel from the FRONT projection's own
        colours, the way a body actually wraps:

          For each horizontal slice of the body (height band), find the colour at
          the two points still visible from the front — the LEFT and RIGHT
          silhouette edges (the extreme +/- theta covered texels) — then for the
          unseen texels in that band interpolate between those two edge colours
          AROUND the back by angle.  The result is a smooth, height-varying skin
          tone derived from the real photo, not one flat slab.

          theta = atan2(x, z): 0 = front (+Z), +90 = char-left (+X), +-180 = back.

        REGION-AWARE (KAN-9): if region_mask is given (True = hair texels), the wrap
        runs SEPARATELY for the body region and the hair region — body texels are
        filled only from body-front edges, hair texels only from hair-front edges.
        That stops blonde hair bleeding onto back-skin and skin bleeding into the
        hair where the hair drapes the sides (the user's front-only fill plan).

        No KD-tree, no dark-rescue, no colour-consistency — fast and stable.
        """
        from scipy.ndimage import gaussian_filter1d
        uv = self.uv_mask
        tp = self.tex_pos_np
        px = tp[..., 0]; py = tp[..., 1]; pz = tp[..., 2]
        theta = np.degrees(np.arctan2(px, pz))            # angle around body
        # Height banding shared across regions so both use the same slices.
        ymin = float(py[uv].min()); ymax = float(py[uv].max())
        yb = np.clip(((py - ymin) / (ymax - ymin + 1e-9) * (n_bands - 1)).astype(np.int32),
                     0, n_bands - 1)
        col = color_np.astype(np.float32)

        def _fill_region(rmask: np.ndarray, rname: str) -> int:
            cov_r = covered & rmask
            fill_r = rmask & (~covered)
            n_fill = int(fill_r.sum())
            if not cov_r.any() or n_fill == 0:
                return 0
            # Per-height edge colours + the angles of those edges (within this region).
            Lc = np.zeros((n_bands, 3), np.float32); Rc = np.zeros((n_bands, 3), np.float32)
            tL = np.full(n_bands, 90.0, np.float32);  tR = np.full(n_bands, -90.0, np.float32)
            has = np.zeros(n_bands, bool)
            for b in range(n_bands):
                m = cov_r & (yb == b)
                if int(m.sum()) < 20:
                    continue
                th = theta[m]; cc = col[m]
                lo = float(th.min()); hi = float(th.max()); rng = max(hi - lo, 1e-3)
                Lm = th >= hi - edge_frac * rng       # left (max-theta) edge
                Rm = th <= lo + edge_frac * rng       # right (min-theta) edge
                Lc[b] = cc[Lm].mean(0); Rc[b] = cc[Rm].mean(0)
                tL[b] = float(th[Lm].mean()); tR[b] = float(th[Rm].mean()); has[b] = True
            if not has.any():
                return 0
            # Bands with no coverage inherit the nearest populated band.
            idx = np.where(has)[0]
            for b in range(n_bands):
                if not has[b]:
                    j = int(idx[np.argmin(np.abs(idx - b))])
                    Lc[b] = Lc[j]; Rc[b] = Rc[j]; tL[b] = tL[j]; tR[b] = tR[j]
            # Smooth the profiles ONLY along height (per channel) — a 2-D blur here
            # would mix R/G/B and desaturate the fill to gray.
            Lc = gaussian_filter1d(Lc, 3.0, axis=0, mode='nearest')
            Rc = gaussian_filter1d(Rc, 3.0, axis=0, mode='nearest')
            tL = gaussian_filter1d(tL, 3.0, axis=0, mode='nearest')
            tR = gaussian_filter1d(tR, 3.0, axis=0, mode='nearest')
            # Interpolate around the BACK between the two edge colours.
            bb = yb[fill_r]; thf = theta[fill_r]
            tLb = tL[bb]; tRb = tR[bb]
            back = 360.0 - (tLb - tRb)
            dleft = np.where(thf >= tLb, thf - tLb, thf - tLb + 360.0)   # arc from left edge
            t = np.clip(dleft / np.maximum(back, 1e-3), 0.0, 1.0)[:, None]
            filled = Lc[bb] * (1.0 - t) + Rc[bb] * t
            color_np[fill_r] = np.clip(filled, 0, 255).astype(np.uint8)
            print(f"[CylindricalFill:{rname}] {n_fill:,} texels filled from "
                  f"{int(has.sum())}/{n_bands} bands")
            return n_fill

        if region_mask is None:
            total = _fill_region(uv, "all")
        else:
            total = _fill_region(uv & (~region_mask), "body")
            if fill_hair:                                  # hair handled by fill_hair_3d when False
                total += _fill_region(uv & region_mask, "hair")
        if total:
            alpha_np[uv] = 255
            print(f"[CylindricalFill] {total:,} uncovered texels filled "
                  f"(photo-derived, height-varying, region-separated — no flat slab)")

    # -------------------------------------------------------------------------
    # HAIR FILL (region-specific, NOT the body cylindrical wrap): every UNSEEN hair texel
    # takes the colour of the nearest VISIBLE hair texel in 3-D. Source and target are both
    # inside hair_mask, so it can NEVER pull skin colour (the bug: the body wrap grabbed the
    # body silhouette on the drape -> skin-toned back hair). Naturally region-dependent —
    # back-of-crown gets crown colour, the back drape gets the nearest visible drape colour.
    # tex_pos is a UV-rasterised position map (valid for unseen texels too), so 3-D nearest
    # is well-defined everywhere in the hair island.
    # -------------------------------------------------------------------------
    def fill_hair_3d(self, color_np: np.ndarray, alpha_np: np.ndarray,
                     covered: np.ndarray, hair_mask: np.ndarray) -> None:
        if hair_mask is None:
            print("  [HairFill3D] no hair mask — skip"); return
        uv = self.uv_mask; tp = self.tex_pos_np
        # KEEP the front image: every VISIBLE (covered) hair texel is left exactly as projected.
        # ONLY fill the hair we CANNOT see (uncovered), from the nearest VISIBLE hair in 3-D — so
        # the unseen back/sides take the colour of the nearby front hair (blonde for the drape;
        # the forehead is far, so nearest-3-D won't pull it onto the drape). No recolour, no
        # overwriting of visible hair.
        src = hair_mask & covered & uv
        tgt = hair_mask & (~covered) & uv
        ns = int(src.sum()); nt = int(tgt.sum())
        if ns < 20 or nt == 0:
            print(f"  [HairFill3D] src={ns} tgt={nt} — nothing to fill"); return
        sidx = np.flatnonzero(src.ravel()); tidx = np.flatnonzero(tgt.ravel())
        sp = tp.reshape(-1, 3)[sidx]; tpos = tp.reshape(-1, 3)[tidx]
        d, nn = cKDTree(sp).query(tpos, k=1, workers=-1)
        colf = color_np.reshape(-1, 3)
        colf[tidx] = colf[sidx[nn]]
        alpha_np.reshape(-1)[tidx] = 255
        print(f"  [HairFill3D] {nt:,} UNSEEN hair texels <- nearest VISIBLE hair "
              f"(front image kept; src {ns:,}, median dist {float(np.median(d)):.4f})")

    # -------------------------------------------------------------------------
    # Limb-aware fill — wrap each ARM around its OWN axis (X), not the spine.
    # The body cylindrical fill wraps around the vertical spine, which is wrong for a
    # sideways arm (every arm texel collapses to ~one angle/height bin → the top/back of
    # the arm gets one smeared tone + a visible transition line). Here we band ALONG the
    # arm (X) and wrap by the angle around the arm cross-section (Y,Z), interpolating the
    # covered front-arm skin around to the uncovered top/back. Run AFTER fill_cylindrical
    # (overwrites the arm texels it filled poorly).
    # -------------------------------------------------------------------------
    def fill_limbs(self, color_np: np.ndarray, covered: np.ndarray,
                   n_bands: int = 120, edge_frac: float = 0.20) -> None:
        from scipy.ndimage import gaussian_filter1d
        uv = self.uv_mask; tp = self.tex_pos_np
        px, py, pz = tp[..., 0], tp[..., 1], tp[..., 2]
        col = color_np.astype(np.float32)
        total = 0
        for side in (-1, +1):
            a = px * side                                        # along-arm coord, +ve outward
            arm = uv & (a > 0.20) & (a < 0.41)                  # this arm, between shoulder and wrist (skip hand)
            if int(arm.sum()) < 200:
                continue
            amin = float(a[arm].min()); amax = float(a[arm].max())
            xb = np.clip(((a - amin) / (amax - amin + 1e-9) * (n_bands - 1)).astype(np.int32), 0, n_bands - 1)
            # per-band cross-section centre (cy,cz) so phi is measured around the arm's own axis
            cy = np.zeros(n_bands, np.float32); cz = np.zeros(n_bands, np.float32); hb = np.zeros(n_bands, bool)
            for b in range(n_bands):
                mb = arm & (xb == b)
                if int(mb.sum()) >= 8:
                    cy[b] = py[mb].mean(); cz[b] = pz[mb].mean(); hb[b] = True
            if not hb.any():
                continue
            jb = np.where(hb)[0]
            for b in range(n_bands):
                if not hb[b]:
                    j = int(jb[np.argmin(np.abs(jb - b))]); cy[b] = cy[j]; cz[b] = cz[j]
            phi = np.degrees(np.arctan2(py - cy[xb], pz - cz[xb]))   # 0=front(+z) 90=top 180=back -90=bottom
            # SOURCE = the front-facing arc only (well-projected skin). FILL = everything beyond it
            # (top/sides/back) — including the GRAZING-DARK top, which the projection "covered" with a
            # poor dark sample. Keying on angle (not the covered flag) is what lets us replace that top.
            FRONT = 35.0          # narrow BRIGHT front arc as the source; re-wrap everything beyond
            cov = arm & (np.abs(phi) < FRONT)        # so the grazing-dark band (35-58) is overwritten too
            fillm = arm & (np.abs(phi) >= FRONT)
            if not cov.any() or not fillm.any():
                continue
            Lc = np.zeros((n_bands, 3), np.float32); Rc = np.zeros((n_bands, 3), np.float32)
            tL = np.full(n_bands, 90.0, np.float32); tR = np.full(n_bands, -90.0, np.float32)
            has = np.zeros(n_bands, bool)
            for b in range(n_bands):
                m = cov & (xb == b)
                if int(m.sum()) < 8:
                    continue
                ph = phi[m]; cc = col[m]
                lo = float(ph.min()); hi = float(ph.max()); rng = max(hi - lo, 1e-3)
                Lm = ph >= hi - edge_frac * rng
                Rm = ph <= lo + edge_frac * rng
                Lc[b] = cc[Lm].mean(0); Rc[b] = cc[Rm].mean(0)
                tL[b] = float(ph[Lm].mean()); tR[b] = float(ph[Rm].mean()); has[b] = True
            if not has.any():
                continue
            ib = np.where(has)[0]
            for b in range(n_bands):
                if not has[b]:
                    j = int(ib[np.argmin(np.abs(ib - b))]); Lc[b] = Lc[j]; Rc[b] = Rc[j]; tL[b] = tL[j]; tR[b] = tR[j]
            Lc = gaussian_filter1d(Lc, 2.0, axis=0, mode='nearest'); Rc = gaussian_filter1d(Rc, 2.0, axis=0, mode='nearest')
            tL = gaussian_filter1d(tL, 2.0, mode='nearest'); tR = gaussian_filter1d(tR, 2.0, mode='nearest')
            bb = xb[fillm]; phf = phi[fillm]
            back = 360.0 - (tL[bb] - tR[bb])
            dleft = np.where(phf >= tL[bb], phf - tL[bb], phf - tL[bb] + 360.0)
            t = np.clip(dleft / np.maximum(back, 1e-3), 0.0, 1.0)[:, None]
            filled = Lc[bb] * (1.0 - t) + Rc[bb] * t
            color_np[fillm] = np.clip(filled, 0, 255).astype(np.uint8)
            total += int(fillm.sum())
        if total:
            print(f"  [LimbFill] re-wrapped {total:,} uncovered arm texels around the arm axis "
                  f"(top/back interpolated from the covered front-arm skin)")

    # -------------------------------------------------------------------------
    # Seam-line removal (user's 3-step spec):
    #   1. find BLACK / very-dark-gray texels (ABSOLUTE -> skin & features never touched)
    #   2. keep only LINE-shaped components (filter out blobs we must keep)
    #   3. replace each with the nearest non-dark skin colour
    # -------------------------------------------------------------------------
    def recolor_seam_borders(self, color_np: np.ndarray, covered: np.ndarray = None) -> None:
        uv = self.uv_mask
        if not uv.any():
            return
        g = color_np.astype(np.float32).mean(2)
        # 1) BLACK or VERY DARK GRAY (absolute). Skin/areola/navel are all >90, so this
        #    threshold cannot touch them — only the actual dark seam pixels qualify.
        dark = uv & (g < 80.0)
        if not dark.any():
            print("  [SeamBorder] no black/dark-gray texels found")
            return
        # 2) keep only LINES — thin or sparse-bbox components; drop compact/large blobs
        #    (a feature, a shadow patch, hair, or a big coverage blob is NOT a line -> kept).
        total = float(uv.sum())
        num, lab, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
        line = np.zeros_like(dark)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area > 0.03 * total:                          # big region (hair/coverage blob) -> keep
                continue
            if min(w, h) <= 12 or area < 0.30 * float(max(w * h, 1)):   # thin strip / sparse bbox = a line
                line |= (lab == i)
        line &= dark
        # 3) replace each dark-line texel with the nearest NON-dark skin colour
        src = uv & ~dark
        if line.any() and src.any():
            _, idx = distance_transform_edt(~src, return_indices=True)
            color_np[line] = color_np[idx[0][line], idx[1][line]]
        print(f"  [SeamBorder] replaced {int(line.sum()):,} black/dark-gray seam-LINE texels with "
              f"nearby skin (skin + features >80 never touched).")

    # -------------------------------------------------------------------------
    # FLAT-FILL detect+replace (user's 3-step plan).  The bad regions (cylindrical-fill
    # smears, grazing fill) share ONE trait the user spotted: they are FLAT — no pixel-to-
    # pixel grain — whereas real projected skin has grain.  So:
    #   detect invalid = LOW local grain, OUTSIDE the protected valid set
    #     (head/face, nipple, and the central-front torso = real smooth skin).
    #   GS_FLATFILL=debug -> paint the mask RED (verify on the real model)
    #   GS_FLATFILL=replace -> replace each flagged texel from the nearest VALID (grainy) skin
    # -------------------------------------------------------------------------
    def _detect_flat(self, color_np: np.ndarray) -> np.ndarray:
        uv = self.uv_mask; tp = self.tex_pos_np
        g = color_np.astype(np.float32).mean(2); M = uv.astype(np.float32)
        blur = lambda im, s: cv2.GaussianBlur(im, (0, 0), s)
        low = blur(g * M, 8) / (blur(M, 8) + 1e-6)
        grain = blur(np.abs(g - low) * M, 5) / (blur(M, 5) + 1e-6)     # high-freq detail (fill lacks it)
        px = tp[:, :, 0]; py = tp[:, :, 1]
        xc = 0.5 * (px[uv].min() + px[uv].max()); xh = 0.5 * (px[uv].max() - px[uv].min()) + 1e-9
        ymn = float(py[uv].min()); ymx = float(py[uv].max())
        xfrac = (px - xc) / xh; pyf = (py - ymn) / (ymx - ymn + 1e-9)
        hsv = cv2.cvtColor(np.clip(color_np, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        S = hsv[..., 1]; Vv = hsv[..., 2]
        torso = uv & (np.abs(xfrac) < 0.18) & (pyf > 0.5) & (pyf < 0.78)
        tS = float(np.median(S[torso])); tV = float(np.median(Vv[torso]))
        flat = grain < np.percentile(grain[uv], 55)
        head = pyf > 0.82
        nipple = (S > 1.25 * tS) & (Vv < tV)
        central = (np.abs(xfrac) < 0.25) & (pyf > 0.45) & (pyf < 0.82)   # chest/belly front = real smooth skin
        inv = uv & flat & ~head & ~nipple & ~central
        inv = binary_closing(inv, iterations=6) & uv                    # consolidate speckly flat regions
        n, lab, st, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(inv)
        for i in range(1, n):
            if st[i, 4] >= 400:
                keep |= (lab == i)
        return keep

    def flat_fill(self, color_np: np.ndarray) -> None:
        mode = os.environ.get('GS_FLATFILL', 'debug')
        inv = self._detect_flat(color_np)
        if not inv.any():
            return
        if mode == 'debug':
            color_np[inv] = (255, 0, 0)                                  # RED = detected invalid (verify)
            print(f"  [FlatFill DEBUG] painted {int(inv.sum()):,} flat-fill texels RED (no replace)")
        else:
            valid = self.uv_mask & ~inv & (color_np.astype(np.float32).mean(2) > 60)
            if valid.any():
                _, idx = distance_transform_edt(~valid, return_indices=True)
                color_np[inv] = color_np[idx[0][inv], idx[1][inv]]
            print(f"  [FlatFill] replaced {int(inv.sum()):,} flat-fill texels from nearest valid skin")

    # -------------------------------------------------------------------------
    # CLEAR BORDER LINES — the distinct seam-border lines (dark OR desaturated vs the
    # torso skin, per the user's colour-picker data: bad areas read ~S31-38 vs skin ~S46).
    # Detect the LINES only (exclude the broad arm/leg tone -> that's a later tone-harmonise),
    # protect face/eyes + nipple + central-front torso, then replace each line texel from the
    # nearest VALID (bright + saturated) skin. Step 2 of the user's plan, scoped to the lines.
    # -------------------------------------------------------------------------
    def replace_clear_borders(self, color_np: np.ndarray) -> None:
        uv = self.uv_mask; tp = self.tex_pos_np
        hsv = cv2.cvtColor(np.clip(color_np, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        S = hsv[..., 1]; Vv = hsv[..., 2]
        px = tp[:, :, 0]; py = tp[:, :, 1]
        xc = 0.5 * (px[uv].min() + px[uv].max()); xh = 0.5 * (px[uv].max() - px[uv].min()) + 1e-9
        ymn = float(py[uv].min()); ymx = float(py[uv].max())
        xfrac = (px - xc) / xh; pyf = (py - ymn) / (ymx - ymn + 1e-9)
        torso = uv & (np.abs(xfrac) < 0.18) & (pyf > 0.5) & (pyf < 0.78)
        tS = float(np.median(S[torso])); tV = float(np.median(Vv[torso]))
        face = (pyf > 0.82) | ((np.abs(xfrac) < 0.22) & (pyf > 0.78))    # head + central-high face/eyes
        nipple = (S > 1.2 * tS) & (Vv < tV)
        central = (np.abs(xfrac) < 0.22) & (pyf > 0.45) & (pyf < 0.82)   # chest/belly front
        bad = uv & ~face & ~nipple & ~central & ((Vv < 0.88 * tV) | (S < 0.83 * tS))
        broad = binary_dilation(binary_erosion(bad, iterations=6), iterations=10) & bad   # wide arm/leg tone
        line = bad & ~broad
        line = binary_closing(line, iterations=2) & uv
        n, lab, st, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(line)
        for i in range(1, n):
            if st[i, 4] >= 100:
                keep |= (lab == i)
        line = keep
        if not line.any():
            print("  [ClearBorder] no border lines found"); return
        valid = uv & ~line & (Vv > 0.88 * tV) & (S > 0.83 * tS)         # good skin = bright + saturated
        if valid.any():
            _, idx = distance_transform_edt(~valid, return_indices=True)
            color_np[line] = color_np[idx[0][line], idx[1][line]]
        print(f"  [ClearBorder] replaced {int(line.sum()):,} clear seam-border line texels from nearest valid skin")

    # -------------------------------------------------------------------------
    # ARM WRAP (user's option 1): take the reliable FRONT-arm colour (the part the camera
    # saw well) and wrap it over the WHOLE arm cross-section (front+top+side+back), per band
    # along the arm. GEOMETRY-targeted to the lateral arm region only (|x|>0.30) -> cannot
    # touch the breast/torso, and stable across per-bake meshes (no colour thresholds).
    # Fixes the browner/darker arm-top by making the whole arm its own reliable front tone.
    # -------------------------------------------------------------------------
    def wrap_arm_front(self, color_np: np.ndarray, n_bands: int = 120) -> None:
        from scipy.ndimage import gaussian_filter1d
        uv = self.uv_mask; tp = self.tex_pos_np
        px, py, pz = tp[..., 0], tp[..., 1], tp[..., 2]
        xc = 0.5 * (px[uv].min() + px[uv].max()); xh = 0.5 * (px[uv].max() - px[uv].min()) + 1e-9
        xfrac = (px - xc) / xh
        col = color_np.astype(np.float32)
        for side in (-1, +1):
            a = xfrac * side
            arm = uv & (a > 0.30) & (a < 0.85)               # lateral arm, shoulder->wrist (skip hand/torso)
            if int(arm.sum()) < 200:
                continue
            amin = float(a[arm].min()); amax = float(a[arm].max())
            xb = np.clip(((a - amin) / (amax - amin + 1e-9) * (n_bands - 1)).astype(np.int32), 0, n_bands - 1)
            cy = np.zeros(n_bands, np.float32); cz = np.zeros(n_bands, np.float32); hb = np.zeros(n_bands, bool)
            for b in range(n_bands):
                mb = arm & (xb == b)
                if int(mb.sum()) >= 8:
                    cy[b] = py[mb].mean(); cz[b] = pz[mb].mean(); hb[b] = True
            if not hb.any():
                continue
            jb = np.where(hb)[0]
            for b in range(n_bands):
                if not hb[b]:
                    j = int(jb[np.argmin(np.abs(jb - b))]); cy[b] = cy[j]; cz[b] = cz[j]
            phi = np.degrees(np.arctan2(py - cy[xb], pz - cz[xb]))    # 0=front(+z)
            front = arm & (np.abs(phi) < 35)                  # the reliable front strip (camera-facing)
            fc = np.zeros((n_bands, 3), np.float32); has = np.zeros(n_bands, bool)
            for b in range(n_bands):
                fb = front & (xb == b)
                if int(fb.sum()) >= 5:
                    fc[b] = np.median(col[fb], 0); has[b] = True
            if not has.any():
                continue
            jh = np.where(has)[0]
            for b in range(n_bands):
                if not has[b]:
                    j = int(jh[np.argmin(np.abs(jh - b))]); fc[b] = fc[j]
            fc = gaussian_filter1d(fc, 2.0, axis=0, mode='nearest')   # smooth along the arm length
            color_np[arm] = np.clip(fc[xb[arm]], 0, 255).astype(np.uint8)
        print("  [ArmWrap] wrapped the reliable front-arm colour over the whole arm (breast/torso untouched)")

    # -------------------------------------------------------------------------
    # SEAM STITCH (user's idea, done in 3-D): a seam is the boundary between two UV
    # islands that meet on the model but sit far apart in the atlas — so the visible
    # line is NOT a paintable line of texels. Instead, for every island-EDGE texel we
    # find its seam PARTNER (the edge texel on the other island sitting at the SAME 3-D
    # position) and set BOTH to their average. That is literally "average the colour on
    # the two sides of the seam", matched in 3-D. Geometry-driven (no colour thresholds)
    # and limited to island edges -> features (nipples/navel/areola = island interiors)
    # are untouched. debug=True paints the matched seam texels green so the result can be
    # eyeballed on the bake (right places hit, no features) before trusting the blend.
    # -------------------------------------------------------------------------
    def stitch_seams(self, color_np: np.ndarray, eps_frac: float = 0.004,
                     uv_far_px: int = 6, ring: int = 2, k: int = 12,
                     debug: bool = False) -> None:
        uv = self.uv_mask; tp = self.tex_pos_np
        H, W = uv.shape
        inner = binary_erosion(uv, iterations=ring)
        boundary = uv & ~inner
        bidx = np.flatnonzero(boundary.ravel())
        if bidx.size < 50:
            print("  [StitchSeams] too few boundary texels — skip"); return
        pos = tp.reshape(-1, 3)[bidx].astype(np.float64)
        xs = (bidx % W).astype(np.int64); ys = (bidx // W).astype(np.int64)
        span = float(np.linalg.norm(tp[uv].max(0) - tp[uv].min(0)))
        eps = eps_frac * span
        tree = cKDTree(pos)
        kk = int(min(k, bidx.size))
        dist, nn = tree.query(pos, k=kk)                       # (N,kk) sorted by 3-D dist
        xnn = xs[nn]; ynn = ys[nn]
        adist2 = (xs[:, None] - xnn) ** 2 + (ys[:, None] - ynn) ** 2
        valid = (dist <= eps) & (adist2 >= uv_far_px * uv_far_px)   # 3-D-near AND atlas-far = other island
        valid[:, 0] = False
        has = valid.any(1)
        first = np.argmax(valid, axis=1)
        partner = np.where(has, nn[np.arange(bidx.size), first], -1)
        ai = np.where(has)[0]; bi = partner[has]
        print(f"  [StitchSeams] boundary={bidx.size}  matched seam texels={int(has.sum())}  eps={eps:.4f}")
        colf = color_np.reshape(-1, 3).astype(np.float32)
        if debug:
            colf[bidx[ai]] = np.array([0, 255, 0], np.float32)
            colf[bidx[bi]] = np.array([0, 255, 0], np.float32)
            color_np[:] = np.clip(colf, 0, 255).astype(np.uint8).reshape(H, W, 3)
            print("  [StitchSeams] DEBUG: painted matched seam texels GREEN")
            return
        avg = 0.5 * (colf[bidx[ai]] + colf[bidx[bi]])
        colf[bidx[ai]] = avg
        colf[bidx[bi]] = avg
        color_np[:] = np.clip(colf, 0, 255).astype(np.uint8).reshape(H, W, 3)
        print("  [StitchSeams] averaged both sides of each seam (island interiors/features untouched)")

    # -------------------------------------------------------------------------
    # Pass 1 — 3-D nearest-neighbour fill for back-facing (uncovered) texels
    # -------------------------------------------------------------------------
    def fill_uncovered_3d(self, color_np: np.ndarray, alpha_np: np.ndarray,
                          covered: np.ndarray, uncovered_uv: np.ndarray) -> None:
        """
        For every UV texel that is inside an island but received no projection
        weight (back-facing, self-occluded, etc.) find its nearest projected
        texel in 3-D world space and copy that colour.

        WHY 3-D instead of UV-space (distance_transform_edt)?
          UV-space distance is geometrically misleading: the back-of-head UV
          island can be 2-D-adjacent to the face-centre UV island even though
          they are far apart in 3-D.  The old 2-D fill spread eye/nose texture
          onto the back of the head and torso-background colours onto the arms.
          3-D fill finds the nearest covered texel in world space, so the back
          of the head gets colour from the 3-D-nearest covered point (side of
          the head → hair/skin tone), not from a UV-adjacent face texel.

        Subsamples source points 1-in-4: spatially adjacent texels are sub-mm
        apart in world space so quality loss is imperceptible, but tree build
        and query time drops ~16×.

        Brightness-filters the source set: arm/leg edges projected at grazing
        angle come in dark even though the surface is skin-coloured.  Without
        this filter the fill propagates dark edge values to uncovered areas
        (arm tops, inner thighs) instead of the correct bright skin.
        Threshold: mean RGB > 70 (≈ 0.27 normalised).
        """
        n_src  = int(covered.sum())
        n_fill = int(uncovered_uv.sum())
        print(f"  Fill pass 1 (3-D): {n_src} projected texels → filling {n_fill} "
              f"back-facing UV texels ({100.0*n_fill/max(self.uv_mask.sum(),1):.1f}% of UV)...")

        if n_src == 0:
            print("  WARNING: no projected pixels found – check front image connection.")
            return
        if n_fill == 0:
            return

        cov_y,   cov_x   = np.where(covered)
        uncov_y, uncov_x = np.where(uncovered_uv)

        stride   = 4
        cov_y_s  = cov_y[::stride]
        cov_x_s  = cov_x[::stride]

        src_brightness = color_np[cov_y_s, cov_x_s].mean(axis=-1)
        bright_src     = src_brightness > 70
        if bright_src.sum() > 500:
            cov_y_s = cov_y_s[bright_src]
            cov_x_s = cov_x_s[bright_src]
            print(f"    (Brightness filter: {bright_src.sum()} / {len(bright_src)} source points kept)")
        else:
            print(f"    (Brightness filter skipped – only {bright_src.sum()} bright points)")

        cov_3d   = self.tex_pos_np[cov_y_s, cov_x_s]
        uncov_3d = self.tex_pos_np[uncov_y,  uncov_x]

        print(f"    Building KD-tree on {len(cov_3d)} source points "
              f"(1-in-{stride} subsample, brightness-filtered)...")
        tree = cKDTree(cov_3d)
        _, nn_idx = tree.query(uncov_3d, workers=-1)

        for c in range(3):
            color_np[uncov_y, uncov_x, c] = color_np[cov_y_s[nn_idx], cov_x_s[nn_idx], c]
        alpha_np[uncov_y, uncov_x] = 255
        print("  Fill pass 1 complete (3-D nearest-neighbour).")

    # -------------------------------------------------------------------------
    # Pass 1b — replace dark texels with nearest bright neighbour
    # -------------------------------------------------------------------------
    def fill_dark_rescue(self, color_np: np.ndarray, alpha_np: np.ndarray,
                         uv_mask: np.ndarray,
                         p1b_dark_thresh: int   = 150,
                         p1b_sat_thresh:  int   = 55,
                         p1b_bright_fraction: float = 0.6,
                         p1b_bright_thresh: int = 180,
                         p1b_relative_mode: bool = True,
                         debug_color=None) -> None:
        """
        Replace UV texels whose mean RGB < dark_thresh with their nearest
        bright, saturated 3-D neighbour.  Rescues shadow-bleed and
        background-leak pixels without touching well-lit skin.

        Source brightness floor — two modes:
          Relative (p1b_relative_mode=True, default):
            Compute the median brightness of all saturated UV pixels and use
            median × p1b_bright_fraction as the floor.  Adapts automatically
            to dark or light skin tones — no manual threshold tuning needed
            per character.
          Absolute (p1b_relative_mode=False):
            Fixed p1b_bright_thresh, same as original behaviour.

        If debug_color=(R,G,B) is passed the dark texels are painted that flat
        colour instead of being filled from neighbours.  Useful to visualise
        which areas are rescued without waiting for the KD-tree.
        """
        mean_bright = color_np.mean(axis=-1)

        _s_bgr = color_np[:, :, ::-1].copy()
        _s_hsv = cv2.cvtColor(_s_bgr, cv2.COLOR_BGR2HSV)
        saturation_map = _s_hsv[:, :, 1].astype(float)

        if p1b_relative_mode:
            _sat_uv_bright = mean_bright[uv_mask & (saturation_map >= p1b_sat_thresh)]
            if len(_sat_uv_bright) > 100:
                median_skin = float(np.median(_sat_uv_bright))
            else:
                median_skin = float(p1b_dark_thresh)
            adaptive_floor = median_skin * p1b_bright_fraction
            # ROOT FIX: "dark" must mean a GENUINE artifact (shadow-bleed / black
            # background leak) — much darker than the character's OWN skin — not
            # normal or dim skin.  The fixed p1b_dark_thresh (150) sat ABOVE normal
            # skin brightness (~120-150), so it flagged the whole face as "dark" and
            # replaced it with flat median → the face lost all its features
            # (measured: face colour-variation collapsed 43 -> 12 at this stage).
            # Tie the dark cutoff to the skin median so it adapts and never eats skin.
            import os as _os
            _dark_frac = float(_os.environ.get('P1B_DARK_FRAC', '0.45'))
            dark_floor = min(float(p1b_dark_thresh), median_skin * _dark_frac)
            dark_in_uv = uv_mask & (mean_bright < dark_floor)
            bright_in_uv = (uv_mask
                            & (saturation_map >= p1b_sat_thresh)
                            & (mean_bright    >= adaptive_floor))
            _mode_str = (f"relative (skin median={median_skin:.0f}; dark<{dark_floor:.0f}, "
                         f"src floor={adaptive_floor:.1f})")
        else:
            dark_in_uv = uv_mask & (mean_bright < p1b_dark_thresh)
            bright_in_uv = (uv_mask
                            & (saturation_map >= p1b_sat_thresh)
                            & (mean_bright    >= p1b_bright_thresh))
            _mode_str = f"absolute (thresh={p1b_dark_thresh})"

        n_dark   = int(dark_in_uv.sum())
        n_bright = int(bright_in_uv.sum())
        print(f"  Fill pass 1b: {n_dark} dark texels (mean<{p1b_dark_thresh}), "
              f"{n_bright} saturated bright sources [{_mode_str}, sat>={p1b_sat_thresh}]")

        if n_dark == 0:
            return

        if debug_color is not None:
            r, g, b = int(debug_color[0]), int(debug_color[1]), int(debug_color[2])
            color_np[dark_in_uv] = np.array([r, g, b], dtype=np.uint8)
            alpha_np[dark_in_uv] = 255
            print(f"  Fill pass 1b: DEBUG colour ({r},{g},{b}) painted on {n_dark} dark texels")
        elif n_bright > 100:
            dark_y,   dark_x   = np.where(dark_in_uv)
            bright_y, bright_x = np.where(bright_in_uv)
            stride2 = max(1, n_bright // 100_000)
            bright_y_s = bright_y[::stride2]
            bright_x_s = bright_x[::stride2]
            bright_3d  = self.tex_pos_np[bright_y_s, bright_x_s]
            dark_3d    = self.tex_pos_np[dark_y,     dark_x]
            print(f"    KD-tree: {len(bright_3d)} sources (1-in-{stride2} subsample)...")
            tree2 = cKDTree(bright_3d)
            _, nn2 = tree2.query(dark_3d, workers=-1)
            for c in range(3):
                color_np[dark_y, dark_x, c] = color_np[bright_y_s[nn2], bright_x_s[nn2], c]
            alpha_np[dark_y, dark_x] = 255
            print(f"  Fill pass 1b complete: {n_dark} dark texels replaced from nearest bright neighbour.")
        else:
            print(f"  Fill pass 1b: skipped — not enough bright sources ({n_bright})")

    # -------------------------------------------------------------------------
    # Pass 1c — 3-D colour-consistency correction (desaturated-outlier fix)
    # -------------------------------------------------------------------------
    def correct_color_consistency(self, color_np: np.ndarray, uv_mask: np.ndarray,
                                  skip: bool = False,
                                  p1c_k: int = 96, p1c_iterations: int = 5,
                                  p1c_frac: float = 0.58, p1c_diff: int = 18,
                                  p1b_relative_mode: bool = True) -> None:
        """
        For each UV texel, look up K nearest 3-D neighbours.  If this texel's
        saturation is far below the neighbourhood median (chalk-white arm tops,
        washed-out shoulders) replace it with the neighbourhood median colour.

        The earlier passes fix uncovered texels and rescue dark ones but leave
        alone any texel that is (a) covered by some view and (b) not dark.
        That includes "chalk-white" / desaturated grey patches on arm tops and
        shoulder crowns that received a grazing-angle projection and ended up
        washed-out even though they are technically covered and bright enough.

        The saturation comparison focuses specifically on the grey-vs-skin
        outlier pattern while leaving legitimate variation (hair, lips, face
        features) completely untouched.

        Runs multiple iterations so corrected pixels from each pass become
        valid colour sources for their neighbours (flood-fill effect).  Stops
        early if no outliers remain.  In relative mode (adaptive floor more
        permissive) grey patches tend to be larger so full K + iterations are
        used (K=96, 5 iterations).  In absolute mode K=24, 1 iteration.
        """
        if skip:
            return

        if p1b_relative_mode:
            CONSISTENCY_K   = p1c_k
            _p1c_iterations = p1c_iterations
        else:
            CONSISTENCY_K   = 24
            _p1c_iterations = 1

        CONSISTENCY_FRAC        = p1c_frac
        CONSISTENCY_DIFF        = p1c_diff
        CONSISTENCY_MIN_NBR_SAT = 18

        _c1c_all_y, _c1c_all_x = np.where(uv_mask)
        _c1c_n = len(_c1c_all_y)

        if _c1c_n <= CONSISTENCY_K * 4:
            return

        _c1c_pos    = self.tex_pos_np[_c1c_all_y, _c1c_all_x]
        _c1c_stride = max(1, _c1c_n // 200_000)
        _c1c_ty     = _c1c_all_y[::_c1c_stride]
        _c1c_tx     = _c1c_all_x[::_c1c_stride]
        _c1c_tp     = self.tex_pos_np[_c1c_ty, _c1c_tx]
        _c1c_k      = min(CONSISTENCY_K, len(_c1c_tp) - 1)
        print(f"  Pass 1c (3-D colour consistency): {_c1c_n} texels, "
              f"tree={len(_c1c_tp)} pts (1-in-{_c1c_stride}), K={_c1c_k}, "
              f"iterations={_p1c_iterations}...")

        _c1c_tree = cKDTree(_c1c_tp)
        _, _c1c_nn = _c1c_tree.query(_c1c_pos, k=_c1c_k, workers=-1)

        _c1c_total_fix = 0
        for _iter in range(_p1c_iterations):
            _c1c_nbr_rgb = color_np[_c1c_ty[_c1c_nn], _c1c_tx[_c1c_nn]]
            _c1c_med_rgb = np.median(_c1c_nbr_rgb, axis=1)

            _c1c_this_rgb = color_np[_c1c_all_y, _c1c_all_x].astype(np.float32)

            _c1c_this_bgr = _c1c_this_rgb[:, ::-1].clip(0, 255).astype(np.uint8).reshape(1, _c1c_n, 3)
            _c1c_med_bgr  = _c1c_med_rgb[:, ::-1].clip(0, 255).astype(np.uint8).reshape(1, _c1c_n, 3)
            _c1c_this_hsv = cv2.cvtColor(_c1c_this_bgr, cv2.COLOR_BGR2HSV)[0]
            _c1c_med_hsv  = cv2.cvtColor(_c1c_med_bgr,  cv2.COLOR_BGR2HSV)[0]

            _c1c_this_sat = _c1c_this_hsv[:, 1].astype(float)
            _c1c_med_sat  = _c1c_med_hsv[:,  1].astype(float)
            _c1c_diff_val = np.abs(_c1c_this_rgb - _c1c_med_rgb).mean(axis=-1)

            _c1c_outlier = (
                (_c1c_med_sat  > CONSISTENCY_MIN_NBR_SAT) &
                (_c1c_this_sat < _c1c_med_sat * CONSISTENCY_FRAC) &
                (_c1c_diff_val > CONSISTENCY_DIFF)
            )
            _c1c_n_fix = int(_c1c_outlier.sum())
            _c1c_total_fix += _c1c_n_fix
            if _c1c_n_fix > 0:
                color_np[_c1c_all_y[_c1c_outlier], _c1c_all_x[_c1c_outlier]] = \
                    _c1c_med_rgb[_c1c_outlier].clip(0, 255).astype(np.uint8)
                print(f"    iter {_iter+1}: {_c1c_n_fix} texels corrected")
            else:
                print(f"    iter {_iter+1}: 0 outliers — converged early.")
                break

        print(f"  Pass 1c complete: {_c1c_total_fix} total corrections over "
              f"{_iter+1} iteration(s) ({100.0*_c1c_total_fix/_c1c_n:.1f}% of UV island).")
        del _c1c_pos, _c1c_nbr_rgb, _c1c_med_rgb, _c1c_this_rgb, _c1c_tree, _c1c_nn

    # -------------------------------------------------------------------------
    # Pass 2 — bleed 8 px outside UV islands (prevents seam black sampling)
    # -------------------------------------------------------------------------
    def seam_bleed(self, color_np: np.ndarray, alpha_np: np.ndarray) -> None:
        """
        Colour-pad 8 px outside UV islands so bilinear sampling at seam edges
        never picks up the black background.

        Alpha is set to 0 in the bleed zone so the Blender material
        (alpha-clip mode) does NOT render those texels independently — this
        eliminates the silhouette "glow/halo" that appeared at the mesh
        perimeter.  The colour value is still available for the GPU's bilinear
        filter to blend correctly at the island boundary.
        """
        uv_mask_u8  = self.uv_mask.astype(np.uint8)
        kernel      = np.ones((3, 3), np.uint8)
        dilated_uv  = cv2.dilate(uv_mask_u8, kernel, iterations=8)
        bleed_mask  = (dilated_uv > 0) & ~self.uv_mask

        n_bleed = int(bleed_mask.sum())
        if n_bleed > 0:
            _, bleed_idx = distance_transform_edt(~self.uv_mask, return_indices=True)
            by, bx = bleed_idx[0], bleed_idx[1]
            for c in range(3):
                color_np[bleed_mask, c] = color_np[by[bleed_mask], bx[bleed_mask], c]
            alpha_np[bleed_mask] = 0
            print(f"  Fill pass 2 (seam bleed): {n_bleed} texels colour-padded "
                  f"(alpha=0 for clean silhouette).")

    # -------------------------------------------------------------------------
    # Interior seam fill — close the black cracks at UV-island boundaries
    # -------------------------------------------------------------------------
    def fill_interior_seams(self, color_np: np.ndarray, alpha_np: np.ndarray,
                            max_px: int = 8, protect_above_frac: float = 0.82) -> None:
        """
        Close the thin BLACK cracks along UV-island seam rings on an OPAQUE
        material.  The mesh is connected across these rings, but the crease faces
        at the seam often project at a grazing angle and stay unpainted (black)
        INSIDE their island.  seam_bleed only pads texels OUTSIDE islands, so it
        never reaches these — hence the surviving cracks.

        This copies the nearest painted colour into every unpainted texel that is
        INSIDE a UV island and within `max_px` of paint.  It is deliberately
        SHORT-RANGE and island-local: it closes seam/edge cracks only, NOT the
        long-range cross-body 3-D fill (front->back bleed).  Larger unpainted
        regions (beyond max_px from any paint) are left untouched.

        HEAD-SAFE: the nearest-colour copy is fine on the body (uniform skin) but
        smears fine FACE detail when the face projects with holes.  So the fill is
        restricted to BELOW the neck (`protect_above_frac` of body height); the
        face/hairline is handled separately (KAN-8), never by this body crack-closer.
        """
        painted = self.uv_mask & (alpha_np > 0)
        holes   = self.uv_mask & ~painted
        if not holes.any() or not painted.any():
            return
        # protect the head/face: skip texels above the neck line (data-driven, no per-char knob)
        yv = self.tex_pos_np[:, :, 1]
        ymin = float(yv[self.uv_mask].min()); ymax = float(yv[self.uv_mask].max())
        neck_y = ymin + protect_above_frac * (ymax - ymin)
        holes = holes & (yv <= neck_y)
        if not holes.any():
            return
        dist, idx = distance_transform_edt(~painted, return_distances=True,
                                           return_indices=True)
        fill = holes & (dist <= max_px)
        if not fill.any():
            return
        ny, nx = idx[0][fill], idx[1][fill]
        for c in range(3):
            color_np[fill, c] = color_np[ny, nx, c]
        alpha_np[fill] = 255
        print(f"  Interior seam fill: {int(fill.sum())} body-seam texels closed "
              f"(<= {max_px}px from paint, below neck y={neck_y:.3f}).")

    # -------------------------------------------------------------------------
    # Colour correction — saturation + brightness within UV islands
    # -------------------------------------------------------------------------
    def colour_correct(self, color_np: np.ndarray, uv_mask: np.ndarray,
                       saturation_boost: float = 1.0,
                       brightness_correction: float = 1.0) -> None:
        """
        Apply HSV saturation boost and/or brightness correction inside UV
        islands only (black background is untouched).

        SV3D renders with flat, washed-out lighting — boosting S restores
        warm skin tones.  Default is (1.0, 1.0) = no change; correction is
        opt-in per run.  Brightness < 1.0 darkens, > 1.0 brightens.  The old
        hardcoded 0.88 was making the back (already darker in SV3D) even
        darker, hence the switch to a tunable widget.
        """
        if saturation_boost == 1.0 and brightness_correction == 1.0:
            return
        color_bgr = color_np[:, :, ::-1].copy()
        color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation_boost != 1.0:
            color_hsv[:, :, 1] = np.clip(color_hsv[:, :, 1] * saturation_boost, 0, 255)
        if brightness_correction != 1.0:
            color_hsv[:, :, 2] = np.clip(color_hsv[:, :, 2] * brightness_correction, 0, 255)
        corrected = cv2.cvtColor(color_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[:, :, ::-1]
        color_np[uv_mask] = corrected[uv_mask]
        print(f"  Colour correction: saturation ×{saturation_boost:.2f}, "
              f"brightness ×{brightness_correction:.2f} applied within UV islands.")
