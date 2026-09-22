"""KAN-9 feature-anchor warp (automatic, scalable, no per-character tuning).

Aligns the projection so the mesh's real features land on the photo's real features:
  anchors = face centre (MediaPipe eyes) + both nipples (auto-detected) -> mesh
  face centre + mesh breast tips.  A thin-plate spline (corners pinned) over those
  few RELIABLE anchors gently warps the sample coordinates so each feature coincides.

Unlike the 478-landmark face warp (which detects on the noisy mesh relief and distorts),
this uses 3 solid anchors + pinned corners, so it can't blow up the face.

Returns a dense warp field (H,W,2) of TARGET sample pixel-positions, indexed by the
texel's current projected (x_pixel, y_pixel); the caller grid_samples it.
"""
import os
import numpy as np
import torch

_MODEL_CANDIDATES = [
    r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models\face_landmarker_v2_with_blendshapes.task",
]
_LMK = None


def _get_lmk():
    global _LMK
    if _LMK is None:
        from mediapipe.tasks import python as _mpp
        from mediapipe.tasks.python import vision as _vis
        m = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), None)
        if m is None:
            raise FileNotFoundError("face_landmarker .task not found")
        _LMK = _vis.FaceLandmarker.create_from_options(_vis.FaceLandmarkerOptions(
            base_options=_mpp.BaseOptions(model_asset_path=m),
            num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))
    return _LMK


ARM_FRACS = (0.45, 0.62, 0.78, 0.92)   # fraction from body centre to fingertip


def _detect_photo_anchors(fimg):
    """fimg: HxWx3 float [0,1] (the front image the projection samples).
    Returns dict with face centre, chin, face height, and L/R nipples (image px)."""
    import mediapipe as mp
    import cv2
    H, W = fimg.shape[:2]
    fg = fimg.sum(-1) > 0.05
    co = np.argwhere(fg)
    if len(co) < 100:
        return None
    fy0, fy1 = co[:, 0].min(), co[:, 0].max()
    icx = float(co[:, 1].mean()); ich = float(fy1 - fy0 + 1); icy = float(fy0 + fy1) / 2.0
    # face via a head crop (face is small in a full-body shot)
    UP = 4
    sy0 = int(fy0); sy1 = int(fy0 + 0.26 * ich)
    sx0 = int(max(0, icx - 0.26 * ich)); sx1 = int(min(W, icx + 0.26 * ich))
    sub = (np.clip(fimg[sy0:sy1, sx0:sx1], 0, 1) * 255).astype(np.uint8)
    if sub.size == 0:
        return None
    subu = cv2.resize(sub, (sub.shape[1] * UP, sub.shape[0] * UP), interpolation=cv2.INTER_LANCZOS4)
    res = _get_lmk().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(subu)))
    if not res.face_landmarks:
        return None
    P = res.face_landmarks[0]; H2, W2 = subu.shape[:2]
    def ic(i): return (sx0 + P[i].x * W2 / UP, sy0 + P[i].y * H2 / UP)
    le = ic(468); re = ic(473)
    face = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    chin = sy0 + max(p.y for p in P) * H2 / UP
    fh = (max(p.y for p in P) - min(p.y for p in P)) * H2 / UP
    # nipples: local-contrast redness blob in the breast band, torso-width, one per side
    by0 = int(chin + 0.5 * fh); by1 = int(chin + 2.0 * fh); half = 0.13 * ich
    red = fimg[:, :, 0] - 0.5 * (fimg[:, :, 1] + fimg[:, :, 2])
    redl = cv2.GaussianBlur(red, (0, 0), 3) - cv2.GaussianBlur(red, (0, 0), max(8.0, 0.06 * ich))
    cols = np.arange(W)[None, :]
    def nip(side):
        m = np.zeros((H, W), bool); m[max(0, by0):min(H, by1)] = True; m &= fg & (np.abs(cols - icx) < half)
        m &= (cols > icx + 0.015 * ich) if side == 'L' else (cols < icx - 0.015 * ich)
        if not m.any():
            return None
        i = np.unravel_index(np.argmax(np.where(m, redl, -9)), m.shape)
        return (float(i[1]), float(i[0]))
    # NAVEL. The warp had only face + two nipples, all in the upper torso, and its influence
    # dies off over ~0.07*height -- so the belly was never corrected and the navel landed low
    # however well the nipples matched. The navel is the mirror of the nipple cue: a local DARK
    # concavity on the midline instead of a local red bump. Band is body-relative (the navel sits
    # ~0.5-0.62 of the way down a standing figure), never in head-heights.
    lum = fimg.mean(-1)
    darkl = (cv2.GaussianBlur(lum, (0, 0), max(8.0, 0.05 * ich))
             - cv2.GaussianBlur(lum, (0, 0), 3))          # >0 where locally darker
    # Band from the figure's OWN proportions, not a fixed body fraction: chin->nipple and
    # nipple->navel are close to equal on a standing figure, so scale the search off the gap we
    # already measured. A fixed 0.50-0.62 band was measured locking onto the CROTCH SHADOW
    # (uv 0.512) instead of the navel (uv ~0.36) -- the crotch is the strongest dark blob on the
    # midline, so the band has to exclude it rather than rely on the peak being the right one.
    _nys = [pa_y for pa_y in (nip('L'), nip('R')) if pa_y is not None]
    if _nys:
        _ny = float(np.mean([q[1] for q in _nys])); _gap = max(_ny - chin, 0.04 * ich)
        ny0 = int(_ny + 0.5 * _gap); ny1 = int(_ny + 1.5 * _gap)
    else:
        ny0 = int(fy0 + 0.33 * ich); ny1 = int(fy0 + 0.47 * ich)
    nm = np.zeros((H, W), bool); nm[max(0, ny0):min(H, ny1)] = True
    nm &= fg & (np.abs(cols - icx) < 0.045 * ich)
    navel = None
    if nm.any():
        resp = np.where(nm, darkl, -9.0)
        i = np.unravel_index(np.argmax(resp), resp.shape)
        peak = float(resp[i]); ref = float(np.median(np.abs(darkl[nm])))
        # Only accept a BLIND find if it clearly dominates its band. On the real projected image
        # the navel measures 0.64x the band median (the belly has plenty of other shading), so
        # this usually declines -- compute_region_offsets then refines using the MESH navel as a
        # prior, which is the reliable route. Never guess from the band alone.
        if peak > max(2.0 * ref, 1e-4):
            navel = (float(i[1]), float(i[0]))
        else:
            print(f"[AnchorWarp] navel: no dominant dark peak in the band "
                  f"(peak {peak:.4f} vs 2x median {2.0*ref:.4f}) -- will try the mesh prior")
    # ARM CENTRELINE ANCHORS. Every anchor so far (face, nipples, navel) is on the TORSO, so the
    # arms are corrected by nothing. Measured: the mesh's arm projects ~24px below the photo's arm
    # while the arm is only ~17px thick, so the front view samples background and the arms come out
    # unpainted. Root cause is proportion: arm-span/height is 0.834 on the mesh vs ~0.91 in the
    # photo, so one global fit that matches the torso cannot match the extremities.
    # Anchors are taken at the same FRACTION from body centre to fingertip on each side, so photo
    # and mesh points correspond anatomically even though the arms are different lengths.
    arms = {}
    _cols_any = np.where(fg.any(0))[0]
    if len(_cols_any):
        for _side, _sgn in (('L', +1), ('R', -1)):
            _tip = _cols_any.max() if _sgn > 0 else _cols_any.min()
            for _i, _f in enumerate(ARM_FRACS):
                _c = int(round(icx + _f * (_tip - icx)))
                if not (0 <= _c < W) or not fg[:, _c].any():
                    continue
                _rows = np.where(fg[:, _c])[0]
                # the arm is a THIN isolated band out here; a tall run means we are still on the
                # torso (or caught hair) and the anchor would be meaningless
                if (_rows.max() - _rows.min() + 1) > 0.12 * ich:
                    continue
                arms[f'A{_side}{_i}'] = (float(_c), float(0.5 * (_rows.min() + _rows.max())))
    out = {'face': face, 'L': nip('L'), 'R': nip('R'), 'N': navel,
           'icx': icx, 'icy': icy, 'ich': ich, '_darkl': darkl, '_fg': fg}
    out.update(arms)
    return out


def _refine_navel(pa, guess, ich):
    """Find the navel in the photo NEAR where the mesh says it is.

    Searching the whole midline band for the darkest spot is unreliable: the navel is a small,
    low-contrast dimple competing with belly shading and the crotch shadow. But the mesh's own
    concavity gives a position that is already close, so only a small window has to be searched.
    Returns (x, y) px or None."""
    darkl = pa.get('_darkl'); fg = pa.get('_fg')
    if darkl is None or guess is None:
        return None
    H, W = darkl.shape
    r = max(6, int(0.035 * ich))
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    x0, x1 = max(0, gx - r), min(W, gx + r + 1)
    y0, y1 = max(0, gy - r), min(H, gy + r + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    win = darkl[y0:y1, x0:x1].copy()
    if fg is not None:
        win = np.where(fg[y0:y1, x0:x1], win, -9.0)
    # Weight by distance from the prior. A flat window lets a stronger blob 25px away win even
    # though the mesh says the navel is here (measured: a 20px-off prior drifted to a wrong spot
    # 28px further). Scoring darkness x a gaussian on distance keeps the choice near the prior
    # unless something much darker is genuinely further out.
    _yy, _xx = np.mgrid[y0:y1, x0:x1]
    _sg = max(4.0, r / 2.0)
    _pri = np.exp(-(((_xx - gx) ** 2 + (_yy - gy) ** 2) / (2.0 * _sg * _sg)))
    win = np.where(win > -8.0, win * _pri, -9.0)
    if not np.isfinite(win).any() or win.max() <= -8.0:
        return None
    j = np.unravel_index(np.argmax(win), win.shape)
    found = (float(x0 + j[1]), float(y0 + j[0]))
    print(f"[AnchorWarp] navel refined near the mesh prior: mesh=({guess[0]:.0f},{guess[1]:.0f}) "
          f"-> photo=({found[0]:.0f},{found[1]:.0f})  (search r={r}px, peak {float(win[j]):.4f})")
    return found


def _mesh_anchors(verts_np, norm_np, icx, icy, ppc, mcx, mcy, s, right=None, up=None):
    """Project the mesh's face centre + breast tips to image px using the same align.
    Breast tip = LOCAL-CONVEXITY peak (the nipple is the sharpest local bump that sticks out past
    its neighbours along the normal). Robust where the old global max-Z grabbed the broad
    sternum/belly; needs no nipple-geometry assumption beyond 'it protrudes'."""
    from scipy.spatial import cKDTree
    hn = (verts_np[:, 1] - verts_np[:, 1].min()) / (verts_np[:, 1].max() - verts_np[:, 1].min() + 1e-9)
    nz = norm_np[:, 2]
    # Project through the ACTUAL camera basis. This used world X and Y directly, which is only
    # the correct projection for az=0/el=0. On the top view the horizontal axis stayed right by
    # luck (the right-vector is +X at az=0 for any elevation) while the vertical used plain Y
    # instead of Y*cos(el) - Z*sin(el): the mesh anchors sat ~155px too high at el=54.4 and ~205px
    # too high once the measured elevation rose to 61, and the warp was shifting that view by up
    # to 166px toward positions computed for a camera that does not exist.
    # The defaults reproduce the previous behaviour exactly, so the front view is unchanged.
    _rv = np.asarray([1.0, 0.0, 0.0]) if right is None else np.asarray(right, dtype=np.float64).ravel()
    _uv = np.asarray([0.0, 1.0, 0.0]) if up    is None else np.asarray(up,    dtype=np.float64).ravel()
    PU = verts_np @ _rv
    PV = verts_np @ _uv
    def pr(pu, pv): return (icx + (pu / (s / 2) - mcx) * ppc, icy - (pv / (s / 2) - mcy) * ppc)
    fm = (nz > 0.5) & (hn > 0.85) & (np.abs(verts_np[:, 0]) < 0.10) & (verts_np[:, 2] > 0)
    face = pr(PU[fm].mean(), PV[fm].mean()) if fm.sum() > 20 else None
    # local convexity: how far each front vertex sticks out past its ~2cm neighbourhood centroid
    conv = np.zeros(len(verts_np)); fr = np.where(nz > 0.1)[0]
    if len(fr) > 100:
        tree = cKDTree(verts_np[fr]); nbs = tree.query_ball_point(verts_np[fr], 0.020, workers=-1)
        for k, j in enumerate(fr):
            nb = nbs[k]
            if len(nb) >= 4:
                conv[j] = float(np.dot(verts_np[j] - verts_np[fr[nb]].mean(0), norm_np[j]))
    # LATERAL BOUND. The nipple search had no limit in x: measured on a real mesh it spanned
    # x = 0.030 .. 0.385, i.e. out to the fingertips, and 639 of its 4511 vertices sat on the arm.
    # It then takes the maximum local convexity, and on an arm that is the elbow -- which is how
    # the left "nipple" anchor landed at image x=792 with the body centre at 456, a 287px error
    # that the warp then applied to the chest. Bound it by the torso's own half-width measured at
    # HIP height, where a T-pose arm cannot reach. Relative to the mesh, no constant.
    _hipband = (hn > 0.33) & (hn < 0.48)
    _halfw = (float(np.percentile(np.abs(verts_np[_hipband, 0]), 98))
              if int(_hipband.sum()) > 50 else float(np.abs(verts_np[:, 0]).max()))
    reg = ((hn > 0.55) & (hn < 0.80) & (verts_np[:, 2] > 0) & (nz > 0.2)
           & (np.abs(verts_np[:, 0]) <= _halfw))
    _hns = {}
    def tip(xs, _key=None):
        sel = reg & xs & (conv > 0)
        if sel.sum() < 5:
            return None
        a = np.where(sel)[0][np.argmax(conv[sel])]
        if _key: _hns[_key] = float(hn[a])
        return pr(PU[a], PV[a])
    # navel = the sharpest local CONCAVITY on the midline below the breast band (conv < 0),
    # i.e. exactly the nipple test with the sign flipped.
    nreg = (hn > 0.40) & (hn < 0.58) & (verts_np[:, 2] > 0) & (nz > 0.5) & (np.abs(verts_np[:, 0]) < 0.03)
    navel = None
    if (nreg & (conv < 0)).sum() >= 5:
        sel = np.where(nreg & (conv < 0))[0]
        a = sel[np.argmin(conv[sel])]
        navel = pr(PU[a], PV[a]); _hns['N'] = float(hn[a])
    _out = {'face': face, 'L': tip(verts_np[:, 0] > 0.03, 'L'),
            'R': tip(verts_np[:, 0] < -0.03, 'R'), 'N': navel}
    # arm anchors: same fractions, measured on the PROJECTED mesh so the pair is anatomically
    # matched rather than pixel-matched (the two arms are different lengths).
    _allpx = icx + (PU / (s / 2) - mcx) * ppc
    _allpy = icy - (PV / (s / 2) - mcy) * ppc
    for _side, _sgn in (('L', +1), ('R', -1)):
        _tip = _allpx.max() if _sgn > 0 else _allpx.min()
        for _i, _f in enumerate(ARM_FRACS):
            _c = icx + _f * (_tip - icx)
            _m = np.abs(_allpx - _c) < max(4.0, 0.004 * abs(_tip - icx))
            if _m.sum() < 12:
                continue
            _out[f'A{_side}{_i}'] = (float(_c), float(np.median(_allpy[_m])))
            _hns[f'A{_side}{_i}'] = float(np.median(hn[_m]))
    _out['hn'] = _hns
    return _out


def compute_global_offset(out_verts, out_normals, fimg, icx, icy, ppc, mcx, mcy, ortho_scale):
    """Return one uniform (dx, dy) pixel shift = MEDIAN of the face + nipple anchor offsets.
    A uniform shift cannot distort the body outline or smear/tear the atlas (a scale would —
    its effect grows with distance from the upper-body anchors and wrecks the legs). On a
    faithful mesh the face and nipple offsets are within a few px, so the median aligns both
    to the detection floor. Returns (0,0) if detection is insufficient; bounded so a bad
    detection can't apply a wild shift."""
    pa = _detect_photo_anchors(fimg)
    if pa is None:
        print("[FeatureOffset] photo face not detected — no shift")
        return (0.0, 0.0)
    ma = _mesh_anchors(out_verts.detach().cpu().numpy(), out_normals.detach().cpu().numpy(),
                       icx, icy, ppc, mcx, mcy, ortho_scale)
    offs = [(pa[k][0] - ma[k][0], pa[k][1] - ma[k][1])
            for k in ('face', 'L', 'R') if pa.get(k) is not None and ma.get(k) is not None]
    if len(offs) < 2:
        print(f"[FeatureOffset] only {len(offs)} anchors — no shift")
        return (0.0, 0.0)
    dx = float(np.median([o[0] for o in offs])); dy = float(np.median([o[1] for o in offs]))
    lim = 0.06 * float(pa['ich'])                       # never more than 6% of body height
    dx = max(-lim, min(lim, dx)); dy = max(-lim, min(lim, dy))
    print(f"[FeatureOffset] global shift dx={dx:.1f} dy={dy:.1f} (median of {len(offs)} anchors)")
    return (dx, dy)


def compute_region_offsets(out_verts, out_normals, fimg, icx, icy, ppc, mcx, mcy, ortho_scale,
                           right=None, up=None):
    """Height-blended alignment: the HEAD lands on the photo's face, the TORSO on the nipples,
    instead of one median shift that (with 2 nipples vs 1 face) gets pulled to the nipples and
    leaves the face a few px low. Returns (face_off, body_off, y_lo, y_hi):
      face_off  = the face anchor's own (dx,dy) px shift  → applied to the head
      body_off  = median of the nipple (dx,dy) shifts      → applied to the torso/legs
      y_lo,y_hi = normalised height [0,1] band (the NECK, head-relative) over which to
                  smoothstep body_off → face_off.
    A pure vertical blend of two uniform shifts: cannot distort the outline or tear the atlas.
    Falls back to a single shift (face==body) if only one region is detected; None if neither."""
    vnp = out_verts.detach().cpu().numpy(); nnp = out_normals.detach().cpu().numpy()
    pa = _detect_photo_anchors(fimg)
    if pa is None:
        print("[RegionOffset] photo face not detected — no shift")
        return None
    ma = _mesh_anchors(vnp, nnp, icx, icy, ppc, mcx, mcy, ortho_scale, right=right, up=up)
    hn = (vnp[:, 1] - vnp[:, 1].min()) / (vnp[:, 1].max() - vnp[:, 1].min() + 1e-9)
    lim = 0.06 * float(pa['ich'])                          # bound: never > 6% of body height
    def _clamp(o): return (max(-lim, min(lim, o[0])), max(-lim, min(lim, o[1])))
    # [AnchorDiag] every anchor pair, so a lying detector is visible in the bake log.
    if pa.get('N') is None and ma.get('N') is not None:
        pa['N'] = _refine_navel(pa, ma['N'], float(pa['ich']))
    for _k in ('face', 'L', 'R', 'N') + tuple(
            f'A{sd}{i}' for sd in ('L', 'R') for i in range(len(ARM_FRACS))):
        _p = pa.get(_k); _m = ma.get(_k)
        if _p is not None and _m is not None:
            print(f"[AnchorDiag] {_k:4s}: photo=({_p[0]:.0f},{_p[1]:.0f}) mesh=({_m[0]:.0f},{_m[1]:.0f}) "
                  f"off=({_p[0]-_m[0]:+.0f},{_p[1]-_m[1]:+.0f})")
        else:
            print(f"[AnchorDiag] {_k:4s}: photo={'ok' if _p is not None else 'MISS'} "
                  f"mesh={'ok' if _m is not None else 'MISS'}")
    # head/face offset + the face anchor's height (drives the neck band, head-relative)
    face_off = None; y_face = 0.90
    if pa.get('face') and ma.get('face'):
        face_off = _clamp((pa['face'][0] - ma['face'][0], pa['face'][1] - ma['face'][1]))
        fm = (nnp[:, 2] > 0.5) & (hn > 0.85) & (np.abs(vnp[:, 0]) < 0.10) & (vnp[:, 2] > 0)
        if fm.sum() > 20:
            y_face = float(hn[fm].mean())
    # torso offset = the detected nipple offsets, VALIDATED BY CONSISTENCY (dynamic, height-
    # relative): all anchors measure the SAME rigid photo<->mesh alignment, so L and R must
    # agree with each other. A rogue redness-blob (tattoo, lips, shadow) shows up as one wild
    # outlier — keep the nipple consistent with the trusted face anchor instead of letting the
    # median drag the torso texture sideways (seen: body dx=-53 pinned at the clamp).
    noffs = [(pa[k][0] - ma[k][0], pa[k][1] - ma[k][1])
             for k in ('L', 'R') if pa.get(k) is not None and ma.get(k) is not None]
    if len(noffs) == 2:
        _tol = 0.035 * float(pa['ich'])
        if max(abs(noffs[0][0] - noffs[1][0]), abs(noffs[0][1] - noffs[1][1])) > _tol:
            if face_off is not None:
                _d = [abs(o[0] - face_off[0]) + abs(o[1] - face_off[1]) for o in noffs]
                _keep = noffs[int(np.argmin(_d))]
                print(f"[RegionOffset] L/R nipple offsets disagree > {_tol:.0f}px — "
                      f"kept the face-consistent one ({_keep[0]:+.0f},{_keep[1]:+.0f})")
                noffs = [_keep]
            else:
                print(f"[RegionOffset] L/R nipple offsets disagree > {_tol:.0f}px, no face anchor — nipples dropped")
                noffs = []
    body_off = _clamp((float(np.median([o[0] for o in noffs])),
                       float(np.median([o[1] for o in noffs])))) if noffs else None
    if face_off is None and body_off is None:
        return None
    if face_off is None: face_off = body_off          # head not found → uniform body shift
    if body_off is None: body_off = face_off          # torso not found → uniform face shift
    # DIFFERENTIAL bound (dynamic): head and torso offsets come from the same rigid pair, so
    # they may differ only slightly (detection floor + slight mesh-proportion drift; good bakes
    # show ~2px). Clamp the body offset to within ±2% of body height AROUND the face offset —
    # a bad nipple detection can no longer shove the torso, while the legit small differential
    # (the whole point of region offsets) passes through untouched.
    if body_off is not face_off:
        _dtol = 0.02 * float(pa['ich'])
        _bdx = face_off[0] + max(-_dtol, min(_dtol, body_off[0] - face_off[0]))
        _bdy = face_off[1] + max(-_dtol, min(_dtol, body_off[1] - face_off[1]))
        if (_bdx, _bdy) != body_off:
            print(f"[RegionOffset] body offset ({body_off[0]:+.1f},{body_off[1]:+.1f}) exceeds "
                  f"face±{_dtol:.0f}px — clamped to ({_bdx:+.1f},{_bdy:+.1f})")
        body_off = (_bdx, _bdy)
    y_hi = y_face - 0.02                               # chin: above here = pure face offset
    y_lo = y_face - 0.13                               # base of neck: below here = pure body offset
    # THIRD ZONE: the belly. Two zones meant the whole torso got the NIPPLE offset, so with the
    # nipples landing right the navel still sat low -- one uniform shift cannot satisfy two
    # features at different heights. Blend the nipple offset into the navel's own offset between
    # their two heights. Still only vertical blending of uniform shifts, so it cannot distort the
    # outline or tear the atlas, exactly like the neck blend above.
    nav_off = None; y_nav = None
    _hnm = ma.get('hn') or {}
    if pa.get('N') is not None and ma.get('N') is not None:
        nav_off = _clamp((pa['N'][0] - ma['N'][0], pa['N'][1] - ma['N'][1]))
        _dtol2 = 0.02 * float(pa['ich'])               # same differential bound as the torso
        nav_off = (face_off[0] + max(-_dtol2, min(_dtol2, nav_off[0] - face_off[0])),
                   face_off[1] + max(-_dtol2, min(_dtol2, nav_off[1] - face_off[1])))
        y_nav = _hnm.get('N')
    _y_nip = [_hnm[k] for k in ('L', 'R') if k in _hnm]
    y_nip = float(np.mean(_y_nip)) if _y_nip else None
    print(f"[RegionOffset] head dx={face_off[0]:.1f} dy={face_off[1]:.1f} | "
          f"body dx={body_off[0]:.1f} dy={body_off[1]:.1f} | neck blend hn=[{y_lo:.2f},{y_hi:.2f}]")
    if nav_off is not None and y_nav is not None and y_nip is not None and y_nip - y_nav > 0.02:
        print(f"[RegionOffset] belly dx={nav_off[0]:.1f} dy={nav_off[1]:.1f} | "
              f"navel blend hn=[{y_nav:.2f},{y_nip:.2f}]")
        return (face_off, body_off, nav_off, float(y_lo), float(y_hi), float(y_nav), float(y_nip))
    if nav_off is None:
        print("[RegionOffset] no navel anchor — torso keeps the nipple offset all the way down")
    return (face_off, body_off, float(y_lo), float(y_hi))


def _offset_at(hn, region_off):
    """The (dx,dy) that compute_region_offsets already applies at normalised height `hn`.
    Mirrors the projector's zone blending so the warp can be made RESIDUAL: without this the
    warp would re-apply a correction the offset stage has already made, doubling it."""
    if not isinstance(region_off, tuple):
        return (0.0, 0.0)
    def _ss(t):
        t = max(0.0, min(1.0, t)); return t * t * (3.0 - 2.0 * t)
    if len(region_off) == 7:
        (fdx, fdy), (bdx, bdy), (ndx, ndy), ylo, yhi, ynav, ynip = region_off
        wb = _ss((hn - ynav) / max(ynip - ynav, 1e-6))
        ox = ndx + (bdx - ndx) * wb; oy = ndy + (bdy - ndy) * wb
        w = _ss((hn - ylo) / max(yhi - ylo, 1e-6))
        return (ox + (fdx - ox) * w, oy + (fdy - oy) * w)
    if len(region_off) == 4:
        (fdx, fdy), (bdx, bdy), ylo, yhi = region_off
        w = _ss((hn - ylo) / max(yhi - ylo, 1e-6))
        return (bdx + (fdx - bdx) * w, bdy + (fdy - bdy) * w)
    if len(region_off) == 2:
        return (float(region_off[0]), float(region_off[1]))
    return (0.0, 0.0)


def compute_feature_anchor_warp(out_verts, out_normals, fimg, icx, icy, ppc, mcx, mcy, ortho_scale,
                                region_off=None, skip_face=False, arms_only=False,
                                right=None, up=None):
    """Returns (warp_field [H,W,2] torch.cuda of target px positions) or None.
    The caller grid_samples it at each texel's projected (x_pixel,y_pixel)."""
    from scipy.interpolate import RBFInterpolator
    H, W = fimg.shape[:2]
    pa = _detect_photo_anchors(fimg)
    if pa is None:
        print("[AnchorWarp] photo face not detected — warp skipped")
        return None
    vnp = out_verts.detach().cpu().numpy(); nnp = out_normals.detach().cpu().numpy()
    ma = _mesh_anchors(vnp, nnp, icx, icy, ppc, mcx, mcy, ortho_scale, right=right, up=up)
    if pa.get('N') is None and ma.get('N') is not None:
        pa['N'] = _refine_navel(pa, ma['N'], float(pa['ich']))
    _hnm = ma.get('hn') or {}
    # The face is already handled per-landmark by the face_landmark_warp; correcting it again
    # here would double-shift it.
    _arm_keys = tuple(f'A{sd}{i}' for sd in ('L', 'R') for i in range(len(ARM_FRACS)))
    # arms_only: a non-front view (e.g. the top view at el~54) sees the face, nipples and
    # navel at a grazing/foreshortened angle where their detection is unreliable, and a bogus
    # torso anchor there would warp geometry the front view already covers correctly. Such a
    # view is only there to reach the arms/shoulders, so restrict it to the arm anchors.
    if arms_only:
        _keys = _arm_keys
    else:
        _keys = (('L', 'R', 'N') if skip_face else ('face', 'L', 'R', 'N')) + _arm_keys
    src = []; tgt = []
    for k in _keys:
        if pa.get(k) is not None and ma.get(k) is not None:
            # RESIDUAL: subtract the uniform/zonal shift the offset stage already applies at this
            # anchor's height, so the warp only supplies what a translation cannot express --
            # e.g. the nipples being 95px apart in the photo but 88px on the mesh, which no single
            # dx can fix (moving one in moves the other out).
            _o = _offset_at(_hnm.get(k, 0.5), region_off)
            src.append(ma[k]); tgt.append((pa[k][0] - _o[0], pa[k][1] - _o[1]))
    _used = [k for k in _keys if pa.get(k) is not None and ma.get(k) is not None]
    print(f"[AnchorWarp] anchors used: {_used}")
    for k in _used:
        print(f"[AnchorDiag] {k:<5}: photo=({pa[k][0]:.0f},{pa[k][1]:.0f}) "
              f"mesh=({ma[k][0]:.0f},{ma[k][1]:.0f}) "
              f"off=({pa[k][0]-ma[k][0]:+.0f},{pa[k][1]-ma[k][1]:+.0f})")
    if len(src) < 2:
        print(f"[AnchorWarp] only {len(src)} usable anchors — warp skipped")
        return None
    src = np.array(src, dtype=np.float32)            # mesh anchor positions (px)
    disp = np.array(tgt, dtype=np.float32) - src     # displacement to each photo feature
    # BOUNDED LOCAL displacement — NOT a global thin-plate spline (that extrapolates
    # wildly over the full atlas and smears the texture). Each anchor's correction is
    # applied only within a gaussian radius and blended; far from every anchor the
    # displacement -> 0 (identity), so legs/arms/back/extremes are untouched and the
    # warp can never blow up. Total shift is capped at max|disp|.
    sigma = max(20.0, 0.07 * float(pa['ich']))
    ys, xs = np.mgrid[0:H, 0:W]
    P = np.stack([xs, ys], -1).astype(np.float32)    # (H,W,2)
    acc = np.zeros((H, W, 2), np.float32); wsum = np.zeros((H, W), np.float32)
    for i in range(len(src)):
        w = np.exp(-((P - src[i]) ** 2).sum(-1) / (2.0 * sigma * sigma)).astype(np.float32)
        acc += w[..., None] * disp[i][None, None, :]; wsum += w
    norm = np.maximum(wsum, 1.0)
    warped = (acc / norm[..., None]).astype(np.float32)   # DISPLACEMENT, added by the caller
    md = float(np.abs(acc / norm[..., None]).max())
    print(f"[AnchorWarp] {len(src)} anchors, LOCAL gaussian (sigma={sigma:.0f}px, max shift={md:.1f}px) — bounded")
    return torch.from_numpy(warped).cuda()
