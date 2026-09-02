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
    return {'face': face, 'L': nip('L'), 'R': nip('R'), 'icx': icx, 'icy': icy, 'ich': ich}


def _mesh_anchors(verts_np, norm_np, icx, icy, ppc, mcx, mcy, s):
    """Project the mesh's face centre + breast tips to image px using the same align.
    Breast tip = LOCAL-CONVEXITY peak (the nipple is the sharpest local bump that sticks out past
    its neighbours along the normal). Robust where the old global max-Z grabbed the broad
    sternum/belly; needs no nipple-geometry assumption beyond 'it protrudes'."""
    from scipy.spatial import cKDTree
    hn = (verts_np[:, 1] - verts_np[:, 1].min()) / (verts_np[:, 1].max() - verts_np[:, 1].min() + 1e-9)
    nz = norm_np[:, 2]
    def pr(px, py): return (icx + (px / (s / 2) - mcx) * ppc, icy - (py / (s / 2) - mcy) * ppc)
    fm = (nz > 0.5) & (hn > 0.85) & (np.abs(verts_np[:, 0]) < 0.10) & (verts_np[:, 2] > 0)
    face = pr(verts_np[fm, 0].mean(), verts_np[fm, 1].mean()) if fm.sum() > 20 else None
    # local convexity: how far each front vertex sticks out past its ~2cm neighbourhood centroid
    conv = np.zeros(len(verts_np)); fr = np.where(nz > 0.1)[0]
    if len(fr) > 100:
        tree = cKDTree(verts_np[fr]); nbs = tree.query_ball_point(verts_np[fr], 0.020, workers=-1)
        for k, j in enumerate(fr):
            nb = nbs[k]
            if len(nb) >= 4:
                conv[j] = float(np.dot(verts_np[j] - verts_np[fr[nb]].mean(0), norm_np[j]))
    reg = (hn > 0.55) & (hn < 0.80) & (verts_np[:, 2] > 0) & (nz > 0.2)
    def tip(xs):
        sel = reg & xs & (conv > 0)
        if sel.sum() < 5:
            return None
        a = np.where(sel)[0][np.argmax(conv[sel])]
        return pr(verts_np[a, 0], verts_np[a, 1])
    return {'face': face, 'L': tip(verts_np[:, 0] > 0.03), 'R': tip(verts_np[:, 0] < -0.03)}


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


def compute_region_offsets(out_verts, out_normals, fimg, icx, icy, ppc, mcx, mcy, ortho_scale):
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
    ma = _mesh_anchors(vnp, nnp, icx, icy, ppc, mcx, mcy, ortho_scale)
    hn = (vnp[:, 1] - vnp[:, 1].min()) / (vnp[:, 1].max() - vnp[:, 1].min() + 1e-9)
    lim = 0.06 * float(pa['ich'])                          # bound: never > 6% of body height
    def _clamp(o): return (max(-lim, min(lim, o[0])), max(-lim, min(lim, o[1])))
    # [AnchorDiag] every anchor pair, so a lying detector is visible in the bake log.
    for _k in ('face', 'L', 'R'):
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
    print(f"[RegionOffset] head dx={face_off[0]:.1f} dy={face_off[1]:.1f} | "
          f"body dx={body_off[0]:.1f} dy={body_off[1]:.1f} | neck blend hn=[{y_lo:.2f},{y_hi:.2f}]")
    return (face_off, body_off, float(y_lo), float(y_hi))


def compute_feature_anchor_warp(out_verts, out_normals, fimg, icx, icy, ppc, mcx, mcy, ortho_scale):
    """Returns (warp_field [H,W,2] torch.cuda of target px positions) or None.
    The caller grid_samples it at each texel's projected (x_pixel,y_pixel)."""
    from scipy.interpolate import RBFInterpolator
    H, W = fimg.shape[:2]
    pa = _detect_photo_anchors(fimg)
    if pa is None:
        print("[AnchorWarp] photo face not detected — warp skipped")
        return None
    vnp = out_verts.detach().cpu().numpy(); nnp = out_normals.detach().cpu().numpy()
    ma = _mesh_anchors(vnp, nnp, icx, icy, ppc, mcx, mcy, ortho_scale)
    src = []; tgt = []
    for k in ('face', 'L', 'R'):
        if pa.get(k) is not None and ma.get(k) is not None:
            src.append(ma[k]); tgt.append(pa[k])
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
    warped = (P + acc / norm[..., None]).astype(np.float32)
    md = float(np.abs(acc / norm[..., None]).max())
    print(f"[AnchorWarp] {len(src)} anchors, LOCAL gaussian (sigma={sigma:.0f}px, max shift={md:.1f}px) — bounded")
    return torch.from_numpy(warped).cuda()
