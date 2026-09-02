"""
Face landmark warp (KAN-9) — per-feature alignment for the front-view projection.

Single front view + one global transform cannot align the photo's face features to
the mesh's, because the Trellis mesh deviates from the photo per-feature and
asymmetrically. This module fixes that for the FACE:

  1. Render the mesh relief in the front-image's own pixel space (same alignment the
     projection uses) -> mesh face features land where the projection currently puts
     them.
  2. Detect 478 MediaPipe FaceLandmarker points on the relief AND on the photo.
  3. Build a thin-plate-spline mapping mesh_landmark -> photo_landmark.
  4. Return a dense warp field (per pixel: where to sample the photo) + a soft face
     mask. The projection loop blends the warped sample position in over the normal
     one inside the face mask only; everything else is untouched.

Validated offline: detection works on the bare relief; warp aligns to ~1px.
All heavy deps (nvdiffrast, mediapipe, scipy) are imported lazily so importing this
module never breaks the texturing node if one is missing.
"""
import os
import numpy as np

_MODEL_CANDIDATES = [
    r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LivePortraitKJ\media_pipe\mp_models\face_landmarker_v2_with_blendshapes.task",
    os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task"),
]

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    model = next((m for m in _MODEL_CANDIDATES if os.path.exists(m)), None)
    if model is None:
        raise FileNotFoundError("face_landmarker_v2_with_blendshapes.task not found")
    _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model),
        num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))
    return _landmarker


def _head_crop_box(fg, row_frac=0.16, core_frac=0.09):
    """Auto head crop from a foreground mask. Rows: top `row_frac` of fg height
    (the head, kept ABOVE the shoulder/arm row of a standing pose). Columns: taken
    from the narrow upper-skull band (top `core_frac`, which has no arms) and padded
    generously so ears + jaw are included without ever catching the arm span."""
    rows = np.where(fg.any(axis=1))[0]
    if len(rows) < 8:
        return None
    ft, fb = int(rows[0]), int(rows[-1]); fh = fb - ft + 1
    cy0, cy1 = ft, ft + max(8, int(row_frac * fh))
    core = fg[ft: ft + max(6, int(core_frac * fh))]      # upper skull only -> no arms
    cols = np.where(core.any(axis=0))[0]
    if len(cols) == 0:
        return None
    pad = int(0.10 * fg.shape[1])                         # generous for ears/jaw
    return (max(0, int(cols[0]) - pad), min(fg.shape[1], int(cols[-1]) + pad), cy0, cy1)


def _detect_478(arr_u8, box, up=3):
    """Detect 478 landmarks inside `box`=(x0,x1,y0,y1), upscaled, mapped to full coords."""
    import cv2
    import mediapipe as mp
    x0, x1, y0, y1 = box
    sub = arr_u8[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    sub = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up), interpolation=cv2.INTER_LANCZOS4)
    res = _get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sub)))
    if not res.face_landmarks:
        return None
    H2, W2 = sub.shape[:2]
    return np.array([[x0 + p.x * W2 / up, y0 + p.y * H2 / up] for p in res.face_landmarks[0]])


def _render_relief(verts, faces, vnormals, ortho_scale, ctx, fg, IH, IW):
    """Render the normalized mesh relief in front-image pixel space (front view,
    body-anchored alignment — identical to the projection). Returns uint8 (IH,IW,3)."""
    import torch
    import nvdiffrast.torch as dr
    s = max(float(ortho_scale), 1e-6)
    uc = verts[:, 0] / (s / 2.0)
    vc = verts[:, 1] / (s / 2.0)
    coords = np.argwhere(fg)
    y0, y1 = int(coords[:, 0].min()), int(coords[:, 0].max())
    img_char_h = float(y1 - y0 + 1)
    img_cx = float(coords[:, 1].mean()); img_cy = float(y0 + y1) / 2.0
    mesh_cx = float((uc.min() + uc.max()) * 0.5)
    mesh_cy = float((vc.min() + vc.max()) * 0.5)
    ppc = img_char_h / float((vc.max() - vc.min()).clamp(min=1e-6))
    xpix = img_cx + (uc - mesh_cx) * ppc
    ypix = img_cy - (vc - mesh_cy) * ppc
    xndc = (xpix + 0.5) / IW * 2 - 1
    yndc = -(((ypix + 0.5) / IH) * 2 - 1)
    d = verts[:, 2]                       # look = -Z, depth ~ +Z
    dmin = d.min(); dsp = (d.max() - dmin).clamp(min=1e-6)
    zndc = 1 - 2 * (d - dmin) / dsp
    clip = torch.stack([xndc, yndc, zndc, torch.ones_like(xndc)], -1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, clip, faces, resolution=[IH, IW])
    hit = rast[0, :, :, 3] > 0
    pn = dr.interpolate(vnormals.unsqueeze(0), rast, faces)[0][0]
    pn = pn / (pn.norm(dim=-1, keepdim=True) + 1e-8)
    lam = (pn[..., 2]).clamp(0, 1)        # dot(normal, -look) = normal.z
    shade = (0.25 + 0.75 * lam)
    hit_np = hit.cpu().numpy()[::-1].copy()
    shade_np = shade.cpu().numpy()[::-1].copy()
    relief = np.full((IH, IW, 3), 0.5, np.float32)
    relief[hit_np] = shade_np[hit_np, None]
    return (relief * 255).clip(0, 255).astype(np.uint8)


def compute_face_landmark_warp(verts, faces, vnormals, front_img_np, ortho_scale, ctx=None):
    """
    verts/faces/vnormals: normalized mesh on cuda (torch). front_img_np: (H,W,3) float[0,1].
    Returns (warp_field, face_mask) as torch cuda tensors:
      warp_field (H,W,2) — for each image pixel, the photo (x,y) the projection should
                           sample instead (identity outside the face bbox).
      face_mask  (H,W)   — soft [0,1] weight; 1 = full warp, 0 = no warp.
    Returns None if detection fails (caller falls back to normal projection).
    """
    import cv2
    import torch
    import nvdiffrast.torch as dr
    from scipy.interpolate import RBFInterpolator

    IH, IW = front_img_np.shape[:2]
    fg = front_img_np.sum(-1) > 0.05
    box = _head_crop_box(fg)
    if box is None:
        return None
    if ctx is None:
        ctx = dr.RasterizeCudaContext()

    relief_u8 = _render_relief(verts, faces, vnormals, ortho_scale, ctx, fg, IH, IW)
    front_u8 = (front_img_np * 255).clip(0, 255).astype(np.uint8)

    mesh_lm = _detect_478(relief_u8, box)
    img_lm = _detect_478(front_u8, box)
    if mesh_lm is None or img_lm is None:
        print(f"[FaceWarp] landmark detection failed "
              f"(mesh={'ok' if mesh_lm is not None else 'NONE'}, "
              f"img={'ok' if img_lm is not None else 'NONE'}) — skipping warp")
        return None

    # residual mismatch the warp will correct (diagnostic)
    _resid = float(np.linalg.norm(mesh_lm - img_lm, axis=1).mean())
    print(f"[FaceWarp] 478 landmarks on relief+photo; mean mesh<->photo offset "
          f"= {_resid:.1f}px (this is the per-feature misalignment being corrected)")

    # TPS: mesh-frame position -> photo position
    rbf = RBFInterpolator(mesh_lm, img_lm, kernel="thin_plate_spline", smoothing=0.0)
    warp = np.stack(np.meshgrid(np.arange(IW), np.arange(IH)), -1).astype(np.float32)  # identity (x,y)
    pad = 12
    bx0 = max(0, int(mesh_lm[:, 0].min()) - pad); bx1 = min(IW, int(mesh_lm[:, 0].max()) + pad)
    by0 = max(0, int(mesh_lm[:, 1].min()) - pad); by1 = min(IH, int(mesh_lm[:, 1].max()) + pad)
    ys, xs = np.mgrid[by0:by1, bx0:bx1]
    tgt = rbf(np.column_stack([xs.ravel(), ys.ravel()])).reshape(by1 - by0, bx1 - bx0, 2)
    warp[by0:by1, bx0:bx1] = tgt.astype(np.float32)

    # soft face mask = blurred convex hull of the mesh landmarks
    mask = np.zeros((IH, IW), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(mesh_lm.astype(np.int32)), 255)
    sigma = max(2.0, (bx1 - bx0) * 0.06)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    face_mask = (mask.astype(np.float32) / 255.0)

    return (torch.from_numpy(warp).cuda(), torch.from_numpy(face_mask).cuda())


def compute_face_region_faces(verts, faces, vnormals, front_img_np, ortho_scale, ctx=None, pad_frac=0.06, mp_image=None):
    """Boolean (n_faces,) marking mesh faces inside the MediaPipe FACE region (eyes/nose/
    mouth/cheeks/jaw — NOT the hairline). MediaPipe recognises a real face on any character,
    so this is robust where the geometric `normal.z > t` guard fails on curved/new faces.
    Used to force the true face out of the hair flag. Returns None if detection fails.

    mp_image: optional uint8 image to run MediaPipe on INSTEAD of the mesh relief. Pass the rembg'd
    front PHOTO here (a real photo MediaPipe detects reliably) so the projection survives when the
    synthetic relief is undetectable. It must be in the SAME pixel frame as front_img_np (so fid maps)."""
    import cv2, torch
    import nvdiffrast.torch as dr
    IH, IW = front_img_np.shape[:2]
    fg = front_img_np.sum(-1) > 0.05
    box = _head_crop_box(fg)
    if box is None:
        return None
    if ctx is None:
        ctx = dr.RasterizeCudaContext()
    # Rasterize the mesh in the SAME front-image alignment the relief uses -> per-pixel face id.
    s = max(float(ortho_scale), 1e-6)
    uc = verts[:, 0] / (s / 2.0); vc = verts[:, 1] / (s / 2.0)
    co = np.argwhere(fg); y0, y1 = int(co[:, 0].min()), int(co[:, 0].max())
    ich = float(y1 - y0 + 1); icx = float(co[:, 1].mean()); icy = float(y0 + y1) / 2.0
    mcx = float((uc.min() + uc.max()) * 0.5); mcy = float((vc.min() + vc.max()) * 0.5)
    ppc = ich / float((vc.max() - vc.min()).clamp(min=1e-6))
    xpix = icx + (uc - mcx) * ppc; ypix = icy - (vc - mcy) * ppc
    xndc = (xpix + 0.5) / IW * 2 - 1; yndc = -(((ypix + 0.5) / IH) * 2 - 1)
    d = verts[:, 2]; dmin = d.min(); dsp = (d.max() - dmin).clamp(min=1e-6); zndc = 1 - 2 * (d - dmin) / dsp
    clip = torch.stack([xndc, yndc, zndc, torch.ones_like(xndc)], -1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, clip, faces, resolution=[IH, IW])
    fid = (rast[0, :, :, 3].long() - 1).cpu().numpy()[::-1].copy()   # flipud to match relief
    if mp_image is not None:                                   # FALLBACK: detect on the real (rembg'd) photo
        lm = _detect_478(mp_image, box)
    else:
        relief = _render_relief(verts, faces, vnormals, ortho_scale, ctx, fg, IH, IW)
        lm = _detect_478(relief, box)
    if lm is None:
        print("[FaceRegion] landmark detection failed — no MediaPipe face region")
        return None
    # convex hull of the face landmarks, padded a touch, filled -> face-region image mask
    hull = cv2.convexHull(lm.astype(np.int32))
    mask = np.zeros((IH, IW), np.uint8); cv2.fillConvexPoly(mask, hull, 255)
    if pad_frac > 0:
        _k = max(1, int(pad_frac * np.sqrt(cv2.contourArea(hull) + 1.0)))
        mask = cv2.dilate(mask, np.ones((_k, _k), np.uint8))
    n_faces = int(faces.shape[0])
    sel = np.zeros(n_faces, bool)
    inside = (mask > 0) & (fid >= 0)
    ids = np.unique(fid[inside])
    sel[ids[ids >= 0]] = True
    print(f"[FaceRegion] MediaPipe face region -> {int(sel.sum())} mesh faces (robust, per-character)")
    return sel


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, cv2, torch, trimesh
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    UV = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer"
    from PIL import Image
    mesh = trimesh.load(os.path.join(UV, "last_seams.glb"), force='mesh')
    front = np.array(Image.open(os.path.join(UV, "_front_real_197.png")).convert("RGB")).astype(np.float32) / 255
    v = torch.from_numpy(mesh.vertices).float().cuda()
    for _ in range(2):
        c = (v.min(0).values + v.max(0).values) * 0.5; v = v - c
        r = torch.sqrt((v ** 2).sum(-1).max()).clamp(min=1e-6); v = v * (1.15 / (r * 2))
    f = torch.from_numpy(mesh.faces).int().cuda()
    vn = torch.from_numpy(np.asarray(mesh.vertex_normals)).float().cuda()
    vn = vn / (vn.norm(dim=-1, keepdim=True) + 1e-8)
    out = compute_face_landmark_warp(v, f, vn, front, 1.15)
    if out is None:
        print("SELF-TEST: returned None"); sys.exit(1)
    warp, mask = out
    warp_np = warp.cpu().numpy(); mask_np = mask.cpu().numpy()
    IH, IW = front.shape[:2]
    sx = np.clip(warp_np[..., 0], 0, IW - 1).astype(np.int32)
    sy = np.clip(warp_np[..., 1], 0, IH - 1).astype(np.int32)
    sampled = front[sy, sx]
    m3 = mask_np[..., None]
    result = (front * (1 - m3) + sampled * m3)
    Image.fromarray((result * 255).astype(np.uint8)).save(os.path.join(UV, "_lw_selftest_corrected.png"))
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(os.path.join(UV, "_lw_selftest_mask.png"))
    print(f"SELF-TEST ok: mask coverage {int((mask_np>0.5).sum())}px; "
          f"saved _lw_selftest_corrected.png + _lw_selftest_mask.png")
