"""Does Trellis's sculpt have HER face? Its proportions against the photo's, measured right after generation (2026-09-25).

Why: Trellis draws some characters with a face that is not the photo's -- female_elf_00006_ / 00010 / 00011 came out with
mouths 28-43% narrower than their photos (share of the face's width), and everything after the generator faithfully
kept it (user: "too small mouth"). Option 3 of the fix (user's choice): check every new sculpt and draw again, with the
next seed, when it disagrees.

How: the raw sculpt already carries Trellis's colour -- a base colour per voxel (attrs, layout["base_color"]) -- so its
head, each vertex coloured by its nearest voxel, rendered straight from the front (nvdiffrast, orthographic), is a face
MediaPipe reads like the textured render the retopo later makes (mouth share measured on both: within 1-13 points,
RMS 6.1). The same landmarks on the photo; each proportion is a share of the face's width (cheek to cheek, 234-454).

What decides: the MOUTH -- it is what varies draw to draw. Tolerance = 3 x the measurement's own noise (the RMS
difference between the textured render and this one on the same sculpts, 6.1 points -> 18%). Measured on 00004/5/7:
-5..+5%; 00006 -40, 00010 -43, 00011 -28 (00008 -10 passes). The nose and the eyes are reported, not decided on:
Trellis draws the nose 7-10% wide of the photo on EVERY good draw (a new seed does not change it -- that is a job for a
correction, not a redraw) and the eyes 6-22% wide, which the eye step rebuilds from the photo anyway.
"""
import os, json
import numpy as np

MOUTH_TOL = 3 * 0.061             # 3 x RMS(textured - sculpt render) of the mouth share, 7 characters (2026-09-25)
HEAD_SHARE = 0.16                 # the top of the body taken as the head for the render (head + neck)
RES = 1024


def _landmarker():
    import sys
    t2 = os.path.dirname(os.path.abspath(__file__))
    if t2 not in sys.path:
        sys.path.insert(0, t2)
    from projection.landmark_warp import _get_landmarker
    return _get_landmarker()


def landmarks(rgb):
    """478 MediaPipe face landmarks (pixels) on an RGB uint8 image, or None. Whole image first, then the face region
    grown from a first coarse detection is not needed here: the sculpt render is a head filling the frame, and the photo
    goes through the pipeline's own small-face search (make_guides.landmarks_on_render) by the caller."""
    import mediapipe as mp
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    H, W = rgb.shape[:2]
    return np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])


def ratios(lm):
    fw = float(np.linalg.norm(lm[234] - lm[454]))
    return dict(mouth=float(np.linalg.norm(lm[61] - lm[291]) / fw),
                nose=float(np.linalg.norm(lm[64] - lm[294]) / fw),
                eye=float(0.5 * (np.linalg.norm(lm[33] - lm[133]) + np.linalg.norm(lm[362] - lm[263])) / fw))


def sculpt_face(mesh, up=2):
    """Render the sculpt's head from the front with Trellis's own colours. Returns an RGB uint8 image with the face
    (both sides along the depth axis are tried; the one MediaPipe finds a face on is the front), or (None, None)."""
    import torch
    import nvdiffrast.torch as dr
    from scipy.spatial import cKDTree
    V = mesh.vertices.detach().float().cpu().numpy()
    F = mesh.faces.detach().long().cpu().numpy()
    top = V[:, up].max(); H = float(np.ptp(V[:, up]))
    keep = V[:, up] > top - HEAD_SHARE * H
    fk = keep[F].all(1)
    idx = np.nonzero(keep)[0]
    remap = -np.ones(len(V), np.int64); remap[idx] = np.arange(len(idx))
    Vh, Fh = V[idx], remap[F[fk]]
    vox = mesh.origin.detach().float().cpu().numpy() + (mesh.coords.detach().float().cpu().numpy() + 0.5) * float(mesh.voxel_size)
    _, nn = cKDTree(vox).query(Vh, workers=-1)
    C = mesh.attrs[:, mesh.layout["base_color"]].detach().float().cpu().numpy()[nn].clip(0, 1)
    lat = [a for a in range(3) if a != up]
    ctx = dr.RasterizeCudaContext()
    Hh = HEAD_SHARE * H                           # the head region's height
    half = 0.45 * Hh                              # the frame: 0.9 of it, centred on it
    cx = float(Vh[:, lat[0]].mean()); cz = float(top - 0.45 * Hh)
    for dep_ax in (lat[1],):
        for sg in (-1.0, 1.0):                    # camera on the -depth side, then the +depth side
            d = Vh[:, dep_ax]
            zc = ((d - d.min()) / max(np.ptp(d), 1e-9)) * 2 - 1
            if sg > 0:
                zc = -zc
            x = (Vh[:, lat[0]] - cx) / half * (-sg)
            y = (Vh[:, up] - cz) / half
            pos = torch.tensor(np.c_[x, y, zc, np.ones(len(x))], dtype=torch.float32, device="cuda")[None]
            tri = torch.tensor(Fh, dtype=torch.int32, device="cuda")
            rast, _ = dr.rasterize(ctx, pos, tri, resolution=[RES, RES])
            col, _ = dr.interpolate(torch.tensor(C, dtype=torch.float32, device="cuda")[None], rast, tri)
            img = (col[0] * (rast[0, ..., 3:4] > 0)).flip(0).clamp(0, 1).cpu().numpy()
            rgb = (img * 255 + 0.5).astype(np.uint8)
            if landmarks(rgb) is not None:
                return rgb, sg
    return None, None


def check(mesh, photo_lm, up=2):
    """(ok, report dict, render). report has both sets of shares and the deviations; ok is None when no face was found
    on the sculpt (the caller keeps the draw: a check that cannot measure must not throw a sculpt away)."""
    rgb, sg = sculpt_face(mesh, up)
    if rgb is None:
        return None, {"error": "no face found on the sculpt render"}, None
    rs = ratios(landmarks(rgb)); rp = ratios(photo_lm)
    dev = {k: rs[k] / rp[k] - 1.0 for k in rs}
    ok = abs(dev["mouth"]) <= MOUTH_TOL
    rep = dict(ok=bool(ok), sculpt=rs, photo=rp, deviation=dev, mouth_tolerance=MOUTH_TOL, front_side=sg,
               decided_on="mouth", reported_only=["nose", "eye"])
    return ok, rep, rgb


# ---- the photo, and a record of the seeds that drew a wrong face -------------------------------------------------------
_RETOPO_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI-Retopo", "scripts")


def photo_landmarks(image):
    """Landmarks on the character's photo (an IMAGE tensor [1,H,W,3+] or a path), through the pipeline's own search
    (make_guides.landmarks_on_render: whole frame, head crop, windows for a small face). None if no face."""
    import sys, tempfile
    if _RETOPO_SCRIPTS not in sys.path:
        sys.path.insert(0, _RETOPO_SCRIPTS)
    import make_guides as mg
    path = image
    if not isinstance(image, str):
        from PIL import Image
        arr = (image[0, ..., :3].detach().float().cpu().numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
        path = os.path.join(tempfile.gettempdir(), "_face_check_photo.png")
        Image.fromarray(arr).save(path)
    try:
        return mg.landmarks_on_render(path)[1]
    except SystemExit:
        return None


def image_key(image):
    """A key for the photo (the same bytes the sculpt cache hashes), so a character's rejected seeds are remembered."""
    import hashlib
    arr = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def record_path(image):
    import folder_paths
    d = os.path.join(folder_paths.get_output_directory(), "_sculpt_cache", "face_check")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, image_key(image) + ".json")


def load_record(image):
    try:
        return json.load(open(record_path(image), encoding="utf-8"))
    except Exception:
        return {"rejected": {}}


def save_record(image, rec):
    try:
        json.dump(rec, open(record_path(image), "w", encoding="utf-8"), indent=1)
    except Exception as e:
        print("[FaceCheck] could not write the record (%r)" % (e,))
