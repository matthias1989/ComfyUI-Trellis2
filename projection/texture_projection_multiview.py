"""
Multi-View Texture Projection for Trellis2

Pipeline:
  A) UV-space rasterization (once, shared across all views):
     - Rasterize mesh using UV coords as clip positions
     - For every texel: interpolate world-space position, normal, face_id

  B) Per-view projection (per view):
     - Rasterize mesh from camera -> face_id per camera pixel (occlusion)
     - Detect character pixel bounds in source image (background removal)
     - Map mesh world-space bounds -> image pixel bounds
     - For each UV texel:
         1. Project 3D position to camera space (world units)
         2. Map world coords -> image pixel coords
         3. Occlusion: face_id match (exact, integer comparison)
         4. Sample colour from source image
         5. Accumulate colour * weight (dot(normal, -look)^exp)

  C) Normalize + inpaint holes

Coordinate system (Y-up world):
  +X = character's LEFT  (from viewer's perspective looking at front)
  +Y = up
  +Z = character's front (toward viewer at az=0)

  Azimuth  0   -> camera at +Z, looking toward -Z (front view)
  Azimuth  90  -> camera at +X, looking toward -X
  Azimuth  180 -> camera at -Z, looking toward +Z (back view)
  Elevation 0  -> horizontal
  Elevation 90 -> camera above, looking down

Camera axes (derived):
  look  = normalize(origin - cam_pos)  (toward origin)
  right = normalize(look x world_up)   -- points to IMAGE right
  up    = normalize(right x look)      -- points to IMAGE up

  At az=0: cam_pos=(0,0,1), look=(0,0,-1)
    right = cross((0,0,-1), (0,1,0)) = (0*0-(-1)*1, (-1)*0-0*0, 0*1-0*0) = (1,0,0)  -> +X
    This means image-right = world +X = character's LEFT from viewer.
    The source images are rendered from the same camera convention, so this is consistent.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import nvdiffrast.torch as dr

_GENSEAMS_VIZ = ('C:/applications/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/uv_transfer/_genseams_viz.npz')
import cv2
import trimesh
import gc
import math

import cumesh as CuMesh

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def get_camera_vectors(azimuth: float, elevation: float, device='cuda'):
    az = math.radians(azimuth)
    el = math.radians(elevation)

    cx = math.sin(az) * math.cos(el)
    cy = math.sin(el)
    cz = math.cos(az) * math.cos(el)

    look = torch.tensor([-cx, -cy, -cz], device=device, dtype=torch.float32)
    look = look / (look.norm() + 1e-8)
    world_up = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)

    right = torch.cross(look, world_up, dim=-1)
    if right.norm() < 1e-6:
        if look[1] > 0.0:  
            world_fwd = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)
        else:              
            world_fwd = torch.tensor([0., 0., -1.], device=device, dtype=torch.float32)
        right = torch.cross(look, world_fwd, dim=-1)        
    right = right / (right.norm() + 1e-8)

    up = torch.cross(right, look, dim=-1)
    up = up / (up.norm() + 1e-8)

    return look, right, up

def build_ortho_clip_verts(vertices, right, up, look, ortho_scale):
    """
    Matches Blender's Orthographic Camera.
    ortho_scale = 2.0 means camera views from -1.0 to +1.0 in world units.
    """
    s = max(float(ortho_scale), 1e-6)
    x = (vertices * right).sum(-1) / (s / 2.0)
    y = (vertices * up).sum(-1)    / (s / 2.0)
    d = -(vertices * look).sum(-1)
    
    # Standard depth remapping for nvdiffrast
    d_min  = d.min()
    d_span = (d.max() - d_min).clamp(min=1e-6)
    z      = 1.0 - 2.0 * (d - d_min) / d_span
    return torch.stack([x, y, z, torch.ones_like(x)], dim=-1).unsqueeze(0)

def project_texels_to_image(tex_pos, right, up, ortho_scale):
    """
    Converts 3D world positions to grid_sample compatible coordinates.
    """
    s = max(float(ortho_scale), 1e-6)
    
    # Map world units to clip space [-1, 1]
    # In Blender, ortho_scale is the full width, so radius is s/2
    u_clip = (tex_pos * right).sum(-1) / (s / 2.0)
    v_clip = (tex_pos * up).sum(-1) / (s / 2.0)
    
    # grid_sample expects v = -1 at the TOP of the image.
    # Our 'up' vector gives positive values at the top, so we invert it for sampling.
    u_samp = u_clip
    v_samp = -v_clip 
    
    return u_samp, v_samp, u_clip, v_clip


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fit_view_scale_offset(verts_np, az_deg, el_deg, fg_mask, tag="", lower_frac=0.55, ds=4):
    """Recover a view's SCALE and POSITION when its silhouette is cropped.

    The standard fit is  pix_per_clip = img_char_h / mesh_v_span  with the mesh centred on the
    silhouette's bbox midpoint. Both inputs are wrong when the figure runs off the top and bottom
    of the frame, which every generated side view does (bbox y[0,1023]): the height is truncated,
    so the scale is too small, and the midpoint of a truncated silhouette is not the body's
    midpoint, so the position is off. Measured symptoms: the ear and the hand land too high on the
    side views, and the chest projects past the profile's forward edge.

    Fits ONE uniform scale and ONE offset for the whole view by maximising silhouette overlap.
    This is deliberately NOT the per-row correction that was tried and rejected: a similarity
    transform has no per-scanline component, so it cannot band the hair.

    Fitted on the LOWER body only (mesh height below `lower_frac`, image rows below the matching
    line). Hair dominates the upper silhouette and mesh hair differs from generated hair, so
    including it would drag the fit; the torso and legs are where the two actually agree.
    Returns (scale, dx, dy) -- (1.0, 0, 0) when the silhouette is not cropped or the fit fails.
    """
    H, W = fg_mask.shape[:2]
    ys, xs = np.nonzero(fg_mask)
    if len(ys) < 200:
        return 1.0, 0.0, 0.0
    y0, y1 = int(ys.min()), int(ys.max())
    if y0 > 0 and y1 < H - 1:
        return 1.0, 0.0, 0.0                       # uncropped: existing fit is sound, leave it
    ich = float(y1 - y0 + 1); icx = float(xs.mean()); icy = float(y0 + y1) / 2.0

    azr = math.radians(float(az_deg)); elr = math.radians(float(el_deg))
    cam = np.array([math.sin(azr) * math.cos(elr), math.sin(elr), math.cos(azr) * math.cos(elr)])
    look = -cam
    right = np.cross(look, np.array([0.0, 1.0, 0.0]))
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        return 1.0, 0.0, 0.0
    right = right / nr
    up = np.cross(right, look)
    Vs = np.asarray(verts_np, dtype=np.float64)
    hn = (Vs[:, 1] - Vs[:, 1].min()) / max(float(Vs[:, 1].max() - Vs[:, 1].min()), 1e-9)
    low = hn < lower_frac
    if int(low.sum()) < 500:
        return 1.0, 0.0, 0.0
    Vl = Vs[low]
    if len(Vl) > 20000:
        Vl = Vl[:: max(1, len(Vl) // 20000)]
    u = Vl @ right; v = Vl @ up
    vall = Vs @ up
    mcx = float((Vs @ right).min() + (Vs @ right).max()) * 0.5
    mcy = float(vall.min() + vall.max()) * 0.5
    span = float(vall.max() - vall.min())
    if span < 1e-9:
        return 1.0, 0.0, 0.0
    ppc = ich / span
    X0 = icx + (u - mcx) * ppc
    Y0 = icy - (v - mcy) * ppc

    ycut = int(y0 + (1.0 - lower_frac) * (y1 - y0))
    img = fg_mask.copy(); img[:max(ycut, 0)] = False
    gh, gw = H // ds, W // ds
    ig = img[:gh * ds, :gw * ds].reshape(gh, ds, gw, ds).any(axis=(1, 3))
    ic = int(ig.sum())
    if ic < 50:
        return 1.0, 0.0, 0.0

    def score(sc, dx, dy):
        gx = np.clip(((icx + (X0 - icx) * sc + dx) / ds).astype(np.int32), 0, gw - 1)
        gy = np.clip(((icy + (Y0 - icy) * sc + dy) / ds).astype(np.int32), 0, gh - 1)
        m = np.zeros((gh, gw), bool); m[gy, gx] = True
        inter = int((m & ig).sum())
        return inter / float(ic + int(m.sum()) - inter + 1e-9)

    best = (score(1.0, 0.0, 0.0), 1.0, 0.0, 0.0)
    for sc in np.arange(0.80, 1.65, 0.05):
        for dx in range(-40, 41, 8):
            for dy in range(-80, 81, 8):
                v_ = score(sc, dx, dy)
                if v_ > best[0]:
                    best = (v_, float(sc), float(dx), float(dy))
    _, bs, bdx, bdy = best
    for sc in np.arange(bs - 0.05, bs + 0.051, 0.01):
        for dx in range(int(bdx) - 8, int(bdx) + 9, 2):
            for dy in range(int(bdy) - 8, int(bdy) + 9, 2):
                v_ = score(sc, dx, dy)
                if v_ > best[0]:
                    best = (v_, float(sc), float(dx), float(dy))
    if best[1] <= 0.81 or best[1] >= 1.63:
        print(f"[FitScale]{tag} optimum pinned at the search boundary (x{best[1]:.2f}) -- that is a "
              f"degenerate fit, not a measurement; keeping the height-based fit")
        return 1.0, 0.0, 0.0
    print(f"[FitScale]{tag} silhouette cropped -> scale x{best[1]:.3f} dx={best[2]:+.0f} "
          f"dy={best[3]:+.0f} (lower-body overlap {100*best[0]:.1f}%, was "
          f"{100*score(1.0,0.0,0.0):.1f}%)")
    return best[1], best[2], best[3]


def _silhouette_iou(verts_np, az_deg, el_deg, fg_mask):
    """Rasterise the mesh silhouette for a candidate camera and score it against the image's
    own foreground. Uses the SAME centre+height fit the projector uses, so the number reflects
    what the projection would actually do."""
    ys, xs = np.nonzero(fg_mask)
    if len(ys) < 100:
        return 0.0
    H, W = fg_mask.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max())
    ich = float(y1 - y0 + 1); icx = float(xs.mean()); icy = float(y0 + y1) / 2.0
    azr = math.radians(float(az_deg)); elr = math.radians(float(el_deg))
    cam = np.array([math.sin(azr) * math.cos(elr), math.sin(elr), math.cos(azr) * math.cos(elr)])
    look = -cam
    right = np.cross(look, np.array([0.0, 1.0, 0.0]))
    nrm = np.linalg.norm(right)
    if nrm < 1e-6:
        return 0.0
    right = right / nrm
    up = np.cross(right, look)
    u = verts_np @ right; v = verts_np @ up
    vs = float(v.max() - v.min())
    if vs < 1e-9:
        return 0.0
    mcx = (u.min() + u.max()) * 0.5; mcy = (v.min() + v.max()) * 0.5
    ppc = ich / vs
    X = np.clip(np.round(icx + (u - mcx) * ppc).astype(np.int64), 0, W - 1)
    Y = np.clip(np.round(icy - (v - mcy) * ppc).astype(np.int64), 0, H - 1)
    m = np.zeros((H, W), dtype=bool); m[Y, X] = True
    for _ in range(3):                      # close the vertex splat into a solid silhouette
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return 100.0 * float((m & fg_mask).sum()) / max(float((m | fg_mask).sum()), 1.0)


def build_silhouette_warp(vx, vy, fg_mask, smooth=9):
    """Per-row correspondence between the MESH silhouette and the IMAGE silhouette.

    The anchor warp needs detectable features (face, nipples, navel, arm centrelines) and is
    therefore limited to near-front views. Side views get no correction at all, and they need one:
    measured on a real bake, the mesh's chest projects 21px PAST the generated profile's forward
    silhouette edge at upper chest, 9px at the nipple line. The generated character is simply
    shallower front-to-back than the mesh. Those forward texels then sample background and are
    dropped ("the side image stops too early"), while the texels just inside sample the shaded
    silhouette rim and paint it at the wrong depth -- the misplaced second nipple.

    A profile needs no feature detection: every row of the silhouette is a correspondence. This
    returns a global vertical shift plus per-row forward/rear edge corrections, which is a far
    denser fit than the 4-8 anchors the front view uses, and is purely relative.

    Returns (dy, meshL, meshR, dxL, dxR, valid) as float32 arrays indexed by image row, or None.
    """
    H, W = fg_mask.shape[:2]
    rows = np.arange(H)
    # image silhouette extents per row
    iL = np.full(H, np.nan); iR = np.full(H, np.nan)
    _any = fg_mask.any(1)
    if int(_any.sum()) < 20:
        return None
    idx = np.where(_any)[0]
    iL[idx] = np.argmax(fg_mask[idx], axis=1)
    iR[idx] = W - 1 - np.argmax(fg_mask[idx][:, ::-1], axis=1)

    # per-row mesh extents, computed ONCE. A vertical shift of the mesh is just a roll of
    # these arrays, so the dy search costs nothing extra.
    yi = np.rint(vy).astype(np.int64)
    ok = (yi >= 0) & (yi < H)
    if int(ok.sum()) < 100:
        return None
    mL0 = np.full(H, np.inf); mR0 = np.full(H, -np.inf)
    np.minimum.at(mL0, yi[ok], vx[ok])
    np.maximum.at(mR0, yi[ok], vx[ok])
    cnt = np.bincount(yi[ok], minlength=H)[:H]
    base_valid = cnt > 4
    mL0[~base_valid] = np.nan; mR0[~base_valid] = np.nan

    def mesh_edges(shift):
        return (np.roll(mL0, shift), np.roll(mR0, shift), np.roll(base_valid, shift))

    # global vertical shift: maximise per-row interval overlap with the image silhouette
    best = (None, 0)
    for sh in range(-60, 61, 2):
        mL, mR, mv = mesh_edges(sh)
        v = mv & _any & ~np.isnan(mL)
        if int(v.sum()) < 20:
            continue
        ov = np.minimum(mR[v], iR[v]) - np.maximum(mL[v], iL[v])
        un = np.maximum(mR[v], iR[v]) - np.minimum(mL[v], iL[v])
        score = float(np.clip(ov, 0, None).sum() / max(un.sum(), 1e-6))
        if best[0] is None or score > best[0]:
            best = (score, sh)
    dy = int(best[1])
    mL, mR, mv = mesh_edges(dy)
    valid = mv & _any & ~np.isnan(mL) & ~np.isnan(iL)
    if int(valid.sum()) < 20:
        return None
    dxL = np.zeros(H); dxR = np.zeros(H)
    # DIRECTION: image - mesh. A texel whose mesh edge projects to mL must sample the image at
    # iL, so the sample position moves by (iL - mL). Measured: with this sign the top view's edge
    # correction was med 8.0px; with the sign flipped it became med 34.8px and coverage fell from
    # 28% to 20% -- the flip doubled the error rather than cancelling it.
    dxL[valid] = iL[valid] - mL[valid]
    dxR[valid] = iR[valid] - mR[valid]
    # smooth along y so a single ragged row cannot tear the projection
    if smooth > 1:
        k = np.ones(smooth) / smooth
        for arr in (dxL, dxR):
            arr[:] = np.convolve(arr, k, mode="same")

    # TRUST TAPER. A row whose correction is close to the view's typical correction is a real
    # measurement of a proportion difference; a row needing many times that is the mesh and the
    # generated image disagreeing about what is even there -- which is the HAIR, where the first
    # version applied 70px corrections against a median of 8px and smeared the back view into
    # horizontal bands. Weight each row by med/|dx|, so corrections at or below the median pass
    # through untouched and outliers fade out as 1/|dx|. Relative to the view's own median, so
    # there is no constant and it adapts to any character.
    _mag = np.maximum(np.abs(dxL), np.abs(dxR))
    _sel = valid & (_mag > 0)
    if int(_sel.sum()) > 10:
        _med = float(np.median(_mag[_sel]))
        if _med > 1e-6:
            _w = np.clip(_med / np.maximum(_mag, _med), 0.0, 1.0)
            _tamed = int((valid & (_w < 0.5)).sum())
            dxL *= _w; dxR *= _w
            print(f"[SilWarp] trust taper: median |dx|={_med:.1f}px; "
                  f"{_tamed} rows damped below half (outliers, typically hair)")
    mL = np.nan_to_num(mL); mR = np.nan_to_num(mR)
    return (dy, mL.astype(np.float32), mR.astype(np.float32),
            dxL.astype(np.float32), dxR.astype(np.float32), valid.astype(np.float32))


def fit_view_azimuth(verts_np, az_declared, el_deg, fg_mask, tag=""):
    """Decide whether a view's declared azimuth is on the correct SIDE of the character.

    Measured on a real bake: the image generated as "left side view" matched the mesh at az=270
    (IoU 73.7%) far better than at its declared az=90 (62.6%), and the "right side view" was the
    same the other way (78.1% vs 64.2%). Every side view was therefore projecting the FRONT of
    the body onto the BACK -- the face onto the back of the skull, the breasts onto the shoulder
    blades. What a generator means by "left" is not guaranteed to match the projector's azimuth
    convention, and it need not be the same for the next character, so measure it.

    Only the MIRROR (360 - az) is considered. That is exactly the ambiguity being resolved, and
    it means this can never turn a front view into a back one: 0 and 180 are their own mirrors
    and are left alone by construction.
    """
    az = float(az_declared) % 360.0
    mirror = (360.0 - az) % 360.0
    if abs(((mirror - az + 180.0) % 360.0) - 180.0) < 1.0:
        return az_declared                       # front/back: nothing to decide
    _here = _silhouette_iou(verts_np, az, el_deg, fg_mask)
    _mirr = _silhouette_iou(verts_np, mirror, el_deg, fg_mask)
    if _mirr > _here + 3.0:                      # margin: only switch on a clear win
        print(f"[AzFit]{tag} declared az={az:.0f} (silhouette IoU {_here:.1f}%) but az={mirror:.0f} "
              f"fits better ({_mirr:.1f}%) -> using az={mirror:.0f}. The generated view is of the "
              f"OPPOSITE side from the declared one.")
        return mirror
    print(f"[AzFit]{tag} az={az:.0f} confirmed (IoU {_here:.1f}% vs {_mirr:.1f}% mirrored)")
    return az_declared


def fit_view_elevation(verts_np, az_deg, el_declared, fg_mask, tag=""):
    """Derive a view's TRUE elevation from its own silhouette, instead of trusting the
    angle declared in the workflow.

    A declared angle goes stale the moment the prompt or the reference image changes, and
    it fails silently -- the projection just lands in the wrong place. This bit us twice:
    a hardcoded 30 that was calibrated before the prompt gained "high angle shot", and then
    a hand-derived 54.4 that was wrong because it modelled the silhouette height as
    Sy*cos(el) and ignored the mesh's own depth. The true relation is

        v_span(el) = extent of (y*cos(el) - z*sin(el))

    so the depth term matters as soon as the camera tilts. Rather than encode that formula
    (and get it wrong again), scan elevations and pick the one whose PROJECTED silhouette
    aspect ratio matches the generated image's. Purely relative -- no constants, no tuning,
    and it re-derives itself every bake.

    Azimuth is trusted: it is what the prompt asks for and the silhouette does not constrain
    it independently. Only the elevation is measured.
    """
    ys, xs = np.nonzero(fg_mask)
    if len(ys) < 100:
        return el_declared
    H, W = fg_mask.shape[:2]
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    # A silhouette running off the top or bottom edge has an unknown true height, so the
    # aspect ratio is meaningless. (This is exactly why the cropped BACK view measures as
    # ~23 deg when it is really 0.) Keep the declared angle rather than fit to a lie.
    if y0 <= 0 or y1 >= H - 1:
        print(f"[ElFit]{tag} silhouette touches the frame edge — height unknown, "
              f"keeping declared el={el_declared}")
        return el_declared
    r_img = float(x1 - x0 + 1) / float(max(y1 - y0 + 1, 1))

    Vs = np.asarray(verts_np, dtype=np.float64)
    if len(Vs) > 40000:                      # extents are broad regions; subsampling is safe
        Vs = Vs[:: max(1, len(Vs) // 40000)]
    azr = math.radians(float(az_deg))
    best_err, best_el = None, el_declared
    for cand in np.arange(0.0, 89.5, 0.5):
        elr = math.radians(float(cand))
        cam = np.array([math.sin(azr) * math.cos(elr), math.sin(elr),
                        math.cos(azr) * math.cos(elr)])
        look = -cam
        right = np.cross(look, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(right)
        if n < 1e-6:
            continue
        right = right / n
        up = np.cross(right, look)
        u = Vs @ right
        v = Vs @ up
        v_span = float(v.max() - v.min())
        if v_span < 1e-9:
            continue
        err = abs(float(u.max() - u.min()) / v_span - r_img)
        if best_err is None or err < best_err:
            best_err, best_el = err, float(cand)
    print(f"[ElFit]{tag} declared el={el_declared} -> measured el={best_el:.1f} "
          f"(silhouette aspect {r_img:.3f}, residual {best_err:.4f})")
    return best_el


def _normalize_verts(verts: torch.Tensor, norm_size: float = 1.15) -> torch.Tensor:
    """
    Two-pass centre-and-scale matching Blender's auto_center_and_scale.
    Pass 1: centre on bounding-box midpoint, scale so max-radius * 2 == norm_size.
    Pass 2: re-centre (floating-point drift) and refine scale.
    """
    for _ in range(2):
        center = (verts.min(dim=0)[0] + verts.max(dim=0)[0]) * 0.5
        verts  = verts - center
        r      = torch.sqrt((verts ** 2).sum(dim=-1).max()).clamp(min=1e-6)
        verts  = verts * (norm_size / (r * 2.0))
    return verts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def texture_mesh_with_multiview(
    high_poly_mesh: trimesh.Trimesh,
    images:     list,
    azimuths:   list,
    elevations: list,
    view_weights: list = None,
    texture_size: int   = 4096,
    mesh_cluster_threshold_cone_half_angle_rad: float = 60.0,
    mesh_cluster_refine_iterations:  int   = 0,
    mesh_cluster_global_iterations:  int   = 1,
    mesh_cluster_smooth_strength:    int   = 1,
    blend_exponent: float = 2.0,
    ortho_scale:    float = 1.0,
    norm_size: float = 1.15,
    fill_holes: bool = False,
    blend_texture: bool = True,
    max_hole_size: int = 10,
    use_metallic: bool = True,
    depth_eps: float = 0.002,
    low_poly_mesh: trimesh.Trimesh = None,
    u_pixel_offset: float = 0.0,
    v_pixel_offset: float = 0.0,
    use_cylindrical_uv: bool = False,
    saturation_boost: float = 1.0,
    brightness_correction: float = 1.0,
    debug_color = None,           # None → normal fix; (R,G,B) tuple → paint that colour instead
    p1b_dark_thresh:        int   = 150,
    p1b_sat_thresh:         int   = 55,
    p1b_bright_fraction:    float = 0.6,   # relative mode: floor = median_sat_bright × fraction
    p1b_bright_thresh:      int   = 180,   # absolute mode: fixed floor
    p1b_relative_mode:      bool  = True,  # True = adaptive/relative, False = absolute
    p1c_k:                  int   = 96,
    p1c_iterations:         int   = 5,
    p1c_frac:               float = 0.58,
    p1c_diff:               int   = 18,
    debug_regions:          bool  = False,  # replace texture with hair-region debug colour map
    # Neck-region filter — suppress detected edges in the forward-facing neck zone.
    neck_y_min:             float = 0.30,  # absolute world-Y bottom of neck zone
    neck_y_max:             float = 0.42,  # absolute world-Y top of neck zone
    # Cyan hairline edge filter
    cyan_y_margin:          float = 0.45,  # floor = T_hair - margin*(T_hair-T_seam_floor)
    cyan_z_frac:            float = 0.72,  # keep edges where Z < Z_max * this fraction
    # Hair debug fill: when not None, paint detected hair pixels with this RGB colour.
    # Use (255, 255, 255) as a white overlay to verify coverage before wiring up the
    # actual hair colour from source images.
    hair_debug_color = None,  # e.g. (255, 255, 0) for yellow; None = disabled
    # Flat skin base colour — when not None, overrides ALL UV pixels with this
    # solid colour before hair_debug_color is applied on top.  Use this to hide
    # the projected skin texture while still evaluating hair coverage.
    skin_base_color = None,   # e.g. (210, 180, 155); None = disabled
    # KAN-9 — face-front weighting. When True, the face region (forward-facing,
    # central, upper head) is painted from the SINGLE most-frontal view only;
    # all other views are suppressed there so eyes/nose/mouth don't ghost from
    # multi-view blending. Off by default → existing behaviour unchanged.
    face_front_only: bool = True,
    # KAN-9 — face landmark warp. When True, the front view's face is warped
    # (478-pt TPS) so its features land on the mesh's per-feature, fixing the
    # asymmetric misalignment a single global transform can't. Off by default.
    face_landmark_warp: bool = False,
    # KAN-9 debug — dump the UV atlas after each post-projection stage (projected,
    # fill, dark-rescue, colour-consistency, final) to uv_transfer/_stage_*.png so
    # we can see exactly which stage alters the face. Off by default.
    dump_stages: bool = False,
    # AUTO-FILL — DEFAULT OFF (user does NOT want any fabricated fill; it hides the
    # real projection and makes the atlas a lie). When False, no cylindrical fill.
    fast_skin_fill: bool = True,
    # KAN-9 — skip depth-occlusion for texels facing the camera by more than this dot.
    # A flat FRONT photo shows every front-facing texel; the depth/occlusion test was
    # only needed for multi-view side projection and it false-rejected the curved torso
    # (sternum/under-breast sit behind the breast tips). Grazing/side texels still use it.
    front_skip_depth_dot: float = 0.35,
    # KAN-9 — legacy heavy post-processing (3-D KD-tree fill of uncovered + dark-rescue
    # + colour-consistency). DEFAULT OFF: the user wants the raw projection only, no
    # fabricated fill and no dark-rescue (which previously flattened the face). With it
    # off, uncovered texels simply keep the mesh's own (Trellis) base texture.
    legacy_fill: bool = False,
    # KAN-9 — feature-anchor warp. OFF (reverted to the known-good 84% baseline). The
    # warp + reclaim experiments are inert; do not re-enable until reworked carefully.
    feature_anchor_warp: bool = True,   # residual per-anchor warp (nipple separation etc.)
    # KAN-9 — global feature-offset. OFF: it's a pixel band-aid — a uniform shift that
    # drags the WHOLE body to chase the features ("everything shifted"). The face is now
    # aligned per-feature by the landmark warp instead. The residual nipple offset is mesh
    # breast GEOMETRY (proven: a perfect chin->feet vertical scale still leaves nipples
    # ~8px low), which no projection-space shift can fix — that's the remesh's job.
    feature_offset: bool = True,    # auto uniform face+nipple alignment shift (convexity mesh anchors + delit photo)
):
    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")

    num_views = len(images)

    if view_weights is None:
        view_weights = [1.0] * num_views

    if len(view_weights) != num_views:
        raise ValueError("view_weights must match number of images")
    print(f"[MultiView] {num_views} views | texture={texture_size} | ortho_scale={ortho_scale}")

    if low_poly_mesh is not None:
        mesh = low_poly_mesh
    else:
        mesh = high_poly_mesh
        
    # =========================================================================
    # STEP 1 – UV unwrap
    # =========================================================================
    #vertices = torch.from_numpy(mesh.vertices).float().cuda()
    #faces = torch.from_numpy(mesh.faces).int().cuda()
    #vertices = mesh.vertices.cuda()
    #faces    = mesh.faces.cuda()

    # If the mesh has ColorVisuals (e.g. loaded from a GLB with UV but no material),
    # convert to TextureVisuals so mesh.visual.uv is accessible.
    if hasattr(mesh, 'visual') and not (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None):
        if hasattr(mesh.visual, 'to_texture'):
            try:
                tex_vis = mesh.visual.to_texture()
                if hasattr(tex_vis, 'uv') and tex_vis.uv is not None:
                    print('[MultiView] Converting ColorVisuals -> TextureVisuals')
                    mesh.visual = tex_vis
            except Exception as _e:
                print(f'[MultiView] visual.to_texture() failed: {_e}')

    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        print('Mesh has UV')
        out_verts = torch.from_numpy(mesh.vertices).float().cuda()
        out_faces = torch.from_numpy(mesh.faces).int().cuda()
        out_uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
        
        #mesh.fix_normals()
        if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
            print('Generating Normals ...')
            # Force generation of face normals first (required for vertex normals)
            mesh.face_normals = mesh.generate_face_normals()
            # Generate vertex normals weighted by face area
            mesh.vertex_normals = mesh.generate_vertex_normals()
        
        out_normals = torch.from_numpy(mesh.vertex_normals).float().cuda()        
        #out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
    else:
        print("[MultiView] UV unwrapping...")
        cumesh = CuMesh.CuMesh()
        cumesh.init(torch.from_numpy(mesh.vertices).float().cuda(), torch.from_numpy(mesh.faces).int().cuda())
        out_verts, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": np.radians(mesh_cluster_threshold_cone_half_angle_rad),
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength":   mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )
        out_verts  = out_verts.cuda()
        out_faces  = out_faces.cuda()
        out_uvs    = out_uvs.cuda()

        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.cuda()]
        out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

        del cumesh; gc.collect(); torch.cuda.empty_cache()

    # =========================================================================
    # Normalize mesh to match Blender's auto_center_and_scale
    # =========================================================================
    out_verts   = _normalize_verts(out_verts, norm_size)
    out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

    print(f"[MultiView] Mesh: {out_verts.shape[0]} verts, {out_faces.shape[0]} faces")
    print(f"  X:[{out_verts[:,0].min():.3f},{out_verts[:,0].max():.3f}]"
          f"  Y:[{out_verts[:,1].min():.3f},{out_verts[:,1].max():.3f}]"
          f"  Z:[{out_verts[:,2].min():.3f},{out_verts[:,2].max():.3f}]")

    # =========================================================================
    # CYLINDRICAL UV OVERRIDE
    # Replaces whatever UV was loaded/computed in STEP 1 with a clean
    # cylindrical parametrisation derived from normalised vertex positions.
    #
    #   U = 0.5  → front of character (+Z, faces viewer at azimuth 0)
    #   U = 0/1  → back of character (-Z, seam)
    #   V = 0    → feet,  V = 1 → head
    #
    # Seam-crossing faces (back centreline) get their minority-side vertices
    # duplicated and their U mirrored so nvdiffrast never sees a face that
    # spans > 0.5 of the texture width.
    #
    # Forces blend_texture=False: diffusion texture lives in a different UV
    # space so it cannot be composited here.
    # =========================================================================
    if use_cylindrical_uv:
        print("[MultiView] Building cylindrical UV...")
        xv, yv, zv = out_verts[:, 0], out_verts[:, 1], out_verts[:, 2]

        u_cyl = (0.5 + torch.atan2(xv, zv) / (2.0 * math.pi)) % 1.0
        y_lo, y_hi = yv.min(), yv.max()
        v_cyl = (yv - y_lo) / (y_hi - y_lo + 1e-8)
        out_uvs = torch.stack([u_cyl, v_cyl], dim=-1)   # (N, 2)

        # ── Seam fix: duplicate vertices on minority side of seam faces ──────
        face_u  = u_cyl[out_faces]                       # (F, 3)
        u_span  = face_u.max(dim=1)[0] - face_u.min(dim=1)[0]
        seam_mask = u_span > 0.5
        n_seam  = int(seam_mask.sum().item())
        print(f"  Cylindrical UV: {n_seam} seam-crossing faces → duplicating minority vertices")

        if n_seam > 0:
            new_faces = out_faces.clone()
            extra_v, extra_n, extra_uv = [], [], []
            dup_map = {}          # orig vertex idx → new vertex idx
            n0 = out_verts.shape[0]

            for fi in seam_mask.nonzero(as_tuple=True)[0]:
                f   = out_faces[fi]
                f_u = u_cyl[f]
                n_low = int((f_u < 0.5).sum().item())

                for i in range(3):
                    vi   = int(f[i].item())
                    vi_u = float(f_u[i].item())
                    # Wrap the lone minority-side vertex to the opposite side
                    needs_wrap = (n_low == 1 and vi_u <  0.5) or \
                                 (n_low == 2 and vi_u >= 0.5)
                    if needs_wrap:
                        if vi not in dup_map:
                            dup_map[vi] = n0 + len(extra_v)
                            extra_v.append(out_verts[vi])
                            extra_n.append(out_normals[vi])
                            new_uv_val = out_uvs[vi].clone()
                            new_uv_val[0] = 1.0 - vi_u   # mirror across seam
                            extra_uv.append(new_uv_val)
                        new_faces[fi, i] = dup_map[vi]

            out_faces   = new_faces
            if extra_v:
                out_verts   = torch.cat([out_verts,   torch.stack(extra_v)])
                out_normals = torch.cat([out_normals, torch.stack(extra_n)])
                out_uvs     = torch.cat([out_uvs,     torch.stack(extra_uv)])
                print(f"  Added {len(extra_v)} seam-duplicate vertices "
                      f"({out_verts.shape[0]} total)")

        blend_texture = False   # cylindrical UV ≠ diffusion UV space
        print(f"  U=[{u_cyl.min():.3f},{u_cyl.max():.3f}]  "
              f"V=[{v_cyl.min():.3f},{v_cyl.max():.3f}]")

    # --- STEP 2: UV RASTERIZATION ---
    ctx = dr.RasterizeCudaContext()
    uvs_clip = torch.cat([out_uvs * 2.0 - 1.0, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uvs_clip, out_faces.int(), resolution=[texture_size, texture_size])
    uv_hit_mask = rast[0, :, :, 3] > 0
    
    tex_pos = dr.interpolate(out_verts.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces.int())[0][0]
    tex_normals = tex_normals / (tex_normals.norm(dim=-1, keepdim=True) + 1e-8)
    tex_face_id = rast[0, :, :, 3].long() - 1
    

    # =========================================================================
    # STEP 2b – Bake tangent-space normal map from mesh geometry
    # =========================================================================
    from .normal_map_utils import bake_normal_map
    normalMapTexture = bake_normal_map(
        out_verts, out_faces, out_uvs,
        tex_normals, tex_face_id,
        uv_hit_mask, texture_size,
    )

    # --- STEP 3: VIEW PROJECTION ---
    # debug_regions only needs UV geometry — skip the expensive projection loop.
    _skip_projection = bool(debug_regions)
    if _skip_projection:
        print("[MultiView] debug_regions=True → skipping view projection + fill passes")

    acc_color  = torch.zeros(texture_size, texture_size, 3, device='cuda')
    acc_weight = torch.zeros(texture_size, texture_size, device='cuda')

    # =========================================================================
    # KAN-9 — Face-front weight + front-view selection
    # Build a soft per-texel weight (1 = face-front, 0 = elsewhere). In the loop
    # below, every view EXCEPT the single most-frontal one is multiplied by
    # (1 - face_weight), so the face is sourced from the front view only and the
    # eyes/nose/mouth stop ghosting. Falls back to normal blending if no view is
    # within the front window.
    # =========================================================================
    face_front_w_t  = None      # torch (H, W) in [0, 1]; None → feature disabled
    _front_view_idx = -1
    if face_front_only and not _skip_projection:
        from .face_detection_utils import FaceDetectionUtils, is_front_view
        _norm_az = [abs(((float(a) + 180.0) % 360.0) - 180.0) for a in azimuths]
        _front_view_idx = int(np.argmin(_norm_az)) if _norm_az else -1
        if _front_view_idx < 0 or not is_front_view(azimuths[_front_view_idx]):
            _closest = azimuths[_front_view_idx] if _front_view_idx >= 0 else None
            print(f"[FaceFront] no view within front window (closest az={_closest}) "
                  f"— face weighting disabled this run")
        else:
            _fdu = FaceDetectionUtils(
                texture_size   = texture_size,
                uv_mask        = uv_hit_mask.cpu().numpy().astype(bool),
                tex_pos_np     = tex_pos.cpu().numpy(),
                tex_normals_np = tex_normals.cpu().numpy(),
                neck_y_min     = neck_y_min,
                neck_y_max     = neck_y_max,
                # No hair_mask here: front-only paints fringe texels with the
                # front view's own (hair) colour, and the seam-hair foundation
                # reclaims hair texels afterward — so there's no skin over-paint.
                hair_mask      = None,
            )
            face_front_w_t = torch.from_numpy(_fdu.get_face_front_weight()).cuda()
            print(f"[FaceFront] enabled — front view idx={_front_view_idx} "
                  f"(az={azimuths[_front_view_idx]})")

    # =========================================================================
    # KAN-9 — Face landmark warp (per-feature alignment, front view only)
    # Render the mesh relief in front-image space, detect 478 FaceLandmarker points
    # on relief + photo, build a TPS warp + soft face mask. Applied in the loop to
    # the front view's sample coords so eyes/nose/mouth land on the mesh per-feature.
    # =========================================================================
    face_warp_field = None      # torch (1,2,H,W) target sample positions in px
    face_mask_t     = None      # torch (1,1,H,W) soft weight
    _warp_front_idx = -1
    if face_landmark_warp and not _skip_projection:
        _waz = [abs(((float(a) + 180.0) % 360.0) - 180.0) for a in azimuths]
        _warp_front_idx = int(np.argmin(_waz)) if _waz else -1
        if _warp_front_idx >= 0 and _waz[_warp_front_idx] <= 45.0:
            try:
                from .landmark_warp import compute_face_landmark_warp
                _fimg = np.array(images[_warp_front_idx].convert('RGB')).astype(np.float32) / 255.0
                _res = compute_face_landmark_warp(
                    out_verts, out_faces.int(), out_normals, _fimg, ortho_scale, ctx)
                if _res is not None:
                    _wf, _fm = _res
                    face_warp_field = _wf.permute(2, 0, 1).unsqueeze(0).contiguous()
                    face_mask_t     = _fm.unsqueeze(0).unsqueeze(0).contiguous()
                    print(f"[FaceWarp] enabled — front view idx={_warp_front_idx} "
                          f"(az={azimuths[_warp_front_idx]})")
            except Exception as _we:
                print(f"[FaceWarp] failed ({_we!r}) — projecting without warp")
        else:
            print(f"[FaceWarp] no front-facing view (closest az="
                  f"{azimuths[_warp_front_idx] if _warp_front_idx>=0 else None}) — disabled")

    _anchor_field = None   # KAN-9 feature-anchor warp field (computed once, front view)
    _anchor_view  = -1
    _g_offset = None       # KAN-9 global feature-offset (dx,dy), per near-front view
    _g_offset_view = -1
    # KAN-9 / arms — the region offset and the anchor warp used to be computed ONCE, on the
    # first near-front view, and applied only to that view (_view_i == _g_offset_view). The
    # added top view (az=0, el~54) is also near-front, but arrived second, so it received
    # NEITHER correction: its arm texels kept the raw uniform fit, landed ~31px above the
    # photo's ~38px-thick arm, sampled background and were rejected -> magenta arm tops.
    # Keyed per view now: every near-front view computes and applies its own fit.
    _g_off_by_view = {}
    _anchor_by_view = {}

    # ── CROSS-VIEW EXPOSURE MATCH (on shared texels) ─────────────────────────
    # Each view's exposure went into the blend untouched (fu.colour_correct runs once on the
    # FINAL atlas, globally). Three horizontal views happened to agree, so it never showed;
    # adding the top view exposed it. Measured on torso skin, like for like:
    #     photo 0.573 | LEFT 0.663  RIGHT 0.682  BACK 0.643 | TOP 0.420   (HSV V)
    # Hue (20-27 deg) and saturation (0.28-0.34) are already identical across every view, so
    # this is purely a V error -- and a scalar RGB gain is the exact operator for it: it leaves
    # hue untouched and leaves S = (max-min)/max unchanged, since both terms scale together.
    #
    # The gain is measured on the texels THIS VIEW AND THE REFERENCE VIEW BOTH SEE -- the same
    # points on the body, so differing hair-to-skin ratios cancel out. Comparing whole frames
    # does not work: the top view looks down on a lot of bright hair, which pulled its frame
    # median up to 0.455 against skin's real 0.420 and under-corrected it to a ~1.09 gain when
    # the skin needed ~1.36. No skin classifier, no thresholds, no constants.
    # Index of the most front-on view (azimuth AND elevation nearest zero). Computed
    # unconditionally and owned by nobody else: _front_view_idx is only assigned when
    # face_front_only is enabled, and borrowing it silently disabled a fix once already.
    _frontmost_i = -1
    try:
        _score = [abs(((float(_a) + 180.0) % 360.0) - 180.0) + abs(float(_e))
                  for _a, _e in zip(azimuths, elevations)]
        _frontmost_i = int(np.argmin(_score)) if _score else -1
        if _frontmost_i >= 0:
            print(f"[Exposure] reference = view {_frontmost_i} "
                  f"(az={azimuths[_frontmost_i]}, el={elevations[_frontmost_i]})")
    except Exception as _ee:
        print(f"[Exposure] disabled ({_ee!r})")
    # -- TRIPWIRE CONFIG (loaded up front so a bad VIEW can abort immediately, mid-loop,
    #    instead of after all five views plus seam-hair plus the atlas pass) --
    import os as _os_twc, json as _json_twc
    _tw_on   = _os_twc.environ.get("TRELLIS_COVERAGE_ABORT", "1").strip() != "0"
    _tw_path = _os_twc.path.join(_os_twc.path.dirname(_os_twc.path.dirname(_os_twc.path.dirname(
        _os_twc.path.dirname(_os_twc.path.abspath(__file__))))), "uv_transfer",
        "_coverage_baseline.json")
    _tw_keep = float(_os_twc.environ.get("TRELLIS_VIEW_REGRESS", "0.75"))
    _tw_base = None
    if _tw_on and _os_twc.environ.get("TRELLIS_COVERAGE_RESET", "0").strip() != "1":
        try:
            with open(_tw_path, "r", encoding="utf-8") as _f:
                _tw_base = _json_twc.load(_f)
        except Exception:
            _tw_base = None
    _tw_views = {}        # this run's per-view COVERED%, keyed az/el
    _dropped_views = []    # views excluded because they collapsed (degrade, never abort)
    _own_w = None          # best per-texel weight so far (view-ownership map)
    _own_i = None          # which view held it
    _front_vis = None      # the reference view's accepted mask
    _front_dot = None      # the reference view's per-texel facing
    _depth_eps_rel = None  # depth tolerance as a FRACTION of a view's own depth range
    _sil_by_view = {}     # per-view silhouette warp (side/back views)
    _fw_warned = False    # one-shot notice if the face warp cannot be bounded
    _ref_lum  = None      # reference view's per-texel luma
    _ref_geom = None      # reference view's per-texel geometric acceptance

    # Visit views NEAREST-FIRST (angular distance from the exposure reference), so every view
    # is exposure-matched against surface that has already been corrected. The back view shares
    # literally zero texels with the front -- opposite sides of a body have no common surface --
    # so in workflow order it could never be matched to anything and was left uncorrected while
    # its neighbours were rescaled. Front -> top -> left/right -> back gives the back view the
    # already-corrected sides to match against.
    # _view_i stays the TRUE index, so everything keyed on it (_front_view_idx, _g_off_by_view,
    # _anchor_by_view, face-front weighting) is unaffected, and the accumulation itself is a
    # weighted sum, which is order-independent.
    _order = sorted(range(len(images)),
                    key=lambda _i: abs(((float(azimuths[_i]) + 180.0) % 360.0) - 180.0)
                                   + abs(float(elevations[_i])))
    if not _skip_projection:
        print(f"[Exposure] view order (nearest-first): {_order}")
    for _view_i in ([] if _skip_projection else _order):
        img, az, el, view_w = (images[_view_i], azimuths[_view_i],
                               elevations[_view_i], view_weights[_view_i])
        img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        img_h, img_w = img_np.shape[:2]

        # --- Dilate foreground colors into background to prevent bilinear bleed ---
        img_rgba = img.convert('RGBA')
        img_np_rgba = np.array(img_rgba)

        # --- Dilate foreground colors into background to prevent bilinear bleed ---
        img_rgba = img.convert('RGBA')
        img_np_rgba = np.array(img_rgba)

        # Build foreground mask
        if img.mode != 'RGBA' or img_np_rgba[:, :, 3].min() == 255:
            bg_color = img_np_rgba[0, 0, :3]
            color_diff = np.abs(img_np_rgba[:, :, :3].astype(int) - bg_color.astype(int)).sum(axis=-1)
            fg_mask = (color_diff > 10).astype(np.uint8)
        else:
            fg_mask = (img_np_rgba[:, :, 3] > 127).astype(np.uint8)

        # ── TRIPWIRE: a matte that covers (almost) the whole frame is not a matte ──
        # An un-matted photo (grey studio backdrop, vignetted corners) makes the
        # corner-pixel background test flag EVERY pixel as foreground.  Two things then
        # break silently and the bake still "succeeds":
        #   1. img_char_h / img_char_cy below become the whole frame instead of the
        #      character, so pix_per_clip is wrong -> this view is projected at the
        #      wrong scale and offset (face does not land on the face).
        #   2. sampled_fg is true everywhere, disabling the background gate, so the
        #      backdrop and floor get painted onto the mesh as grey blobs.
        # Route the image through RMBG (background -> Color #000000) like the other views.
        _fg_frac = float(fg_mask.mean())
        if _fg_frac > 0.95:
            print("=" * 78)
            print(f"[MultiView] !! NO BACKGROUND MATTE on view az={az:.0f} "
                  f"— foreground = {100.0 * _fg_frac:.1f}% of the frame")
            print(f"[MultiView] !! corner bg={img_np_rgba[0, 0, :3].tolist()}; alignment scale AND the "
                  f"background gate are both meaningless for this view.")
            print(f"[MultiView] !! The backdrop WILL be baked onto the mesh. Feed this "
                  f"image through RMBG (background=Color #000000) first.")
            print("=" * 78)
        elif _fg_frac < 0.005:
            print(f"[MultiView] !! view az={az:.0f}: foreground is only "
                  f"{100.0 * _fg_frac:.2f}% of the frame — matte looks empty.")

        # Dilate the foreground mask and inpaint the background band
        # Scale dilation with image resolution to prevent bilinear bleed
        base_res = 1024
        scale = max(img_h, img_w) / base_res
        dilate_px = int(max(5, 5 * scale))
        kernel = np.ones((3, 3), np.uint8)
        dilated_fg = cv2.dilate(fg_mask, kernel, iterations=dilate_px)
        band_to_fill = cv2.bitwise_and(dilated_fg, cv2.bitwise_not(fg_mask))

        # ACCEPTANCE mask: ERODE, don't dilate.
        # The RMBG matte fades into the black background over a few pixels, and that halo is
        # INSIDE fg_mask -- `color_diff > 10` against a black corner counts every non-black
        # pixel as foreground -- so the anti-bleed inpaint above (which only fills OUTSIDE fg)
        # never touches it, and the absolute `bright_enough > 0.08` test lets it through.
        # Measured on QWEN_BACK: core brightness median 0.528, but 48.6% of the 6px rim sits
        # between 0.08 and 0.5, with the dark band reaching 7px inside (p90). Projected, that
        # rim is the black line along every arm / leg / torso silhouette.
        # Eroding the acceptance mask by the same margin drops the rim; those texels are then
        # filled from a good neighbour by the fill passes instead of from the halo.
        # Drop only the rim pixels that are ACTUALLY dark, not the whole rim: a flat erosion
        # also discards good silhouette skin, and on a side view (profile ~190px wide) that
        # is a large fraction of the subject -- which then has to be invented by the fills.
        # Dark is judged RELATIVE to this view's own foreground core, so it adapts to the
        # character's tone instead of a fixed threshold, and interior dark content (hair,
        # shadow) is untouched because only the rim band is considered at all.
        erode_px  = int(max(3, 7 * scale))
        _core     = cv2.erode(fg_mask, kernel, iterations=erode_px)
        _rim      = cv2.bitwise_and(fg_mask, cv2.bitwise_not(_core)).astype(bool)
        _lum      = img_np.mean(axis=2)
        _core_med = float(np.median(_lum[_core.astype(bool)])) if _core.any() else 0.0
        _halo     = _rim & (_lum < 0.60 * _core_med)
        # REPAIR the halo instead of DROPPING it.
        # Dropping assumed "the fill passes will recover these from a good neighbour" -- but the
        # fill is off (TRELLIS_FILL=none), so every dropped rim pixel becomes permanent magenta.
        # On a ~100px-wide leg a 6-7px rim on both contours is >10% of it, removed as a stripe
        # along the edge; between the legs no other view can cover it, which is the inner-leg
        # magenta stripe.
        # These pixels are NOT missing data: the surface is plainly visible in the image. Only
        # its colour is wrong, because the matte alpha-blends it toward black over the last few
        # pixels. So undo the blend rather than discard the texel -- a per-pixel gain back toward
        # this view's own foreground core. Hue is preserved (a scalar gain cannot shift it), the
        # gain is relative to the view's own core so it adapts to the character, and it is capped
        # so a near-black pixel cannot be amplified into noise.
        # Nothing is invented here: every repaired pixel is one a camera actually saw.
        import os as _os_hg
        eroded_fg = fg_mask.copy()
        if _os_hg.environ.get("TRELLIS_HALO_DROP", "0").strip() == "1":
            eroded_fg[_halo] = 0                       # previous behaviour, kept as an escape
            _halo_note = f"dropped {int(_halo.sum()):,} dark rim px"
        elif bool(_halo.any()) and _core_med > 1e-6:
            _g = np.clip(_core_med / np.maximum(_lum, 1e-4), 1.0, 4.0)[..., None]
            img_np = np.where(_halo[..., None], np.clip(img_np * _g, 0.0, 1.0), img_np)
            _halo_note = (f"repaired {int(_halo.sum()):,} dark rim px "
                          f"(gain<=4x toward core {_core_med:.3f}), none dropped")
        else:
            _halo_note = "no dark rim found"
        print(f"    [halo-guard az={az:.0f}] {_halo_note}; fg {100.0 * fg_mask.mean():.1f}% -> accept "
              f"{100.0 * eroded_fg.mean():.1f}%  (rim {erode_px}px, core median {_core_med:.3f}, "
              f"dropped {int(_halo.sum()):,} dark rim px)")

        # Inpaint the color channels into the dilated band
        img_rgb_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if int(band_to_fill.sum()) > 0:
            for c in range(3):
                img_rgb_u8[..., c] = cv2.inpaint(img_rgb_u8[..., c], band_to_fill, 3, cv2.INPAINT_NS)
        img_np = img_rgb_u8.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_np).cuda().permute(2, 0, 1).unsqueeze(0).contiguous()

        # Measure this view's elevation from its own silhouette rather than trusting the
        # declared angle (see fit_view_elevation). Must happen before the camera basis is
        # built, since every downstream step -- clip verts, alignment, view_min_dot -- uses it.
        try:
            _vnp = out_verts.detach().cpu().numpy()
            _fgb = fg_mask.astype(bool)
            az = fit_view_azimuth(_vnp, az, el, _fgb, tag=f" az={az:.0f}")
            # Only re-measure elevation where it was DECLARED non-zero, i.e. where we are genuinely
            # unsure (the custom/top view). front/back/left/right are eye-level by construction.
            # Measured on the reframed bake: once the images had margins the fit finally ran on those
            # views and returned el=29 for the BACK and el=9.5 for a side view -- both are 0. It
            # misattributes the arm-span difference (mesh 0.834 vs image ~0.91) to elevation, because a
            # wider silhouette looks like a tilted camera to that formula. The crop had hidden the bug
            # by making the fit refuse.
            if abs(float(el)) > 1e-6:
                el = fit_view_elevation(_vnp, az, el, _fgb, tag=f" az={az:.0f}")
            else:
                print(f"[ElFit] az={az:.0f} declared el=0 (eye-level by construction) - not refitted")
        except Exception as _efe:
            print(f"[ViewFit] failed ({_efe!r}) — keeping declared az={az} el={el}")

        look, right, up = get_camera_vectors(az, el, device='cuda')

        # Create a moderately high-resolution occlusion map.
        # Side views are especially sensitive to self-occlusion leaks where
        # background/body texels can slip through around limbs.
        occ_res = min(2048, img_h, img_w)
        cam_clip = build_ortho_clip_verts(out_verts, right, up, look, ortho_scale)
        
        cam_rast, _ = dr.rasterize(ctx, cam_clip, out_faces.int(), resolution=[occ_res, occ_res])
        cam_hit = cam_rast[0, :, :, 3] > 0

        # Dilate the silhouette by a few pixels so thin features (legs, fingers,
        # hair) don't get dropped by bilinear sampling at the raster boundary.
        cam_hit_np_dil = cam_hit.cpu().numpy().astype(np.uint8)
        cam_hit_np_dil = cv2.dilate(cam_hit_np_dil, np.ones((5, 5), np.uint8), iterations=2)
        cam_hit_dilated = torch.from_numpy(cam_hit_np_dil).bool().cuda()
        cam_hit_img = cam_hit_dilated.float().unsqueeze(0).unsqueeze(0)

        # depth buffer
        cam_depth = dr.interpolate(
            (-(out_verts * look).sum(-1)).unsqueeze(0).unsqueeze(-1),
            cam_rast,
            out_faces.int()
        )[0][0]

        cam_depth = cam_depth.permute(2,0,1).unsqueeze(0)  # for grid_sample
        inf_depth = torch.full_like(cam_depth, torch.finfo(cam_depth.dtype).max)
        cam_depth_occ = torch.where(cam_hit.unsqueeze(0).unsqueeze(0), cam_depth, inf_depth)
        cam_depth_occ = -F.max_pool2d(-cam_depth_occ, kernel_size=3, stride=1, padding=1)

        # Map texels to camera clip space
        _, _, u_clip, v_clip = project_texels_to_image(tex_pos, right, up, ortho_scale)

        # Map mesh world-space bounds -> image pixel bounds
        cam_clip_verts = cam_clip[0]
        mesh_u_min = cam_clip_verts[:, 0].min()
        mesh_u_max = cam_clip_verts[:, 0].max()
        mesh_v_min = cam_clip_verts[:, 1].min()
        mesh_v_max = cam_clip_verts[:, 1].max()

        mesh_v_span = (mesh_v_max - mesh_v_min).clamp(min=1e-6)

        # Find character bounds in image using the foreground mask
        coords = np.argwhere(fg_mask)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            y_min, x_min = 0, 0
            y_max, x_max = img_h - 1, img_w - 1

        # ── Auto-alignment: centre + uniform height-based scale ───────────────
        # Scale: derived from foreground character height (most stable axis).
        img_char_h  = float(y_max - y_min + 1)

        # X: foreground centroid — tracks the character's horizontal position in
        #    each SV3D frame independently (character may not be image-centred).
        # Y: bounding-box midpoint — aligns the mesh top/bottom to the character's
        #    actual top/bottom in the image, eliminating per-view vertical drift
        #    without needing manual v_pixel_offset tuning per character/view.
        #    (Using image-centre here was the old behaviour that caused the drift.)
        img_char_cx = float(coords[:, 1].mean())            # centroid x
        img_char_cy = float(y_min + y_max) / 2.0           # bbox midpoint y

        mesh_cx = float((mesh_u_min + mesh_u_max) * 0.5)  # mesh clip-space centre-x
        mesh_cy = float((mesh_v_min + mesh_v_max) * 0.5)  # mesh clip-space centre-y

        # Single uniform scale derived from character height
        pix_per_clip = img_char_h / float(mesh_v_span)

        # A cropped silhouette makes BOTH the scale and the centre above unreliable. Recover one
        # uniform scale + offset for the whole view. Scaling about the fit centre is exactly
        # ppc *= s with the centre shifted, so everything downstream stays consistent.
        # Never the reference view: it is the colour and geometry anchor for every other
        # view, its silhouette has real margins, and its fit demonstrably works.
        if (_view_i != _frontmost_i
                and _os_twc.environ.get("TRELLIS_FIT_SCALE", "0").strip() == "1"):
            # OFF: the fit is CIRCULAR. It maximises overlap against the very silhouette whose
            # truncation it is meant to correct, so scaling the mesh UP fills more of the
            # cropped legs region and scores better while pushing real geometry out of frame.
            # Measured: scale x1.200 raised the fit metric 63->79% while actual coverage fell
            # 79.3%->61.4% (az270) and 72.6%->46.0% (az90), inbounds 100%->89%.
            try:
                _fs, _fdx, _fdy = fit_view_scale_offset(
                    out_verts.detach().cpu().numpy(), az, el, fg_mask.astype(bool),
                    tag=f" az={az:.0f}")
                pix_per_clip *= _fs
                img_char_cx += _fdx
                img_char_cy += _fdy
            except Exception as _fse:
                print(f"[FitScale] failed ({_fse!r}) — keeping the height-based fit")

        print(f"    [align az={az:.0f} el={el:.0f}] img_centre=({img_char_cx:.1f},{img_char_cy:.1f}) "
              f"char_h={img_char_h:.0f}px  pix/clip={pix_per_clip:.1f}")

        # v_clip positive = up in world = smaller y in image (origin top-left)
        x_pixel = img_char_cx + (u_clip - mesh_cx) * pix_per_clip + u_pixel_offset
        y_pixel = img_char_cy - (v_clip - mesh_cy) * pix_per_clip + v_pixel_offset

        # Heuristic (pre-offset) sample coords. The face-landmark warp field is built in
        # THIS space (the relief is rendered without the body offset), so the warp must be
        # indexed here — not after the body offset shifts x/y, which would mis-look-up the
        # field by the offset amount. Lets the body-offset (below) and face-warp compose.
        _xpix_heur = x_pixel
        _ypix_heur = y_pixel

        # KAN-9 — GLOBAL FEATURE-OFFSET (auto, bounded): one uniform (dx,dy) shift so the
        # mesh's face+nipple anchors land on the photo's. Uniform-only: cannot distort the
        # body outline or smear/tear the atlas (a scale would, by extrapolating to the legs).
        # Computed once on the front-most view.
        _g_offset = _g_off_by_view.get(_view_i)
        if feature_offset and (_g_offset is None) and _view_i == _frontmost_i:
            try:
                from .feature_anchor_warp import compute_region_offsets
                _r = compute_region_offsets(out_verts, out_normals, img_np,
                                            img_char_cx, img_char_cy, pix_per_clip,
                                            mesh_cx, mesh_cy, ortho_scale,
                                            right=right.detach().cpu().numpy(),
                                            up=up.detach().cpu().numpy())
                _g_offset = (0.0, 0.0) if _r is None else _r
                # NOTE: no blanket sanity guard here any more — it discarded the GOOD face offset
                # whenever the body one was bad (face stayed shifted). compute_region_offsets now
                # self-validates per anchor (L/R consistency + body-differential clamp to the face).
                _g_offset_view = _view_i
                _g_off_by_view[_view_i] = _g_offset
            except Exception as _ge:
                print(f"[FeatureOffset] failed ({_ge!r})")
                _g_offset = (0.0, 0.0); _g_off_by_view[_view_i] = _g_offset
        if feature_offset and (_g_offset is not None):
            if isinstance(_g_offset, tuple) and len(_g_offset) == 7:
                # three zones: head -> face, chest -> nipples, belly/legs -> navel
                (_fdx, _fdy), (_bdx, _bdy), (_ndx, _ndy), _ylo, _yhi, _ynav, _ynip = _g_offset
                _ymn = out_verts[:, 1].min(); _yrng = (out_verts[:, 1].max() - _ymn).clamp(min=1e-9)
                _hn = ((tex_pos[..., 1] - _ymn) / _yrng)
                _w = ((_hn - _ylo) / max(_yhi - _ylo, 1e-6)).clamp(0.0, 1.0)
                _w = _w * _w * (3.0 - 2.0 * _w)                    # neck: body -> face
                _wb = ((_hn - _ynav) / max(_ynip - _ynav, 1e-6)).clamp(0.0, 1.0)
                _wb = _wb * _wb * (3.0 - 2.0 * _wb)                # belly: navel -> body
                _ox = _ndx + (_bdx - _ndx) * _wb
                _oy = _ndy + (_bdy - _ndy) * _wb
                x_pixel = x_pixel + _ox + (_fdx - _ox) * _w
                y_pixel = y_pixel + _oy + (_fdy - _oy) * _w
            elif isinstance(_g_offset, tuple) and len(_g_offset) == 4:
                # height-blended: head→face_off, torso→body_off, smoothstep across the neck
                (_fdx, _fdy), (_bdx, _bdy), _ylo, _yhi = _g_offset
                _ymn = out_verts[:, 1].min(); _yrng = (out_verts[:, 1].max() - _ymn).clamp(min=1e-9)
                _hn = ((tex_pos[..., 1] - _ymn) / _yrng)                  # per-texel height [0,1]
                _w = ((_hn - _ylo) / max(_yhi - _ylo, 1e-6)).clamp(0.0, 1.0)
                _w = _w * _w * (3.0 - 2.0 * _w)                           # smoothstep
                x_pixel = x_pixel + _bdx + (_fdx - _bdx) * _w
                y_pixel = y_pixel + _bdy + (_fdy - _bdy) * _w
            else:
                x_pixel = x_pixel + _g_offset[0]
                y_pixel = y_pixel + _g_offset[1]

        # KAN-9 — face landmark warp (front view only): inside the soft face mask,
        # bend each texel's sample position from the mesh-feature location to the
        # photo-feature location (TPS over 478 MediaPipe landmarks). Fixes per-
        # feature / asymmetric face misalignment that no global transform can.
        if face_warp_field is not None and _view_i == _warp_front_idx:
            _gx = (_xpix_heur / float(img_w - 1)) * 2.0 - 1.0   # index in heuristic space
            _gy = (_ypix_heur / float(img_h - 1)) * 2.0 - 1.0   # (pre body-offset)
            _grid = torch.stack([_gx, _gy], dim=-1).unsqueeze(0)
            _tgt = F.grid_sample(face_warp_field, _grid, mode='bilinear',
                                 padding_mode='border', align_corners=True)[0]   # (2,H,W)
            _fw = F.grid_sample(face_mask_t, _grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)[0, 0]   # (H,W)
            # Bound the warp to REAL face geometry. face_mask_t is a purely IMAGE-SPACE convex
            # hull of the landmarks, so on its own it bends any texel whose sample position lands
            # inside that hull -- and on a closed head the temple, the ear and the hair at the
            # BACK of the skull all project into the same image region as the face. Those texels
            # were being dragged onto eyes/nose/mouth, which is the face appearing on the back of
            # the hair, and the lip-shaped marks by the ear.
            # face_front_w_t is per-texel and built from n_z x height x lateral ramps, so it is
            # 3D-aware and front-facing only: geometry that is not on the front of the head gets
            # weight 0 and cannot be warped regardless of where it projects.
            # Geometric bound, ALWAYS available: a texel whose normal points away from this
            # camera is not on the face, whatever its sample position lands on in the image.
            # This is what stops the back of the skull and the hair behind it being dragged onto
            # eyes/nose/mouth. Previously this depended on face_front_w_t, which only exists when
            # the unrelated face_front_only option is enabled -- and it is off in practice, so the
            # bound silently did nothing. Never gate a fix on a feature flag it does not own.
            _facing = (tex_normals * (-look)).sum(-1).clamp(min=0.0)
            _fw = _fw * _facing
            if face_front_w_t is not None:
                _fw = _fw * face_front_w_t          # extra 3-D bound when it is available
            elif not _fw_warned:
                print("[FaceWarp] face_front_only is off — using the geometric facing bound only "
                      "(still 3-D, back-facing geometry cannot be warped)")
                _fw_warned = True
            x_pixel = x_pixel * (1.0 - _fw) + _tgt[0] * _fw
            y_pixel = y_pixel * (1.0 - _fw) + _tgt[1] * _fw

        # KAN-9 — FEATURE-ANCHOR WARP (automatic, scalable): build a sparse thin-plate
        # spline that maps the mesh's face-centre + breast-tips onto the photo's detected
        # face + nipples (corners pinned), then bend every sample coord through it so
        # those features land exactly. Corrects breasts-too-high / face-shift per
        # character with no manual numbers. Computed once on the front-most view.
        _anchor_field = _anchor_by_view.get(_view_i)
        if feature_anchor_warp and (_anchor_field is None) and abs(((az + 180.0) % 360.0) - 180.0) <= 45.0:
            try:
                from .feature_anchor_warp import compute_feature_anchor_warp
                _anchor_field = compute_feature_anchor_warp(
                    out_verts, out_normals, img_np,
                    img_char_cx, img_char_cy, pix_per_clip, mesh_cx, mesh_cy, ortho_scale,
                    region_off=_g_offset if feature_offset else None,
                    skip_face=bool(face_warp_field is not None),
                    arms_only=bool(_view_i != _frontmost_i),
                    right=right.detach().cpu().numpy(),
                    up=up.detach().cpu().numpy())
                _anchor_view = _view_i
                _anchor_by_view[_view_i] = _anchor_field
            except Exception as _awe:
                print(f"[AnchorWarp] failed ({_awe!r}) — projecting without anchor warp")
                _anchor_field = None
        if feature_anchor_warp and (_anchor_field is not None):
            # The field is a RESIDUAL DISPLACEMENT and its gaussians are centred on the MESH
            # anchor pixels, so it must be sampled at the PRE-offset position (same coords the
            # face warp uses) and ADDED -- sampling post-offset would mis-centre every gaussian,
            # and replacing would throw away the region offset entirely.
            _af = _anchor_field.permute(2, 0, 1).unsqueeze(0).contiguous()
            _gx = (_xpix_heur / float(img_w - 1)) * 2.0 - 1.0
            _gy = (_ypix_heur / float(img_h - 1)) * 2.0 - 1.0
            _grid = torch.stack([_gx, _gy], dim=-1).unsqueeze(0)
            _tw = F.grid_sample(_af, _grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0]
            x_pixel = x_pixel + _tw[0]; y_pixel = y_pixel + _tw[1]

        # ── SILHOUETTE WARP (TRELLIS_SIL_WARP=0 to disable) ────────────────────────────
        # It fixes the silhouette (mesh chest projected 21px past the image's forward edge) but
        # it corrects toward the OUTLINE, and interior features are not on the outline: between
        # the forward and rear edges it linearly blends the two edge corrections, which has no
        # reason to be anatomically right and measurably moved the nipple the wrong way. Needs an
        # interior falloff before it can be trusted on by default.
        # The anchor warp needs detectable features and is gated to within 45 deg of front, so
        # side views were projected with NO correction at all. See build_silhouette_warp.
        import os as _os_sw
        _sil_applies = abs(((float(az) + 180.0) % 360.0) - 180.0) > 45.0   # == anchor warp's complement
        # OFF by default (TRELLIS_SIL_WARP=1 to enable). It does fix a real error -- the mesh
        # chest projected 21px past the generated profile's forward silhouette edge -- but an
        # A/B showed it is what smeared the hair on the back view into horizontal bands. The
        # per-row correspondence is least trustworthy exactly where mesh and generated
        # silhouettes disagree most, which is the hair, and there it applies its largest
        # corrections (70px against a median of 8). Before re-enabling it needs to fade out
        # where that disagreement is large, instead of trusting it most where it is worst.
        # TRELLIS_SIL_WARP:  "dy" (default) = vertical shift only | "1"/"full" = also the
        #                    per-row horizontal correction | "0" = off entirely.
        # The two halves of this warp behave completely differently. The per-row HORIZONTAL
        # correction fixes the chest depth error (21px -> 0.5px) but varies row to row, and that
        # is what smeared the hair into horizontal bands -- twice, and the trust taper did not
        # save it because the MEDIAN correction is ~20px, not just the outliers.
        # The VERTICAL part is a single number for the whole view. A rigid translation cannot
        # produce per-row artefacts, so it carries none of that risk, and it is worth having:
        # measured +24 and +26px on the two side views, which is uncorrected vertical drift that
        # puts the side image's EAR about 25px below the mesh's ear -- the ear-shaped patch
        # appearing down in the hair. Side views get no anchor warp, so nothing else fixes it.
        _sil_mode = _os_sw.environ.get("TRELLIS_SIL_WARP", "0").strip().lower()
        if _sil_applies and _sil_mode != "0":
            if _view_i not in _sil_by_view:
                _swv = None
                try:
                    _cc = cam_clip[0]
                    _vx = (img_char_cx + (_cc[:, 0] - mesh_cx) * pix_per_clip).detach().cpu().numpy()
                    _vy = (img_char_cy - (_cc[:, 1] - mesh_cy) * pix_per_clip).detach().cpu().numpy()
                    _swv = build_silhouette_warp(_vx, _vy, fg_mask.astype(bool))
                except Exception as _swe:
                    print(f"[SilWarp] az={az:.0f}: failed ({_swe!r}) — projecting uncorrected")
                    _swv = None
                if _swv is not None:
                    _dy0, _mL0, _mR0, _dxL0, _dxR0, _vld0 = _swv
                    _sel = _vld0 > 0
                    _swv = (_dy0,
                            torch.from_numpy(_mL0).cuda(), torch.from_numpy(_mR0).cuda(),
                            torch.from_numpy(_dxL0).cuda(), torch.from_numpy(_dxR0).cuda(),
                            torch.from_numpy(_vld0).cuda())
                    print(f"[SilWarp] mode={_sil_mode} az={az:.0f} el={el:.0f}: dy={_dy0:+d}px over {int(_sel.sum())} "
                          f"matched rows; edge correction med={float(np.median(np.abs(_dxR0[_sel]))):.1f}px "
                          f"max={float(np.abs(_dxR0[_sel]).max()):.1f}px")
                _sil_by_view[_view_i] = _swv
            _swv = _sil_by_view[_view_i]
            if _swv is not None:
                _dy0, _tL, _tR, _tdL, _tdR, _tv = _swv
                y_pixel = y_pixel + float(_dy0)          # rigid: cannot band
                if _sil_mode in ("1", "full"):
                    _ri = y_pixel.clamp(0, img_h - 1).long()
                    _Lr = _tL[_ri]; _Rr = _tR[_ri]; _wr = _tv[_ri]
                    _tt = ((x_pixel - _Lr) / (_Rr - _Lr).clamp(min=1.0)).clamp(0.0, 1.0)
                    x_pixel = x_pixel + _wr * (_tdL[_ri] + (_tdR[_ri] - _tdL[_ri]) * _tt)

        # ── ALIGN A SIDE VIEW AGAINST THE FRONT VIEW (photometric, report-only) ───────
        # Silhouette overlap and on-foreground coverage BOTH have a degenerate direction:
        # overlap is maximised by scaling UP to fill a truncated silhouette (coverage then fell
        # 79%->61%), coverage is maximised by scaling DOWN so every sample lands inside the
        # silhouette (that pushed the ear higher while the numbers hit record highs). Neither
        # knows WHICH pixel a texel should receive, so neither can locate an alignment.
        #
        # The front view does know: it is well-aligned (anchors within ~8px) and it shares a
        # large band of surface with each side view -- 0.8-1.2M texels, per the exposure match.
        # So score a candidate by how well the side view's sampled colour AGREES with what the
        # front view got for the SAME texels. Normalised cross-correlation, so the exposure
        # difference between views cannot influence it.
        # This cannot be gamed by scaling either way: shrink and the samples come from the wrong
        # places, so agreement drops; grow and they leave the frame, so agreement drops. Only
        # real alignment makes two views agree about the same surface.
        # REPORT ONLY unless TRELLIS_ALIGN=apply -- measure first, act only on a sharp peak.
        if (_view_i != _frontmost_i and _ref_lum is not None and _ref_geom is not None
                and _os_sw.environ.get("TRELLIS_ALIGN", "report").strip() != "0"):
            try:
                _both0 = uv_hit_mask & _ref_geom & (( (tex_normals * (-look)).sum(-1) ) > 0.20)
                _nb = int(_both0.sum().item())
                if _nb > 20000:
                    _ixb = torch.nonzero(_both0, as_tuple=False)
                    if _ixb.shape[0] > 150000:
                        _ixb = _ixb[torch.randperm(_ixb.shape[0], device=_ixb.device)[:150000]]
                    _by, _bx = _ixb[:, 0], _ixb[:, 1]
                    _tgt = _ref_lum[_by, _bx]
                    _tgt = (_tgt - _tgt.mean()) / _tgt.std().clamp(min=1e-6)
                    _xb = x_pixel[_by, _bx]; _yb = y_pixel[_by, _bx]
                    _lum_img = torch.from_numpy(img_np.mean(axis=2).astype(np.float32)).cuda()
                    _fgb = torch.from_numpy(eroded_fg.astype(np.bool_)).cuda()
                    def _ncc(_sc, _dx, _dy):
                        _X = img_char_cx + (_xb - img_char_cx) * _sc + _dx
                        _Y = img_char_cy + (_yb - img_char_cy) * _sc + _dy
                        _ok = (_X >= 0) & (_X < img_w) & (_Y >= 0) & (_Y < img_h)
                        _xi = _X.clamp(0, img_w - 1).long(); _yi = _Y.clamp(0, img_h - 1).long()
                        _ok = _ok & _fgb[_yi, _xi]
                        if int(_ok.sum().item()) < 5000:
                            return -1.0
                        _v = _lum_img[_yi[_ok], _xi[_ok]]
                        _v = (_v - _v.mean()) / _v.std().clamp(min=1e-6)
                        return float((_v * _tgt[_ok]).mean().item())
                    _id0 = _ncc(1.0, 0.0, 0.0)
                    _bb = (_id0, 1.0, 0.0, 0.0)
                    for _sc in (0.92, 0.96, 1.00, 1.04, 1.08, 1.12):
                        for _dx in (-24, -12, 0, 12, 24):
                            for _dy in (-40, -20, -10, 0, 10, 20, 40):
                                _v2 = _ncc(_sc, _dx, _dy)
                                if _v2 > _bb[0]:
                                    _bb = (_v2, _sc, float(_dx), float(_dy))
                    _peak = _bb[0] - _id0
                    print(f"    [Align az={az:.0f} el={el:.0f}] NCC vs front on {_nb:,} shared "
                          f"texels: identity {_id0:+.3f} -> best {_bb[0]:+.3f} "
                          f"(scale x{_bb[1]:.2f} dx={_bb[2]:+.0f} dy={_bb[3]:+.0f}, gain {_peak:+.3f})")
                    if _os_sw.environ.get("TRELLIS_ALIGN", "report").strip() == "apply" and _peak > 0.02:
                        x_pixel = img_char_cx + (x_pixel - img_char_cx) * _bb[1] + _bb[2]
                        y_pixel = img_char_cy + (y_pixel - img_char_cy) * _bb[1] + _bb[3]
                        print(f"    [Align az={az:.0f}] applied")
                else:
                    print(f"    [Align az={az:.0f}] only {_nb:,} texels shared with the front view "
                          f"-- too few to align on")
            except Exception as _ale:
                print(f"    [Align az={az:.0f}] skipped: {_ale!r}")

        # ── DEPTH SCALE for profile views (horizontal only) ───────────────────────────
        # In a profile the image's HORIZONTAL axis is the body's front-back axis. The generated
        # character is shallower front-to-back than the mesh -- measured: the mesh's chest projects
        # 21px PAST the image's forward silhouette edge at upper chest, 9px at the nipple line.
        # So mesh front texels sample from too far back and rear texels from too far forward: the
        # content lands compressed toward the middle. Every earlier fit used a UNIFORM scale and
        # so could not express "wrong in depth, right in height", which is why they all failed.
        # Measured, not optimised: the ratio of the image's body width to the mesh's projected
        # body width over the TORSO rows only (hair dominates the head band and mesh hair differs
        # from generated hair). Vertical untouched, no per-row term, so it cannot band.
        if (abs(math.sin(math.radians(float(az))) * math.cos(math.radians(float(el)))) > 0.7
                and _os_sw.environ.get("TRELLIS_DEPTH_SCALE", "1").strip() != "0"):
            try:
                _ymnD = out_verts[:, 1].min(); _yrgD = (out_verts[:, 1].max() - _ymnD).clamp(min=1e-9)
                _hnD = ((tex_pos[..., 1] - _ymnD) / _yrgD)
                _bandD = uv_hit_mask & (_hnD > 0.42) & (_hnD < 0.72)
                if int(_bandD.sum().item()) > 20000:
                    _bxD = x_pixel[_bandD]; _byD = y_pixel[_bandD]
                    _mlo = float(torch.quantile(_bxD, 0.005).item())
                    _mhi = float(torch.quantile(_bxD, 0.995).item())
                    _r0D = int(max(0, min(img_h - 1, torch.quantile(_byD, 0.05).item())))
                    _r1D = int(max(0, min(img_h - 1, torch.quantile(_byD, 0.95).item())))
                    _colsD = np.where(eroded_fg[_r0D:_r1D + 1].any(axis=0))[0]
                    if len(_colsD) > 10 and (_mhi - _mlo) > 1.0:
                        _ilo = float(_colsD.min()); _ihi = float(_colsD.max())
                        _sx = (_ihi - _ilo) / (_mhi - _mlo)
                        _mcD = 0.5 * (_mlo + _mhi); _icD = 0.5 * (_ilo + _ihi)
                        if 0.5 < _sx < 2.0:
                            x_pixel = _icD + (x_pixel - _mcD) * _sx
                            print(f"    [DepthScale az={az:.0f}] torso width mesh {_mhi-_mlo:.0f}px "
                                  f"vs image {_ihi-_ilo:.0f}px -> x-scale {_sx:.3f}, centre "
                                  f"{_mcD:.0f}->{_icD:.0f} (vertical untouched)")
                        else:
                            print(f"    [DepthScale az={az:.0f}] implausible ratio {_sx:.2f} -- skipped")
            except Exception as _dse:
                print(f"    [DepthScale az={az:.0f}] skipped: {_dse!r}")

        # ── EAR MATCH for profile views ───────────────────────────────────────────────
        # The depth scale is measured on the TORSO, because at head height the silhouette is hair,
        # and mesh-hair vs generated-hair is the mismatch that dragged every earlier fit. The head
        # therefore needs its own correspondence, and a real one exists: the ear is the
        # skin-coloured region inside the head. The mesh knows where its ear is (the head's lateral
        # extreme); in the image it is a skin-coloured blob surrounded by hair. Matching the two
        # centroids gives the head's correction directly -- a feature match, not a silhouette
        # statistic. Applied to the head only and faded out by height, so the torso (already
        # handled by the depth scale) is untouched. TRELLIS_EAR_MATCH=0 disables.
        if (abs(math.sin(math.radians(float(az))) * math.cos(math.radians(float(el)))) > 0.7
                and _os_sw.environ.get("TRELLIS_EAR_MATCH", "1").strip() != "0"):
            try:
                from scipy import ndimage as _ndi
                _ymnE = out_verts[:, 1].min(); _yrgE = (out_verts[:, 1].max() - _ymnE).clamp(min=1e-9)
                _hnE = ((tex_pos[..., 1] - _ymnE) / _yrgE)
                _headE = uv_hit_mask & (_hnE > 0.82)
                _absxE = tex_pos[..., 0].abs()
                if int(_headE.sum().item()) > 5000:
                    _hwE = float(_absxE[_headE].median().item())
                    _facE = (tex_normals * (-look)).sum(-1)
                    _earE = _headE & (_absxE > 1.6 * max(_hwE, 1e-6)) & (_facE > 0.25)
                    if int(_earE.sum().item()) > 200:
                        _emx = float(x_pixel[_earE].median().item())
                        _emy = float(y_pixel[_earE].median().item())
                        _hyE = y_pixel[_headE]
                        _r0E = int(max(0, torch.quantile(_hyE, 0.02).item()))
                        _r1E = int(min(img_h - 1, torch.quantile(_hyE, 0.98).item()))
                        _subE = img_np[_r0E:_r1E + 1]
                        _mE = eroded_fg[_r0E:_r1E + 1].astype(bool)
                        if int(_mE.sum()) > 500:
                            _chE = _subE / np.maximum(_subE.sum(axis=2, keepdims=True), 1e-6)
                            _medE = np.median(_chE[_mE], axis=0)      # head band is mostly hair
                            _dE = np.linalg.norm(_chE - _medE[None, None, :], axis=2)
                            _skinE = _mE & (_dE > float(np.quantile(_dE[_mE], 0.90)))
                            _lab, _nlab = _ndi.label(_skinE)
                            if _nlab >= 2:
                                _sz = np.asarray(_ndi.sum(_skinE, _lab, range(1, _nlab + 1)), dtype=float)
                                # Pick the ear by POSITION, not size. The face is always the FRONTMOST skin region in a
                                # profile and the ear sits behind it; picking 'second largest' grabbed another piece of
                                # the face whenever it split into two regions, which dragged the ear forward onto the
                                # cheek. 'Forward' is known from the camera basis: right[2] > 0 means the character's
                                # front (+Z) maps to increasing image x.
                                _fwd_pos = bool(float(right[2].item()) > 0)
                                _cents = _ndi.center_of_mass(_skinE, _lab, range(1, _nlab + 1))
                                _cxs = [float(c[1]) for c in _cents]
                                _face_i = int(np.argmax(_cxs) if _fwd_pos else np.argmin(_cxs))
                                _cand = [q for q in range(_nlab) if q != _face_i and _sz[q] > 0.02 * float(_sz.max())]
                                if not _cand:
                                    raise ValueError('no skin region behind the face to use as the ear')
                                _pick = int(max(_cand, key=lambda q: _sz[q])) + 1
                                _cyE, _cxE = _ndi.center_of_mass(_skinE, _lab, _pick)
                                _icx2 = float(_cxE); _icy2 = float(_cyE) + _r0E
                                _dxE = _icx2 - _emx; _dyE = _icy2 - _emy
                                if abs(_dxE) < 0.25 * img_w and abs(_dyE) < 0.25 * img_h:
                                    _wE = ((_hnE - 0.62) / 0.20).clamp(0.0, 1.0)
                                    _wE = _wE * _wE * (3.0 - 2.0 * _wE)
                                    x_pixel = x_pixel + _wE * _dxE
                                    y_pixel = y_pixel + _wE * _dyE
                                    print(f"    [EarMatch az={az:.0f}] mesh ear ({_emx:.0f},{_emy:.0f})"
                                          f" -> image ear ({_icx2:.0f},{_icy2:.0f})  shift "
                                          f"({_dxE:+.0f},{_dyE:+.0f}) on the head band "
                                          f"(blob {int(_sz[_pick - 1]):,}px of {_nlab} skin regions)")
                                else:
                                    print(f"    [EarMatch az={az:.0f}] implausible shift "
                                          f"({_dxE:+.0f},{_dyE:+.0f}) -- not applied")
                            else:
                                print(f"    [EarMatch az={az:.0f}] {_nlab} skin region(s) -- cannot "
                                      f"separate ear from face")
                    else:
                        print(f"    [EarMatch az={az:.0f}] too few mesh ear texels face this view")
            except Exception as _eme:
                print(f"    [EarMatch az={az:.0f}] skipped: {_eme!r}")

        u_samp_raw = ((x_pixel + 0.5) / float(img_w)) * 2.0 - 1.0
        v_samp_raw = ((y_pixel + 0.5) / float(img_h)) * 2.0 - 1.0

        # in_bounds MUST be checked on the raw (unclamped) coordinates.
        # If we check after clamping, every pixel appears "in bounds" and
        # T-pose arm UV texels that project far outside the source image silently
        # sample the dark border pixel → the entire arm gets dark color baked in
        # as "covered" and the 3-D fill never fixes it.
        in_bounds = (u_samp_raw.abs() <= 1.0) & (v_samp_raw.abs() <= 1.0)

        # Clamp only for grid_sample (which requires coords in [-1,1])
        eps = 1e-4
        u_samp = u_samp_raw.clamp(-1 + eps, 1 - eps)
        v_samp = v_samp_raw.clamp(-1 + eps, 1 - eps)

        # Occlusion check — use a low threshold because the silhouette was dilated;
        # 0.05 retains all genuine hits while still rejecting clear background.
        grid_occ = torch.stack([u_clip, v_clip], dim=-1).unsqueeze(0)
        sampled_hit = F.grid_sample(
            cam_hit_img,
            grid_occ,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )[0, 0] > 0.05

        sampled_depth = F.grid_sample(
            cam_depth_occ,
            grid_occ,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[0,0]

        tex_depth = -(tex_pos * look).sum(-1)

        # RELATIVE depth tolerance. `depth_eps` is an absolute world-space number, but the depth
        # range being compared differs per view: for the front view the axis is the body's Z
        # extent (~0.21), for a profile it is the whole arm span (~0.87). The same 0.002 is
        # therefore ~0.95% of the range on the front view and ~0.23% on a side view -- four times
        # stricter, on exactly the views that then reported depth=62-75% while the front reported
        # 99%. Measured consequence: 93.4% of the magenta on the finished model faces a camera
        # above its angle gate, i.e. it should have been painted and was false-rejected here.
        # The fraction is not a new constant: it is taken from the FRONT view's existing
        # behaviour, so that view is bit-identical and every other view simply gets the same
        # relative tolerance instead of an accidentally harsher one.
        _dvalid = tex_depth[uv_hit_mask] if bool(uv_hit_mask.any()) else tex_depth
        _dspan = float((_dvalid.max() - _dvalid.min()).item()) if _dvalid.numel() else 0.0
        if _depth_eps_rel is None or _view_i == _frontmost_i:
            _eps_here = float(depth_eps)
            if _dspan > 1e-9:
                _depth_eps_rel = float(depth_eps) / _dspan
        else:
            _eps_here = _depth_eps_rel * _dspan if _dspan > 1e-9 else float(depth_eps)
        depth_match = tex_depth >= sampled_depth - _eps_here

        # Foreground check: reject texels that project onto the image background.
        # Use the DILATED foreground mask (not the raw one) so that soft edges
        # in SV3D-generated frames don't create dark halos at island boundaries.
        fg_mask_t    = torch.from_numpy(eroded_fg.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        grid_fg      = torch.stack([u_samp, v_samp], dim=-1).unsqueeze(0)
        sampled_fg   = F.grid_sample(
            fg_mask_t, grid_fg, mode='bilinear', padding_mode='zeros', align_corners=False
        )[0, 0] > 0.5

        # Sample colours first so we can check brightness before marking visible
        grid_col = torch.stack([u_samp, v_samp], dim=-1).unsqueeze(0)
        sampled_colors = F.grid_sample(img_t, grid_col, mode='bilinear', padding_mode='border', align_corners=False)[0].permute(1, 2, 0)

        # Reject very dark pixels (shadows / background bleed) – let KD-tree fill
        # propagate correct skin colour instead of projecting a near-black value.
        brightness = sampled_colors.mean(dim=-1)   # [H_tex, W_tex], range [0,1]
        bright_enough = brightness > 0.08

        # Reject grazing-angle projections.  Surfaces whose normal is nearly
        # perpendicular to this camera (dot < MIN_DOT) should not receive a
        # colour from this view – they'll be filled later by the KD-tree from
        # a well-projected neighbour.  Arm tops (normal≈+Y) have dot=0 with
        # any horizontal camera, so they are always excluded here and left truly
        # uncovered, allowing KD-tree to propagate correct skin from arm sides.
        dot_raw = (tex_normals * (-look)).sum(-1)          # signed, no clamp yet

        # Per-view minimum dot threshold — auto-scaled from elevation angle.
        #
        # Horizontal views (el=0):  min_dot = 0.12 (unchanged).
        # Top-down view  (el=90):   min_dot = 0.50 — only surfaces within ~60° of
        #   straight-up get the top-view projection.  Without this, any back or side
        #   surface that tilts even slightly upward (dot≈0.15) gets painted with
        #   whatever the Trellis texture looks like from above, which includes hair
        #   shadows and wrong-angle colour bleeding onto the upper back.
        # Intermediate (e.g. el=30): min_dot ≈ max(0.12, sin(30°)×0.55) ≈ 0.28.
        el_rad       = math.radians(el)
        view_min_dot = max(0.12, abs(math.sin(el_rad)) * 0.55)
        good_angle   = dot_raw > view_min_dot

        # KAN-9 — a flat FRONT photo shows every front-facing texel, but the
        # depth/occlusion test (designed for multi-view side projection) false-rejects
        # the curved torso: the sternum / under-breast skin sits a little behind the
        # breast tips and gets flagged "self-occluded". Skip depth for texels that
        # clearly face this camera (they can't be a hidden/back surface); keep it for
        # grazing/side texels. Fixes the missing chest with no back-leak.
        # Gated to the FRONT view. The skip exists because a flat front photo shows every
        # front-facing texel, so a self-occlusion test there only produces false rejects. On
        # ANY OTHER view it switches occlusion off precisely where a surface faces that camera
        # hard -- the torso flank or the side of the skull on a profile, the scalp from above --
        # so hair draped in front of that surface gets sampled straight onto the body. That was
        # the smeared hair-coloured block on the flanks, and it is why the profile views report
        # depth=95%/83% while the back view (whose surfaces face it squarely, unoccluded) is 99%.
        if _view_i == _frontmost_i:
            depth_ok = depth_match | (dot_raw > front_skip_depth_dot)
        else:
            depth_ok = depth_match

        # ── NEAR-ARM OCCLUDER (profile views only) ────────────────────────────────────
        # In a T-pose profile the near arm points straight AT the camera, so it appears as a
        # foreshortened fist sitting over the hair and shoulder. The depth test does not remove
        # what it hides, because the mesh's arm and the generated image's hand do not occupy the
        # same pixels -- the arms are the worst-aligned part of the mesh (21-61px on the front
        # view) -- so the mesh shadows one region while the image's hand covers another, and every
        # texel in the gap samples HAND. That is the "ear pointing down" in the hair, on both
        # sides, and no amount of vertical or silhouette correction touches it.
        # A profile view's own near arm is therefore unusable for anything except the arm itself.
        # Reject any texel that samples inside the projected near-arm footprint while not being
        # on that arm. Purely geometric: the arm is identified by the torso's own half-width,
        # measured at hip height where a T-pose arm cannot reach, so there is no tuned distance.
        _nearm_rej = None
        # OFF. It rejects using the MESH's arm footprint, but the image's hand sits at a
        # different height -- confirmed on the bake: the hand painted on the hair is ABOVE
        # the real hand. So it removed coverage where the mesh's arm is while the image's
        # hand kept bleeding: new magenta, artefact untouched. Enable with TRELLIS_NEARARM=1.
        _camx = abs(math.sin(math.radians(float(az))) * math.cos(math.radians(float(el))))
        if _camx > 0.7 and _os_sw.environ.get('TRELLIS_NEARARM', '0').strip() == '1':
            try:
                _v3 = out_verts.detach().cpu().numpy()
                _hnv = ((_v3[:, 1] - _v3[:, 1].min())
                        / max(float(_v3[:, 1].max() - _v3[:, 1].min()), 1e-9))
                _hipb = (_hnv > 0.33) & (_hnv < 0.48)
                _halfw = (float(np.percentile(np.abs(_v3[_hipb, 0]), 98)) if int(_hipb.sum()) > 50
                          else float(np.abs(_v3[:, 0]).max()))
                _sgn = 1.0 if math.sin(math.radians(float(az))) > 0 else -1.0
                _armv = (_v3[:, 0] * _sgn) > _halfw
                if int(_armv.sum()) > 50:
                    _cc = cam_clip[0]
                    _apx = (img_char_cx + (_cc[:, 0] - mesh_cx) * pix_per_clip).detach().cpu().numpy()
                    _apy = (img_char_cy - (_cc[:, 1] - mesh_cy) * pix_per_clip).detach().cpu().numpy()
                    _mk = np.zeros((img_h, img_w), np.uint8)
                    _mk[np.clip(np.rint(_apy[_armv]).astype(int), 0, img_h - 1),
                        np.clip(np.rint(_apx[_armv]).astype(int), 0, img_w - 1)] = 1
                    _mk = cv2.dilate(_mk, np.ones((3, 3), np.uint8), iterations=3)
                    _mt = torch.from_numpy(_mk.astype(bool)).cuda()
                    _px = x_pixel.round().long().clamp(0, img_w - 1)
                    _py = y_pixel.round().long().clamp(0, img_h - 1)
                    _is_arm = (tex_pos[..., 0] * _sgn) > _halfw
                    _nearm_rej = _mt[_py, _px] & (~_is_arm)
                    print(f"    [NearArm az={az:.0f}] footprint {int(_mk.sum()):,}px "
                          f"(|x|>{_halfw:.3f}); rejecting {int(_nearm_rej.sum()):,} non-arm texels "
                          f"sampling through it")
            except Exception as _nae:
                print(f"    [NearArm az={az:.0f}] skipped: {_nae!r}")

        # Geometric acceptance -- everything except the brightness test, which depends on the
        # colours we are about to correct (using `visible` here would be circular).
        _geom_ok = uv_hit_mask & in_bounds & sampled_hit & depth_ok & sampled_fg & good_angle
        if _nearm_rej is not None:
            _geom_ok = _geom_ok & (~_nearm_rej)

        if _ref_lum is None:
            # first view visited is the reference; it defines the exposure everything matches
            _ref_lum  = sampled_colors.mean(dim=-1).clone()
            _ref_geom = _geom_ok.clone()
            print(f"[Exposure] view {_view_i} az={az:.0f} el={el:.0f} = REFERENCE "
                  f"({int(_ref_geom.sum().item())} texels)")
        else:
            _both = _geom_ok & _ref_geom
            _nsh  = int(_both.sum().item())
            if _nsh >= 500:
                _gm = float(torch.median(_ref_lum[_both])
                            / torch.clamp(torch.median(sampled_colors.mean(dim=-1)[_both]), min=1e-6))
                sampled_colors = torch.clamp(sampled_colors * _gm, 0.0, 1.0)
                print(f"[Exposure] view {_view_i} az={az:.0f} el={el:.0f} "
                      f"gain={_gm:.3f}  (measured on {_nsh} shared texels)")
            else:
                print(f"[Exposure] view {_view_i} az={az:.0f} el={el:.0f} "
                      f"— only {_nsh} shared texels, left as generated")
            # Grow the reference with this view's now-corrected colour, on surface no earlier
            # view reached. This is what lets the back view match: by the time it is visited,
            # the sides have contributed the flanks it actually shares.
            _fresh = _geom_ok & (~_ref_geom)
            if bool(_fresh.any()):
                _ref_lum  = torch.where(_fresh, sampled_colors.mean(dim=-1), _ref_lum)
                _ref_geom = _ref_geom | _geom_ok

        # brightness must be re-derived from the CORRECTED colours
        brightness    = sampled_colors.mean(dim=-1)
        bright_enough = brightness > 0.08

        # Visibility Mask
        visible = _geom_ok & bright_enough

        # ── REJECTION DUMP (TRELLIS_DUMP_REJECT=1) ────────────────────────────────────
        # Writes, per view, the exact image the projector sampled plus a map of WHERE each
        # rejected texel sampled, colour-coded by the term that rejected it. Reconstructing the
        # matte offline gave answers that disagreed with the bake's own fg% by 4x, so this reads
        # the real arrays instead of approximating them.
        #   white = accepted   red = fg (sampled background)   blue = depth   green = angle
        #   yellow = bright    magenta = out of bounds / no raster hit
        # default "1": ON for one diagnostic run; set back to "0" afterwards.
        if _os_sw.environ.get("TRELLIS_DUMP_REJECT", "0").strip() == "1":
            try:
                import os as _od
                from PIL import Image as _PI
                _dd = _os_sw.path.join(_os_sw.path.dirname(_os_sw.path.dirname(
                    _os_sw.path.dirname(_os_sw.path.dirname(_os_sw.path.abspath(__file__))))),
                    "uv_transfer")
                _tag = f"az{az:.0f}_el{el:.0f}"
                _PI.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    _od.path.join(_dd, f"_rej_{_tag}_image.png"))
                _PI.fromarray((eroded_fg.astype(np.uint8) * 255)).save(
                    _od.path.join(_dd, f"_rej_{_tag}_matte.png"))
                _canvas = np.zeros((img_h, img_w, 3), np.uint8)
                _xi = x_pixel.round().long().clamp(0, img_w - 1)
                _yi = y_pixel.round().long().clamp(0, img_h - 1)
                _sel = uv_hit_mask & (dot_raw > 0.12)
                for _m, _col in ((visible, (255, 255, 255)),
                                 (_sel & ~good_angle, (0, 255, 0)),
                                 (_sel & good_angle & ~depth_ok, (0, 0, 255)),
                                 (_sel & good_angle & depth_ok & ~sampled_fg, (255, 0, 0)),
                                 (_sel & good_angle & depth_ok & sampled_fg & ~bright_enough,
                                  (255, 255, 0)),
                                 (_sel & ~(in_bounds & sampled_hit), (255, 0, 255))):
                    _mm = (_sel & _m)
                    if not bool(_mm.any()):
                        continue
                    _yy = _yi[_mm].cpu().numpy(); _xx = _xi[_mm].cpu().numpy()
                    _canvas[_yy, _xx] = _col
                _PI.fromarray(_canvas).save(_od.path.join(_dd, f"_rej_{_tag}_map.png"))
                print(f"    [RejectDump] {_tag}: wrote _rej_{_tag}_image/matte/map.png")

                # WHERE DOES EACH TEXEL ACTUALLY SAMPLE? Plot every accepted texel's sample
                # position on the very image the projector read, tinted by its HEIGHT on the
                # mesh (blue = feet -> red = top of head), over a dimmed copy of that image.
                # If the mapping is vertically correct the red band sits on the head in the
                # photo; if it is shifted the bands sit off the body by exactly that amount.
                # Ear texels are drawn bright green: the head's laterally-extreme geometry,
                # found relative to the head's own width so it needs no per-character number.
                _ymn = tex_pos[..., 1].min(); _yrg = (tex_pos[..., 1].max() - _ymn).clamp(min=1e-9)
                _hn = ((tex_pos[..., 1] - _ymn) / _yrg)
                _ov = (np.clip(img_np, 0, 1) * 0.40 * 255).astype(np.uint8)
                _m = uv_hit_mask & visible
                if bool(_m.any()):
                    _h = _hn[_m].cpu().numpy()
                    _yy = _yi[_m].cpu().numpy(); _xx = _xi[_m].cpu().numpy()
                    _ov[_yy, _xx, 0] = (255 * np.clip(_h, 0, 1)).astype(np.uint8)
                    _ov[_yy, _xx, 2] = (255 * np.clip(1.0 - _h, 0, 1)).astype(np.uint8)
                _headband = uv_hit_mask & (_hn > 0.88)
                if bool(_headband.any()):
                    _absx = tex_pos[..., 0].abs()
                    _hw = float(_absx[_headband].median().item())
                    _ear = _headband & (_absx > 1.6 * max(_hw, 1e-6))
                    if bool(_ear.any()):
                        _yy = _yi[_ear].cpu().numpy(); _xx = _xi[_ear].cpu().numpy()
                        _ov[_yy, _xx] = (0, 255, 0)
                        print(f"    [RejectDump] {_tag}: {int(_ear.sum()):,} ear texels marked "
                              f"green (head median |x|={_hw:.3f})")
                _PI.fromarray(_ov).save(_od.path.join(_dd, f"_rej_{_tag}_where.png"))
            except Exception as _rde:
                print(f"    [RejectDump] failed: {_rde!r}")

        # KAN-9 — per-mask coverage diagnostic on the REAL bake mesh (the offline
        # reproduction kept disagreeing with the bake; this prints the ground truth).
        try:
            _ff = uv_hit_mask & (dot_raw > 0.12)
            _n  = int(_ff.sum().item())
            if _n > 0:
                _p = lambda m: 100.0 * int((_ff & m).sum().item()) / _n
                _tw_views["az{:.0f}_el{:.0f}".format(az, el)] = _p(visible)
                print(f"[CoverageDiag az={az:.0f} el={el:.0f}] front-facing={_n}  "
                      f"inbounds={_p(in_bounds):.0f}% occ={_p(sampled_hit):.0f}% "
                      f"depth={_p(depth_ok):.0f}% fg={_p(sampled_fg):.0f}% "
                      f"bright={_p(bright_enough):.0f}% angle={_p(good_angle):.0f}% "
                      f"eps={_eps_here:.4f}  "
                      f"=> COVERED={_p(visible):.0f}%")
        except Exception as _cde:
            print(f"[CoverageDiag] skipped: {_cde!r}")

        # -- PER-VIEW TRIPWIRE: reject a bad bake here, at the earliest point it is detectable,
        # rather than letting four more views, the seam-hair pass, the atlas pass, the FBX export
        # and the rigging run first. Relative to what this view has achieved before.
        # PER-VIEW TRIPWIRE: DROP the view, never abort the bake.
        # Aborting here turns a one-view problem into a total loss after ~20 minutes of Qwen and
        # Trellis work that cannot be cached for a new character. A collapsed view is excluded
        # from the blend and the other views carry on, so the run still produces an asset with a
        # known, reported gap. Only a total collapse (every view gone) is fatal, because then
        # there is nothing to paint with.
        _drop_this_view = False
        if _tw_on and _tw_base:
            _vk = "az{:.0f}_el{:.0f}".format(az, el)
            _prev = (_tw_base.get("views") or {}).get(_vk)
            _cur = _tw_views.get(_vk)
            if isinstance(_prev, (int, float)) and _prev > 0 and _cur is not None:
                _floor = float(_prev) * _tw_keep
                print("[Tripwire] view {} covered {:.1f}%  (best {:.1f}%, floor {:.1f}%)".format(
                    _vk, _cur, float(_prev), _floor))
                if _cur < _floor:
                    _drop_this_view = True
                    _dropped_views.append((_vk, _cur, float(_prev)))
                    print("[Tripwire] !! view {} COLLAPSED ({:.1f}% vs floor {:.1f}%) even after "
                          "repair -- DROPPING it and continuing. The bake completes; the surface "
                          "it covered stays unpainted.".format(_vk, _cur, _floor))

        dot = dot_raw.clamp(min=0.0) ** 1.5
        weights = (dot ** blend_exponent) * visible.float() * view_w
        if _drop_this_view:
            weights = torch.zeros_like(weights)

        # ── THE BEST-PLACED VIEW OWNS THE SURFACE ─────────────────────────────────────
        # A nipple faces the front camera almost head-on and is nearly edge-on to a side camera,
        # so the side view has no business colouring it -- but it still contributed, and because
        # the generated profile is shallower than the mesh, what it contributed was the shaded
        # breast rim landing at the wrong depth: the second, misplaced nipple.
        # Fade a non-reference view in proportion to how much better the reference view faces the
        # same surface. Purely geometric -- no regions, no thresholds, and it applies to the whole
        # body rather than a hand-drawn chest area.
        # CRITICAL: gated on the reference view having actually PAINTED that texel. Where the
        # front view missed, nothing is faded, so this cannot open a new hole -- which is the
        # whole point, given everything visible has to stay painted.
        if _view_i == _frontmost_i:
            _front_vis = visible.clone()
            _front_dot = dot_raw.clone()
        elif (_front_vis is not None) and (_front_dot is not None):
            # Cubic, not linear. Linear left the side view at 25% weight on a nipple (front dot
            # ~0.95 vs side ~0.20), and 25% of a wrongly-placed nipple is still a visible nipple.
            # Cubed that becomes ~1.6%, while surfaces both views see comparably (adv ~ 0) are
            # untouched -- so it is a sharp handoff exactly where one view is clearly better
            # placed, and no change at all anywhere else.
            _adv = (_front_dot - dot_raw).clamp(0.0, 1.0)     # how much better the front sees it
            _keep = (1.0 - _adv) ** 3
            weights = weights * torch.where(_front_vis, _keep, torch.ones_like(_keep))

        # KAN-9 — in the face region, keep ONLY the front view: every other view's
        # weight is faded out by (1 - face_weight) so it cannot contribute colour
        # to eyes/nose/mouth. The front view keeps its full weight.
        if face_front_w_t is not None and _view_i != _front_view_idx:
            weights = weights * (1.0 - face_front_w_t)

        # VIEW-OWNERSHIP MAP: remember which view contributed most to each texel, so an artefact
        # can be traced to the view that painted it instead of guessed at.
        if _own_w is None:
            _own_w = torch.zeros_like(weights)
            _own_i = torch.full(weights.shape, -1, dtype=torch.int16, device=weights.device)
        _better = weights > _own_w
        _own_w = torch.where(_better, weights, _own_w)
        _own_i = torch.where(_better, torch.full_like(_own_i, int(_view_i)), _own_i)

        acc_color += sampled_colors * weights.unsqueeze(-1)
        acc_weight += weights

    # =========================================================================
    # STEP 4 – Normalize and blend over existing PBR texture
    # =========================================================================
    if _dropped_views:
        print("[Tripwire] {} view(s) dropped this bake: ".format(len(_dropped_views))
              + "; ".join("{} at {:.1f}% (best {:.1f}%)".format(*d) for d in _dropped_views))
        if len(_dropped_views) >= len(images):
            raise RuntimeError("[Tripwire] every view collapsed -- nothing left to paint with.")

    if _own_i is not None and _os_twc.environ.get("TRELLIS_DUMP_OWNER", "0").strip() == "1":
        try:
            from PIL import Image as _PI2
            _pal = np.array([[255, 80, 80], [80, 160, 255], [80, 255, 120],
                             [255, 220, 80], [230, 100, 255], [255, 255, 255]], np.uint8)
            _oi = _own_i.cpu().numpy()
            _img = np.zeros(_oi.shape + (3,), np.uint8)
            for _k in range(len(images)):
                _img[_oi == _k] = _pal[_k % len(_pal)]
            _op = _os_twc.path.join(_os_twc.path.dirname(_os_twc.path.dirname(
                _os_twc.path.dirname(_os_twc.path.dirname(_os_twc.path.abspath(__file__))))),
                "uv_transfer", "_view_owner.png")
            _PI2.fromarray(_img).save(_op)
            _leg = "  ".join(f"view{_k}(az={azimuths[_k]:.0f},el={elevations[_k]:.0f})="
                             f"{('red','blue','green','yellow','magenta','white')[_k % 6]}"
                             for _k in range(len(images)))
            print(f"[ViewOwner] wrote _view_owner.png -- {_leg}")
            for _k in range(len(images)):
                print(f"[ViewOwner]   view{_k} az={azimuths[_k]:.0f} el={elevations[_k]:.0f} owns "
                      f"{100.0 * float((_oi == _k).sum()) / max(float((_oi >= 0).sum()), 1):.1f}% of painted texels")
        except Exception as _oe:
            print(f"[ViewOwner] failed: {_oe!r}")

    print("[MultiView] Finalising texture...")

    total_w = acc_weight.sum().item()
    print(f"  Total accumulated weight: {total_w:.1f}")
    if total_w < 1.0:
        print("  WARNING: almost no weight accumulated – check camera setup!")

    # Mask for safe division (avoid div-by-zero)
    valid_mask = acc_weight > 1e-6
    projected_color = torch.zeros_like(acc_color)
    projected_color[valid_mask] = acc_color[valid_mask] / acc_weight[valid_mask].unsqueeze(-1)

    # Mask for compositing: only replace original texture where at least one view
    # had a confident projection (normal dot product weight above a real threshold).
    # This prevents near-black grazing-angle texels from overwriting the original.
    # Threshold is relative to the maximum accumulated weight in the texture,
    # so it adapts automatically regardless of blend_exponent or number of views.
    # ------------------------------------------------------------------
    # Compute confidence from accumulated weights
    # ------------------------------------------------------------------

    max_w = acc_weight.max().clamp(min=1e-6)

    # Normalize weights
    confidence = acc_weight / max_w

    # Smoothstep curve (removes harsh edges)
    confidence = confidence * confidence * (3.0 - 2.0 * confidence)

    # Optional: boost confident projections
    confidence = confidence ** 0.5

    confidence3 = confidence.unsqueeze(-1)

    #-- Load and resample the existing PBR base color texture ----------------
    existing_base = None
    try:
        mat = high_poly_mesh.visual.material
        existing_base = getattr(mat, 'baseColorTexture', None)
        if existing_base is None:
            # Fallback: try accessing via image attribute (SimpleMaterial / PBRMaterial variants)
            existing_base = getattr(mat, 'image', None)
    except AttributeError:
        pass

    if existing_base and blend_texture:
        print("  Blending projected views over existing PBR texture...")
        if existing_base.mode != 'RGBA':
            existing_base = existing_base.convert('RGBA')
        # Resize to target texture size if needed
        if existing_base.size != (texture_size, texture_size):
            print(f"  Resampling existing texture from {existing_base.size} -> ({texture_size},{texture_size})")
            existing_base = existing_base.resize((texture_size, texture_size), Image.LANCZOS)

        existing_np = np.array(existing_base).astype(np.float32) / 255.0   # (H, W, 4)
        existing_np = np.flip(existing_np, axis=0).copy()
        existing_rgb = torch.from_numpy(existing_np[..., :3]).cuda()        # (H, W, 3)
        existing_alpha = torch.from_numpy(existing_np[..., 3:4]).cuda()     # (H, W, 1)

        # Luminance-gated composite:
        #   - Where projection is bright (not a dark edge artifact) AND confident:
        #     use projected color (accurate front-image colours).
        #   - Where projection is dark (sampled background / silhouette edge):
        #     defer entirely to the diffusion base texture.
        #   - Where projection has no coverage: use diffusion base.
        #
        # This prevents dark edge samples from overwriting good diffusion skin tones.
        proj_lum = (0.299 * projected_color[..., 0]
                  + 0.587 * projected_color[..., 1]
                  + 0.114 * projected_color[..., 2])          # (H,W) in [0,1]
        exist_lum = (0.299 * existing_rgb[..., 0]
                   + 0.587 * existing_rgb[..., 1]
                   + 0.114 * existing_rgb[..., 2])

        # Gate: projection must be at least 50% as bright as the diffusion base
        # (prevents near-black edge samples from dominating).
        bright_gate = (proj_lum > 0.05) & (proj_lum >= exist_lum * 0.5)
        bright_gate3 = bright_gate.unsqueeze(-1).float()

        gated_conf3 = confidence3 * bright_gate3

        blended_rgb = (
            projected_color * gated_conf3 +
            existing_rgb   * (1.0 - gated_conf3)
        )

        composite_mask = (confidence > 0.01) & bright_gate

        color_np = (blended_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # Preserve the original alpha everywhere (mesh already has full coverage)
        alpha_np = (existing_alpha.squeeze(-1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        n_projected = int(composite_mask.sum().item())
        print(f"  Projected texels: {n_projected} / {texture_size*texture_size}"
              f"  ({100.0*n_projected/(texture_size*texture_size):.1f}%)")

    else:
        # No existing texture found – fall back to projection-only output
        print(
            "  WARNING: No existing PBR baseColorTexture found on mesh, "
            "outputting projection-only texture (holes will be transparent)."
        )

        # RGB from projection — back-facing areas (projected_color=0) will be
        # overwritten by the nearest-neighbour fill below.
        color_np = (projected_color.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Alpha: fully opaque for every projected pixel; the fill pass will
        # set alpha=255 for back-facing areas too.  The final texture is opaque
        # everywhere inside the UV islands (alphaMode='OPAQUE' in the material).
        alpha_np = np.where(valid_mask.cpu().numpy(), np.uint8(255), np.uint8(0))

    # =========================================================================
    # FILL – delegated to FillUtils
    # =========================================================================
    from .fill_utils import FillUtils

    uv_mask_np_bool = uv_hit_mask.cpu().numpy().astype(bool)
    covered         = valid_mask.cpu().numpy()
    uncovered_uv    = uv_mask_np_bool & ~covered

    tex_pos_np = tex_pos.cpu().numpy()
    fu = FillUtils(uv_mask_np_bool, tex_pos_np)

    # ── [AtlasDiag] ROOT-CAUSE instrumentation: how the 4096 atlas area is split across UV
    # islands, with each island's 3-D centre (paint space: X=lat, Y=height, Z=depth) so we can
    # see whether the BODY island is being squeezed while hair/genital/face hog the atlas, and
    # whether the covered (real-projected) fraction is healthy. Compare a GOOD vs BAD bake.
    try:
        from scipy.ndimage import label as _ad_label
        _ad_lab, _ad_n = _ad_label(uv_mask_np_bool)
        _ad_sz = np.bincount(_ad_lab.ravel())
        _ad_tot = int(uv_mask_np_bool.sum())
        _ad_cov = int((uv_mask_np_bool & covered).sum())
        print(f"[AtlasDiag] uv_hit={_ad_tot:,} ({100.0*_ad_tot/uv_mask_np_bool.size:.1f}% of atlas)  "
              f"covered(real-projected)={_ad_cov:,} ({100.0*_ad_cov/max(_ad_tot,1):.1f}% of uv_hit)  islands={_ad_n}")
        _ad_order = (np.argsort(_ad_sz[1:])[::-1] + 1) if _ad_n > 0 else []
        for _ad_li in _ad_order[:8]:
            _ad_m = (_ad_lab == _ad_li)
            _ad_c = tex_pos_np[_ad_m].mean(0)
            _ad_cf = 100.0 * int((_ad_m & covered).sum()) / max(int(_ad_m.sum()), 1)
            print(f"   island {int(_ad_li):3d}: {int(_ad_sz[_ad_li]):8,d} tex "
                  f"({100.0*_ad_sz[_ad_li]/max(_ad_tot,1):4.1f}% of uv_hit, covered {_ad_cf:3.0f}%)  "
                  f"centre x={_ad_c[0]:+.2f} y={_ad_c[1]:+.2f} z={_ad_c[2]:+.2f}")
    except Exception as _ade:
        print(f"[AtlasDiag] skipped: {_ade!r}")

    # ── SEAM-DRIVEN HAIR FOUNDATION (KAN-8) ───────────────────────────────────
    # Use the hair region from gen_seams (the seam we tuned) as the hair mask so the
    # seam work drives the colour.  Map gen_seams' per-tri hair flag (_genseams_viz
    # .npz, its NORMALIZED space) onto the paint's texels by nearest centroid --
    # normalize the paint's tex_pos the same way gen_seams normalizes (own bbox).
    seam_hair_mask = None
    try:
        from scipy.spatial import cKDTree as _cKDT
        from .hair_detection_utils_v2 import HairDetectionUtils
        _vz = np.load(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz')
        _vco = _vz['co'].astype(np.float64); _vtris = _vz['tris']; _vth = _vz['tris_hair'].astype(bool)
        _vtc = _vco[_vtris].mean(axis=1)
        _ov = out_verts.detach().cpu().numpy().astype(np.float64)
        _of = out_faces.detach().cpu().numpy()
        # AXIS: the paint mesh is Y-up (Y=height, Z=depth); gen_seams is Z-up.  Map the
        # paint mesh into gen's normalized space the same way gen_seams normalized.
        _hmn, _hmx = float(_ov[:, 1].min()), float(_ov[:, 1].max())   # height = paint Y
        _lmn, _lmx = float(_ov[:, 0].min()), float(_ov[:, 0].max())   # lateral = paint X
        _zs = 0.91 / (_hmx - _hmn); _zo = -0.41 - _hmn * _zs
        _xs = 0.92 / (_lmx - _lmn); _xc = (_lmn + _lmx) / 2.0
        # PER-FACE hair, NOT per-texel: map each paint FACE centroid to the nearest gen
        # tri.  tex_pos is garbage for back-facing texels (the unseen crown/back of the
        # head), which silently dropped them; per-face + rasterize can't drop them.
        _pfc = _ov[_of].mean(axis=1)
        _pfn = np.empty_like(_pfc)
        _pfn[:, 0] = (_pfc[:, 0] - _xc) * _xs    # gen lateral
        _pfn[:, 1] = -_pfc[:, 2] * _zs           # gen depth (Y-up -> Z-up flip)
        _pfn[:, 2] = _pfc[:, 1] * _zs + _zo      # gen height
        _, _fi = _cKDT(_vtc).query(_pfn, k=1, workers=-1)
        _facehair = _vth[_fi] & (np.abs(_pfc[:, 0]) < 0.20)   # guard: never the arms
        # rasterize the hair FACES straight into the atlas (fill their UV triangles)
        _hdu_ear = HairDetectionUtils(
            texture_size=texture_size, uv_mask=uv_mask_np_bool, tex_pos_np=tex_pos_np,
            tex_normals_np=tex_normals.cpu().numpy(),
            neck_y_min=neck_y_min, neck_y_max=neck_y_max,
            cyan_y_margin=cyan_y_margin, cyan_z_frac=cyan_z_frac, trimesh=mesh,
            _norm_verts=_ov, _norm_faces=_of, _norm_uvs=out_uvs.detach().cpu().numpy())
        seam_hair_mask = _hdu_ear._rasterize_face_mask(_facehair, dilate_px=1)
        if seam_hair_mask is None:
            seam_hair_mask = np.zeros_like(uv_mask_np_bool)
        seam_hair_mask &= uv_mask_np_bool
        print(f"[SeamHair] {int(_facehair.sum()):,} hair faces -> {int(seam_hair_mask.sum()):,} texels (per-face rasterised, tex_pos-free)")
        # (#1) remove EARS (the rasterized hair can still cover the thin ear)
        try:
            _Tear = _hdu_ear._compute_thresholds()
            _earf = _hdu_ear._detect_ear_faces(_Tear)
            if _earf is not None and _earf.any():
                _earuv = _hdu_ear._rasterize_face_mask(_earf, dilate_px=2)
                if _earuv is not None:
                    _nb = int(seam_hair_mask.sum()); seam_hair_mask &= ~_earuv
                    print(f"[SeamHair] -ears: {_nb} -> {int(seam_hair_mask.sum())} (ear px {int(_earuv.sum()):,})")
        except Exception as _ee:
            print(f"[SeamHair] ear removal skipped: {_ee!r}")
        # ── KAN-9: RECLAIM THE FACE from the hair mis-flag ────────────────────
        # gen_seams over-flags the FRONT of the face (forehead/eyes/nose/cheeks) as
        # hair, so the face gets hair treatment + the FBX Hair material -> "no face".
        # The true face skin = faces above the neck, facing the camera, on the front
        # half.  Pull them out of the hair mask AND write the set so the FBX exporter
        # excludes them from the Hair material too.  Real hair is untouched: the scalp
        # points up (normal_z low), sides/back point away (front half / normal_z fail).
        try:
            _ovf = out_verts.detach().cpu().numpy(); _off = out_faces.detach().cpu().numpy()
            _onf = out_normals.detach().cpu().numpy()
            _fcen = _ovf[_off].mean(axis=1)
            _fnrm = _onf[_off].mean(axis=1)
            _fnz  = _fnrm[:, 2] / (np.linalg.norm(_fnrm, axis=1) + 1e-8)
            # ROBUST face region via MediaPipe (recognises the real face on ANY character and
            # distinguishes face from hairline) — replaces the fragile per-character angle
            # guard that let curved faces slip through and be flagged as hair. Falls back to
            # the geometric guard only if the detector can't find a face on the relief.
            _faceskin = None
            try:
                from .landmark_warp import compute_face_region_faces
                _wfi = int(np.argmin([abs(((float(_a) + 180.0) % 360.0) - 180.0) for _a in azimuths]))
                _ffi = np.array(images[_wfi].convert('RGB')).astype(np.float32) / 255.0
                _faceskin = compute_face_region_faces(out_verts, out_faces.int(), out_normals, _ffi, ortho_scale, ctx)
            except Exception as _fre:
                print(f"[FaceSkin] MediaPipe face region failed ({_fre!r}) — geometric fallback")
            if _faceskin is None:
                _faceskin = (_fnz > 0.35) & (_fcen[:, 2] > 0.0) & (_fcen[:, 1] > float(neck_y_min))
            np.save(r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable'
                    r'\ComfyUI\uv_transfer\last_seams_faceskin.npy', _faceskin)
            _fsuv = _hdu_ear._rasterize_face_mask(_faceskin, dilate_px=1)
            if _fsuv is not None:
                _nb = int(seam_hair_mask.sum()); seam_hair_mask &= ~_fsuv
                print(f"[SeamHair] -face skin reclaimed (hair mis-flag): {_nb} -> {int(seam_hair_mask.sum())}")
            print(f"[FaceSkin] {int(_faceskin.sum())} front-face faces reclaimed from hair "
                  f"-> last_seams_faceskin.npy")
        except Exception as _fse:
            print(f"[FaceSkin] reclaim skipped: {_fse!r}")
    except Exception as _se:
        print(f"[SeamHair] failed ({_se!r}) — no seam-hair foundation this run")
        import traceback as _tb; _tb.print_exc()

    def _dump_stage(_tag):
        if not dump_stages:
            return
        try:
            from PIL import Image as _PImg
            import os as _os
            _p = _os.path.join(r'C:\applications\ComfyUI_windows_portable_nvidia'
                               r'\ComfyUI_windows_portable\ComfyUI\uv_transfer',
                               '_stage_' + _tag + '.png')
            _PImg.fromarray(color_np).save(_p)
            print(f"[DumpStage] {_tag} -> {_os.path.basename(_p)}")
        except Exception as _de:
            print(f"[DumpStage] {_tag} failed: {_de!r}")

    _dump_stage('0_projected')
    # ── COVERAGE REPORT ───────────────────────────────────────────────────────
    # Principle (user, 2026-09-06): if a surface is not in any image we add an image, we do not
    # invent one. So say plainly how much was never seen, and where, instead of quietly filling it.
    _cov_stats = None
    try:
        import os as _os_cov
        _uvn = int(uv_mask_np_bool.sum()); _unc = int(uncovered_uv.sum())
        _cov_stats = {"total": 100.0 * _unc / max(_uvn, 1)}
        print(f"[Coverage] {_unc:,} of {_uvn:,} UV texels were seen by NO view "
              f"({100.0*_unc/max(_uvn,1):.1f}%)")
        _tp = tex_pos_np if 'tex_pos_np' in dir() else None
        if _tp is not None:
            _ax = np.abs(_tp[..., 0])
            for _nm, _sel in (("arms |x|>0.30", _ax > 0.30),
                              ("torso+head |x|<=0.30", _ax <= 0.30)):
                _a = uv_mask_np_bool & _sel; _u = uncovered_uv & _sel
                if _a.sum():
                    _cov_stats[_nm.split()[0]] = 100.0 * _u.sum() / _a.sum()
                    print(f"[Coverage]   {_nm:<22} {int(_u.sum()):>9,} / {int(_a.sum()):>9,} "
                          f"unseen ({100.0*_u.sum()/_a.sum():.1f}%)")
            if seam_hair_mask is not None:
                _a = uv_mask_np_bool & seam_hair_mask; _u = uncovered_uv & seam_hair_mask
                if _a.sum():
                    _cov_stats["hair"] = 100.0 * _u.sum() / _a.sum()
                    print(f"[Coverage]   {'hair':<22} {int(_u.sum()):>9,} / {int(_a.sum()):>9,} "
                          f"unseen ({100.0*_u.sum()/_a.sum():.1f}%)")
    except Exception as _cve:
        print(f"[Coverage] report failed: {_cve!r}")

    # -- COVERAGE TRIPWIRE ----------------------------------------------------
    # A bake full of holes must FAIL HERE, not sail on through FBX export, eye placement and
    # rigging to produce a broken asset several minutes later. The primary rule is RELATIVE:
    # compare against the best coverage this pipeline has ever reached and refuse a run that is
    # meaningfully worse, so a regression is caught in the bake that introduces it rather than
    # weeks later. The absolute ceiling is only a catastrophe backstop.
    #   TRELLIS_COVERAGE_ABORT=0   disable entirely
    #   TRELLIS_COVERAGE_REGRESS   allowed multiple of the best-ever unseen% (default 1.5)
    #   TRELLIS_COVERAGE_MAX_PCT   hard ceiling regardless of history (default 15.0)
    #   TRELLIS_COVERAGE_RESET=1   forget the stored baseline, after an intentional trade-off
    if (not _skip_projection) and _cov_stats is not None:
        import os as _os_tw, json as _json_tw, time as _time_tw
        if _os_tw.environ.get("TRELLIS_COVERAGE_ABORT", "1").strip() != "0":
            _root = _os_tw.path.dirname(_os_tw.path.dirname(_os_tw.path.dirname(
                _os_tw.path.dirname(_os_tw.path.abspath(__file__)))))   # -> ComfyUI/
            _bl_path = _os_tw.path.join(_root, "uv_transfer", "_coverage_baseline.json")
            _regress = float(_os_tw.environ.get("TRELLIS_COVERAGE_REGRESS", "1.5"))
            _hardmax = float(_os_tw.environ.get("TRELLIS_COVERAGE_MAX_PCT", "15.0"))
            _base = None
            if _os_tw.environ.get("TRELLIS_COVERAGE_RESET", "0").strip() != "1":
                try:
                    with open(_bl_path, "r", encoding="utf-8") as _f:
                        _base = _json_tw.load(_f)
                except Exception:
                    _base = None
            _now = float(_cov_stats["total"])
            _fail = []
            if _now > _hardmax:
                _fail.append("total unseen {:.1f}% exceeds the hard ceiling {:.1f}%".format(
                    _now, _hardmax))
            if _base and isinstance(_base.get("total"), (int, float)):
                _bt = float(_base["total"]); _lim = _bt * _regress
                print("[Tripwire] unseen {:.2f}%  vs best-ever {:.2f}%  (allowed up to {:.2f}% "
                      "= {}x)".format(_now, _bt, _lim, _regress))
                for _k in sorted(_cov_stats):
                    if _k != "total" and isinstance(_base.get(_k), (int, float)):
                        print("[Tripwire]   {:<14} {:6.2f}%  (was {:6.2f}%, delta {:+6.2f})".format(
                            _k, float(_cov_stats[_k]), float(_base[_k]),
                            float(_cov_stats[_k]) - float(_base[_k])))
                if _now > _lim:
                    _fail.append("total unseen {:.1f}% is worse than {}x the best-ever "
                                 "{:.2f}%".format(_now, _regress, _bt))
            else:
                print("[Tripwire] no baseline yet - recording {:.2f}% unseen as the first "
                      "one".format(_now))
            if _fail:
                _worst = max(((_k, float(_v)) for _k, _v in _cov_stats.items() if _k != "total"),
                             key=lambda kv: kv[1], default=(None, 0.0))
                _msg = ["[Tripwire] ABORTING BAKE - texture coverage is too poor to ship."]
                _msg += ["  " + _r for _r in _fail]
                if _worst[0]:
                    _msg.append("  worst region: {} at {:.1f}% unseen".format(_worst[0], _worst[1]))
                _msg.append("  Stopped before FBX export / rigging, so no broken asset is written.")
                _msg.append("  Baseline file: " + _bl_path)
                _msg.append("  If this run is an intentional trade-off, re-run once with "
                            "TRELLIS_COVERAGE_RESET=1, or TRELLIS_COVERAGE_ABORT=0 to disable.")
                if _os_tw.environ.get("TRELLIS_COVERAGE_HARD", "0").strip() == "1":
                    raise RuntimeError("\n".join(_msg))
                # Otherwise WARN and finish. Aborting here throws a completed bake away: generation, mesh
                # prep and projection are already paid for, and for a NEW character none of it is cached.
                # An asset you can look at beats nothing after 13 minutes.
                print("[Tripwire] !! COVERAGE IS POOR: " + "; ".join(_fail)
                      + " -- continuing so the bake still produces an asset; the baseline is NOT updated."
                      + " Set TRELLIS_COVERAGE_HARD=1 to abort instead.")
            if (not _fail) and ((not _base) or _now < float(_base.get("total", 1e9))):
                try:
                    _cov_stats["views"] = _tw_views
                    _cov_stats["_recorded"] = _time_tw.strftime("%Y-%m-%d %H:%M:%S")
                    with open(_bl_path, "w", encoding="utf-8") as _f:
                        _json_tw.dump(_cov_stats, _f, indent=2)
                    print("[Tripwire] new best coverage - baseline updated to {:.2f}% "
                          "unseen".format(_now))
                except Exception as _bwe:
                    print("[Tripwire] could not write baseline: {!r}".format(_bwe))

    import os as _os_fill
    _FILL_MODE = _os_fill.environ.get("TRELLIS_FILL", "none").strip().lower()
    if (not _skip_projection) and _FILL_MODE == "none":
        # NO FILL. Only pixels that a camera actually saw end up on the model. Anything unseen is
        # left unpainted and MARKED, so the texture itself shows where an image is missing instead
        # of hiding it behind invented or copied colour. This is what removes the need for a hair
        # mask entirely: nothing has to guess what a surface "is" in order to colour it.
        # The marker is opaque on purpose -- seam_bleed pads into alpha==0 regions, so leaving the
        # gaps transparent would let the padding quietly fill them back in.
        _mark = _os_fill.environ.get("TRELLIS_MARK_UNCOVERED", "1") != "0"
        _n_un = int(uncovered_uv.sum())
        if _mark and _n_un:
            color_np[uncovered_uv] = np.array([255, 0, 255], dtype=color_np.dtype)
            alpha_np[uncovered_uv] = 255
            print(f"  Fill: NONE — {_n_un:,} unseen texels left unpainted and marked MAGENTA "
                  f"(TRELLIS_MARK_UNCOVERED=0 to leave them transparent)")
        else:
            print(f"  Fill: NONE — {_n_un:,} unseen texels left unpainted")
        _dump_stage('1_no_fill')
    elif (not _skip_projection) and fast_skin_fill and _FILL_MODE == "geometric":
        # GEOMETRIC FILL. Every uncovered texel takes the colour of its nearest PROJECTED texel
        # in 3-D -- a real observed pixel, never an invented one.
        #
        # This replaces two fills that were written for the front-image-only era and say so in
        # their own comments: fill_cylindrical ("front-only ... interpolating the two still-visible
        # silhouette-edge colours AROUND the body", 523,069 texels last bake) and wrap_arm_front
        # ("wrap the reliable front-arm colour over the whole arm", 396,700 texels = 33% of the
        # arms). Both synthesise colour rather than copy it, which is what produced the brown arm
        # banding; and fill_hair_3d only did the right thing INSIDE the hair mask, so hanging hair
        # outside that mask (mask floor z=0.093, hair reaches far below) got skin-toned fill --
        # the pink locks. The 3-D nearest-visible rule needs no class mask at all: hanging hair's
        # nearest visible neighbour is the hair above it.
        fu.fill_uncovered_3d(color_np, alpha_np, covered, uncovered_uv)
        _dump_stage('1_fill_uncovered_3d')
    elif (not _skip_projection) and fast_skin_fill:
        # ── CYLINDRICAL FILL (front-only): keep the projected front; fill every
        # uncovered (back/side) texel by interpolating the two still-visible
        # silhouette-edge colours AROUND the body, per height band.  Photo-derived,
        # smooth, height-varying skin — NOT a flat slab.  Skips the 3-D KD-tree
        # fill, dark-rescue and colour-consistency (the ~2-3 min cost, and the
        # dark-rescue that flattened the face).  Hair = its own FBX material.
        # NOTE: a global erode of `covered` (border-margin idea) was REVERTED — it also grows the
        # INTERNAL coverage holes (curved chest false-rejected by the depth test) into front patches.
        # The border-margin must only nibble the TRUE OUTER grazing edge, not internal holes (TODO).
        # Body: cylindrical wrap (hair OFF — the wrap assumes a cylinder and grabbed the body
        # silhouette on the drape -> skin-toned back hair). Hair gets its own 3-D fill below.
        fu.fill_cylindrical(color_np, alpha_np, covered, region_mask=seam_hair_mask, fill_hair=False)
        _dump_stage('1_cylindrical_fill')
        # HAIR: keep the front image for visible hair; ONLY fill the hair we cannot see, from the
        # nearest VISIBLE hair in 3-D (unseen back/sides take the nearby front-hair colour).
        fu.fill_hair_3d(color_np, alpha_np, covered, seam_hair_mask)
        _dump_stage('1b_hair_fill')
        # ARM WRAP (user's option 1): wrap the reliable front-arm colour over the whole arm
        # so the browner/darker top+back take the good front tone. GEOMETRY-targeted to the
        # lateral arm region (|x|>0.30) only -> cannot touch the breast/torso, and stable
        # across per-bake meshes (no colour thresholds — the thing that broke the breast).
        fu.wrap_arm_front(color_np, covered)
        _dump_stage('2_arm_wrap')
        # SEAM STITCH disabled — reverted to the known-good arm-wrap-only state while we
        # diagnose the upstream UV-unwrap collapse (atlas progressively emptying across
        # bakes). stitch_seams only recolours existing texels, so it is not the cause; kept
        # in fill_utils for when the unwrap is stable again.
        # fu.stitch_seams(color_np, debug=bool(int(_os_st.environ.get('STITCH_DEBUG', '1'))))
        # Other per-texel detect-replace passes stay DISABLED (over-caught the breast on a
        # different per-bake mesh). Seams are fixed at the SOURCE in gen_seams instead.
        # fu.recolor_seam_borders(color_np)
        # fu.replace_clear_borders(color_np)
    elif (not _skip_projection) and legacy_fill:
        fu.fill_uncovered_3d(color_np, alpha_np, covered, uncovered_uv)
        _dump_stage('1_fill_uncovered')
        # snapshot the projected colour BEFORE the dark-rescue, so the seam's hair
        # texels can be kept at their real projected colour (1b/1c would repaint the
        # darker hair with skin).
        _pre_rescue = color_np.copy() if seam_hair_mask is not None else None
        fu.fill_dark_rescue(color_np, alpha_np, uv_mask_np_bool,
                            p1b_dark_thresh, p1b_sat_thresh,
                            p1b_bright_fraction, p1b_bright_thresh,
                            p1b_relative_mode, debug_color)
        _dump_stage('2_dark_rescue')
        fu.correct_color_consistency(color_np, uv_mask_np_bool,
                                     skip=False,
                                     p1c_k=p1c_k, p1c_iterations=p1c_iterations,
                                     p1c_frac=p1c_frac, p1c_diff=p1c_diff,
                                     p1b_relative_mode=p1b_relative_mode)
        _dump_stage('3_color_consistency')
        # FOUNDATION + ENFORCE version A as a HARD boundary: BLONDE up to the line,
        # SKIN after it.  The projection strips the blonde (only ~144 of 2.8M hair
        # texels survived as blonde), so filling FROM blonde can't work -- instead we
        # RECOLOUR the whole hair region to blonde: keep each texel's brightness (so
        # shading/shape is preserved) but set the hue to the blonde tone.  Then cut any
        # hair colour that sits PAST the line.
        if False:  # DISABLED: hair colour now comes from the per-face FBX Hair material, not this texture recolour
            color_np[seam_hair_mask] = _pre_rescue[seam_hair_mask]
            alpha_np[seam_hair_mask] = 255
            from scipy.spatial import cKDTree as _CKDe
            _hair = seam_hair_mask & uv_mask_np_bool
            BLONDE = (np.array(hair_base_color, dtype=np.float32)
                      if 'hair_base_color' in dir() and hair_base_color is not None
                      else np.array([228.0, 198.0, 120.0], dtype=np.float32))   # natural blonde (tunable)
            if int(_hair.sum()) > 0:
                _Rf = color_np[:, :, 0].astype(np.float32)
                _Gf = color_np[:, :, 1].astype(np.float32)
                _Bf = color_np[:, :, 2].astype(np.float32)
                # SOLID uniform blonde: center on the MEAN hair brightness and clamp
                # tightly, so the mangled projection brightness can't blotch it into
                # streaks.  Reads as one blonde mane with only a whisper of form.
                _lum = (0.299 * _Rf + 0.587 * _Gf + 0.114 * _Bf)[_hair]
                _mean = max(float(_lum.mean()), 1.0)
                _scale = np.clip(_lum / _mean, 0.90, 1.10)[:, None]
                color_np[_hair] = np.clip(BLONDE[None, :] * _scale, 0, 255).astype(np.uint8)
                alpha_np[_hair] = 255
            # HARD CUT after the line: any hair-coloured texel OUTSIDE the region -> skin
            _Gc = color_np[:, :, 1].astype(np.float32); _Bc = color_np[:, :, 2].astype(np.float32)
            _ishaircol = (_Bc < 0.85 * _Gc) & (_Gc > 50.0)
            _bleed = (~seam_hair_mask) & _ishaircol & uv_mask_np_bool
            _skin_out = (~seam_hair_mask) & (~_ishaircol) & uv_mask_np_bool
            if int(_bleed.sum()) and int(_skin_out.sum()):
                _sy, _sx = np.where(_skin_out)
                if len(_sy) > 800000:
                    _sub = np.random.default_rng(0).choice(len(_sy), 800000, replace=False)
                    _sy, _sx = _sy[_sub], _sx[_sub]
                _tr = _CKDe(tex_pos_np[_sy, _sx].astype(np.float32))
                _qy, _qx = np.where(_bleed)
                _, _jj = _tr.query(tex_pos_np[_qy, _qx].astype(np.float32), k=1, workers=-1)
                color_np[_qy, _qx] = color_np[_sy[_jj], _sx[_jj]]
            print(f"[SeamHair-recolour] blonde up to line: {int(_hair.sum()):,} texels  "
                  f"tone={BLONDE.astype(int).tolist()}  cut after line: {int(_bleed.sum()):,}")
            # DIAG: trace the CROWN (paint height Y>0.40) and ARMS (|X|>0.20) texels
            _cy = uv_mask_np_bool & (tex_pos_np[:, :, 1] > 0.40)
            _ar = uv_mask_np_bool & (np.abs(tex_pos_np[:, :, 0]) > 0.20)
            _Gd = color_np[:, :, 1].astype(np.float32); _Bd = color_np[:, :, 2].astype(np.float32)
            _isbl = (_Bd < 0.85 * _Gd) & (_Gd > 50.0)
            print(f"[DIAG-crown] uv_texels={int(_cy.sum()):,}  flagged_hair={int((_cy & seam_hair_mask).sum()):,}  blonde_now={int((_cy & _isbl).sum()):,}")
            print(f"[DIAG-arms ] uv_texels={int(_ar.sum()):,}  flagged_hair={int((_ar & seam_hair_mask).sum()):,}  blonde_now={int((_ar & _isbl).sum()):,}")

    if not _skip_projection:
        # Close black cracks at UV-island seam rings (short-range, island-local;
        # NOT the cross-body 3-D fill). Runs before seam_bleed so the outside pad
        # then bleeds the already-closed edge colour.
        fu.fill_interior_seams(color_np, alpha_np)
    # ── SKIN PAINTED ONTO HAIR ────────────────────────────────────────────────────
    # A side view paints the ear / the near hand onto HAIR geometry, because the mesh's head and
    # arm do not sit where the generated image's do. That is a LOCAL mismatch: NCC against the
    # front view measured the left view as already optimally aligned (gain +0.000), so no global
    # scale/offset can fix it, and four attempts at one confirmed that the hard way.
    # Fix it where the knowledge exists instead. gen_seams already flags which faces are hair, so
    # a hair texel whose colour is closer to the BODY'S OWN SKIN median than to the HAIR median
    # has been mis-painted. Two medians measured from this character's own atlas -- no threshold,
    # no assumed hair or skin colour, works for any character.
    # Only the outliers move; the hair itself is untouched. This is NOT the disabled blanket
    # recolour above, which flattened all hair to a fixed blonde.
    try:
        import os as _os_hs
        if (seam_hair_mask is not None and _os_hs.environ.get("TRELLIS_HAIR_SKIN_FIX", "1") != "0"):
            _hairm = seam_hair_mask & uv_mask_np_bool & (alpha_np > 0)
            _skinm = (~seam_hair_mask) & uv_mask_np_bool & (alpha_np > 0)
            if int(_hairm.sum()) > 5000 and int(_skinm.sum()) > 5000:
                _c = color_np.astype(np.float32)
                _l = _c.sum(axis=2, keepdims=True).clip(min=1.0)
                _chroma = _c / _l                      # brightness-independent: shading cannot flip it
                _mh = np.median(_chroma[_hairm], axis=0)
                _ms = np.median(_chroma[_skinm], axis=0)
                if float(np.abs(_mh - _ms).sum()) > 1e-3:
                    _dh = np.linalg.norm(_chroma - _mh[None, None, :], axis=2)
                    _ds = np.linalg.norm(_chroma - _ms[None, None, :], axis=2)
                    _wrong = _hairm & (_ds < _dh)      # closer to skin than to hair
                    _n = int(_wrong.sum())
                    if _n > 0:
                        _good = _hairm & ~_wrong
                        if int(_good.sum()) > 100:
                            from scipy.spatial import cKDTree as _CK
                            _gy, _gx = np.where(_good)
                            if len(_gy) > 400000:
                                _sel = np.random.default_rng(0).choice(len(_gy), 400000, replace=False)
                                _gy, _gx = _gy[_sel], _gx[_sel]
                            _t = _CK(tex_pos_np[_gy, _gx].astype(np.float32))
                            _wy, _wx = np.where(_wrong)
                            _, _j = _t.query(tex_pos_np[_wy, _wx].astype(np.float32), k=1, workers=-1)
                            color_np[_wy, _wx] = color_np[_gy[_j], _gx[_j]]
                            print(f"  [HairSkinFix] {_n:,} hair texels were closer to skin than to "
                                  f"hair ({100.0*_n/max(int(_hairm.sum()),1):.1f}% of hair) -- "
                                  f"repainted from the nearest real hair texel in 3D")
                        else:
                            print(f"  [HairSkinFix] {_n:,} suspect texels but too little clean hair "
                                  f"to source from -- left alone")
                    else:
                        print("  [HairSkinFix] no hair texel is closer to skin than to hair -- clean")
    except Exception as _hse:
        print(f"  [HairSkinFix] skipped: {_hse!r}")

    fu.seam_bleed(color_np, alpha_np)
    fu.colour_correct(color_np, uv_mask_np_bool, saturation_boost, brightness_correction)
    _dump_stage('4_final')

    # =========================================================================
    # FLAT SKIN BASE COLOUR — overrides every UV pixel with a solid colour so
    # the projected skin texture is hidden.  hair_debug_color is applied on top
    # afterwards, so yellow hair is still visible over the flat skin.
    # =========================================================================
    if skin_base_color is not None and not _skip_projection:
        _skin_arr = np.array(skin_base_color, dtype=np.uint8)
        color_np[uv_mask_np_bool] = _skin_arr
        alpha_np[uv_mask_np_bool] = 255
        print(f"[SkinBase] {int(uv_mask_np_bool.sum()):,} UV px → flat skin {skin_base_color}")

    # =========================================================================
    # HAIR DEBUG FILL — paint detected hair pixels with a solid debug colour.
    # Enabled when hair_debug_color is not None (e.g. (255, 255, 255) = white).
    # Runs a silent HairDetectionUtils pass to build the 3D-based hair mask, then
    # overrides those texels in color_np.  Alpha is forced to 255 for coverage.
    # =========================================================================
    if hair_debug_color is not None and not _skip_projection:
        if seam_hair_mask is not None:
            # seam drives the hair mask -> skip the slow old detector + fixture save
            _hair_mask = seam_hair_mask
            print(f"[HairDebug] using SEAM hair mask ({int(_hair_mask.sum()):,} px) — old detector + fixture skipped")
        else:
            from .hair_detection_utils_v2 import HairDetectionUtils
            print("[HairDebug] Computing hair mask for debug fill...")
            _hdu_fill = HairDetectionUtils(
                texture_size   = texture_size,
                uv_mask        = uv_mask_np_bool,
                tex_pos_np     = tex_pos_np,
                tex_normals_np = tex_normals.cpu().numpy(),
                neck_y_min     = neck_y_min,
                neck_y_max     = neck_y_max,
                cyan_y_margin  = cyan_y_margin,
                cyan_z_frac    = cyan_z_frac,
                trimesh        = mesh,             # per-face ear detection (uses rasterized mesh's UVs)
                _norm_verts    = out_verts.detach().cpu().numpy(),
                _norm_faces    = out_faces.detach().cpu().numpy(),
                _norm_uvs      = out_uvs.detach().cpu().numpy(),
            )
            # Capture fixture so the old mask can be tuned offline without a re-run.
            import os as _os_hd
            _fx_hd = _os_hd.path.join(_os_hd.path.dirname(__file__), 'hair_detection_fixture.npz')
            np.savez_compressed(
                _fx_hd,
                tex_pos      = tex_pos_np,
                tex_normals  = tex_normals.cpu().numpy(),
                uv_mask      = uv_mask_np_bool,
                texture_size = np.array([texture_size]),
                norm_verts   = out_verts.detach().cpu().numpy(),
                norm_faces   = out_faces.detach().cpu().numpy(),
                norm_uvs     = out_uvs.detach().cpu().numpy(),
            )
            print(f"[Fixture] Saved (hair_debug path) → {_fx_hd}")
            _hair_mask = _hdu_fill.get_hair_mask()
        _n_hair    = int(_hair_mask.sum())
        # KAN-8: save the CLEAN projected base (skin + real hair colour, BEFORE the
        # debug-yellow is painted) so mask changes can be previewed OFFLINE with no
        # pipeline run — the removed-hair texels still hold their real skin colour.
        try:
            from PIL import Image as _PBImg
            _PBImg.fromarray(color_np).save(
                _os_hd.path.join(_os_hd.path.dirname(__file__), '_proj_base.png'))
            print("[HairDebug] saved clean projected base → _proj_base.png")
        except Exception as _be:
            print(f"[HairDebug] base save failed: {_be}")
        print(f"[HairDebug] {_n_hair:,} hair px → painting {hair_debug_color}")
        color_np[_hair_mask] = np.array(hair_debug_color, dtype=np.uint8)
        alpha_np[_hair_mask] = 255

        # ── EAR + NECK OVERRIDE (OLD detector path only) ──────────────────────
        # The seam already excludes the ear/neck from hair, so this redundant
        # override is skipped when the seam drives the mask (and _hdu_fill isn't
        # built then -- skipping it avoids the UnboundLocalError too).
        if seam_hair_mask is None:
            _skin_color = (np.array(skin_base_color, dtype=np.uint8)
                           if skin_base_color is not None
                           else np.array((210, 180, 155), dtype=np.uint8))
            _ear_uv = _hdu_fill.get_ear_uv_mask()
            if _ear_uv is not None and _ear_uv.any():
                _n_ear = int(_ear_uv.sum())
                color_np[_ear_uv] = _skin_color
                alpha_np[_ear_uv] = 255
                print(f"[HairDebug] EAR OVERRIDE: {_n_ear:,} px → skin {tuple(_skin_color.tolist())}")
            _neck_uv = _hdu_fill.get_neck_uv_mask()
            if _neck_uv is not None and _neck_uv.any():
                _n_neck = int(_neck_uv.sum())
                color_np[_neck_uv] = _skin_color
                alpha_np[_neck_uv] = 255
                print(f"[HairDebug] NECK OVERRIDE: {_n_neck:,} px → skin {tuple(_skin_color.tolist())}")

        # Re-bleed seams AFTER hair/ear/neck paint.  The earlier seam_bleed
        # (before hair paint) padded the inter-island gutters with the old
        # SKIN colour, so the hair's back-center UV seam sampled skin → visible
        # white band.  Re-running now re-pads each gutter with its island's
        # CURRENT colour (yellow on the hair side) → the seam band disappears.
        fu.seam_bleed(color_np, alpha_np)

    # In debug_regions mode override color/alpha with geometry-only values now
    # that all fill passes have been (mostly) skipped.
    if _skip_projection:
        tex_pos_np  = tex_pos.cpu().numpy()
        alpha_np    = uv_hit_mask.cpu().numpy().astype(np.uint8) * 255
        color_np    = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        print("[MultiView] debug_regions: geometry-only mode active — projection skipped")

    # =========================================================================
    # DEBUG – Paint UV texture with region colours based on 3-D world position.
    # Replaces the real texture so you can visually verify region boundaries
    # before committing to per-region colour correction.
    #
    # Coordinate system (after normalization, norm_size=1.15):
    #   +Y = up    (full character: ~-0.575 to +0.575)
    #   +Z = front (toward viewer at azimuth 0)
    #   +X = character's LEFT (from viewer's perspective)
    # =========================================================================
    if debug_regions:
        from .hair_detection_utils_v2 import HairDetectionUtils
        print("[DebugRegions] Running HairDetectionUtils...")

        # ── KAN-8 island-separation A/B metric (prints, no files) ─────────────
        # Quantifies whether ear surface got its own UV island for the current
        # `mesh_cluster_threshold_cone_half_angle_rad`.  Watch head_x_max climb
        # and frac|X|>0.10 go 0→>0 as the cone angle is lowered.
        try:
            from .uv_island_diag import analyse_arrays
            analyse_arrays(tex_pos_np, uv_mask_np_bool, texture_size,
                           label=f"cone={mesh_cluster_threshold_cone_half_angle_rad}deg "
                                 f"cyl={use_cylindrical_uv}")
        except Exception as _e:
            print(f"[uv_island_diag] skipped: {_e}")

        # ── One-time fixture save for offline unit testing ────────────────────
        import os as _os
        _fixture_path = _os.path.join(_os.path.dirname(__file__),
                                      'hair_detection_fixture.npz')
        if True:  # KAN-8: ALWAYS refresh so the offline fixture matches the latest run
            import numpy as _np_fix
            _np_fix.savez_compressed(
                _fixture_path,
                tex_pos      = tex_pos_np,
                tex_normals  = tex_normals.cpu().numpy(),
                uv_mask      = uv_mask_np_bool,
                texture_size = _np_fix.array([texture_size]),
                # Per-face data — lets offline tools reproduce the ear subtraction
                # (which needs normalized verts/faces/uvs), so ear-boundary tuning
                # can be verified without a full pipeline run.
                norm_verts   = out_verts.detach().cpu().numpy(),
                norm_faces   = out_faces.detach().cpu().numpy(),
                norm_uvs     = out_uvs.detach().cpu().numpy(),
            )
            print(f"[Fixture] Saved → {_fixture_path}")
        # ─────────────────────────────────────────────────────────────────────

        hdu = HairDetectionUtils(
            texture_size   = texture_size,
            uv_mask        = uv_mask_np_bool,
            tex_pos_np     = tex_pos_np,
            tex_normals_np = tex_normals.cpu().numpy(),
            neck_y_min     = neck_y_min,
            neck_y_max     = neck_y_max,
            cyan_y_margin  = cyan_y_margin,
            cyan_z_frac    = cyan_z_frac,
            trimesh        = mesh,             # per-face ear detection (uses rasterized mesh's UVs)
            _norm_verts    = out_verts.detach().cpu().numpy(),
            _norm_faces    = out_faces.detach().cpu().numpy(),
            _norm_uvs      = out_uvs.detach().cpu().numpy(),
        )
        color_np = hdu.run(color_np)

        # ── Optional flat skin base colour in debug_regions mode ─────────────
        # Paint UV pixels that the detection left black (background) so the
        # atlas isn't a pure black backdrop.  Detection overlay colours stay.
        if skin_base_color is not None:
            _skin_arr      = np.array(skin_base_color, dtype=np.uint8)
            _unpainted_sk  = np.all(color_np == 0, axis=2) & uv_mask_np_bool
            color_np[_unpainted_sk] = _skin_arr
            print(f"[SkinBase] {int(_unpainted_sk.sum()):,} unpainted UV px → flat skin {skin_base_color}")

        # ── Optional hair-fill overlay on top of the debug canvas ────────────
        # When hair_debug is also enabled, paint detected hair pixels with the
        # debug colour, but ONLY on pixels the detection left black (unpainted).
        # This shows the hair fill region while keeping cyan/yellow/green/magenta
        # borders visible on top.
        if hair_debug_color is not None:
            print("[HairDebug] Computing hair mask for debug overlay...")
            _hair_mask = seam_hair_mask if seam_hair_mask is not None else hdu.get_hair_mask()
            _unpainted = np.all(color_np == 0, axis=2) & uv_mask_np_bool
            _fill_here = _hair_mask & _unpainted
            color_np[_fill_here] = np.array(hair_debug_color, dtype=np.uint8)
            print(f"[HairDebug] {int(_hair_mask.sum()):,} hair px → "
                  f"{int(_fill_here.sum()):,} filled on unpainted px")

        alpha_np = uv_hit_mask.cpu().numpy().astype(np.uint8) * 255
        print("[DebugRegions] Done — HairDetectionUtils completed.")


    # =========================================================================
    # STEP 5 – Build output textures and trimesh
    # =========================================================================
    baseColorTexture = Image.fromarray(np.dstack([color_np, alpha_np]))
    metallicRoughnessTexture = None
    
    if hasattr(high_poly_mesh, 'visual') and hasattr(high_poly_mesh.visual, 'material') and isinstance(high_poly_mesh.visual.material, trimesh.visual.material.PBRMaterial):
        if high_poly_mesh.visual.material.metallicRoughnessTexture:
            metallicRoughnessTexture = high_poly_mesh.visual.material.metallicRoughnessTexture
            
    if metallicRoughnessTexture is None or not use_metallic:
        mr_np = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        # GLTF: G = roughness, B = metallic.
        # Skin roughness ~0.55 gives natural specularity; 0.9 was too matte/washed.
        mr_np[..., 1] = int(0.55 * 255)   # roughness 0.55
        mr_np[..., 2] = 0                  # metallic  0.0
        metallicRoughnessTexture = Image.fromarray(mr_np)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=baseColorTexture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=metallicRoughnessTexture,
        metallicFactor=0.0,
        roughnessFactor=1.0,   # let texture drive roughness directly
        alphaMode='OPAQUE',
        doubleSided=True,
    )

    verts_np = out_verts.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np   = out_uvs.cpu().numpy()
    uvs_np[:, 1] = 1.0 - uvs_np[:, 1]   # flip V: GL bottom-up -> image top-down

    textured_mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )

    return textured_mesh, baseColorTexture, metallicRoughnessTexture, normalMapTexture


# ---------------------------------------------------------------------------
# Alignment preview
# ---------------------------------------------------------------------------

def render_alignment_preview(
    mesh: trimesh.Trimesh,
    image,                          # PIL Image (any mode)
    azimuth:        float = 0.0,
    elevation:      float = 0.0,
    ortho_scale:    float = 1.15,
    norm_size:      float = 1.15,
    u_pixel_offset: float = 0.0,
    v_pixel_offset: float = 0.0,
    overlay_color:  tuple = (0, 230, 180),   # cyan-green outline
    overlay_alpha:  float = 0.30,
) -> Image.Image:
    """
    Render the mesh silhouette overlaid on a source image using the exact same
    auto-alignment math as texture_mesh_with_multiview.  Fast — no texture work.
    Use to tune u/v pixel offsets before running the full projection.
    """
    # ── 1. Normalise mesh (identical two-pass to projection) ─────────────────
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    verts = _normalize_verts(verts, norm_size)

    # ── 2. Camera vectors ────────────────────────────────────────────────────
    look, right, up = get_camera_vectors(azimuth, elevation, device='cuda')

    s = max(float(ortho_scale), 1e-6)

    # ── 3. Per-vertex clip-space coords (same formula as build_ortho_clip_verts) ──
    u_clip_v = (verts * right).sum(-1) / (s / 2.0)   # (N,)
    v_clip_v = (verts * up).sum(-1)    / (s / 2.0)   # (N,)

    # ── 4. Mesh bounds in clip space → alignment params ──────────────────────
    mesh_u_min = u_clip_v.min().item()
    mesh_u_max = u_clip_v.max().item()
    mesh_v_min = v_clip_v.min().item()
    mesh_v_max = v_clip_v.max().item()
    mesh_cx    = (mesh_u_min + mesh_u_max) / 2.0
    mesh_cy    = (mesh_v_min + mesh_v_max) / 2.0
    mesh_v_span = max(mesh_v_max - mesh_v_min, 1e-6)

    # ── 5. Foreground detection in source image (same as projection code) ─────
    img_rgba  = np.array(image.convert('RGBA'))
    img_h, img_w = img_rgba.shape[:2]

    if img_rgba[:, :, 3].min() < 255:
        fg_mask = img_rgba[:, :, 3] > 127
    else:
        bg_color   = img_rgba[0, 0, :3]
        color_diff = np.abs(img_rgba[:, :, :3].astype(int) - bg_color.astype(int)).sum(axis=-1)
        fg_mask    = color_diff > 10

    coords = np.argwhere(fg_mask)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img_char_h  = float(y_max - y_min + 1)
        img_char_cx = float(coords[:, 1].mean())
        img_char_cy = float(y_min + y_max) / 2.0   # bbox midpoint — consistent with projection code
    else:
        img_char_h  = float(img_h)
        img_char_cx = float(img_w) * 0.5
        img_char_cy = float(img_h) * 0.5

    pix_per_clip = img_char_h / mesh_v_span

    print(f"[AlignPreview] az={azimuth:.0f} el={elevation:.0f}  "
          f"char_cx={img_char_cx:.1f} char_h={img_char_h:.0f}  "
          f"pix/clip={pix_per_clip:.1f}  "
          f"u_off={u_pixel_offset:.1f} v_off={v_pixel_offset:.1f}")

    # ── 6. Build image-space NDC clip verts and rasterise at image resolution ─
    # The projection maps clip → image as:
    #   x_pix = img_char_cx + (u_clip - mesh_cx) * pix_per_clip + u_pixel_offset
    #   y_pix = img_char_cy - (v_clip - mesh_cy) * pix_per_clip + v_pixel_offset
    # Invert to NDC (GL convention: y=-1 bottom, y=+1 top):
    #   x_ndc = (x_pix + 0.5) / img_w * 2 - 1
    #   y_ndc = -((y_pix + 0.5) / img_h * 2 - 1)

    x_pix = img_char_cx + (u_clip_v - mesh_cx) * pix_per_clip + u_pixel_offset
    y_pix = img_char_cy - (v_clip_v - mesh_cy) * pix_per_clip + v_pixel_offset

    x_ndc = (x_pix + 0.5) / img_w * 2.0 - 1.0
    y_ndc = -(((y_pix + 0.5) / img_h) * 2.0 - 1.0)

    # Depth (same remapping as build_ortho_clip_verts)
    d     = -(verts * look).sum(-1)
    d_min = d.min()
    d_span = (d.max() - d_min).clamp(min=1e-6)
    z_ndc = 1.0 - 2.0 * (d - d_min) / d_span

    aligned_clip = torch.stack(
        [x_ndc, y_ndc, z_ndc, torch.ones_like(x_ndc)], dim=-1
    ).unsqueeze(0)

    ctx  = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, aligned_clip, faces.int(), resolution=[img_h, img_w])
    hit_mask = (rast[0, :, :, 3] > 0).cpu().numpy()   # (img_h, img_w), y=0 at bottom

    # nvdiffrast y=0 is bottom; image y=0 is top → flip
    hit_mask = hit_mask[::-1].copy()

    # ── 7. Draw overlay on source image ──────────────────────────────────────
    img_rgb = np.array(image.convert('RGB')).astype(np.float32)
    oc = np.array(overlay_color, dtype=np.float32)

    # Semi-transparent fill inside the mesh silhouette
    img_rgb[hit_mask] = img_rgb[hit_mask] * (1.0 - overlay_alpha) + oc * overlay_alpha

    # Solid 2-px outline at the silhouette boundary
    hit_u8   = hit_mask.astype(np.uint8)
    kernel   = np.ones((3, 3), np.uint8)
    outline  = cv2.dilate(hit_u8, kernel, iterations=2).astype(bool) & ~hit_mask.astype(bool)
    img_rgb[outline] = oc

    return Image.fromarray(img_rgb.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Standalone view renderer
# ---------------------------------------------------------------------------

def render_mesh_from_view(
    mesh: trimesh.Trimesh,
    azimuth: float,
    elevation: float,
    ortho_scale: float = 1.15,
    norm_size: float   = 1.15,
    render_size: int   = 1024,
) -> Image.Image:
    """
    Render the existing mesh base-colour texture from a given camera angle.

    Uses identical vertex normalisation and orthographic camera maths as
    texture_mesh_with_multiview, so the framing is pixel-perfect consistent
    with the projection stage.

    Returns a PIL RGB image (white background on non-mesh pixels) that can be
    fed directly as top_image / front_image etc. to the texturing node.

    Raises ValueError if the mesh has no UV or no base-colour texture.
    """
    # ── 1. Extract mesh data ─────────────────────────────────────────────────
    vis = getattr(mesh, 'visual', None)
    if not (vis is not None and hasattr(vis, 'uv') and vis.uv is not None):
        raise ValueError("render_mesh_from_view: mesh has no UV coordinates")

    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    uvs   = torch.from_numpy(vis.uv).float().cuda()          # GL convention: V=0 bottom

    # ── 2. Normalise (two-pass, identical to texture_mesh_with_multiview) ────
    verts = _normalize_verts(verts, norm_size)

    # ── 3. Load existing base-colour texture ─────────────────────────────────
    tex_pil = None
    try:
        tex_pil = mesh.visual.material.baseColorTexture
    except AttributeError:
        pass
    if tex_pil is None:
        raise ValueError("render_mesh_from_view: mesh has no baseColorTexture")

    # Flip V so grid_sample convention matches: after flip, UV v=0 → image row 0.
    tex_np = np.array(tex_pil.convert('RGB')).astype(np.float32) / 255.0
    tex_np = np.flip(tex_np, axis=0).copy()
    tex_t  = torch.from_numpy(tex_np).cuda().permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

    # ── 4. Camera + rasterize ────────────────────────────────────────────────
    ctx      = dr.RasterizeCudaContext()
    look, right, up = get_camera_vectors(azimuth, elevation)
    cam_clip = build_ortho_clip_verts(verts, right, up, look, ortho_scale)

    rast, _  = dr.rasterize(ctx, cam_clip, faces.int(), resolution=[render_size, render_size])
    hit_mask = (rast[0, :, :, 3] > 0).cpu().numpy()          # bool (H, W) in raster space

    # ── 5. Interpolate UV at each camera pixel ───────────────────────────────
    uv_at_pix = dr.interpolate(uvs.unsqueeze(0), rast, faces.int())[0][0]  # (H, W, 2)

    # Map GL UV [0,1] → grid_sample [-1,1].  No extra V-flip here because
    # the texture image was already flipped in step 3.
    u_g  = uv_at_pix[:, :, 0] * 2.0 - 1.0
    v_g  = uv_at_pix[:, :, 1] * 2.0 - 1.0
    grid = torch.stack([u_g, v_g], dim=-1).unsqueeze(0)       # (1, H, W, 2)

    sampled = F.grid_sample(tex_t, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
    rendered = sampled[0].permute(1, 2, 0).cpu().numpy()      # (H, W, 3)

    # ── 6. White background + Y-flip (nvdiffrast Y=0 is bottom, PIL Y=0 top) ─
    rendered[~hit_mask] = 1.0
    rendered = rendered[::-1].copy()

    print(f"[RenderView] az={azimuth:.0f} el={elevation:.0f}  "
          f"hit={hit_mask.sum()} / {render_size*render_size} px")
    return Image.fromarray((rendered * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Fast per-feature alignment preview (KAN-9)
# ---------------------------------------------------------------------------

def render_face_projection_preview(
    mesh: trimesh.Trimesh,
    image,                              # PIL source image (e.g. the front view)
    azimuth:           float = 0.0,
    elevation:         float = 0.0,
    ortho_scale:       float = 1.15,
    norm_size:         float = 1.15,
    u_pixel_offset:    float = 0.0,
    v_pixel_offset:    float = 0.0,
    blend:             float = 0.0,     # 0 = photo only; >0 fills shaded relief over photo
    show_mesh_edges:   bool  = True,    # orange  = mesh GEOMETRIC features (nose/eyes/lips)
    show_source_edges: bool  = True,    # cyan    = PHOTO features
    mesh_edge_color:   tuple = (255, 140, 0),
    source_edge_color: tuple = (0, 230, 230),
) -> Image.Image:
    """
    Fast per-feature alignment check for KAN-9 — NO projection bake, NO fill,
    NO UV.  Runs in well under a second (one rasterise + one normal interpolate).

    The real bake misplaces colour when the mesh point that is geometrically a
    feature (lip, eye) gets sampled from an image pixel that is NOT that feature.
    So the thing to compare is the MESH's own features against the PHOTO's
    features, both placed by the *identical* auto-alignment mapping the texturing
    node uses (height-bbox scale + foreground centroid).  This renders the mesh
    with simple front-lit shading so nose / eye-sockets / lips / jaw relief shows,
    frames it onto the photo with that mapping, and overlays:

      • orange edges = where the MESH's geometric features sit
      • cyan   edges = where the PHOTO's features sit

    If orange sits on cyan, alignment is correct.  Any orange→cyan offset is the
    misalignment that will smear colour onto the wrong geometry during the bake —
    read its direction/size straight off the image, and tune u/v_pixel_offset (or
    diagnose a scale/proportion mismatch) without a 10-minute pipeline run.
    """
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    verts = _normalize_verts(verts, norm_size)

    # Vertex normals (trigger trimesh computation if absent)
    vn = np.asarray(mesh.vertex_normals)
    if vn is None or len(vn) != len(mesh.vertices):
        mesh.vertex_normals  # property access forces recompute
        vn = np.asarray(mesh.vertex_normals)
    nrm = torch.from_numpy(vn).float().cuda()
    nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-8)

    look, right, up = get_camera_vectors(azimuth, elevation, device='cuda')

    # Per-vertex clip coords (same formula as build_ortho_clip_verts)
    s = max(float(ortho_scale), 1e-6)
    u_clip_v = (verts * right).sum(-1) / (s / 2.0)
    v_clip_v = (verts * up).sum(-1)    / (s / 2.0)

    mesh_u_min = u_clip_v.min().item(); mesh_u_max = u_clip_v.max().item()
    mesh_v_min = v_clip_v.min().item(); mesh_v_max = v_clip_v.max().item()
    mesh_cx     = (mesh_u_min + mesh_u_max) / 2.0
    mesh_cy     = (mesh_v_min + mesh_v_max) / 2.0
    mesh_v_span = max(mesh_v_max - mesh_v_min, 1e-6)

    # Source image + foreground bbox (identical to the bake's body-anchored align)
    img_rgb = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    img_h, img_w = img_rgb.shape[:2]

    img_rgba = np.array(image.convert('RGBA'))
    if img_rgba[:, :, 3].min() < 255:
        fg_mask = img_rgba[:, :, 3] > 127
    else:
        bg = img_rgba[0, 0, :3]
        fg_mask = np.abs(img_rgba[:, :, :3].astype(int) - bg.astype(int)).sum(-1) > 10

    coords = np.argwhere(fg_mask)
    if len(coords) > 0:
        y0, _x0 = coords.min(0); y1, _x1 = coords.max(0)
        img_char_h  = float(y1 - y0 + 1)
        img_char_cx = float(coords[:, 1].mean())
        img_char_cy = float(y0 + y1) / 2.0
    else:
        img_char_h = float(img_h); img_char_cx = img_w * 0.5; img_char_cy = img_h * 0.5

    pix_per_clip = img_char_h / mesh_v_span

    x_pix = img_char_cx + (u_clip_v - mesh_cx) * pix_per_clip + u_pixel_offset
    y_pix = img_char_cy - (v_clip_v - mesh_cy) * pix_per_clip + v_pixel_offset
    x_ndc = (x_pix + 0.5) / img_w * 2.0 - 1.0
    y_ndc = -(((y_pix + 0.5) / img_h) * 2.0 - 1.0)

    d      = -(verts * look).sum(-1)
    d_min  = d.min(); d_span = (d.max() - d_min).clamp(min=1e-6)
    z_ndc  = 1.0 - 2.0 * (d - d_min) / d_span

    clip = torch.stack([x_ndc, y_ndc, z_ndc, torch.ones_like(x_ndc)], dim=-1).unsqueeze(0)

    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, clip, faces.int(), resolution=[img_h, img_w])
    hit = rast[0, :, :, 3] > 0

    pix_nrm = dr.interpolate(nrm.unsqueeze(0), rast, faces.int())[0][0]
    pix_nrm = pix_nrm / (pix_nrm.norm(dim=-1, keepdim=True) + 1e-8)
    lambert = (pix_nrm * (-look)).sum(-1).clamp(0.0, 1.0)
    shade   = (0.25 + 0.75 * lambert)            # soft ambient floor

    # nvdiffrast Y=0 is bottom; flip to image (top-down) orientation
    hit_np   = hit.cpu().numpy()[::-1].copy()
    shade_np = shade.cpu().numpy()[::-1].copy()

    out = img_rgb.copy()
    b = float(np.clip(blend, 0.0, 1.0))
    if b > 0.0:
        shaded_rgb = np.repeat(shade_np[..., None], 3, axis=2)
        out[hit_np] = (1.0 - b) * img_rgb[hit_np] + b * shaded_rgb[hit_np]
    out_u8 = (out * 255).clip(0, 255).astype(np.uint8)

    # Orange = mesh geometric-feature edges (Canny on the shaded relief)
    if show_mesh_edges:
        sh_u8 = (shade_np * 255).clip(0, 255).astype(np.uint8)
        me = cv2.Canny(sh_u8, 40, 110)
        me = cv2.dilate(me, np.ones((2, 2), np.uint8), 1).astype(bool) & hit_np
        out_u8[me] = np.array(mesh_edge_color, dtype=np.uint8)

    # Cyan = photo feature edges
    if show_source_edges:
        g  = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        se = cv2.Canny(g, 60, 150)
        se = cv2.dilate(se, np.ones((2, 2), np.uint8), 1).astype(bool) & fg_mask
        out_u8[se] = np.array(source_edge_color, dtype=np.uint8)

    print(f"[FaceAlignPreview] az={azimuth:.0f} el={elevation:.0f}  "
          f"hit={int(hit_np.sum())}px  pix/clip={pix_per_clip:.1f}")
    return Image.fromarray(out_u8)
