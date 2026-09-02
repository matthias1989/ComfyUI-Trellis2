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

    for img, az, el, view_w in (zip(images, azimuths, elevations, view_weights)
                                  if not _skip_projection else []):
        img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        img_h, img_w = img_np.shape[:2]

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

        # Dilate the foreground mask and inpaint the background band
        # Scale dilation with image resolution to prevent bilinear bleed
        base_res = 1024
        scale = max(img_h, img_w) / base_res
        dilate_px = int(max(5, 5 * scale))
        kernel = np.ones((3, 3), np.uint8)
        dilated_fg = cv2.dilate(fg_mask, kernel, iterations=dilate_px)
        band_to_fill = cv2.bitwise_and(dilated_fg, cv2.bitwise_not(fg_mask))

        # Inpaint the color channels into the dilated band
        img_rgb_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if int(band_to_fill.sum()) > 0:
            for c in range(3):
                img_rgb_u8[..., c] = cv2.inpaint(img_rgb_u8[..., c], band_to_fill, 3, cv2.INPAINT_NS)
        img_np = img_rgb_u8.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_np).cuda().permute(2, 0, 1).unsqueeze(0).contiguous()

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

        print(f"    [align az={az:.0f}] img_centre=({img_char_cx:.1f},{img_char_cy:.1f}) "
              f"char_h={img_char_h:.0f}px  pix/clip={pix_per_clip:.1f}")

        # v_clip positive = up in world = smaller y in image (origin top-left)
        x_pixel = img_char_cx + (u_clip - mesh_cx) * pix_per_clip + u_pixel_offset
        y_pixel = img_char_cy - (v_clip - mesh_cy) * pix_per_clip + v_pixel_offset

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

        depth_match = tex_depth >= sampled_depth - depth_eps

        # Foreground check: reject texels that project onto the image background.
        # Use the DILATED foreground mask (not the raw one) so that soft edges
        # in SV3D-generated frames don't create dark halos at island boundaries.
        fg_mask_t    = torch.from_numpy(dilated_fg.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
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

        # Visibility Mask
        visible = uv_hit_mask & in_bounds & sampled_hit & depth_match & sampled_fg & bright_enough & good_angle

        dot = dot_raw.clamp(min=0.0) ** 1.5
        weights = (dot ** blend_exponent) * visible.float() * view_w

        acc_color += sampled_colors * weights.unsqueeze(-1)
        acc_weight += weights

    # =========================================================================
    # STEP 4 – Normalize and blend over existing PBR texture
    # =========================================================================
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

    if not _skip_projection:
        fu.fill_uncovered_3d(color_np, alpha_np, covered, uncovered_uv)
        fu.fill_dark_rescue(color_np, alpha_np, uv_mask_np_bool,
                            p1b_dark_thresh, p1b_sat_thresh,
                            p1b_bright_fraction, p1b_bright_thresh,
                            p1b_relative_mode, debug_color)
        fu.correct_color_consistency(color_np, uv_mask_np_bool,
                                     skip=False,
                                     p1c_k=p1c_k, p1c_iterations=p1c_iterations,
                                     p1c_frac=p1c_frac, p1c_diff=p1c_diff,
                                     p1b_relative_mode=p1b_relative_mode)

    fu.seam_bleed(color_np, alpha_np)
    fu.colour_correct(color_np, uv_mask_np_bool, saturation_boost, brightness_correction)

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
        # Capture fixture (write-once, incl per-face data) so the hair mask and
        # ear subtraction can be tuned offline without a full pipeline re-run.
        import os as _os_hd
        _fx_hd = _os_hd.path.join(_os_hd.path.dirname(__file__), 'hair_detection_fixture.npz')
        if True:  # KAN-8: ALWAYS refresh so the offline fixture matches the latest run
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

        # ── EAR + NECK OVERRIDE: paint these UV pixels skin colour to defeat ─
        # the hair-yellow that leaks via shared UV islands.  Applied AFTER
        # hair_mask paint so the override wins at those texels.
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
            _hair_mask = hdu.get_hair_mask()
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
