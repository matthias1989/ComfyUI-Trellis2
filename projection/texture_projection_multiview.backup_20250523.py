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
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

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
    debug_regions:          bool  = False,  # replace texture with K-means cluster colour map
    debug_regions_k:        int   = 6,      # number of colour clusters
):
    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")
        
    if not (len(images) == len(azimuths) == len(elevations)):
        raise ValueError("images, azimuths, and elevations must have the same length")

    num_views = len(images)

    if view_weights is None:
        view_weights = [1.0] * num_views

    if len(view_weights) != num_views:
        raise ValueError("view_weights must match number of images")        

    num_views = len(images)
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
    # NEW: Replicate blender_render.py normalization
    # =========================================================================
    # 1. Center the geometry (Blender does this in auto_center_and_scale)
    bbox_min = out_verts.min(dim=0)[0]
    bbox_max = out_verts.max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    out_verts = out_verts - center

    # 2. Scale to match Blender's norm_size
    # Blender calculates Max Radius from the new center
    current_max_radius = torch.sqrt((out_verts**2).sum(dim=-1).max())
    norm_size = 1.15  # This must match --norm_size in blender_render.py
    scale_user = current_max_radius * 2.0
    scale_factor = norm_size / scale_user

    out_verts = out_verts * scale_factor

    print(f"[MultiView] Mesh: {out_verts.shape[0]} verts, {out_faces.shape[0]} faces")
    print(f"  X:[{out_verts[:,0].min():.3f},{out_verts[:,0].max():.3f}]"
          f"  Y:[{out_verts[:,1].min():.3f},{out_verts[:,1].max():.3f}]"
          f"  Z:[{out_verts[:,2].min():.3f},{out_verts[:,2].max():.3f}]")


    current_max_radius = torch.sqrt((out_verts**2).sum(dim=-1).max())
    scale_factor = norm_size / (current_max_radius * 2.0)
    out_verts = out_verts * scale_factor # Scale to 1.15 units
    out_normals = out_normals / (out_normals.norm(dim=-1, keepdim=True) + 1e-8)

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
    # This encodes the smooth vertex normals (which carry curvature detail from
    # the original high-poly voxel mesh) into tangent space relative to the UV
    # layout. UE5 DirectX convention: R=+X, G=-Y (flipped), B=+Z.
    print("[MultiView] Baking geometry normal map...")

    # Per-face edge vectors and UV deltas
    v0_f   = out_verts[out_faces[:, 0]]    # (F, 3)
    v1_f   = out_verts[out_faces[:, 1]]
    v2_f   = out_verts[out_faces[:, 2]]
    uv0_f  = out_uvs[out_faces[:, 0]]      # (F, 2)
    uv1_f  = out_uvs[out_faces[:, 1]]
    uv2_f  = out_uvs[out_faces[:, 2]]

    dp1_f  = v1_f - v0_f                   # (F, 3)
    dp2_f  = v2_f - v0_f
    duv1_f = uv1_f - uv0_f                 # (F, 2)
    duv2_f = uv2_f - uv0_f

    det_f   = duv1_f[:, 0] * duv2_f[:, 1] - duv1_f[:, 1] * duv2_f[:, 0]  # (F,)
    inv_det = torch.where(det_f.abs() > 1e-8, 1.0 / det_f, torch.zeros_like(det_f))

    T_face = (dp1_f * duv2_f[:, 1:2] - dp2_f * duv1_f[:, 1:2]) * inv_det.unsqueeze(-1)
    B_face = (-dp1_f * duv2_f[:, 0:1] + dp2_f * duv1_f[:, 0:1]) * inv_det.unsqueeze(-1)
    T_face = F.normalize(T_face, dim=-1)   # (F, 3)
    B_face = F.normalize(B_face, dim=-1)

    # Flat face normal – tangent-space reference direction
    N_geom_face = F.normalize(torch.cross(dp1_f, dp2_f, dim=-1), dim=-1)  # (F, 3)

    # Per-texel TBN lookup
    fid_clamped = tex_face_id.clamp(0)              # (H, W), safe index
    T_tex       = T_face[fid_clamped]               # (H, W, 3)
    N_geom_tex  = N_geom_face[fid_clamped]          # (H, W, 3)
    N_smooth    = tex_normals                        # (H, W, 3), smooth vertex normals

    # Gram-Schmidt: re-orthogonalise T against smooth N
    T_gs = F.normalize(T_tex - (T_tex * N_smooth).sum(-1, keepdim=True) * N_smooth, dim=-1)
    B_gs = torch.cross(N_smooth, T_gs, dim=-1)       # (H, W, 3)

    # Project smooth normal into tangent space
    nx = (N_smooth * T_gs).sum(-1, keepdim=True)     # (H, W, 1)
    ny = (N_smooth * B_gs).sum(-1, keepdim=True)
    nz = (N_smooth * N_geom_tex).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    # Encode: (n+1)/2*255, flip Y for UE5 DirectX convention
    normal_ts  = torch.cat([nx, -ny, nz], dim=-1)   # flip G
    normal_enc = ((normal_ts + 1.0) * 0.5 * 255.0).clamp(0, 255)

    # Fill with flat-normal default, overwrite UV-hit texels
    nm_img = torch.full((texture_size, texture_size, 3), 128.0, device='cuda', dtype=torch.float32)
    nm_img[..., 2] = 255.0
    nm_img[uv_hit_mask] = normal_enc[uv_hit_mask]
    nm_np = nm_img.byte().cpu().numpy()

    # Pad outward from UV island boundary (prevents seam sampling artifacts)
    uv_bool_nm = uv_hit_mask.cpu().numpy().astype(bool)
    outside_nm = ~uv_bool_nm
    if outside_nm.any():
        _, nm_idx = distance_transform_edt(outside_nm, return_indices=True)
        ry, rx = nm_idx[0], nm_idx[1]
        for c in range(3):
            nm_np[outside_nm, c] = nm_np[ry[outside_nm], rx[outside_nm], c]

    normalMapTexture = Image.fromarray(nm_np, mode='RGB')
    print(f"  Normal map: {texture_size}x{texture_size} (DirectX/UE5 convention)")

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
    # FILL – two-pass nearest-neighbour propagation
    # In debug_regions mode the expensive KD-tree passes are skipped; alpha_np
    # and color_np are set directly from the UV rasterisation result below.
    # =========================================================================
    uv_mask_np_bool = uv_hit_mask.cpu().numpy().astype(bool)
    covered         = valid_mask.cpu().numpy()          # bool (H,W): any weight > 0
    uncovered_uv    = uv_mask_np_bool & ~covered        # back-facing UV texels

    n_src  = int(covered.sum())
    n_fill = int(uncovered_uv.sum())
    if not _skip_projection:
        print(f"  Fill pass 1 (3-D): {n_src} projected texels → filling {n_fill} "
              f"back-facing UV texels ({100.0*n_fill/max(uv_mask_np_bool.sum(),1):.1f}% of UV)...")

    if not _skip_projection and n_src == 0:
        print("  WARNING: no projected pixels found – check front image connection.")
    elif not _skip_projection and n_fill > 0:
        # 3-D world-space nearest-neighbour fill.
        #
        # WHY NOT UV-space (distance_transform_edt)?
        #   UV-space distance is misleading: the back-of-head UV region can be
        #   2-D-adjacent to the face-centre UV region even though they are
        #   3-D-far apart.  The old 2-D fill therefore spread eye/nose texture
        #   onto the back of the head, and torso-background colours onto the
        #   arms.
        #
        # HOW 3-D FILL WORKS:
        #   For every back-facing UV texel we know its 3-D world position
        #   (tex_pos, interpolated by the UV rasteriser).  We build a KD-tree
        #   of all *covered* UV texels and for each uncovered texel find its
        #   nearest covered 3-D neighbour, then copy that colour.  The back of
        #   the head therefore gets colours from the 3-D-nearest covered point
        #   (the side of the head → hair / skin tone), not from the face centre.
        tex_pos_np = tex_pos.cpu().numpy()   # (H, W, 3)

        cov_y,   cov_x   = np.where(covered)
        uncov_y, uncov_x = np.where(uncovered_uv)

        # Subsample source points for the KD-tree.
        # Neighbouring covered texels are spatially very close (sub-mm in world
        # space) so taking every 4th point loses no perceptible quality while
        # cutting tree build + query time by ~16×.
        stride = 4
        cov_y_s  = cov_y[::stride]
        cov_x_s  = cov_x[::stride]

        # Brightness-filter the source points: only build the KD-tree from
        # well-lit pixels.  Arm/leg edges projected at a grazing angle come in
        # dark even though the surface itself is skin-coloured; without this
        # filter the fill propagates those dark edge values to uncovered areas
        # (arm tops, inner thighs, etc.) instead of the correct bright skin.
        src_brightness = color_np[cov_y_s, cov_x_s].mean(axis=-1)  # uint8 → [0,255]
        bright_src     = src_brightness > 70   # ≈ 0.27 normalised – raises bar so dark oblique-angle
                                              # arm/shoulder sides are not used as fill sources for arm tops.
        if bright_src.sum() > 500:
            cov_y_s = cov_y_s[bright_src]
            cov_x_s = cov_x_s[bright_src]
            print(f"    (Brightness filter: {bright_src.sum()} / {len(bright_src)} source points kept)")
        else:
            print(f"    (Brightness filter skipped – only {bright_src.sum()} bright points)")

        cov_3d   = tex_pos_np[cov_y_s, cov_x_s]   # (N_bright, 3)
        uncov_3d = tex_pos_np[uncov_y, uncov_x]    # (N_uncov,  3)

        print(f"    Building KD-tree on {len(cov_3d)} source points (1-in-{stride} subsample, brightness-filtered)...")
        tree = cKDTree(cov_3d)
        _, nn_idx = tree.query(uncov_3d, workers=-1)  # parallel query

        for c in range(3):
            color_np[uncov_y, uncov_x, c] = color_np[cov_y_s[nn_idx], cov_x_s[nn_idx], c]
        alpha_np[uncov_y, uncov_x] = 255
        print("  Fill pass 1 complete (3-D nearest-neighbour).")

    # Pass 1b: replace every dark UV texel with its nearest bright 3-D neighbour.
    #
    # "Dark" = mean RGB < DARK_THRESH (genuine shadow / background bleed).
    # Source = saturated UV texels above the brightness floor (relative or absolute mode).
    # Uses 3-D world-space nearest-neighbour so the replacement colour is always
    # a geometrically close surface pixel, not a random UV-adjacent one.
    #
    # If debug_color=(R,G,B) is passed, the dark texels are painted that flat
    # colour instead of being filled from neighbours.  Useful to visualise which
    # areas are being rescued without waiting for the KD-tree.
    DARK_THRESH = p1b_dark_thresh    # configurable from node widget
    SAT_THRESH  = p1b_sat_thresh    # Rejects grey pixels (sat≈0) so they can never become
                                    # the fill colour for a dark area.

    mean_bright = color_np.mean(axis=-1)                          # (H, W)

    # Compute saturation for source filtering (one full-image HSV conversion)
    _s_bgr = color_np[:, :, ::-1].copy()
    _s_hsv = cv2.cvtColor(_s_bgr, cv2.COLOR_BGR2HSV)
    saturation_map = _s_hsv[:, :, 1].astype(float)               # 0-255

    dark_in_uv = uv_mask_np_bool & (mean_bright < DARK_THRESH)

    # ── Source brightness floor ──────────────────────────────────────────────
    # Relative mode (default, p1b_relative_mode=True):
    #   Compute the median brightness of all saturated UV pixels and use
    #   that × p1b_bright_fraction as the floor.  Adapts automatically to
    #   dark or light skin tones — no manual threshold tuning needed per character.
    # Absolute mode (p1b_relative_mode=False):
    #   Use fixed p1b_bright_thresh, same as old behaviour.
    if p1b_relative_mode:
        # Only consider saturated UV pixels as the representative skin sample
        _sat_uv_bright = mean_bright[uv_mask_np_bool & (saturation_map >= SAT_THRESH)]
        if len(_sat_uv_bright) > 100:
            adaptive_floor = float(np.median(_sat_uv_bright)) * p1b_bright_fraction
        else:
            # Fallback if not enough saturated pixels (e.g. very desaturated scene)
            adaptive_floor = float(p1b_bright_thresh)
        bright_in_uv = (uv_mask_np_bool
                        & (saturation_map >= SAT_THRESH)
                        & (mean_bright    >= adaptive_floor))
        _mode_str = f"relative (median×{p1b_bright_fraction:.2f} → floor={adaptive_floor:.1f})"
        _floor_val = adaptive_floor
    else:
        bright_in_uv = (uv_mask_np_bool
                        & (saturation_map >= SAT_THRESH)
                        & (mean_bright    >= p1b_bright_thresh))
        _mode_str = f"absolute (thresh={p1b_bright_thresh})"
        _floor_val = p1b_bright_thresh

    n_dark   = int(dark_in_uv.sum())
    n_bright = int(bright_in_uv.sum())
    print(f"  Fill pass 1b: {n_dark} dark texels (mean<{DARK_THRESH}), "
          f"{n_bright} saturated bright sources [{_mode_str}, sat>={SAT_THRESH}]")

    if n_dark > 0:
        if debug_color is not None:
            # Debug mode: paint flat colour so you can see exactly what gets rescued
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
            bright_3d  = tex_pos_np[bright_y_s, bright_x_s]
            dark_3d    = tex_pos_np[dark_y,     dark_x]
            print(f"    KD-tree: {len(bright_3d)} sources (1-in-{stride2} subsample)...")
            tree2 = cKDTree(bright_3d)
            _, nn2 = tree2.query(dark_3d, workers=-1)
            for c in range(3):
                color_np[dark_y, dark_x, c] = color_np[bright_y_s[nn2], bright_x_s[nn2], c]
            alpha_np[dark_y, dark_x] = 255
            print(f"  Fill pass 1b complete: {n_dark} dark texels replaced from nearest bright neighbour.")
        else:
            print(f"  Fill pass 1b: skipped — not enough bright sources ({n_bright})")

    # Pass 1c: 3-D spatial colour-consistency correction
    #
    # The earlier passes fill UNCOVERED texels and rescue DARK texels, but leave
    # alone any texel that is (a) covered by some view and (b) not dark.  That
    # includes "chalk-white" or desaturated grey patches — arm tops / shoulder
    # crowns that received a grazing-angle projection and ended up washed-out even
    # though they are technically "covered" and "bright enough".
    #
    # Strategy: for each UV texel, find its K nearest 3-D neighbours.  If this
    # texel's colour saturation is far below the neighbourhood median AND the mean
    # per-channel colour distance is large enough, replace it with the neighbourhood
    # median.  The saturation comparison focuses specifically on the grey-vs-skin
    # outlier pattern while leaving legitimate variation (hair, lips, face features)
    # completely untouched.
    #
    # Parameters (conservative defaults — tune CONSISTENCY_FRAC upward if grey
    # patches remain, downward if good skin is being over-corrected):
    # In relative mode the grey patches tend to be larger (adaptive floor is more
    # permissive), so we use the full K + multi-iteration flood-fill.
    # In absolute mode the old fast settings (K=24, 1 iteration) are sufficient.
    if p1b_relative_mode:
        CONSISTENCY_K        = p1c_k
        _p1c_iterations      = p1c_iterations
    else:
        CONSISTENCY_K        = 24
        _p1c_iterations      = 1
    CONSISTENCY_FRAC        = p1c_frac
    CONSISTENCY_DIFF        = p1c_diff
    CONSISTENCY_MIN_NBR_SAT = 18    # skip correction if neighbourhood is already grey

    _c1c_all_y, _c1c_all_x = np.where(uv_mask_np_bool)
    _c1c_n = len(_c1c_all_y)

    if _c1c_n > CONSISTENCY_K * 4 and not _skip_projection:
        _c1c_pos = tex_pos_np[_c1c_all_y, _c1c_all_x]   # (N, 3) world positions

        # Subsample tree — every M-th point; aim for ~200K tree points max.
        # More points = denser tree = corrected pixels more likely to be sources
        # in subsequent iterations, so the flood-fill propagates further inward.
        # The tree is built ONCE and reused across iterations (positions don't change).
        _c1c_stride = max(1, _c1c_n // 200_000)
        _c1c_ty = _c1c_all_y[::_c1c_stride]
        _c1c_tx = _c1c_all_x[::_c1c_stride]
        _c1c_tp = tex_pos_np[_c1c_ty, _c1c_tx]
        _c1c_k  = min(CONSISTENCY_K, len(_c1c_tp) - 1)
        print(f"  Pass 1c (3-D colour consistency): {_c1c_n} texels, "
              f"tree={len(_c1c_tp)} pts (1-in-{_c1c_stride}), K={_c1c_k}, "
              f"iterations={_p1c_iterations}...")

        _c1c_tree = cKDTree(_c1c_tp)
        _, _c1c_nn = _c1c_tree.query(_c1c_pos, k=_c1c_k, workers=-1)   # (N, K) — fixed

        _c1c_total_fix = 0
        for _iter in range(_p1c_iterations):
            # Neighbourhood median colour — re-sampled each iteration so corrected
            # pixels from the previous pass become valid sources for their neighbours.
            _c1c_nbr_rgb = color_np[_c1c_ty[_c1c_nn], _c1c_tx[_c1c_nn]]   # (N, K, 3)
            _c1c_med_rgb = np.median(_c1c_nbr_rgb, axis=1)                  # (N, 3) float

            _c1c_this_rgb = color_np[_c1c_all_y, _c1c_all_x].astype(np.float32)

            _c1c_this_bgr = _c1c_this_rgb[:, ::-1].clip(0, 255).astype(np.uint8).reshape(1, _c1c_n, 3)
            _c1c_med_bgr  = _c1c_med_rgb[:, ::-1].clip(0, 255).astype(np.uint8).reshape(1, _c1c_n, 3)
            _c1c_this_hsv = cv2.cvtColor(_c1c_this_bgr, cv2.COLOR_BGR2HSV)[0]
            _c1c_med_hsv  = cv2.cvtColor(_c1c_med_bgr,  cv2.COLOR_BGR2HSV)[0]

            _c1c_this_sat = _c1c_this_hsv[:, 1].astype(float)
            _c1c_med_sat  = _c1c_med_hsv[:,  1].astype(float)
            _c1c_diff     = np.abs(_c1c_this_rgb - _c1c_med_rgb).mean(axis=-1)

            _c1c_outlier = (
                (_c1c_med_sat  > CONSISTENCY_MIN_NBR_SAT) &
                (_c1c_this_sat < _c1c_med_sat * CONSISTENCY_FRAC) &
                (_c1c_diff     > CONSISTENCY_DIFF)
            )
            _c1c_n_fix = int(_c1c_outlier.sum())
            _c1c_total_fix += _c1c_n_fix
            if _c1c_n_fix > 0:
                color_np[_c1c_all_y[_c1c_outlier], _c1c_all_x[_c1c_outlier]] = \
                    _c1c_med_rgb[_c1c_outlier].clip(0, 255).astype(np.uint8)
                print(f"    iter {_iter+1}: {_c1c_n_fix} texels corrected")
            else:
                print(f"    iter {_iter+1}: 0 outliers — converged early.")
                break   # no point continuing if nothing changed

        print(f"  Pass 1c complete: {_c1c_total_fix} total corrections over "
              f"{_iter+1} iteration(s) ({100.0*_c1c_total_fix/_c1c_n:.1f}% of UV island).")
        del _c1c_pos, _c1c_nbr_rgb, _c1c_med_rgb, _c1c_this_rgb, _c1c_tree, _c1c_nn

    # Pass 2: bleed 8 px outside islands so seam edges never sample black background
    uv_mask_u8  = uv_mask_np_bool.astype(np.uint8)
    kernel      = np.ones((3, 3), np.uint8)
    dilated_uv  = cv2.dilate(uv_mask_u8, kernel, iterations=8)
    bleed_mask  = (dilated_uv > 0) & ~uv_mask_np_bool   # narrow band just outside islands

    n_bleed = int(bleed_mask.sum())
    if n_bleed > 0:
        _, bleed_idx = distance_transform_edt(~uv_mask_np_bool, return_indices=True)
        by, bx = bleed_idx[0], bleed_idx[1]
        for c in range(3):
            color_np[bleed_mask, c] = color_np[by[bleed_mask], bx[bleed_mask], c]
        # Keep colour in the bleed zone so bilinear sampling at seam edges gets the
        # correct hue, but set alpha=0 so the Blender material (alpha-clip mode) does
        # NOT render those pixels — this eliminates the silhouette "glow/halo" that
        # appeared at the mesh perimeter.  The colour is still available for the GPU's
        # bilinear filter to blend at the island boundary; only the alpha=0 prevents
        # those texels from being shaded independently.
        alpha_np[bleed_mask] = 0
        print(f"  Fill pass 2 (seam bleed): {n_bleed} texels colour-padded (alpha=0 for clean silhouette).")

    # Colour correction – compensates for SV3D's flat, washed-out lighting.
    # Applied only inside UV islands so the black background is untouched.
    # SV3D drains warmth and saturation; we boost S and slightly darken V so
    # skin looks like proper golden/warm tones instead of chalk-white.
    if saturation_boost != 1.0 or brightness_correction != 1.0:
        color_bgr  = color_np[:, :, ::-1].copy()
        color_hsv  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Boost saturation to compensate for SV3D's washed-out colours.
        if saturation_boost != 1.0:
            color_hsv[:, :, 1] = np.clip(color_hsv[:, :, 1] * saturation_boost, 0, 255)
        # Optional brightness correction (< 1 darkens, > 1 brightens).
        # Default 1.0 = no change. The old hardcoded 0.88 was making the back
        # (which SV3D already renders darker than the front) even darker.
        if brightness_correction != 1.0:
            color_hsv[:, :, 2] = np.clip(color_hsv[:, :, 2] * brightness_correction, 0, 255)
        corrected  = cv2.cvtColor(color_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[:, :, ::-1]
        color_np[uv_mask_np_bool] = corrected[uv_mask_np_bool]
        print(f"  Colour correction: saturation ×{saturation_boost:.2f}, brightness ×{brightness_correction:.2f} applied within UV islands.")

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
        # ── Hair/face seam detection — mean normal-Z + KNN smoothing ─────────
        # Purely geometric (no colour).  Key insight:
        #   Face/neck skin normals point forward (+Z) → high normal.Z per pixel.
        #   Hair geometry normals point upward / sideways / backward → low normal.Z.
        # Nose and eye sockets are also forward-facing → correctly stay as face.
        # After a cell-level Otsu threshold, iterative KNN smoothing (5 passes)
        # refines the boundary to pixel precision in 3D space.
        print("[DebugRegions] Hair seam detection starting (mean-nz + KNN smoothing)...")

        _REGION_NAMES  = ["foot","lower_leg","upper_leg","torso_back",
                          "torso_front","arms","head","hair"]
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

        # ── Gather pixels ─────────────────────────────────────────────────────
        _uv_y, _uv_x = np.where(uv_mask_np_bool)
        _uv_pixels   = color_np[_uv_y, _uv_x].astype(np.float32)
        _pos3d       = tex_pos_np[_uv_y, _uv_x]
        p_y = _pos3d[:, 1];  p_x = _pos3d[:, 0];  p_z = _pos3d[:, 2]

        # ── Mesh extents ──────────────────────────────────────────────────────
        _y_max = float(p_y.max());  _y_min = float(p_y.min())
        _y_rng = _y_max - _y_min
        _x_half = max(float(np.abs(p_x).max()), 1e-6)

        _T_head    = _y_max - 0.28 * _y_rng   # chin/ear level
        # Hairline = 20 % from the TOP OF THE HEAD (not body).
        # Head height = y_max - T_head.  20% of that from y_max puts the
        # hairline at the upper forehead, well above nose/eyes.
        # Old formula (0.12 * body_height) was landing below nose level.
        _T_hair    = _y_max - 0.20 * (_y_max - _T_head)   # hairline (20% from crown)
        _T_torso_b = _y_min + 0.42 * _y_rng
        _T_knee    = _y_min + 0.22 * _y_rng
        _T_foot    = _y_min + 0.07 * _y_rng
        _T_x_arm   = _x_half * 0.38

        print(f"  [Thresholds] Y: hair>{_T_hair:.3f}  head>{_T_head:.3f}  "
              f"torso>{_T_torso_b:.3f}  knee>{_T_knee:.3f}  foot>{_T_foot:.3f}  "
              f"|X|arm>{_T_x_arm:.3f}  seam_floor (computed below after T_hair set)")

        # ── Base labels ───────────────────────────────────────────────────────
        _labels = np.zeros(len(_uv_y), dtype=np.uint8)
        _labels[p_y > _T_foot]                        = 1   # lower_leg
        _labels[p_y > _T_knee]                        = 2   # upper_leg
        _labels[p_y > _T_torso_b]                     = 3   # torso_back
        _labels[(p_y > _T_torso_b) & (p_z > 0.01)]   = 4   # torso_front
        _labels[p_y > _T_head]                        = 6   # head (hair overrides below)
        _labels[np.abs(p_x) > _T_x_arm]              = 5   # arms

        # ── Seam detection: head zone (Y > T_head, not arms) ─────────────────
        # Signal: mean surface normal Z component per XZ cell.
        #   Face / neck skin faces the viewer → normal.Z ≈ +0.5 to +0.9  (HIGH).
        #   Hair geometry on crown / sides / back faces upward or sideways →
        #     normal.Z ≈ 0 to –0.5  (LOW).
        #   Unlike normal-variance, nose/eye-socket normals are still mostly
        #   forward-pointing → they stay high-nz (face), not misclassified as hair.
        #
        # Pipeline:
        #   Phase 1 — XZ cell grid, mean_nz per cell, Otsu threshold.
        #   Phase 2 — per-pixel initial label from each pixel's own normal.Z.
        #   Phase 3 — iterative KNN smoothing (5 passes, K=25) using a 4-D
        #             KD-tree (x,y,z,nz_weight) so the boundary converges to
        #             a geometrically precise, smooth seam.
        #   Phase 4 — pixel-level seam drawn directly in UV space (1 px wide).
        # Full head zone: T_head → top of head. Includes neck, braid over neck, etc.
        # Problem: neck pixels have very high nz (face strongly toward viewer) and
        # dominate the Otsu distribution, pushing the threshold so high that even the
        # forehead is mis-labelled as hair.
        # Fix: compute the Otsu threshold ONLY from the upper skull zone (above
        # T_head_seam), where the distribution is cleanly bimodal (face-skin vs
        # scalp/hair). Then apply that threshold to ALL head-zone pixels, so the
        # braid / hair hanging over the neck is also classified correctly.
        _T_head_seam = _T_head + 0.35 * (_T_hair - _T_head)
        print(f"  [Thresholds] seam_floor>{_T_head_seam:.3f}  "
              f"(Otsu computed from skull zone Y>{_T_head_seam:.3f}, "
              f"applied to full head zone Y>{_T_head:.3f} incl. neck/braid)")

        # Seam detection head zone: start high enough to exclude chest/shoulder
        # pixels that creep in when T_head_seam (0.290) is too low.
        # Use 60 % of the way from T_head to T_hair (≈ lower-cheek level).
        _T_seam_floor = _T_head + 0.60 * (_T_hair - _T_head)
        print(f"  [SeamFloor] seam_floor={_T_seam_floor:.3f}  "
              f"(T_head={_T_head:.3f} + 60% × (T_hair-T_head={_T_hair - _T_head:.3f}))")
        _head_mask = (p_y > _T_seam_floor) & (np.abs(p_x) <= _T_x_arm)
        _hi = np.where(_head_mask)[0]

        _seam_pxmap    = None   # will hold bool (texture_size, texture_size) seam pixels

        if len(_hi) >= 100:
            import time as _time
            import scipy.ndimage as _spndimage
            import scipy.spatial as _spspatial
            _t0 = _time.perf_counter()
            # local aliases — avoid 'from x import y' which makes y a local var
            # in the whole function and breaks earlier cKDTree references at line 826
            binary_dilation  = _spndimage.binary_dilation
            _cKDTree         = _spspatial.cKDTree

            _hy = p_y[_hi];  _hx = p_x[_hi];  _hz = p_z[_hi]

            # ── Diagnostic: what do normals look like at known 3-D locations? ──
            _tex_normals_np = tex_normals.cpu().numpy()
            _hn_x_all = _tex_normals_np[_uv_y[_hi], _uv_x[_hi], 0].astype(np.float32)
            _hn_y_all = _tex_normals_np[_uv_y[_hi], _uv_x[_hi], 1].astype(np.float32)
            _hn_z_all = _tex_normals_np[_uv_y[_hi], _uv_x[_hi], 2].astype(np.float32)
            # Top-50 frontmost pixels (highest p_z) → should be nose/forehead
            _f50 = np.argsort(_hz)[-50:]
            print(f"  [SeamDiag] Frontmost-50 (face?) avg:"
                  f"  p_z={float(_hz[_f50].mean()):.3f}"
                  f"  nx={float(_hn_x_all[_f50].mean()):.3f}"
                  f"  ny={float(_hn_y_all[_f50].mean()):.3f}"
                  f"  nz={float(_hn_z_all[_f50].mean()):.3f}")
            # Top-50 topmost pixels (highest p_y) → should be crown
            _t50 = np.argsort(_hy)[-50:]
            print(f"  [SeamDiag] Topmost-50   (crown?) avg:"
                  f"  p_y={float(_hy[_t50].mean()):.3f}"
                  f"  nx={float(_hn_x_all[_t50].mean()):.3f}"
                  f"  ny={float(_hn_y_all[_t50].mean()):.3f}"
                  f"  nz={float(_hn_z_all[_t50].mean()):.3f}")
            # Top-50 most-central pixels (lowest |p_x| + low p_y) → chin/neck
            _dist_chin = np.abs(_hx) + np.abs(_hy - float(_hy.min()))
            _c50 = np.argsort(_dist_chin)[:50]
            print(f"  [SeamDiag] Chin-area-50 (face?) avg:"
                  f"  p_y={float(_hy[_c50].mean()):.3f}"
                  f"  nx={float(_hn_x_all[_c50].mean()):.3f}"
                  f"  ny={float(_hn_y_all[_c50].mean()):.3f}"
                  f"  nz={float(_hn_z_all[_c50].mean()):.3f}")

            # ── Phase 4: coordinate-based face / hair classification ────────
            # No normals.  Pure world-space position:
            #   Face = forward-facing (p_z > 0) AND below hairline (p_y ≤ T_hair)
            #   Hair = backward/sideward (p_z ≤ 0) OR above hairline (p_y > T_hair)
            # This naturally produces:
            #   • Forehead seam at y ≈ T_hair (face meets crown hair)
            #   • Temple seam at z ≈ 0 (cheek meets side-hair)
            #   • No false seam lines across nose/eyes/lips

            _T_crown = _y_max - 0.04 * _y_rng
            _pts3d = np.column_stack([
                _hx.astype(np.float64),
                _hy.astype(np.float64),
                _hz.astype(np.float64),
            ])

            _is_hair = (_hz <= 0.0) | (_hy > _T_hair)
            _is_hair[_hy > _T_crown] = True   # explicit crown (top 4%)

            _n_h = int(_is_hair.sum())
            print(f"  [SeamWalk] coord-based  T_hair={_T_hair:.3f}  T_crown={_T_crown:.3f}"
                  f"  z_split=0.0")
            print(f"  [SeamWalk] {len(_hi) - _n_h:,} face px "
                  f"({100*(1-_is_hair.mean()):.0f}%), "
                  f"{_n_h:,} hair px ({100*_is_hair.mean():.0f}%)"
                  f"  ({_time.perf_counter()-_t0:.1f}s)")

            _labels[_hi[_is_hair]]  = 7
            _labels[_hi[~_is_hair]] = 6

            # ── Seam detection in 3D space (NOT UV space) ─────────────────────
            # Find face pixels (z > 0) whose nearest hair pixel (z ≤ 0 or
            # y ≥ T_hair) is within a 3-D bandwidth.  The bandwidth is set
            # relative to total character height so it scales with mesh size.
            # At the forehead hairline the face→hair step is ~0.02 in y;
            # at the temple it's ~0.01–0.02 in z.  We need bw ≥ 0.025.
            _seam_pxmap = np.zeros((texture_size, texture_size), dtype=bool)
            _face_local = np.where(~_is_hair)[0]   # indices into _hi → face pixels
            _hair_local = np.where( _is_hair)[0]   # indices into _hi → hair pixels
            if len(_face_local) > 0 and len(_hair_local) > 0:
                # 1.5 % of the head-zone vertical span (tight — only catch
                # pixels truly adjacent to the hair boundary in 3D)
                _seam_bw = (_y_max - _T_seam_floor) * 0.015
                _hair_tree = _cKDTree(_pts3d[_hair_local])
                _face_pts3d = _pts3d[_face_local]
                _dist, _nn_idx = _hair_tree.query(_face_pts3d, k=1,
                                                  distance_upper_bound=_seam_bw)
                # Clamp sentinel (scipy returns len(data) when no neighbor found)
                _safe_idx = np.minimum(_nn_idx, len(_hair_local) - 1)
                # Only flag pixels where a hair neighbor was actually found
                _is_hairline = _dist < _seam_bw

                # ── Exclude central face interior (nose, eyes, mouth) ──────────
                # Three conditions (ALL must be true to suppress a seam candidate):
                #   1. |p_x| < 0.25 × head_x_range       → central 50% of head width
                #   2. p_y < floor + 0.80 × (T_hair-floor) → below 80% of face-zone height
                #      (leaves the top 20% — the forehead near hairline — as seam candidates)
                #   3. p_z > hz_min + 0.70 × hz_range     → front 30% of head depth
                #      (nose/eyes are the most forward pixels; neck/chest are NOT in front 30%)
                _hx_rng    = float(_hx.max() - _hx.min())
                _hz_hd_min = float(_hz.min())
                _hz_hd_rng = max(float(_hz.max()) - _hz_hd_min, 1e-6)
                _head_y_span = _T_hair - _T_seam_floor

                _excl_x_half   = 0.18 * _hx_rng                           # ±18% width
                _excl_y_ceil   = _T_seam_floor + 0.85 * _head_y_span      # 85% up from floor
                _excl_z_thresh = _hz_hd_min + 0.70 * _hz_hd_rng           # front 30% in Z

                _face_central = (
                    (np.abs(_face_pts3d[:, 0]) < _excl_x_half) &
                    (_face_pts3d[:, 1] < _excl_y_ceil) &
                    (_face_pts3d[:, 2] > _excl_z_thresh)
                )
                _is_hairline &= ~_face_central
                print(f"  [SeamWalk] Central-face excl: {int(_face_central.sum()):,} px  "
                      f"x<±{_excl_x_half:.3f}  y<{_excl_y_ceil:.3f}  z>{_excl_z_thresh:.3f}")

                _seam_hi_idx = _hi[_face_local[_is_hairline]]
                _seam_pxmap[_uv_y[_seam_hi_idx], _uv_x[_seam_hi_idx]] = True
                print(f"  [SeamWalk] 3-D seam: {int(_seam_pxmap.sum()):,} px"
                      f"  bw={_seam_bw:.4f}"
                      f"  ({_time.perf_counter()-_t0:.1f}s)")
        else:
            print("  [SeamWalk] Too few head-zone texels — keeping Y-threshold only")

        # ── Re-apply arms (final override) ────────────────────────────────────
        _labels[np.abs(p_x) > _T_x_arm] = 5

        # ── Paint region colours ──────────────────────────────────────────────
        debug_canvas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        print("[DebugRegions] Region summary:")
        for ri, (rname, rcolor) in enumerate(zip(_REGION_NAMES, _REGION_COLORS)):
            _mask_r = (_labels == ri)
            debug_canvas[_uv_y[_mask_r], _uv_x[_mask_r]] = rcolor
            _avg = _uv_pixels[_mask_r].mean(axis=0) if _mask_r.any() else np.zeros(3)
            print(f"  [{ri}] {rname:12s}: {rcolor}  "
                  f"avg_RGB=({_avg[0]:.0f},{_avg[1]:.0f},{_avg[2]:.0f})  "
                  f"px={int(_mask_r.sum()):,}")

        # ═══════════════════════════════════════════════════════════════════
        # HAIRLINE / HAIR-REGION DETECTION — research notes
        # ═══════════════════════════════════════════════════════════════════
        #
        # GOAL
        #   Detect the hair region on the Trellis mesh so we can fill it with
        #   hair colour and the rest of the head with skin colour in the final
        #   texture.  We do NOT need a perfect 1-px hairline — we need a
        #   reliable hair-region MASK.  The boundary of that mask is the hairline.
        #
        # COORDINATE SYSTEM  (Y-up world)
        #   +X = character LEFT,  +Y = UP,  +Z = toward viewer (front)
        #   T_seam_floor ≈ 0.372  — bottom of head zone (nape of neck)
        #   T_hair       ≈ 0.467  — Y level of the front hairline
        #   y_max        ≈ 0.525  — top of head zone
        #
        # Z-VALUE DISTRIBUTION (this character)
        #   75 % of safe head pixels have Z < 0 (sides/back, clipped to black).
        #   Only the front-facing cap + face have Z > 0.  Z_max ≈ 0.070.
        #   Hair cap and forehead Z values are very close together (Δ ≈ 0.01–0.02).
        #
        # WHAT WAS TRIED AND WHY IT FAILED
        #
        #   1. Per-column largest-Z-jump scan
        #      Fixed T_hair window missed side hairline (hairline drops at temples).
        #      Switching to full-column scan was noisy.  ABANDONED.
        #
        #   2. Binary Z-threshold + grid boundary  (grid 80×80 world-space cells)
        #      Used 0.5 * Z_pixel_max as threshold → only 1 cyan dot because
        #      cell AVERAGES are always lower than pixel max.  Switched to p60 of
        #      cell averages → fired all over the hair cap, not just the boundary,
        #      because Canny detects ALL Z gradients incl. internal hair ridges.
        #      ABANDONED.
        #
        #   3. UV-space Canny on Z-grayscale — fingerprint noise
        #      Canny in UV texture space fires at UV island seam boundaries where
        #      two adjacent UV patches have slightly different Z values.
        #      Produces a dense fingerprint pattern all over the hair cap.
        #      BUT: the front hairline IS detectable (cyan blur=5 σ2, Canny 5/15).
        #      KEPT AS CYAN BASELINE — do not change these params.
        #
        #   4. Grid-space Canny (80×80 Z-grid scaled 8×, then Canny)
        #      Avoided UV seam artifacts but detected internal hair ridges/braids
        #      instead of the outer hairline.  ABANDONED.
        #
        #   5. Connected-component filter on cyan edges
        #      2979 components, ALL < 100 px.  UV mapping fragments the hairline
        #      across islands so even the real hairline has tiny components.
        #      CC filtering in UV space fundamentally doesn't work.  ABANDONED.
        #
        #   6. World-Y range filter on cyan edges
        #      Kept edges in [T_seam_floor, T_hair+0.05].  Most UV-seam noise
        #      is in the same Y band as the hairline → didn't filter enough.
        #      ABANDONED.
        #
        #   7. Local Z-contrast filter on cyan edges
        #      9×9 window max-min.  Contrast at ALL edge pixels: p50=20, max=94.
        #      Hair cap and face have nearly identical Z → contrast can't separate
        #      hairline from UV-seam noise.  ABANDONED.
        #
        #   8. Binary-threshold blob → fill_holes → outer boundary
        #      fill_holes leaked through the face exclusion-zone hole and flooded
        #      the entire face area.  Large dilation also bled onto ears/neck.
        #      ABANDONED.
        #
        #   9. World-Y threshold (Y > T_hair) as direct hair mask
        #      Produced straight horizontal lines in UV space — the UV layout maps
        #      world-Y to horizontal strips so the boundary was a flat stripe.
        #      Not useful as a hairline.  ABANDONED.
        #
        #  10. Normal-Y Canny (geometry edges, no UV seam artifacts)  ← CURRENT
        #      tex_normals are purely geometric → no fingerprint noise.
        #      Fires on hair geometry edges (ridges, cap rim, braids).
        #      Covers areas cyan misses (sides, back of cap).
        #      Filter v2: Y >= T_seam_floor (full head circumference) AND
        #                 NOT (Y < T_hair AND Z > 0)  → removes front-face below
        #                 hairline (forehead/neck) while keeping the lower back of
        #                 the hair cap (same Y band but Z <= 0, facing away).
        #      Still fires on elf ears (Z <= 0, same Y band) — filter |X| later.
        #
        # CURRENT STATE
        #   Cyan  = Z-based Canny (UV seam fingerprint, best params locked in)
        #   Yellow = Normal-Y Canny filtered to Y >= T_hair (geometry edges)
        #   Together they aim to cover the full hair cap region.
        #   Body-zone hanging hair (below T_seam_floor) → Step 4 (separate pass).
        #
        # NEXT STEPS
        #   - Combine cyan + yellow into a single hair-region mask (dilate+close)
        #   - Filter ear false-positives by |world-X| < threshold
        #   - Handle body-zone hanging hair via Step 4 elevation detection
        #   - Once mask is solid: use it to assign hair colour vs skin colour
        #     in the final texture fill pass
        #
        # ═══════════════════════════════════════════════════════════════════
        # HAIRLINE DETECTION — Step 1: safe mask  /  Step 2: Z-surface viz
        # ═══════════════════════════════════════════════════════════════════
        if len(_hi) >= 100:
            # ── Step 1: safe mask = head zone MINUS exclusion zone ────────
            _hl_hx_rng = float(_hx.max() - _hx.min())
            _hl_hz_min = float(_hz.min())
            _hl_hz_rng = max(float(_hz.max()) - _hl_hz_min, 1e-6)
            _hl_y_span = _T_hair - _T_seam_floor

            _hl_excl_x = 0.18 * _hl_hx_rng
            _hl_excl_y = _T_seam_floor + 0.85 * _hl_y_span
            _hl_excl_z = _hl_hz_min + 0.70 * _hl_hz_rng

            _hl_excl = (
                (np.abs(_hx) < _hl_excl_x) &
                (_hy < _hl_excl_y) &
                (_hz > _hl_excl_z)
            )
            _hl_safe = ~_hl_excl
            print(f"  [Hairline-S1] head={len(_hi):,}  "
                  f"excl={int(_hl_excl.sum()):,}  "
                  f"safe={int(_hl_safe.sum()):,}")

            # ── Step 2: paint safe pixels as grayscale by Z value ─────────
            # Dark = low Z (back/sides), Bright = high Z (protrudes forward)
            # The hairline boundary should be visible as a brightness edge.
            _hl_safe_hi = _hi[_hl_safe]
            _hl_excl_hi = _hi[_hl_excl]
            _hz_s       = _hz[_hl_safe]
            # Clip normalization to positive-Z range only.
            # Back-of-head pixels have negative Z and are included in the safe mask,
            # but normalizing from the full min would compress the face/hair boundary
            # (which lives in positive Z) into a tiny bright sliver — barely visible.
            # By clamping _z_lo to 0.0, the grayscale range spans only the forward
            # hemisphere: back-of-head → black, face skin → bright, hairline edge →
            # visible drop.  Pixels with hz < 0 get clipped to 0 (black) by np.clip.
            _z_lo = 0.0
            _z_hi = float(_hz_s.max())
            _z_rng = max(_z_hi - _z_lo, 1e-6)
            _gray = np.clip((_hz_s - _z_lo) / _z_rng * 255,
                            0, 255).astype(np.uint8)
            # Overdraw safe area with grayscale
            debug_canvas[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = \
                np.column_stack([_gray, _gray, _gray])
            # Excluded zone → black (so its shape stays visible)
            debug_canvas[_uv_y[_hl_excl_hi], _uv_x[_hl_excl_hi]] = (0, 0, 0)

            # ── Step 2b: build 2D Z-grid for gradient in Step 3 ──────────
            _hl_Nx = 80;  _hl_Ny = 80
            _hl_xb = np.linspace(float(_hx.min()), float(_hx.max()), _hl_Nx + 1)
            _hl_yb = np.linspace(float(_T_seam_floor), float(_y_max),  _hl_Ny + 1)
            _hl_xi = np.clip(np.searchsorted(_hl_xb[1:], _hx[_hl_safe]), 0, _hl_Nx - 1)
            _hl_yi = np.clip(np.searchsorted(_hl_yb[1:], _hy[_hl_safe]), 0, _hl_Ny - 1)
            _hl_zs = np.zeros((_hl_Ny, _hl_Nx), dtype=np.float64)
            _hl_zn = np.zeros((_hl_Ny, _hl_Nx), dtype=np.float64)
            np.add.at(_hl_zs, (_hl_yi, _hl_xi), _hz[_hl_safe])
            np.add.at(_hl_zn, (_hl_yi, _hl_xi), 1.0)
            _hl_zgrid = np.where(_hl_zn > 0, _hl_zs / _hl_zn, np.nan)
            _hl_filled = int((~np.isnan(_hl_zgrid)).sum())
            _z_actual_lo = float(_hz_s.min())
            print(f"  [Hairline-S2] Z-grid {_hl_Ny}×{_hl_Nx}: "
                  f"{_hl_filled}/{_hl_Ny*_hl_Nx} cells filled  "
                  f"Z_actual=[{_z_actual_lo:.3f}, {_z_hi:.3f}]  "
                  f"norm_floor=0.0 (positive-Z only)")

            # ── Step 3: binary Z-threshold + 2D boundary trace ───────────────
            # The grayscale already shows the answer visually: hair cap = BRIGHT
            # (high Z, protrudes forward), face/back/sides = DARK (low Z).
            # The seam is simply the boundary between bright and dark regions.
            #
            # Algorithm:
            #   1. Threshold the Z-grid: cell is "hair" if Z > 50% of Z_max.
            #   2. A hair cell is a BOUNDARY cell if any of its 4 grid neighbours
            #      is non-hair (low Z) or empty (no pixels in that cell).
            #   3. Project boundary grid cells back to UV pixels → paint cyan.
            #
            # This naturally handles the curved hairline (forehead, temples,
            # sides, back) in a single 2D pass — no per-column scanning needed.
            # The 50% threshold corresponds to mid-grey in the grayscale image.

            _hl_hair_row   = int(np.searchsorted(_hl_yb[1:], _T_hair))

            # Threshold from CELL averages (not pixel max — cell averages are
            # lower because they blend hair + non-hair pixels within each cell).
            # Use the 60th percentile of filled cell Z values: top 40% = hair.
            _hl_filled_z   = _hl_zgrid[_hl_zn > 0]   # 1-D array of cell means
            _hl_z_p25  = float(np.percentile(_hl_filled_z, 25))
            _hl_z_p50  = float(np.percentile(_hl_filled_z, 50))
            _hl_z_p60  = float(np.percentile(_hl_filled_z, 60))
            _hl_z_p75  = float(np.percentile(_hl_filled_z, 75))
            _hl_hair_thresh = _hl_z_p60   # tune: higher → less hair classified
            print(f"  [Hairline-S3] cell-Z dist: "
                  f"p25={_hl_z_p25:.3f}  p50={_hl_z_p50:.3f}  "
                  f"p60={_hl_z_p60:.3f}  p75={_hl_z_p75:.3f}  "
                  f"Z_px_max={_z_hi:.3f}  thresh={_hl_hair_thresh:.3f}")

            # ── Step 3: Canny on UV-space Z-grayscale ────────────────────────
            # Build a clean single-channel grayscale from the safe pixels
            # (bright = high Z = hair cap, dark = face/back).  Run Canny on it.
            # Works at full texture resolution, handles asymmetric hair, detects
            # edges in both X and Y simultaneously.
            #
            # UV-seam fingerprint noise is a known side-effect; we accept it for
            # now and will filter it afterwards.  This version (cyan sigma=5) was
            # the best at tracing the actual front hairline.

            import cv2 as _cv2_hl
            _hl_tex_size = debug_canvas.shape[0]
            _hl_gray_img = np.zeros((_hl_tex_size, _hl_tex_size), dtype=np.uint8)
            _hl_gray_img[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = _gray

            # Interior mask: erode safe-pixel region inward so Canny cannot fire
            # at the outer mesh boundary (neck, elf ears, etc.)
            _hl_safe_uv = np.zeros((_hl_tex_size, _hl_tex_size), dtype=bool)
            _hl_safe_uv[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = True
            _hl_interior = _spndimage.binary_erosion(_hl_safe_uv, iterations=12)

            # ── LAST KNOWN GOOD (cyan baseline) ──────────────────────────────
            # blur=(5,5) σ2  |  Canny 5/15  |  interior erosion=12px
            # Catches front hairline well. UV-seam fingerprint noise present
            # but hairline signal is clearly there. Do NOT change these.
            # ─────────────────────────────────────────────────────────────────

            # ── Cyan: blur=5 σ2 — catches hairline detail ────────────────────
            _hl_blurA = _cv2_hl.GaussianBlur(_hl_gray_img, (5, 5), 2.0)
            _hl_edgesA = (_cv2_hl.Canny(_hl_blurA, 5, 15) > 0) & _hl_interior

            debug_canvas[_hl_edgesA] = (0, 255, 255)
            debug_canvas[_uv_y[_hl_excl_hi], _uv_x[_hl_excl_hi]] = (0, 0, 0)

            print(f"  [Hairline-S3] Canny cyan(blur=5,5/15): "
                  f"edge_px={int(_hl_edgesA.sum()):,}")
                  
                  
            # =========================================================================
            # NEW INSERTION: Filter the Cyan Border & Create Brown Duplicate
            # =========================================================================
            print("\n" + "="*50)
            print("!!! DEBUG: FILTERING CYAN AREA VIA EXACT VARIABLE MATCH !!!")
            print("="*50)

            try:
                # Pull torch from the top-level global module scope to protect line 211
                _torch = globals().get('torch')
                
                # Pull your spatial mesh context arrays safely from the running local frame
                pos_tensor = locals().get('tex_pos', locals().get('world_pos', None))
                canvas_tensor = locals().get('img_out', locals().get('region_img', None))

                if _torch is not None and pos_tensor is not None:
                    # Safely cast your NumPy array (_hl_edgesA) directly onto the GPU workspace
                    active_cyan_pixels = _torch.from_numpy(_hl_edgesA).to(device=pos_tensor.device) > 0
                    
                    if active_cyan_pixels.sum() > 0:
                        cyan_positions = pos_tensor[active_cyan_pixels] # Shape: [N, 3]
                        
                        # 2. Extract boundaries based on your spatial limits logic
                        # +Z is depth towards the face (Bangs/Forehead transition)
                        # -Y is vertical descent (The bottom fringe tips of the strands)
                        z_coords = cyan_positions[:, 2]
                        y_coords = cyan_positions[:, 1]
                        
                        max_z_cyan = _torch.max(z_coords)
                        min_y_cyan = _torch.min(y_coords)
                        
                        # 3. Apply a tight 3cm filtering tolerance strip for the extreme perimeter
                        filter_tolerance = 0.03
                        is_front_edge = _torch.abs(pos_tensor[..., 2] - max_z_cyan) < filter_tolerance
                        is_bottom_edge = _torch.abs(pos_tensor[..., 1] - min_y_cyan) < filter_tolerance
                        
                        # 4. Combine the masks safely as matching GPU Tensors
                        brown_border_mask = active_cyan_pixels & (is_front_edge | is_bottom_edge)
                        
                        # 5. Overwrite the visual preview canvas by finding it dynamically
                        if canvas_tensor is None:
                            # Let's hunt for any local 3D or 4D tensor that looks like an image array
                            for var_name, var_val in locals().items():
                                if hasattr(var_val, 'ndim') and var_val.ndim >= 3:
                                    # If it matches the spatial width/height of our mask, we found it!
                                    if var_val.shape[0] == brown_border_mask.shape[0]:
                                        canvas_tensor = var_val
                                        print(f"    [Auto-Hook] Found and attached to active canvas: '{var_name}'")
                                        break

                        if canvas_tensor is not None:
                            # Dynamically match data format (float32 vs uint8)
                            if canvas_tensor.dtype == _torch.float32:
                                brown_color = _torch.tensor([0.25, 0.13, 0.05], device=canvas_tensor.device, dtype=_torch.float32)
                            else:
                                brown_color = _torch.tensor([64, 33, 13], device=canvas_tensor.device, dtype=_torch.uint8)
                            
                            canvas_tensor[brown_border_mask] = brown_color
                            print(f"    [Success] Successfully isolated and painted {int(brown_border_mask.sum())} border pixels as Brown.")
                        else:
                            print("    [Filter Warning] Border calculated, but no active visual image canvas tensor could be detected in this frame.")
                    else:
                        print("    [Filter Error] Target '_hl_edgesA' mask contained zero active pixels.")
                else:
                    print("    [Filter Error] Environment check failed. Missing global 'torch' or local 'tex_pos'.")
                    
            except Exception as e:
                print(f"    [Filter Error]: {str(e)}")
                
            print("="*50 + "\n")
            # =========================================================================      

            # ── Green: local Z-contrast filter on cyan edges ─────────────────
            # At the real hairline, one side is bright (hair cap, high Z) and the
            # other is darker (face, lower Z) → large local gray contrast.
            # At a UV-seam edge inside the hair cap, both sides have similar Z
            # (both within the cap) → small local contrast.
            # Strategy: dilate/erode the gray image in a small window to get
            # local max and min; keep only edge pixels where (max-min) > threshold.
            #
            # Non-safe pixels have gray=0 which would corrupt erode/dilate near
            # the mask boundary → fix by setting non-safe to 255 for the erode
            # input (so it can't pull the local min down) and to 0 for the dilate
            # input (so it can't pull the local max up).
            _hl_kc = np.ones((9, 9), np.uint8)   # 9×9 neighbourhood

            _hl_g_for_max = _hl_gray_img.copy()
            _hl_g_for_min = _hl_gray_img.copy()
            _hl_g_for_max[~_hl_safe_uv] = 0    # non-safe won't inflate local max
            _hl_g_for_min[~_hl_safe_uv] = 255  # non-safe won't deflate local min

            _hl_lmax = _cv2_hl.dilate(_hl_g_for_max, _hl_kc).astype(np.int16)
            _hl_lmin = _cv2_hl.erode(_hl_g_for_min,  _hl_kc).astype(np.int16)
            _hl_contrast = (_hl_lmax - _hl_lmin).clip(0, 255).astype(np.uint8)

            # ── Yellow: Canny on normal-Y grayscale (geometry edges, no UV noise) ─
            # Cyan fires on UV seam boundaries (fingerprint pattern, mostly top cap).
            # Normal-Y Canny fires on actual geometry edges: hair ridges, cap rim,
            # braids — wherever the surface changes orientation.  No UV seam
            # artifacts since normals are purely geometric.
            # Together cyan + yellow should cover the full hair surface.
            _hl_norm_np = tex_normals.cpu().numpy()   # (H, W, 3)
            _hl_ny_img  = np.zeros((_hl_tex_size, _hl_tex_size), dtype=np.float32)
            _hl_ny_img[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = \
                _hl_norm_np[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi], 1]
            # ny in [-1,+1] → [0,255]
            _hl_ny_u8   = np.clip((_hl_ny_img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

            _hl_blurY   = _cv2_hl.GaussianBlur(_hl_ny_u8, (5, 5), 2.0)
            _hl_edgesY  = (_cv2_hl.Canny(_hl_blurY, 5, 15) > 0) & _hl_interior

            # Yellow filter: back/sides of hair cap ONLY.
            # Z <= 0  → back/sides, no forehead.
            # Y >= T_seam_floor → within head zone (not body).
            # |X| < 80% of head half-width → no ears.

            _hl_wy_img2 = np.zeros((_hl_tex_size, _hl_tex_size), dtype=np.float32)
            _hl_wy_img2[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = _hy[_hl_safe]
            _hl_wz_img2 = np.zeros((_hl_tex_size, _hl_tex_size), dtype=np.float32)
            _hl_wz_img2[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = _hz[_hl_safe]
            _hl_wx_img2 = np.zeros((_hl_tex_size, _hl_tex_size), dtype=np.float32)
            _hl_wx_img2[_uv_y[_hl_safe_hi], _uv_x[_hl_safe_hi]] = _hx[_hl_safe]

            _hl_x_head_half = float(np.abs(_hx).max())
            _hl_ear_thresh  = _hl_x_head_half * 0.80

            _hl_y_mask = (
                (_hl_wy_img2 >= float(_T_seam_floor)) &  # within head zone
                (_hl_wz_img2 <= 0.0) &                   # back/sides only — no forehead
                (np.abs(_hl_wx_img2) < _hl_ear_thresh)   # no ears
            )
            _hl_edgesY  = _hl_edgesY & _hl_y_mask

            # Green REMOVED — front/side hanging hair detection moved to Step 4
            # (per-row Z-elevation, same approach as body-zone hanging hair).

            debug_canvas[_hl_edgesA] = (0, 255, 255)                   # cyan baseline
            debug_canvas[_hl_edgesY & ~_hl_edgesA] = (255, 255, 0)     # yellow = back/sides
            debug_canvas[_uv_y[_hl_excl_hi], _uv_x[_hl_excl_hi]] = (0, 0, 0)

            print(f"  [Hairline-S3] Normal-Y Canny yellow"
                  f"(Y>={float(_T_seam_floor):.3f} & Z<=0 & |X|<{_hl_ear_thresh:.3f}): "
                  f"edge_px={int(_hl_edgesY.sum()):,}  "
                  f"yellow_only={int(((_hl_edgesY & ~_hl_edgesA)).sum()):,}")

            # ── Step 4: Hanging-hair boundary detection (body zone) ───────────
            # Hair hanging away from the body (braids, ponytails, long strands)
            # floats in FRONT of the body surface → high Z relative to other
            # surfaces at the same (X, Y) world position.
            #
            # Build a Z-grid over the body zone (below _T_seam_floor).
            # Per Y-row baseline = median Z of all filled cells in that row
            # (body surface dominates; hair is the outlier).
            # Elevated cells: Z > row_median + 1.5 × row_std → floating hair.
            #
            # Paint elevated body pixels YELLOW to visualise the hanging hair
            # region.  For each X-column, the bottom-most elevated row is the
            # hanging-hair bottom anchor → painted CYAN (same colour as the
            # front hairline anchors so all anchors are consistent).

            _hh_Nx = 80;  _hh_Ny = 100
            _hh_y_lo = float(_y_min)
            _hh_y_hi = float(_T_seam_floor)

            # All pixels in the body zone (uses full p_x/p_y/p_z)
            _hh_body_mask = (p_y >= _hh_y_lo) & (p_y < _hh_y_hi)
            _hh_bi        = np.where(_hh_body_mask)[0]

            if len(_hh_bi) >= 100:
                _hh_by = p_y[_hh_bi];  _hh_bx = p_x[_hh_bi]
                _hh_bz = p_z[_hh_bi]

                _hh_xb = np.linspace(float(p_x.min()), float(p_x.max()),
                                     _hh_Nx + 1)
                _hh_yb = np.linspace(_hh_y_lo, _hh_y_hi, _hh_Ny + 1)

                _hh_bxi = np.clip(
                    np.searchsorted(_hh_xb[1:], _hh_bx), 0, _hh_Nx - 1)
                _hh_byi = np.clip(
                    np.searchsorted(_hh_yb[1:], _hh_by), 0, _hh_Ny - 1)

                # Mean Z per grid cell (vectorised)
                _hh_zs = np.zeros((_hh_Ny, _hh_Nx), dtype=np.float64)
                _hh_zn = np.zeros((_hh_Ny, _hh_Nx), dtype=np.float64)
                np.add.at(_hh_zs, (_hh_byi, _hh_bxi), _hh_bz)
                np.add.at(_hh_zn, (_hh_byi, _hh_bxi), 1.0)
                _hh_zgrid = np.where(_hh_zn > 0, _hh_zs / _hh_zn, np.nan)

                # Per-row baseline (median) and spread (std) of filled cells
                _hh_row_med = np.full(_hh_Ny, np.nan)
                _hh_row_std = np.full(_hh_Ny, np.nan)
                for _ry in range(_hh_Ny):
                    _rv = _hh_zgrid[_ry, _hh_zn[_ry, :] > 0]
                    if len(_rv) >= 3:
                        _hh_row_med[_ry] = float(np.median(_rv))
                        _hh_row_std[_ry] = float(np.std(_rv))

                # Elevation threshold broadcast across grid
                _hh_med_g = _hh_row_med[:, np.newaxis]   # (Ny, 1)
                _hh_std_g = _hh_row_std[:, np.newaxis]
                _hh_elevated = (
                    (~np.isnan(_hh_med_g)) &
                    (_hh_zn > 0) &
                    (_hh_zgrid > _hh_med_g + 1.5 * _hh_std_g)
                )

                _hh_n_elevated = int(_hh_elevated.sum())
                print(f"  [HangHair-S4] body_px={len(_hh_bi):,}  "
                      f"elevated_cells={_hh_n_elevated}")

                # Paint elevated body pixels yellow so the region is visible
                _hh_elev_px = _hh_elevated[_hh_byi, _hh_bxi]
                _hh_elev_idx = _hh_bi[_hh_elev_px]
                debug_canvas[_uv_y[_hh_elev_idx],
                             _uv_x[_hh_elev_idx]] = (255, 255, 0)   # yellow

                # Per X-column: bottom-most elevated row = hanging-hair anchor
                # Note: row 0 = _hh_y_lo (feet), row Ny-1 = _T_seam_floor (neck).
                # Hair hangs downward, so the bottom anchor is the LOWEST row.
                _hh_anchor_row  = np.full(_hh_Nx, -1, dtype=int)
                for _cx in range(_hh_Nx):
                    _elev_rows = np.where(_hh_elevated[:, _cx])[0]
                    if len(_elev_rows) > 0:
                        _hh_anchor_row[_cx] = int(_elev_rows[0])  # lowest row first

                # Consensus filter: anchors clustered too far apart are noise
                _hh_valid_anchors = _hh_anchor_row >= 0
                _hh_n_anchors = int(_hh_valid_anchors.sum())
                if _hh_n_anchors >= 3:
                    _hh_anch_med = int(np.median(
                        _hh_anchor_row[_hh_valid_anchors]))
                    # Allow wider tolerance (±20 rows) since dangling hair can
                    # end at very different heights across X
                    _hh_valid_anchors &= (
                        np.abs(_hh_anchor_row - _hh_anch_med) <= 20)

                # Paint bottom-anchor pixels cyan
                _hh_n_cyan_anchors = 0
                for _cx in range(_hh_Nx):
                    if not _hh_valid_anchors[_cx]:
                        continue
                    _ar = _hh_anchor_row[_cx]
                    # Find body pixels that land in this cell and are elevated
                    _cell_px = (_hh_byi == _ar) & (_hh_bxi == _cx) & _hh_elev_px
                    _cell_idx = _hh_bi[_cell_px]
                    if len(_cell_idx) > 0:
                        debug_canvas[_uv_y[_cell_idx],
                                     _uv_x[_cell_idx]] = (0, 255, 255)
                        _hh_n_cyan_anchors += len(_cell_idx)

                print(f"  [HangHair-S4] anchors: raw={_hh_n_anchors}  "
                      f"after_filter={int(_hh_valid_anchors.sum())}  "
                      f"cyan_anchor_px={_hh_n_cyan_anchors}")

            # ── Step 4b: Dual-Z-layer detection — HEAD zone ──────────────────
            # Build a 2-D grid (X × Y) of Z_max per cell.  Cells with
            # Z_max > hair_floor are "hair".  Find hair cells whose inward
            # horizontal neighbour (toward X=0) is non-hair — that is the
            # face-side inner edge of the hanging hair.
            # Restrict to front-facing pixels (Z > 0).
            _hf_Nx = 160;  _hf_Ny = 160
            _hf_y_lo = float(_T_seam_floor)   # head-mesh bottom
            _hf_y_hi = float(_T_hair)         # up to scalp hairline
            _hf_x_half = float(_hl_ear_thresh) # stay inside ears

            _hf_face_mask = (
                (p_y >= _hf_y_lo) & (p_y < _hf_y_hi) &
                (np.abs(p_x) < _hf_x_half) &
                (p_z > 0.0)                    # front-facing only
            )
            _hf_fi = np.where(_hf_face_mask)[0]
            print(f"  [HangHair-S4b] head_zone px={len(_hf_fi):,}  "
                  f"Y:[{_hf_y_lo:.3f},{_hf_y_hi:.3f}]  |X|<{_hf_x_half:.3f}  Z>0")

            if len(_hf_fi) >= 100:
                _hf_fy = p_y[_hf_fi]
                _hf_fx = p_x[_hf_fi]
                _hf_fz = p_z[_hf_fi]

                _hf_xb = np.linspace(-_hf_x_half, _hf_x_half, _hf_Nx + 1)
                _hf_yb = np.linspace(_hf_y_lo, _hf_y_hi, _hf_Ny + 1)

                _hf_fxi = np.clip(
                    np.searchsorted(_hf_xb[1:], _hf_fx), 0, _hf_Nx - 1)
                _hf_fyi = np.clip(
                    np.searchsorted(_hf_yb[1:], _hf_fy), 0, _hf_Ny - 1)

                # Per-cell Z min and max
                _INF = float('inf')
                _hf_zmin = np.full((_hf_Ny, _hf_Nx),  _INF, dtype=np.float64)
                _hf_zmax = np.full((_hf_Ny, _hf_Nx), -_INF, dtype=np.float64)
                np.minimum.at(_hf_zmin, (_hf_fyi, _hf_fxi), _hf_fz)
                np.maximum.at(_hf_zmax, (_hf_fyi, _hf_fxi), _hf_fz)

                _hf_zrange = np.where(
                    _hf_zmax > _hf_zmin, _hf_zmax - _hf_zmin, 0.0)

                # Classify each cell as hair, face-skin, or empty.
                # Adaptive skin ceiling from the bottom rows which are pure face-skin
                # (logs show zrange_max≈0.004-0.005 there; ×2.5 gives safe ceiling).
                _hf_n_bot = max(4, _hf_Ny // 40)
                _hf_bot_valid = _hf_zmax[:_hf_n_bot][_hf_zmax[:_hf_n_bot] < float('inf')]
                _hf_skin_ceil = min(float(_hf_bot_valid.max()) * 2.5, 0.018) \
                                if len(_hf_bot_valid) else 0.012
                _hf_hair_floor = _hf_skin_ceil * 2.5

                _hf_has_px  = _hf_zmax > -float('inf')
                _hf_is_hair = _hf_has_px & (_hf_zmax > _hf_hair_floor)
                _hf_not_hair = ~_hf_is_hair   # empty cells OR face cells (non-hair)

                _cx_arr = np.arange(_hf_Nx, dtype=int)[np.newaxis, :]  # (1, Nx)
                _cx_mid = _hf_Nx // 2

                # Inner-edge: hair cell whose inward X neighbour (toward X=0)
                # is non-hair.  Finds the face-side gap of the hair cap.
                _hf_right_nh = np.zeros_like(_hf_not_hair)
                _hf_right_nh[:, :-1] = _hf_not_hair[:, 1:]
                _hf_left_nh  = np.zeros_like(_hf_not_hair)
                _hf_left_nh[:, 1:]  = _hf_not_hair[:, :-1]

                _hf_boundary = (
                    (_hf_is_hair & (_cx_arr <  _cx_mid) & _hf_right_nh) |
                    (_hf_is_hair & (_cx_arr >= _cx_mid) & _hf_left_nh)
                )

                # Filter pixels to actual hair layer only (Z > hair_floor).
                # Boundary cells can contain face-mesh pixels (Z < skin_ceil)
                # whose UV coords point to the face UV region — excluding them
                # prevents green dots appearing on eye/nose in the viewport.
                _hf_bound_px = _hf_boundary[_hf_fyi, _hf_fxi] & (_hf_fz > _hf_hair_floor)
                _hf_elev_idx = _hf_fi[_hf_bound_px]

                _hf_hair_col_min = [int(np.where(_hf_is_hair[r])[0].min()) if _hf_is_hair[r].any() else -1
                                    for r in range(_hf_Ny)]
                _hf_hair_col_max = [int(np.where(_hf_is_hair[r])[0].max()) if _hf_is_hair[r].any() else -1
                                    for r in range(_hf_Ny)]

                print(f"  [HangHair-S4b] hair_floor={_hf_hair_floor:.4f}  "
                      f"hair_cells={int(_hf_is_hair.sum())}  "
                      f"boundary_cells={int(_hf_boundary.sum())}  "
                      f"elevated_px={len(_hf_elev_idx):,}")
                for _ry in range(0, _hf_Ny, max(1, _hf_Ny // 12)):
                    _nh = int(_hf_is_hair[_ry].sum())
                    _nb = int(_hf_boundary[_ry].sum())
                    print(f"    row {_ry:3d} Y={float(_hf_yb[_ry]):.3f}  "
                          f"hair={_nh}  boundary={_nb}  "
                          f"hair_cols=[{_hf_hair_col_min[_ry]},{_hf_hair_col_max[_ry]}]")

                # Paint hair-layer pixels GREEN — but exclude the nose/chin
                # exclusion zone from Step 1 (_hl_excl_hi) which covers
                # |X|<0.031 & Y<0.453 & Z>0.030 (high-Z protrusion = nose/chin).
                # Those pixels UV-map to the face viewport area; filtering them
                # here suppresses the scattered green on eye/nose.
                _hf_excl_flag = np.zeros(len(p_y), dtype=bool)
                _hf_excl_flag[_hl_excl_hi] = True
                _hf_final_idx = _hf_elev_idx[~_hf_excl_flag[_hf_elev_idx]]
                print(f"  [HangHair-S4b] excl_filter: "
                      f"{len(_hf_elev_idx)} → {len(_hf_final_idx)} px "
                      f"({len(_hf_elev_idx)-len(_hf_final_idx)} removed)")
                debug_canvas[_uv_y[_hf_final_idx],
                             _uv_x[_hf_final_idx]] = (0, 220, 0)   # green

        # ── Step 4c: Back hair cap bottom rim ────────────────────────────────
        # In the back zone (Z<0, Y:[T_seam_floor,T_hair]) all pixels are hair
        # cap — no face skin competes here.  Per X column find the LOWEST Y row
        # with any pixels: that is the bottom rim of the hair cap where it ends
        # at the nape of the neck.  Paint those cells green.
        _hb_Nx = 160;  _hb_Ny = 160
        _hb_y_lo = float(_T_seam_floor)
        _hb_y_hi = float(_T_hair)
        _hb_x_half = float(_hl_ear_thresh)

        _hb_back_mask = (
            (p_y >= _hb_y_lo) & (p_y < _hb_y_hi) &
            (np.abs(p_x) < _hb_x_half) &
            (p_z < 0.0)
        )
        _hb_bi = np.where(_hb_back_mask)[0]
        print(f"  [BackHair-S4c] back_zone px={len(_hb_bi):,}  "
              f"Y:[{_hb_y_lo:.3f},{_hb_y_hi:.3f}]  |X|<{_hb_x_half:.3f}  Z<0")

        if len(_hb_bi) >= 100:
            _hb_by = p_y[_hb_bi]
            _hb_bx = p_x[_hb_bi]

            _hb_xb = np.linspace(-_hb_x_half, _hb_x_half, _hb_Nx + 1)
            _hb_yb = np.linspace(_hb_y_lo, _hb_y_hi, _hb_Ny + 1)

            _hb_bxi = np.clip(np.searchsorted(_hb_xb[1:], _hb_bx), 0, _hb_Nx - 1)
            _hb_byi = np.clip(np.searchsorted(_hb_yb[1:], _hb_by), 0, _hb_Ny - 1)

            # Pixel count per cell (only need presence)
            _hb_count = np.zeros((_hb_Ny, _hb_Nx), dtype=np.int32)
            np.add.at(_hb_count, (_hb_byi, _hb_bxi), 1)
            _hb_has_px = _hb_count >= 3   # require ≥3 px per cell (noise filter)

            # Per column: first True along rows = lowest Y row = bottom rim
            _hb_has_col  = _hb_has_px.any(axis=0)
            _hb_col_bot  = np.where(_hb_has_col,
                                    np.argmax(_hb_has_px, axis=0), -1).astype(int)

            _hb_boundary = np.zeros((_hb_Ny, _hb_Nx), dtype=bool)
            _hb_vcols = np.where(_hb_col_bot >= 0)[0]
            _hb_boundary[_hb_col_bot[_hb_vcols], _hb_vcols] = True

            _hb_bound_px = _hb_boundary[_hb_byi, _hb_bxi]
            _hb_elev_idx = _hb_bi[_hb_bound_px]

            print(f"  [BackHair-S4c] boundary_cols={int(_hb_has_col.sum())}  "
                  f"elevated_px={len(_hb_elev_idx):,}")
            for _ry in range(0, _hb_Ny, max(1, _hb_Ny // 10)):
                _nh = int(_hb_has_px[_ry].sum())
                _nb = int(_hb_boundary[_ry].sum())
                _hc = np.where(_hb_has_px[_ry])[0]
                _cmi = int(_hc.min()) if len(_hc) else -1
                _cma = int(_hc.max()) if len(_hc) else -1
                print(f"    row {_ry:3d} Y={float(_hb_yb[_ry]):.3f}  "
                      f"cells={_nh}  rim_here={_nb}  col_span=[{_cmi},{_cma}]")

            debug_canvas[_uv_y[_hb_elev_idx],
                         _uv_x[_hb_elev_idx]] = (0, 220, 0)   # green

        # ── Step 4d: Back-hanging hair in body zone ──────────────────────────
        # Braid hangs BEHIND the body (large negative code Z = Blender +Y direction).
        # Filter to Z<0 (back-facing pixels). Within that space:
        #   Body back surface: Z_min ≈ -0.10 to -0.20 (just the torso back face)
        #   Braid (much further behind): Z_min ≈ -0.30 to -0.50
        # Discriminator: Z_min per cell — flag cells that are unusually DEEP
        # (Z_min << row median = braid protruding far behind the body).
        _hbd_Nx = 120;  _hbd_Ny = 120
        _hbd_y_lo = float(_y_min)
        _hbd_y_hi = float(_T_seam_floor)

        # Z<0 filter: braid is behind the body
        _hbd_back_mask = (
            (p_y >= _hbd_y_lo) & (p_y < _hbd_y_hi) &
            (np.abs(p_x) < 0.40) &
            (p_z < 0.0)
        )
        _hbd_bi = np.where(_hbd_back_mask)[0]
        print(f"  [BackHair-S4d] Z<0 body px={len(_hbd_bi):,}  "
              f"Y:[{_hbd_y_lo:.3f},{_hbd_y_hi:.3f}]")

        if len(_hbd_bi) >= 100:
            _hbd_by = p_y[_hbd_bi]
            _hbd_bx = p_x[_hbd_bi]
            _hbd_bz = p_z[_hbd_bi]

            _hbd_xb = np.linspace(-0.40, 0.40, _hbd_Nx + 1)
            _hbd_yb = np.linspace(_hbd_y_lo, _hbd_y_hi, _hbd_Ny + 1)

            _hbd_bxi = np.clip(np.searchsorted(_hbd_xb[1:], _hbd_bx), 0, _hbd_Nx - 1)
            _hbd_byi = np.clip(np.searchsorted(_hbd_yb[1:], _hbd_by), 0, _hbd_Ny - 1)

            # Z_min per cell: most negative Z = deepest behind-body point
            _hbd_zmin = np.full((_hbd_Ny, _hbd_Nx), np.inf, dtype=np.float64)
            _hbd_cnt  = np.zeros((_hbd_Ny, _hbd_Nx), dtype=np.int32)
            np.minimum.at(_hbd_zmin, (_hbd_byi, _hbd_bxi), _hbd_bz)
            np.add.at(_hbd_cnt,  (_hbd_byi, _hbd_bxi), 1)

            _hbd_has_px    = _hbd_cnt > 0
            _hbd_zmin_safe = np.where(_hbd_has_px, _hbd_zmin, np.nan)

            # Per-row: median and std of Z_min (body back surface baseline)
            _hbd_row_med = np.full(_hbd_Ny, np.nan)
            _hbd_row_std = np.full(_hbd_Ny, np.nan)
            for _ry in range(_hbd_Ny):
                _rv = _hbd_zmin_safe[_ry, _hbd_has_px[_ry]]
                if len(_rv) >= 3:
                    _hbd_row_med[_ry] = float(np.nanmedian(_rv))
                    _hbd_row_std[_ry] = float(np.nanstd(_rv))

            _hbd_med_g = _hbd_row_med[:, np.newaxis]
            _hbd_std_g = _hbd_row_std[:, np.newaxis]

            # Braid cells: Z_min well BELOW row median (deeper behind body)
            _hbd_elevated = (
                (~np.isnan(_hbd_med_g)) &
                _hbd_has_px &
                (_hbd_zmin_safe < _hbd_med_g - 1.5 * _hbd_std_g)
            )

            _hbd_n_elev   = int(_hbd_elevated.sum())
            _hbd_elev_px  = _hbd_elevated[_hbd_byi, _hbd_bxi]
            _hbd_elev_idx = _hbd_bi[_hbd_elev_px]
            print(f"  [BackHair-S4d] elevated_cells={_hbd_n_elev}  "
                  f"elevated_px={len(_hbd_elev_idx):,}")

            # Per-pixel Z filter: a cell is flagged because its Z_MIN is extreme,
            # but the cell also contains shallow neck/body pixels with moderate Z.
            # Keep only pixels whose OWN Z is below the row baseline — they are
            # the true deep-braid pixels; the rest are neck/body surface noise.
            _hbd_orig_count  = len(_hbd_elev_idx)
            _hbd_elev_bz     = _hbd_bz[_hbd_elev_px]        # Z per elevated pixel
            _hbd_elev_byrow  = _hbd_byi[_hbd_elev_px]       # row index per elevated px
            _hbd_px_med      = _hbd_row_med[_hbd_elev_byrow]
            _hbd_px_std_raw  = _hbd_row_std[_hbd_elev_byrow]
            _hbd_px_std      = np.where(np.isnan(_hbd_px_std_raw),
                                        0.01, _hbd_px_std_raw)
            # Threshold: pixel Z < row_median − 1.0σ  (same spirit as cell flag)
            _hbd_px_keep = (
                (~np.isnan(_hbd_px_med)) &
                (_hbd_elev_bz < _hbd_px_med - 1.0 * _hbd_px_std)
            )
            _hbd_elev_idx = _hbd_elev_idx[_hbd_px_keep]
            print(f"  [BackHair-S4d] after per-px Z-filter: "
                  f"{len(_hbd_elev_idx):,} px  "
                  f"(dropped {_hbd_orig_count - len(_hbd_elev_idx):,} shallow px)")

            for _ry in range(0, _hbd_Ny, max(1, _hbd_Ny // 12)):
                _ne  = int(_hbd_elevated[_ry].sum())
                _nm  = int(_hbd_has_px[_ry].sum())
                _med = _hbd_row_med[_ry]
                # X span of elevated cells
                _ex  = np.where(_hbd_elevated[_ry])[0]
                _x_span = (f"[{float(_hbd_xb[_ex.min()]):.3f},"
                           f"{float(_hbd_xb[_ex.max()+1]):.3f}]"
                           if len(_ex) else "none")
                # Deepest Z_min this row (p10 = most braid-like cells)
                _zv  = _hbd_zmin_safe[_ry, _hbd_has_px[_ry]]
                _p10 = float(np.nanpercentile(_zv, 10)) if len(_zv) >= 5 else float('nan')
                _med_s = f"{_med:.3f}" if not np.isnan(_med) else "nan"
                _p10_s = f"{_p10:.3f}" if not np.isnan(_p10) else "nan"
                print(f"    row {_ry:3d} Y={float(_hbd_yb[_ry]):.3f}  "
                      f"cells={_nm}  elev={_ne}  "
                      f"Zmin_med={_med_s}  p10={_p10_s}  "
                      f"elev_X={_x_span}")

            debug_canvas[_uv_y[_hbd_elev_idx],
                         _uv_x[_hbd_elev_idx]] = (0, 220, 0)   # green

        # Red seam overlay removed — replaced by green Y-threshold boundary

        # ── Purple seed point: pixel closest to (p_x=0, p_y=T_head) ────────
        # Single dot to verify the hairline-walk starting location.
        # Using normalised distance in X and Y so neither axis dominates.
        if len(_hi) >= 1:
            _sp_hx   = p_x[_hi]
            _sp_hy   = p_y[_hi]
            _sp_x_sc = float(_sp_hx.std()) + 1e-6
            _sp_y_sc = float(_sp_hy.std()) + 1e-6
            _sp_dist = (_sp_hx / _sp_x_sc) ** 2 + \
                       ((_sp_hy - _T_head) / _sp_y_sc) ** 2
            _best    = int(np.argmin(_sp_dist))
            _seed_px = _hi[_best]
            # Paint a small 9×9 cross so it's visible at any texture resolution
            _sy0, _sx0 = int(_uv_y[_seed_px]), int(_uv_x[_seed_px])
            _r = 4
            _cy = np.clip(np.arange(_sy0-_r, _sy0+_r+1), 0, texture_size-1)
            _cx = np.clip(np.arange(_sx0-_r, _sx0+_r+1), 0, texture_size-1)
            debug_canvas[_cy, _sx0] = (180, 0, 255)   # vertical bar
            debug_canvas[_sy0, _cx] = (180, 0, 255)   # horizontal bar
            print(f"  [Seed] Purple seed dot at UV=({_sx0},{_sy0})  "
                  f"3D=({float(p_x[_seed_px]):.3f}, "
                  f"{float(p_y[_seed_px]):.3f}, "
                  f"{float(p_z[_seed_px]):.3f})")

        # ── Hair Border: contour of full hair region — no flood fill ─────────────
        # Hair = Z≤0 (back/sides, always hair) OR Y>_T_hair (top of head, covers
        # the front cap including the forehead hairline).
        # Together these form the complete hair cap. Contour = full hairline border.

        # Hair region: Z ≤ 0 (back and sides of head — never face).
        # Y > _T_hair intentionally excluded: it draws a flat horizontal line
        # across the forehead regardless of actual hair shape. Front cap handled separately.
        _hm_hair_raw = np.zeros((texture_size, texture_size), dtype=np.uint8)
        _hm_pz = p_z[_hl_safe_hi]
        _hm_hair_sel = (_hm_pz <= 0.0)
        _hm_hair_idx = _hl_safe_hi[_hm_hair_sel]
        _hm_hair_raw[_uv_y[_hm_hair_idx], _uv_x[_hm_hair_idx]] = 255
        print(f"  [HairBorder] hair px (Z≤0): {int(_hm_hair_sel.sum()):,}")

        # Close UV sparsity gaps so the Z≤0 back/side region is solid
        _hm_kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        _hm_closed = cv2.morphologyEx(_hm_hair_raw, cv2.MORPH_CLOSE, _hm_kern)

        # Add the cyan Canny region (front hair cap) to fill the front gap:
        # The cyan edges already trace the actual hair boundary from texture.
        # Close them into a solid blob, then merge with the back region.
        # This gives us the full hair cap without any artificial Y threshold.
        _hm_canny_u8 = _hl_edgesA.astype(np.uint8) * 255
        _hm_canny_closed = cv2.morphologyEx(_hm_canny_u8, cv2.MORPH_CLOSE, _hm_kern)
        # Only keep the Canny blob that overlaps or is adjacent to the Z≤0 back region
        # (avoids picking up body/neck Canny edges far from head)
        _hm_kern_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
        _hm_back_dilated = cv2.dilate(_hm_closed, _hm_kern_big)
        _hm_canny_near = _hm_canny_closed & (_hm_back_dilated > 0)

        _hm_full = np.clip(_hm_closed.astype(np.uint16) + _hm_canny_near.astype(np.uint16),
                           0, 255).astype(np.uint8)
        # Final close to merge back+front into one solid region
        _hm_full = cv2.morphologyEx(_hm_full, cv2.MORPH_CLOSE, _hm_kern)
        print(f"  [HairBorder] back={int((_hm_closed>0).sum()):,}  "
              f"front_canny={int((_hm_canny_near>0).sum()):,}  "
              f"merged={int((_hm_full>0).sum()):,}")

        # Outer contour of the merged region = full hairline, no forehead lines
        _hm_cnts, _ = cv2.findContours(_hm_full, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        _hm_border_img = np.zeros((texture_size, texture_size), dtype=np.uint8)
        cv2.drawContours(_hm_border_img, _hm_cnts, -1, 255, thickness=8)
        print(f"  [HairBorder] contours={len(_hm_cnts)}  "
              f"border px={int((_hm_border_img > 0).sum()):,}")

        _hm_clr = np.array([220, 0, 220], dtype=np.uint8)
        debug_canvas[_hm_border_img > 0] = _hm_clr
        print(f"  [HairBorder] done — back(Z≤0) + front(Canny) merged, no Y threshold")

        color_np = debug_canvas
        print("[DebugRegions] Done — seam-walk region map applied.")

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

    for _ in range(2):
        bbox_min = verts.min(dim=0)[0]
        bbox_max = verts.max(dim=0)[0]
        verts = verts - (bbox_min + bbox_max) * 0.5
        r = torch.sqrt((verts ** 2).sum(dim=-1).max())
        verts = verts * (norm_size / (r * 2.0))

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
    for _ in range(2):
        bbox_min = verts.min(dim=0)[0]
        bbox_max = verts.max(dim=0)[0]
        verts    = verts - (bbox_min + bbox_max) * 0.5
        r        = torch.sqrt((verts ** 2).sum(dim=-1).max())
        verts    = verts * (norm_size / (r * 2.0))

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
