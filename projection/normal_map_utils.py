"""
NormalMapUtils — tangent-space normal-map baking extracted from
texture_projection_multiview.py.

Versioning:
  normal_map_utils.py    — v1: geometry normal map baked from smooth vertex normals
  normal_map_utils_v2.py — create alongside this file when experimenting; swap import.

Usage:
  from .normal_map_utils import bake_normal_map
  normalMapTexture = bake_normal_map(
      out_verts, out_faces, out_uvs,
      tex_normals, tex_face_id,
      uv_hit_mask, texture_size,
  )
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt


def bake_normal_map(
    out_verts:    torch.Tensor,   # (V, 3) float32 cuda
    out_faces:    torch.Tensor,   # (F, 3) int32  cuda
    out_uvs:      torch.Tensor,   # (V, 2) float32 cuda
    tex_normals:  torch.Tensor,   # (H, W, 3) float32 cuda — per-texel smooth normals
    tex_face_id:  torch.Tensor,   # (H, W) int64   cuda — face index per texel (-1 = no hit)
    uv_hit_mask:  torch.Tensor,   # (H, W) bool    cuda
    texture_size: int,
) -> Image.Image:
    """
    Bake a tangent-space normal map from the mesh geometry.

    Encodes smooth vertex normals (which carry curvature detail from the
    original high-poly voxel mesh) into tangent space relative to the UV
    layout.  UE5 / DirectX convention: R=+X, G=-Y (flipped), B=+Z.

    Returns a PIL RGB Image ready to embed in the GLB material.
    """
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

    det_f   = duv1_f[:, 0] * duv2_f[:, 1] - duv1_f[:, 1] * duv2_f[:, 0]
    inv_det = torch.where(det_f.abs() > 1e-8, 1.0 / det_f, torch.zeros_like(det_f))

    T_face = (dp1_f * duv2_f[:, 1:2] - dp2_f * duv1_f[:, 1:2]) * inv_det.unsqueeze(-1)
    B_face = (-dp1_f * duv2_f[:, 0:1] + dp2_f * duv1_f[:, 0:1]) * inv_det.unsqueeze(-1)
    T_face = F.normalize(T_face, dim=-1)   # (F, 3)
    B_face = F.normalize(B_face, dim=-1)

    # Flat face normal — tangent-space reference direction
    N_geom_face = F.normalize(torch.cross(dp1_f, dp2_f, dim=-1), dim=-1)  # (F, 3)

    # Per-texel TBN lookup
    fid_clamped = tex_face_id.clamp(0)              # (H, W), safe index
    T_tex       = T_face[fid_clamped]               # (H, W, 3)
    N_geom_tex  = N_geom_face[fid_clamped]          # (H, W, 3)
    N_smooth    = tex_normals                        # (H, W, 3)

    # Gram-Schmidt: re-orthogonalise T against smooth N
    T_gs = F.normalize(T_tex - (T_tex * N_smooth).sum(-1, keepdim=True) * N_smooth, dim=-1)
    B_gs = torch.cross(N_smooth, T_gs, dim=-1)       # (H, W, 3)

    # Project smooth normal into tangent space
    nx = (N_smooth * T_gs).sum(-1, keepdim=True)     # (H, W, 1)
    ny = (N_smooth * B_gs).sum(-1, keepdim=True)
    nz = (N_smooth * N_geom_tex).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    # Encode: (n+1)/2*255, flip Y for UE5 DirectX convention
    normal_ts  = torch.cat([nx, -ny, nz], dim=-1)   # flip G channel
    normal_enc = ((normal_ts + 1.0) * 0.5 * 255.0).clamp(0, 255)

    # Fill with flat-normal default (128, 128, 255), overwrite UV-hit texels
    nm_img = torch.full((texture_size, texture_size, 3), 128.0,
                        device='cuda', dtype=torch.float32)
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

    print(f"  Normal map: {texture_size}x{texture_size} (DirectX/UE5 convention)")
    return Image.fromarray(nm_np, mode='RGB')
