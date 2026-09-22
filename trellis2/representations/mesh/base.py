from typing import *
import torch
from ..voxel import Voxel
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d

import numpy as np
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import gc


def _repair_hole_centres(old_v, new_v, new_f):
    """Recompute the vertex CuMesh's fill_holes appends at the centre of every filled loop.

    Measured 2026-09-15 on character_posed_00003_'s cached sculpt: the kernel does not always WRITE
    those positions. Fresh processes, identical input: faces and every original vertex identical, but
    the 3,903 appended centres sat at the hole centres in one run and held uninitialised GPU memory
    (up to 6.9e26) in others. Their fan triangles span the whole volume, the dual-contouring remesh
    wraps sheets around them (3.48M -> 29M faces) and the bake ends with 36,697 holes. Nothing
    downstream can see it: the remesh output stays inside its grid, so its coordinate check passes.

    Each appended vertex is the apex of a fan over one loop; set it to the mean of that loop's
    vertices -- what the kernel produces when it does write them (9e-6 difference on a clean run).
    """
    n_old = old_v.shape[0]
    n_new = new_v.shape[0] - n_old
    if n_new <= 0:
        return new_v
    f = new_f.long()
    isnew = f >= n_old
    one = isnew.sum(1) == 1
    fan = f[one]; m = isnew[one]
    if fan.shape[0] == 0:
        return new_v
    c = fan[m] - n_old                                # the centre each fan triangle belongs to
    rim = fan[~m].view(-1, 2)                         # its two loop vertices
    N = n_old + n_new
    key = torch.unique(torch.cat([c, c]) * N + torch.cat([rim[:, 0], rim[:, 1]]))   # each loop vertex once
    kc = key // N; kr = key % N
    acc = torch.zeros((n_new, 3), dtype=new_v.dtype, device=new_v.device)
    cnt = torch.zeros((n_new,), dtype=new_v.dtype, device=new_v.device)
    acc.index_add_(0, kc, new_v[kr])
    cnt.index_add_(0, kc, torch.ones_like(kc, dtype=new_v.dtype))
    mean = acc / cnt.clamp(min=1)[:, None]
    # how many the kernel left unwritten: farther from the loop centre than the loop's own farthest vertex
    reach = torch.zeros((n_new,), dtype=new_v.dtype, device=new_v.device)
    reach.scatter_reduce_(0, kc, (new_v[kr] - mean[kc]).norm(dim=1), reduce="amax")
    idx = torch.nonzero(cnt > 0).squeeze(1)
    old = new_v[idx + n_old]
    bad = ~torch.isfinite(old).all(1) | ((old - mean[idx]).norm(dim=1) > reach[idx])
    out = new_v.clone()
    out[idx + n_old] = mean[idx]
    nb = int(bad.sum())
    if nb:
        print(f"[fill_holes] repaired {nb} of {n_new} hole-centre vertices CuMesh left unwritten "
              f"(worst |coord| {float(old[bad].abs().max()):.3g})", flush=True)
    return out


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs
        
    @property
    def device(self):
        return self.vertices.device
        
    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )
        
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
        
    def cpu(self):
        return self.to('cpu')
    
    def fill_holes(self, max_hole_perimeter=3e-2):
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.get_edges()
        mesh.get_boundary_info()
        if mesh.num_boundaries == 0:
            del mesh
            gc.collect()
            return
        mesh.get_vertex_edge_adjacency()
        mesh.get_vertex_boundary_adjacency()
        mesh.get_manifold_boundary_adjacency()
        mesh.read_manifold_boundary_adjacency()
        mesh.get_boundary_connected_components()
        mesh.get_boundary_loops()
        if mesh.num_boundary_loops == 0:
            del mesh
            gc.collect()
            return
        mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
        new_vertices, new_faces = mesh.read()
        new_vertices = _repair_hole_centres(vertices, new_vertices, new_faces)
        
        del mesh
        gc.collect()         
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def remove_faces(self, face_mask: torch.Tensor):
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_faces(face_mask)
        new_vertices, new_faces = mesh.read()
        
        del mesh
        gc.collect()         
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def simplify_with_cumesh(self, target=1000000, verbose: bool=True, options: dict={}):        
        current_faces_num = len(self.faces)
        print(f'Current Faces Number: {current_faces_num}')
        
        if current_faces_num<target:
            return
        
        vertices = self.vertices.cuda()
        faces = self.faces.cuda()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()
        
        del mesh
        gc.collect()         
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def simplify_with_meshlib(self, target=1000000):
        current_faces_num = len(self.faces)
        print(f'Current Faces Number: {current_faces_num}')
        
        if current_faces_num<target:
            return

        settings = mrmeshpy.DecimateSettings()
        faces_to_delete = current_faces_num - target
        settings.maxDeletedFaces = faces_to_delete                        
        settings.packMesh = True
        
        print('Generating Meshlib Mesh ...')
        mesh = mrmeshnumpy.meshFromFacesVerts(self.faces.cpu().numpy(), self.vertices.cpu().numpy())
        print('Packing Optimally ...')
        mesh.packOptimally()
        print('Decimating ...')
        mrmeshpy.decimateMesh(mesh, settings)
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
        
        print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
        
        self.vertices = torch.from_numpy(new_vertices).float().to(self.device)
        self.faces = torch.from_numpy(new_faces).int().to(self.device)

        del mesh
        gc.collect()


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)
