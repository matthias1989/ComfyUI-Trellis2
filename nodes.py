import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm
import time
import shutil
import uuid
import triton
import triton.compiler

import folder_paths
import node_helpers
import hashlib
import cv2
import gc
import time as _time_mod
import copy

import pymeshlab

import cumesh as CuMesh

# ── UV-chart tripwire ────────────────────────────────────────────────────────────────────────
# cumesh clusters the mesh and hands every cluster to xatlas as its own chart; nothing on that
# path compares the count to the atlas it must fit into. Measured on this pipeline, same node,
# same 4096 atlas: 200,447 charts added in 0:14 and packed to 84 texels each, while 5,766,859
# added in 9:51, cost ~15 minutes end to end, and packed to 2.9 texels each -- a micro-island
# mosaic that cannot hold an eyelid. A chart needs roughly a 4x4 patch to be worth packing once
# xatlas has spent a texel of padding around it, so the floor is texture_size**2 / 16 charts.
# Above it the mesh is at fault, not the atlas: that many charts means the cone-angle clustering
# could not merge neighbouring faces. Nothing is salvageable, so fail in seconds, not minutes.
_CHART_MIN_TEXELS = 16                    # a 4x4 patch -- the smallest chart worth packing
_CHART_GUARD = {"budget": None, "texture_size": None}   # set per call by the unwrap nodes

def _dc_sane(v, scale):
    """Reject a dual-contouring draw whose COORDINATES are destroyed, not just its face count.

    The retry loop below guards on face-count ratio only. But this DC's documented failure mode
    (see the comment there) is finishing "with vertex coordinates around 1e29 -- geometry
    numerically destroyed". Such a draw can have a perfectly ordinary face count, so it sails past
    the ratio guard, and then two things happen 20 minutes later and 400 lines away:

      * xatlas cannot normalise a normal from it -> thousands of
        "ASSERT: isNormalized(normal) xatlas.cpp 1263" lines that bury the console; and
      * (pos - aabb[0]) / voxel_size lands astronomically out of range, which is an illegal
        memory access inside grid_sample_3d and ABORTS the ComfyUI process -- losing the run,
        the caches and the log with it.

    Verified 2026-09-11: the raw sculpt is clean (full offline unwrap, 597,173 charts, zero
    asserts, zero degenerate triangles), so the damage is introduced here, not by the generator.

    The bound is deliberately loose. The DC is called with center 0 and scale*1.1, so its output
    belongs within about +-0.55; past 2x the scale is not a slightly-off draw but a broken one.
    """
    if v is None or v.numel() == 0:
        return False, "empty result"
    finite = torch.isfinite(v)
    if not bool(finite.all()):
        return False, f"{int((~finite).any(dim=-1).sum()):,} non-finite vertices"
    m = float(v.abs().max())
    lim = float(scale) * 2.0
    if m > lim:
        return False, f"max |coord| {m:.3e}, limit {lim:.3f}"
    return True, f"max |coord| {m:.4f}"


def _safe_grid(pos, aabb, voxel_size, grid_size, where):
    """Validate and repair the sample coordinates before they reach the CUDA kernel.

    grid_sample_3d is a raw kernel with no bounds check, so ONE non-finite or out-of-range
    coordinate is an illegal memory access -- which aborts the PROCESS. ComfyUI dies, the run is
    lost, the caches with it, and the traceback names only the kernel, so there is nothing to
    diagnose from. That cost two 30-minute bakes on 2026-09-11.

    The bound is deliberately [0, grid_size], the volume itself, and NOT [0, grid_size - 1].
    Healthy meshes routinely sample the outermost voxel band -- the good 2026-09-11 sculpt spans
    -0.4997..0.4996, i.e. up to 1023.6 on a 1024 grid -- and those bakes work. A guard that
    clamped at grid_size - 1 fired on 4 of 1000 healthy samples in testing and would have shifted
    edge texels on EVERY bake. Only coordinates genuinely outside the volume are touched, so a
    healthy bake is bit-identical to before.

    `pos` is a barycentric combination of BVH triangle vertices, so it should already lie inside
    the volume; outside means the BVH returned something wrong. Clamping puts such a sample at the
    volume face, which is what sampling the volume means there. A non-finite one cannot be
    repaired and is parked at the centre: one wrong texel instead of a dead bake. Everything is
    counted and printed so the real cause shows up in the log.
    """
    g = (pos - aabb[0]) / voxel_size
    gs = grid_size.to(device=g.device, dtype=g.dtype)
    n = int(g.shape[0])
    if n == 0:
        print(f"[grid guard] {where}: EMPTY sample set -- nothing rasterised into the atlas")
        return g.reshape(1, -1, 3)
    finite = torch.isfinite(g).all(dim=-1)
    n_bad = int((~finite).sum())
    n_lo = int(((g < 0).any(dim=-1) & finite).sum())
    n_hi = int(((g > gs).any(dim=-1) & finite).sum())
    if n_bad or n_lo or n_hi:
        fg = g[finite]
        rng = ((fg.min(0).values.tolist(), fg.max(0).values.tolist())
               if fg.numel() else (None, None))
        print(f"[grid guard] {where}: {n:,} samples OUTSIDE the volume -- non-finite {n_bad:,}, "
              f"below 0: {n_lo:,}, above {[round(v) for v in gs.tolist()]}: {n_hi:,}")
        print(f"[grid guard]   finite range {rng[0]} .. {rng[1]}")
        print(f"[grid guard]   repaired instead of letting the kernel abort the process")
        g = torch.where(finite.unsqueeze(-1), g, gs * 0.5)
        g = torch.clamp(g, torch.zeros_like(gs), gs)
    return g.reshape(1, -1, 3)


_cm_read_atlas_charts = CuMesh.CuMesh.read_atlas_charts


def _cm_read_atlas_charts_guarded(self, *args, **kwargs):
    out = _cm_read_atlas_charts(self, *args, **kwargs)
    budget = _CHART_GUARD.get("budget")
    if budget:
        n = max(int(out[0]), 1)
        ts = _CHART_GUARD.get("texture_size")
        per = ts * ts / float(n)
        if n > budget:
            raise RuntimeError(
                f"CHART TRIPWIRE: clustering produced {n:,} charts for a {ts}x{ts} atlas -- "
                f"{per:.1f} texels per chart, against the {_CHART_MIN_TEXELS} (a 4x4 patch) "
                f"needed to hold any detail. Packing this many takes ~15 minutes and the atlas "
                f"comes out a micro-island mosaic. The mesh is the problem, not the atlas: this "
                f"many charts means the cone-angle clustering could not merge neighbouring "
                f"faces, so the surface is noise. Aborting before the unwrap instead of after."
            )
        print(f"[chart guard] {n:,} charts, {per:.0f} texels each (budget {budget:,})")
    return out


CuMesh.CuMesh.read_atlas_charts = _cm_read_atlas_charts_guarded
import o_voxel

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

import nvdiffrast.torch as dr
from flex_gemm.ops.grid_sample import grid_sample_3d

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .trellis2.pipelines import Trellis2ImageTo3DPipeline
from .trellis2.representations import Mesh, MeshWithVoxel
from .trellis2.modules.attention import config
from .trellis2.modules.sparse import config as sparseconfig
from .trellis2.pipelines import samplers
from .trellis2.modules.sparse import SparseTensor

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

BASE_CACHE_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "triton_caches"
#os.environ["TRITON_ALWAYS_COMPILE"] = "1"
#os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"]="1"

to_pil = transforms.ToPILImage()

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

def rotate_triton_cache():
    """
    Creates a new cache directory and attempts to clean up old ones.
    """
    # 1. Create the base directory if it doesn't exist
    BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Generate a unique ID for this specific run
    run_id = f"cache_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    new_cache_path = BASE_CACHE_DIR / run_id
    new_cache_path.mkdir()

    # 3. Point Triton to this NEW empty folder
    # This forces a recompile without needing to delete the locked file immediately
    os.environ["TRITON_CACHE_DIR"] = str(new_cache_path)
    print(f"[TrellisNode] 🔄 Switched to fresh Triton cache: {new_cache_path.name}")

    # 4. Garbage Collection: Try to delete OLD cache folders
    # We wrap this in a try/except so if Windows locks a file, we just skip it
    # and leave it for the next cleanup cycle.
    cleanup_old_caches(current_active=new_cache_path)                

def cleanup_old_caches(current_active):
    """
    Iterates through the cache folder and deletes anything that isn't the current active one.
    If a file is locked by Windows, it silently fails and leaves it for later.
    """
    for item in BASE_CACHE_DIR.iterdir():
        if item.is_dir() and item != current_active:
            try:
                shutil.rmtree(item)
                print(f"[TrellisNode] 🧹 Cleaned up old cache: {item.name}")
            except OSError:
                # This is expected on Windows! The file is locked.
                # We just ignore it and try again next time the node runs.
                pass 

def parse_string_to_int_list(number_string):
  """
  Parses a string containing comma-separated numbers into a list of integers.

  Args:
    number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

  Returns:
    A list of integers parsed from the input string.
    Returns an empty list if the input string is empty or None.
  """
  if not number_string:
    return []

  try:
    # Split the string by comma and convert each part to an integer
    int_list = [int(num.strip()) for num in number_string.split(',')]
    return int_list
  except ValueError as e:
    print(f"Error converting string to integer: {e}. Please ensure all values are valid numbers.")
    return []

def reset_cuda():    
    # Synchronize to ensure all GPU operations complete
    torch.cuda.synchronize()     
    
    # Force garbage collection of Python objects
    gc.collect()    
    
    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Accepts either:
      - (H,W,C)
      - (1,H,W,C)
    Returns a PIL RGB/RGBA image depending on channels.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            # Expect (B,H,W,C); allow only B==1 here
            if t.shape[0] != 1:
                raise ValueError(f"tensor2pil expects batch of 1, got batch={t.shape[0]}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"tensor2pil expects (H,W,C) or (1,H,W,C), got shape={tuple(t.shape)}")

        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"tensor2pil expected torch.Tensor, got {type(image)}")    
    
def tensor_batch_to_pil_list(images: torch.Tensor, max_views: int = 4) -> list[Image.Image]:
    """
    Converts a ComfyUI IMAGE tensor (B,H,W,C) into a list of PIL images.
    Caps to max_views for safety.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for IMAGE, got {type(images)}")

    if images.ndim == 4:
        b = int(images.shape[0])
        n = min(b, int(max_views))
        return [tensor2pil(images[i:i+1]) for i in range(n)]

    if images.ndim == 3:
        return [tensor2pil(images)]

    raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(images.shape)}")    
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array
    
def simplify_with_meshlib(vertices, faces, target=1000000):
    current_faces_num = len(faces)
    print(f'Current Faces Number: {current_faces_num}')
    
    if current_faces_num<target:
        return

    settings = mrmeshpy.DecimateSettings()
    faces_to_delete = current_faces_num - target
    settings.maxDeletedFaces = faces_to_delete                        
    settings.packMesh = True
    
    print('Generating Meshlib Mesh ...')
    mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
    print('Packing Optimally ...')
    mesh.packOptimally()
    print('Decimating ...')
    mrmeshpy.decimateMesh(mesh, settings)
    
    new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
    new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
    
    print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
        
    return new_vertices, new_faces

def remove_floater(mesh):
    print('Removing floater ...')
    faces = mesh.faces.cpu().numpy()
    print(f"Current faces: {len(faces)}")
    mesh_set = pymeshlab.MeshSet()
    mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.vertices.cpu().numpy(), face_matrix=faces)
    mesh_set.add_mesh(mesh_pymeshlab, "converted_mesh")
    mesh_set = pymeshlab_remove_floater(mesh_set)
    
    mesh_pymeshlab = mesh_set.current_mesh()    
    
    new_faces = mesh_pymeshlab.face_matrix()
    print(f"After removing floater: {len(new_faces)}")
    
    new_vertices = torch.from_numpy(mesh_pymeshlab.vertex_matrix()).contiguous().float()
    new_faces = torch.from_numpy(new_faces).contiguous().int()   
    
    mesh.vertices = new_vertices
    mesh.faces = new_faces
    
    return mesh
    
def remove_floater2(vertices, faces):
    print('Removing floater ...')
    #faces = faces.cpu().numpy()
    print(f"Current faces: {len(faces)}")
    mesh_set = pymeshlab.MeshSet()
    mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    mesh_set.add_mesh(mesh_pymeshlab, "converted_mesh")
    mesh_set = pymeshlab_remove_floater(mesh_set)
    
    mesh_pymeshlab = mesh_set.current_mesh()    
    
    new_faces = mesh_pymeshlab.face_matrix()
    print(f"After removing floater: {len(new_faces)}")
    
    new_vertices = mesh_pymeshlab.vertex_matrix()
    
    return new_vertices, new_faces

def remove_mesh_infinite_vertices(mesh):
    print('Removing infinite vertices ...')
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    trimesh = Trimesh.Trimesh(vertices=vertices,faces=faces)
    print(f"Original vertex count: {len(trimesh.vertices)}")
    
    # Remove anything outside a reasonable bounding box
    limit = 1e10 
    valid_mask = (np.abs(trimesh.vertices) < limit).all(axis=1)
    
    trimesh.update_vertices(valid_mask)
    
    # Removing vertices can leave "degenerate" faces or orphan nodes
    trimesh.update_faces(trimesh.nondegenerate_faces())
    trimesh.remove_unreferenced_vertices()  

    print(f"Cleaned vertex count: {len(trimesh.vertices)}")
    
    new_vertices = torch.from_numpy(trimesh.vertices).float()
    new_faces = torch.from_numpy(trimesh.faces).int()   
    
    mesh.vertices = new_vertices
    mesh.faces = new_faces    
    
    return mesh
    
def pymeshlab_remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh 
    
def _batched_unsigned_distance(bvh, positions, batch_size=100000, return_uvw=False):
    """
    Batch unsigned_distance queries to avoid GPU kernel timeout on large meshes.
    When processing high-resolution textures (e.g., 2048x2048 = ~4M pixels) on complex
    meshes, a single BVH query can cause GPU watchdog timeout. This function splits
    the query into smaller batches.
    Args:
        bvh: The BVH structure from cumesh
        positions: (N, 3) tensor of query positions
        batch_size: Maximum number of queries per batch (default 100K, matching
            the rasterization chunk size used elsewhere in this file)
        return_uvw: Whether to return barycentric coordinates
    Returns:
        Same as bvh.unsigned_distance()
    """
    import torch
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)

    distances_list = []
    face_id_list = []
    uvw_list = [] if return_uvw else None

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        d, f, u = bvh.unsigned_distance(positions[i:end], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)

    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None
    )    

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": (["microsoft/TRELLIS.2-4B","visualbruno/TRELLIS.2-4B-FP8"],{"default":"microsoft/TRELLIS.2-4B"}),
                "backend": (["flash_attn","xformers","sdpa","flash_attn_3"],{"default":"flash_attn"}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "low_vram": ("BOOLEAN",{"default":True}),
                "keep_models_loaded": ("BOOLEAN", {"default":True}),
                "conv_backend": (["spconv","torchsparse","flex_gemm"],{"default":"flex_gemm"}),
                "sparse_backend": (["xformers","flash_attn"],{"default":"flash_attn"}),
                "use_reconviagen": ("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, modelname, backend, device, low_vram, keep_models_loaded, conv_backend, sparse_backend, use_reconviagen):    
        import requests
        
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
        #os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
        #os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'        
        os.environ['ATTN_BACKEND'] = backend
        
        config.set_backend(backend)
        sparseconfig.set_attn_backend(sparse_backend)
        sparseconfig.set_conv_backend(conv_backend)
        
        reset_cuda()
        
        torch.backends.cudnn.benchmark = False        
            
        model_path = os.path.join(folder_paths.models_dir, modelname)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=modelname,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        
        reconviagen_pipeline_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','reconviagen_pipeline.json')
        if not os.path.exists(reconviagen_pipeline_file):
            source_reconviagen_pipeline_file = os.path.join(script_directory,'reconviagen_pipeline.json')
            shutil.copyfile(source_reconviagen_pipeline_file,reconviagen_pipeline_file)
        
        dinov3_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m","model.safetensors")
        if not os.path.exists(dinov3_model_path):
            raise Exception("Facebook Dinov3 model not found in models/facebook/dinov3-vitl16-pretrain-lvd1689m folder")
        
        trellis_image_large_path = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")
        if not os.path.exists(trellis_image_large_path):
            print('Trellis-Image-Large ss_dec_conv3d_16l8_fp16 files not found. Trying to download the files from huggingface ...')            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.json")
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.json complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.json")
            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")

            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.safetensors complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.safetensors")
        
        if use_reconviagen:
            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_vggt_cond.safetensors')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_vggt_cond.safetensors not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_vggt_cond.safetensors?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_vggt_cond.safetensors")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_vggt_cond.safetensors complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_vggt_cond.safetensors")
            
            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_vggt_cond.json')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_vggt_cond.json not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_vggt_cond.json?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_vggt_cond.json")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_vggt_cond.json complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_vggt_cond.json") 

            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_flow_img_dit_L_16l8_fp16.safetensors')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_flow_img_dit_L_16l8_fp16.safetensors not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_flow_img_dit_L_16l8_fp16.safetensors")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_flow_img_dit_L_16l8_fp16.safetensors complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_flow_img_dit_L_16l8_fp16.safetensors")      

            reconviagen_file = os.path.join(folder_paths.models_dir,'microsoft','TRELLIS.2-4B','ckpts','ss_flow_img_dit_L_16l8_fp16.json')
            if not os.path.exists(reconviagen_file):
                print('ReconViaGen file ss_flow_img_dit_L_16l8_fp16.json not found. Trying to download the files from huggingface ...')            
                url = "https://huggingface.co/Stable-X/trellis-vggt-v0-2/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.json?download=true"
                filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS.2-4B","ckpts","ss_flow_img_dit_L_16l8_fp16.json")
                path = Path(filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print("Download ss_flow_img_dit_L_16l8_fp16.json complete!")
                else:
                    raise Exception("Cannot download ReconViaGen file ss_flow_img_dit_L_16l8_fp16.json")                       
        
        if modelname == "visualbruno/TRELLIS.2-4B-FP8":
            use_fp8 = True
            if use_reconviagen:
                raise Exception("ReconViaGen cannot be used with TRELLIS.2-4B-FP8. Select microsoft/TRELLIS.2-4B")
        else:
            use_fp8 = False
                
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path, keep_models_loaded = keep_models_loaded, use_fp8=use_fp8, use_reconviagen=use_reconviagen)
        pipeline.low_vram = low_vram
        
        if device=="cuda":
            if low_vram:
                pipeline.cuda()
            else:
                pipeline.to(device)
        else:
            pipeline.to(device)
        
        return (pipeline,)
        
class Trellis2MeshWithVoxelGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "max_num_tokens": ("INT",{"default":49152,"min":0,"max":999999}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, shape_steps, texture_steps, max_num_tokens, max_views, sparse_structure_resolution, generate_texture_slat, use_tiled_decoder, sampler, fill_holes, hole_iterations, hole_fill_algorithm, keep_only_shell):
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images

        sparse_structure_sampler_params = {"steps":sparse_structure_steps}
        shape_slat_sampler_params = {"steps":shape_steps}
        tex_slat_sampler_params = {"steps":texture_steps}
        
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)        
        
        mesh = pipeline.run(image=image_in, 
                            seed=seed, 
                            pipeline_type=pipeline_type, 
                            sparse_structure_sampler_params = sparse_structure_sampler_params, 
                            shape_slat_sampler_params = shape_slat_sampler_params, 
                            tex_slat_sampler_params = tex_slat_sampler_params, 
                            max_num_tokens = max_num_tokens, 
                            sparse_structure_resolution = sparse_structure_resolution, 
                            max_views = max_views, 
                            generate_texture_slat = generate_texture_slat, 
                            use_tiled=use_tiled_decoder, 
                            pbar=pbar, 
                            sampler=sampler,
                            fill_holes=fill_holes,
                            hole_iterations=hole_iterations,
                            hole_fill_algorithm=hole_fill_algorithm,
                            keep_only_shell=keep_only_shell)[0]
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()        
       
        # Build BVH for the current mesh to guide remeshing
        if generate_texture_slat:
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None        
        
        return (mesh, bvh,)    

class Trellis2LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Trellis2Wrapper"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "image_with_alpha", "character_name")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        output_images_ori = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            output_images_ori.append(pil2tensor(i))

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        character_name = Path(image_path).stem   # use image_path: `image` got clobbered to a Tensor in the loop above
        return (output_image, output_mask, output_image_ori, character_name)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True  

class Trellis2SimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method):        
        mesh_copy = copy.deepcopy(mesh)
        if method=="Cumesh":
            # internal testing future release
            # options = {
                # 'method': 'legacy'
            # }             
            mesh_copy.simplify_with_cumesh(target = target_face_num)
        elif method=="Meshlib":
            mesh_copy.simplify_with_meshlib(target = target_face_num)
        else:
            raise Exception("Unknown simplification method")             
        
        return (mesh_copy,)     

class Trellis2SimplifyTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, target_face_num, method):        
        mesh_copy = copy.deepcopy(trimesh)
        if method=="Cumesh":
            # internal testing future release
            # options = {
                # 'options': 'legacy'
            # }            
            cumesh = CuMesh.CuMesh()
            cumesh.init(torch.from_numpy(mesh_copy.vertices).float().cuda(), torch.from_numpy(mesh_copy.faces).int().cuda())
            cumesh.simplify(target_face_num, verbose=True)
            new_vertices, new_faces = cumesh.read()
            mesh_copy.vertices = new_vertices.cpu().numpy()
            mesh_copy.faces = new_faces.cpu().numpy()
            
            del cumesh
        elif method=="Meshlib":
            new_vertices, new_faces = simplify_with_meshlib(mesh_copy.vertices, mesh_copy.faces, target = target_face_num)
            mesh_copy.vertices = new_vertices
            mesh_copy.faces = new_faces
        else:
            raise Exception("Unknown simplification method")             
        
        return (mesh_copy,)   

class Trellis2ProgressiveSimplify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_edge_length": ("FLOAT",{"default":0.00,"min":0.00,"max":99999.99,"step":0.01}),
                "max_triangle_aspect_ratio": ("FLOAT",{"default":20.00,"min":0.01,"max":99999.99,"step":0.01}),
                "strategy": (["Minimal Error First","Shortest Edge First"],{"default":"Minimal Error First"}),
                "stabilizer": ("FLOAT",{"default":0.000001,"min":0.0,"max":0.999999,"step":0.000001}),
                "touch_near_boundary_edges": ("BOOLEAN",{"default":True}),
                "optimize_vertex_positions": ("BOOLEAN",{"default":True}),
                "angle_based_weights": ("BOOLEAN",{"default":False}),                
            },
            "optional": {
                "trimesh": ("TRIMESH",),
                "mesh": ("MESHWITHVOXEL",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESHWITHVOXEL",)
    RETURN_NAMES = ("trimesh", "mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights, trimesh = None, mesh = None):        
        if trimesh is not None:
            trimesh = copy.deepcopy(trimesh)
            
            vertices = trimesh.vertices
            faces = trimesh.faces
            
            vertices, faces = self.simplify(vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights)
            trimesh.vertices = vertices
            trimesh.faces = faces
            
        if mesh is not None:
            mesh = copy.deepcopy(mesh)
            
            vertices = mesh.vertices.cpu().numpy()
            faces = mesh.faces.cpu().numpy()
            
            vertices, faces = self.simplify(vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights)
            mesh.vertices = torch.from_numpy(vertices).float()
            mesh.faces = torch.from_numpy(faces).int()
        
        return (trimesh, mesh) 

    def simplify(self, vertices, faces, max_edge_length, max_triangle_aspect_ratio, strategy, stabilizer, touch_near_boundary_edges, optimize_vertex_positions, angle_based_weights):
        current_faces_num = len(faces)
        print(f'Current Faces Number: {current_faces_num}')

        settings = mrmeshpy.DecimateSettings()
        
        if strategy == "Minimal Error First":
            settings.strategy = mrmeshpy.DecimateStrategy.MinimizeError
        else:
            settings.strategy = mrmeshpy.DecimateStrategy.ShortestEdgeFirst
            
        settings.maxTriangleAspectRatio = max_triangle_aspect_ratio
        settings.stabilizer = stabilizer
        settings.touchNearBdEdges = touch_near_boundary_edges
        settings.optimizeVertexPos = optimize_vertex_positions
        settings.angleWeightedDistToPlane = angle_based_weights        
        settings.packMesh = True
        
        print('Generating Meshlib Mesh ...')
        mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
        
        if max_edge_length == 0.0:
            max_edge_length = 2.0
            # for edge_id in mesh.topology.allValidEdges():
                    # edge_len = mesh.computeEdgeLen(edge_id)
                    # if edge_len > max_edge_length:
                        # max_edge_length = edge_len
            # print(f"Calculated Max Edge Length: {max_edge_length}")
            
        settings.maxEdgeLen = max_edge_length   
        settings.maxError = max_edge_length / 1000
        
        print('Packing Optimally ...')
        mesh.packOptimally()
        print('Decimating ...')
        mrmeshpy.decimateMesh(mesh, settings)
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)               
        
        print(f"Reduced faces, resulting in {len(new_vertices)} vertices and {len(new_faces)} faces")
            
        del mesh
        gc.collect()
        
        return new_vertices, new_faces        
        
class Trellis2MeshWithVoxelToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, reorient_vertices):       
        mesh_copy = copy.deepcopy(mesh)
        
        vertices_np = mesh_copy.vertices.cpu().numpy()
        
        if reorient_vertices == '90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        elif reorient_vertices == '-90 degrees':
            vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2], vertices_np[:, 1]
        
        trimesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=mesh_copy.faces.cpu().numpy(),
            process=False
        )
        
        return (trimesh,)
        
class Trellis2ExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Trellis2"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "character_name": ("STRING", {"default": "",
                                              "tooltip": "When set, saves to output/characters/{character_name}/ "
                                                         "and ignores filename_prefix. Type the character image name here."}),
                # A SOCKET (forceInput), deliberately not a widget: adding a widget would change
                # widgets_values on every existing ExportMesh node and the saved graph would
                # refuse to load. Wire Run Paths here so the file lands in this run's folder
                # instead of a path typed into a widget that drifts out of date.
                "path_prefix": ("STRING", {"forceInput": True,
                                "tooltip": "Output prefix relative to the output folder, e.g. "
                                           "characters/<char>/run_<stamp>/TRELLIS_TEXTURED. "
                                           "Overrides filename_prefix and character_name."}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("glb_path","relative_path",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, character_name="",
                path_prefix=""):
        pp = (path_prefix or "").strip()
        if pp:
            filename_prefix = pp.replace("\\", "/").lstrip("/")
            print(f"[ExportMesh] path_prefix -> {filename_prefix}")
            character_name = ""
        # If character_name is supplied, override prefix → output/characters/{name}/
        char = (character_name or "").strip()
        if char:
            # Strip any file extension the user may have typed (e.g. "elf.png" → "elf")
            char = Path(char).stem
            filename_prefix = f"characters/{char}"
            print(f"[ExportMesh] character_name={char!r} → saving to characters/{char}/")

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format=='obj':
            materialName = f"{filename}_{counter:05}_.mtl"
            if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material') and trimesh.visual.material is not None:
                trimesh.visual.material.name = f"{filename}_{counter:05}"

            trimesh.export(output_glb_path, file_type=file_format, mtl_name=materialName)
        else:
            trimesh.export(output_glb_path, file_type=file_format)

        relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'

        return (str(output_glb_path), str(relative_path), )        
        
class Trellis2PostProcessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remove_duplicate_faces": ("BOOLEAN",{"default":False}),
                "repair_non_manifold_edges": ("BOOLEAN", {"default":False}),
                "remove_non_manifold_faces": ("BOOLEAN", {"default":False}),
                "remove_small_connected_components": ("BOOLEAN", {"default":False}),
                "remove_small_connected_components_size": ("FLOAT", {"default":0.00001,"min":0.00001,"max":9.99999,"step":0.00001}),
                "unify_faces_orientation": ("BOOLEAN", {"default":False}),
                "remove_floaters": ("BOOLEAN",{"default":False}),
                "remove_infinite_vertices": ("BOOLEAN",{"default":False}),
                "merge_vertices": ("BOOLEAN",{"default":False}),
                "merge_distance": ("FLOAT",{"default":0.0010,"min":0.0001,"max":999.9999,"step":0.0001}),
                "remove_nan_vertices": ("BOOLEAN",{"default":False}),                
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        mesh, 
        remove_duplicate_faces, 
        repair_non_manifold_edges, 
        remove_non_manifold_faces, 
        remove_small_connected_components, 
        remove_small_connected_components_size,
        unify_faces_orientation,
        remove_floaters,
        remove_infinite_vertices,
        merge_vertices,
        merge_distance,
        remove_nan_vertices):
            
        mesh_copy = copy.deepcopy(mesh)

        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        if remove_infinite_vertices:
            mesh_copy = remove_mesh_infinite_vertices(mesh_copy)                    

        vertices = mesh_copy.vertices
        faces = mesh_copy.faces

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
            
        if remove_duplicate_faces:
            print('Removing duplicate faces ...')
            cumesh.remove_duplicate_faces()
            
        if repair_non_manifold_edges:
            print('Repairing non manifold edges ...')
            cumesh.repair_non_manifold_edges()
            
        if remove_non_manifold_faces:
            print('Removing non manifold faces ...')
            cumesh.remove_non_manifold_faces()
            
        if remove_small_connected_components:
            print('Removing small connected components ...')
            cumesh.remove_small_connected_components(remove_small_connected_components_size)        
        
        if unify_faces_orientation:
            print('Unifying faces orientation ...')
            cumesh.unify_face_orientations()            
        
        if merge_vertices or remove_nan_vertices:
            import open3d            
            open3d_mesh = open3d.geometry.TriangleMesh()
            open3d_mesh.vertices = open3d.utility.Vector3dVector(vertices.cpu().numpy())
            open3d_mesh.triangles = open3d.utility.Vector3iVector(faces.cpu().numpy().astype(np.int32))

            # NaN check
            print('Removing NaN vertices ...')
            verts = np.asarray(open3d_mesh.vertices)
            if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                print('NaN found. Cleaning them ...')
                verts = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
                open3d_mesh.vertices = open3d.utility.Vector3dVector(verts)
                open3d_mesh = open3d_mesh.remove_duplicated_vertices()
                open3d_mesh = open3d_mesh.remove_duplicated_triangles()
                open3d_mesh = open3d_mesh.remove_degenerate_triangles()
                open3d_mesh = open3d_mesh.remove_unreferenced_vertices()

            #bbox = open3d_mesh.get_axis_aligned_bounding_box()
            #max_extent = np.max(bbox.get_extent())
            #safe_merge_distance = max_extent * 0.0005  # More conservative
            #print(f"Auto-calculated merge distance: {safe_merge_distance:.6f}")
            
            if merge_vertices:
                # Merge and cleanup
                open3d_mesh = open3d_mesh.merge_close_vertices(merge_distance)
                open3d_mesh = open3d_mesh.remove_duplicated_vertices()
                open3d_mesh = open3d_mesh.remove_duplicated_triangles()
                open3d_mesh = open3d_mesh.remove_degenerate_triangles()
                open3d_mesh = open3d_mesh.remove_unreferenced_vertices()

                # Proper normal computation sequence
                open3d_mesh.compute_triangle_normals()
                open3d_mesh.compute_vertex_normals()
                open3d_mesh.normalize_normals()
                open3d_mesh.orient_triangles()  # Orient based on computed normals
                open3d_mesh.compute_vertex_normals()  # Recompute after orientation

                # Gentler smoothing
                open3d_mesh = open3d_mesh.filter_smooth_taubin(number_of_iterations=3)
                open3d_mesh.compute_vertex_normals()  # Final recompute            
            
            cumesh.init(torch.from_numpy(np.asarray(open3d_mesh.vertices)).cuda().float(), torch.from_numpy(np.asarray(open3d_mesh.triangles)).cuda().int())
            del open3d_mesh
        
        print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                                                   
        
        new_vertices, new_faces = cumesh.read()
        
        mesh_copy.vertices = new_vertices.to(mesh_copy.device)
        mesh_copy.faces = new_faces.to(mesh_copy.device) 
        
        del cumesh
        gc.collect()
                
        return (mesh_copy,)
       
class Trellis2UnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":4096, "min":512, "max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "bvh": ("BVH",),
                "inpainting": (["telea","ns"],{"default":"telea"}),
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture", "metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices,use_custom_normals,bvh,inpainting):
        mesh_copy = copy.deepcopy(mesh)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size  
        
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)

        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size       
        
            print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")        
        
        vertices = vertices.cuda()
        faces = faces.cuda()        
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        
        # Build BVH for the current mesh to guide remeshing
        # if bvh == None:
        # print(f"Building BVH for current mesh...")
        # bvh = CuMesh.cuBVH(vertices, faces) 
        # bvh.vertices = vertices
        # bvh.faces = faces
        
        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            print('Baking colors on vertices...')
            out_vertices, out_faces = cumesh.read()
            out_vertices = out_vertices.cuda()
            out_faces = out_faces.cuda()
            cumesh.compute_vertex_normals()
            out_normals = cumesh.read_vertex_normals()
            
            # Sample attributes directly at vertex positions from the voxel grid
            # No BVH mapping needed - the voxel grid contains all the color information
            vertex_attrs = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=_safe_grid(out_vertices, aabb, voxel_size, grid_size, "vertex bake"),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = attr_layout['base_color']
            alpha_idx = attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals.cpu().numpy()
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            del cumesh
            gc.collect()
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
            return (textured_mesh, placeholder_texture, placeholder_texture,)        
        
        _CHART_GUARD.update(budget=(texture_size * texture_size) // _CHART_MIN_TEXELS,
                            texture_size=texture_size)
        print('Unwrapping ...')        
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,                
            },
            return_vmaps=True,
            verbose=True,
        )
        
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()
        out_vmaps = out_vmaps.cuda()
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps]        

        print("Sampling attributes...")
        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        
        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)
        
        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0
        
        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        
        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)        
        
        torch.cuda.synchronize()
        
        # Trilinear sampling from the attribute volume (Color, Material props)
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=_safe_grid(valid_pos, aabb, voxel_size, grid_size, "texture bake"),
            mode='trilinear',
        )      
        
        # --- Texture Post-Processing & Material Construction ---
        print("Finalizing mesh...")
        
        mask = mask.cpu().numpy()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = texture_alpha_mode
        
        if inpainting == 'telea':
            inpainting = cv2.INPAINT_TELEA
        else:
            inpainting = cv2.INPAINT_NS
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, inpainting)
        metallic = cv2.inpaint(metallic, mask_inv, 1, inpainting)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, inpainting)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, inpainting)[..., None]
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        baseColorTexture_np = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture_np = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture_np,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture_np,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=double_side_material,
        )        
        
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
        else:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )            

        del cumesh
        gc.collect()    

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
                
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )
        
class Trellis2MeshWithVoxelAdvancedGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),                
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),                
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,        
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        max_num_tokens,
        max_views,
        sparse_structure_resolution,
        generate_texture_slat,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        sampler,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell
        ):
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images

        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)
        
        mesh = pipeline.run(image=image_in, 
                            seed=seed, 
                            pipeline_type=pipeline_type, 
                            sparse_structure_sampler_params = sparse_structure_sampler_params, 
                            shape_slat_sampler_params = shape_slat_sampler_params, 
                            tex_slat_sampler_params = tex_slat_sampler_params, 
                            max_num_tokens = max_num_tokens, 
                            sparse_structure_resolution = sparse_structure_resolution, 
                            max_views = max_views, 
                            generate_texture_slat=generate_texture_slat, 
                            use_tiled=use_tiled_decoder, 
                            pbar=pbar, 
                            sampler=sampler,
                            fill_holes = fill_holes,
                            hole_iterations = hole_iterations,
                            verbose = verbose,
                            dino_lock = dino_lock,
                            dino_substeps = dino_substeps,
                            hole_fill_algorithm=hole_fill_algorithm,
                            dino_foundation_cap=dino_foundation_cap,
                            keep_only_shell=keep_only_shell)[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()                
        
        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None
        
        return (mesh,bvh,)         

class Trellis2MeshWithVoxelMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),                 
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),                
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    # NOT an output node: the Sculpt Cache asks for this draw lazily, only on a miss. As an output node ComfyUI ran
    # it on EVERY bake anyway, after the rig -- a full Qwen-view + Trellis draw that nothing used (~3 min a
    # character, and its axis warnings in the log looked like a failed bake; 2026-09-24).
    OUTPUT_NODE = False

    def process(self, pipeline, front_image, seed, pipeline_type, sparse_structure_steps,
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,        
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        max_num_tokens,
        sparse_structure_resolution,
        generate_texture_slat,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        front_axis,
        blend_temperature,
        sampler,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        back_image=None,
        left_image=None,
        right_image=None):

        reset_cuda()
        
        # Convert front image tensor to PIL
        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None        
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)

        # THE FACE CHECK (2026-09-25, user: option 3 -- "check every new sculpt's face proportions against the photo and
        # regenerate automatically when they disagree"). Trellis drew 00006 / 00010 / 00011 with mouths 28-43% narrower
        # than their photos. Each draw's face is measured against the photo (face_check.py: the raw sculpt's head in its own
        # colours, MediaPipe, the mouth's share of the face width, tolerance 3 x the measurement's noise); a draw that
        # fails is drawn again with the next seed, up to TRELLIS_FACE_CHECK_TRIES draws; the best one is kept if none
        # passes. Seeds that drew a wrong face for this photo are remembered and skipped next time. Off:
        # TRELLIS_FACE_CHECK=0. No widget is added on purpose: a widget would change every cached sculpt's key.
        _fc_on = os.environ.get("TRELLIS_FACE_CHECK", "1") == "1"
        _fc_tries = max(1, int(os.environ.get("TRELLIS_FACE_CHECK_TRIES", "4")))
        _photo_lm, _fc_rec = None, None
        if _fc_on:
            try:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("trellis2_face_check", os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_check.py"))
                _fc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_fc)
                _photo_lm = _fc.photo_landmarks(front_image)
                _fc_rec = _fc.load_record(front_image)
                if _photo_lm is None:
                    print("[FaceCheck] no face found on the photo -- the draw is not checked")
            except Exception as _e:
                print("[FaceCheck] unavailable (%r) -- the draw is not checked" % (_e,))
                _photo_lm = None
        _best, _best_dev, _seed_used, _drawn = None, None, None, 0
        for _k in range(1000):
            if _drawn >= (_fc_tries if _photo_lm is not None else 1):
                break
            _seed = seed + _k
            if _photo_lm is not None and str(_seed) in _fc_rec.get("rejected", {}):
                print("[FaceCheck] seed %d drew a wrong face for this photo before -- skipped" % _seed)
                continue
            _drawn += 1
            mesh = self._draw(pipeline, front_pil, back_pil, left_pil, right_pil, _seed, pipeline_type,
                              sparse_structure_sampler_params, shape_slat_sampler_params, tex_slat_sampler_params,
                              max_num_tokens, sparse_structure_resolution, generate_texture_slat, use_tiled_decoder,
                              pbar, front_axis, blend_temperature, sampler, fill_holes, hole_iterations, verbose,
                              dino_lock, dino_substeps, hole_fill_algorithm, dino_foundation_cap, keep_only_shell)
            if _photo_lm is None:
                _best, _seed_used = mesh, _seed
                break
            try:
                _ok, _rep, _ = _fc.check(mesh, _photo_lm)
            except Exception as _e:
                _ok, _rep = None, {"error": repr(_e)}
            _d = _rep.get("deviation", {})
            print("[FaceCheck] seed %d: %s -- mouth %s, nose %s, eyes %s of the photo's (share of the face width; the "
                  "mouth decides, +-%.0f%%)" % (_seed, {True: "PASS", False: "FAIL", None: "NOT MEASURED"}[_ok],
                                                 *("%+.0f%%" % (100 * _d[k]) if k in _d else "?" for k in ("mouth", "nose", "eye")),
                                                 100 * getattr(_fc, "MOUTH_TOL", 0)))
            mesh._face_check = dict(_rep, seed=_seed)
            if _ok is None or _ok:
                _best, _seed_used = mesh, _seed
                break
            _fc_rec.setdefault("rejected", {})[str(_seed)] = _rep
            _fc.save_record(front_image, _fc_rec)
            if _best is None or abs(_d["mouth"]) < _best_dev:
                _best, _best_dev, _seed_used = mesh, abs(_d["mouth"]), _seed
            else:
                del mesh
            torch.cuda.empty_cache()
        else:
            pass
        mesh = _best
        if _photo_lm is not None and getattr(mesh, "_face_check", {}).get("ok") is False:
            mesh._face_check["best_of"] = _drawn
            print("[FaceCheck] *** none of %d draws has her mouth -- keeping the closest (seed %d); raise "
                  "TRELLIS_FACE_CHECK_TRIES to search further ***" % (_drawn, _seed_used))
        if _photo_lm is not None:
            _fc_rec["accepted"] = _seed_used
            _fc.save_record(front_image, _fc_rec)

        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()

        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None

        return (mesh,bvh,)

    def _draw(self, pipeline, front_pil, back_pil, left_pil, right_pil, seed, pipeline_type,
              sparse_structure_sampler_params, shape_slat_sampler_params, tex_slat_sampler_params,
              max_num_tokens, sparse_structure_resolution, generate_texture_slat, use_tiled_decoder,
              pbar, front_axis, blend_temperature, sampler, fill_holes, hole_iterations, verbose,
              dino_lock, dino_substeps, hole_fill_algorithm, dino_foundation_cap, keep_only_shell):
        """One Trellis draw (the call process() always made)."""
        return pipeline.run_multiview(
            front=front_pil,
            back=back_pil,
            left=left_pil,
            right=right_pil,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            max_num_tokens=max_num_tokens,
            sparse_structure_resolution=sparse_structure_resolution,
            generate_texture_slat=generate_texture_slat,
            use_tiled=use_tiled_decoder,
            pbar=pbar,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            sampler=sampler,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell
        )[0]

class Trellis2PostProcessAndUnWrapAndRasterizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
                "texture_size": ("INT",{"default":4096, "min":512, "max":16384}),
                "remesh": ("BOOLEAN",{"default":True}),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "target_face_num": ("INT",{"default":2000000,"min":1,"max":16000000}),
                "simplify_method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"1024"}),
                "double_side_material": ("BOOLEAN",{"default":False}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals":("BOOLEAN",{"default":False}),
                "bvh": ("BVH",),
                "remove_inner_faces": ("BOOLEAN",{"default":True}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength, texture_size, remesh, remesh_band, remesh_project, target_face_num, simplify_method, fill_holes, texture_alpha_mode, dual_contouring_resolution, double_side_material, remove_floaters, bake_on_vertices,use_custom_normals,bvh,remove_inner_faces,inpainting):
        pbar = ProgressBar(5 if not bake_on_vertices else 4)
        mesh_copy = copy.deepcopy(mesh)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size  
        
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)

        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size
        
        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
            
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        
        vertices = vertices.cuda()
        faces = faces.cuda()                
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")        
        _pre_remesh_faces = int(cumesh.num_faces)

        pbar.update(1)
            
        print("Cleaning mesh...")        
        # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
        if not remesh:            
            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num * 3, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())        
            
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            cumesh.remove_small_connected_components(1e-5)
            
            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())
            
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            cumesh.remove_small_connected_components(1e-5)         
            
            print(f"After initial cleanup: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")                            
                
            # Step 2: Unify face orientations
            print('Unifying faces orientation ...')
            cumesh.unify_face_orientations()
        
        # --- Branch 2: Remeshing Pipeline ---
        else:
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            
            if dual_contouring_resolution == "Auto":
                resolution = grid_size.max().item()
                print(f"Dual Contouring resolution: {resolution}")
            else:
                resolution = int(dual_contouring_resolution)
            
            print('Performing Dual Contouring ...')
            # Perform Dual Contouring remeshing (rebuilds topology).
            # BLOW-UP FALLBACK: DC normally lands near 1.5x the input face count and was
            # bit-identical across every single-view bake (2,993,958 -> 4,629,724, three runs).
            # Multi-view conditioning raises the variance: one bake produced 27,039,966 faces
            # from 2,784,426 (9.7x), which then had 8,398 boundary loops, stalled the meshlib
            # filler 20+ min, and came out with vertex coordinates around 1e29 -- geometry
            # numerically destroyed, whole bake wasted. Everything here is decimated to ~135k
            # regardless, so on that path a lower-resolution DC is strictly better than a
            # self-destroying one. Normal bakes never enter the retry.
            _dc_res = resolution
            for _dc_try in range(3):
                cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc_quad(
                    vertices, faces,
                    center = center,
                    scale = scale * 1.1,
                    resolution = _dc_res,
                    band = remesh_band,
                    project_back = remesh_project, # Snaps vertices back to original surface
                    verbose = True,
                    remove_inner_faces = remove_inner_faces,
                    #bvh = bvh,
                ))
                _ratio = cumesh.num_faces / max(_pre_remesh_faces, 1)
                print(f"[Remesh] dual contouring @{_dc_res}: {_pre_remesh_faces:,} -> "
                      f"{cumesh.num_faces:,} faces ({_ratio:.2f}x input)")
                _max_dc = int(os.environ.get("TRELLIS_MAX_DC_FACES", "12000000"))
                if (_ratio <= 4.0 and len(_out_f) <= _max_dc) or _dc_res <= 256:
                    break
                _dc_res = max(256, _dc_res // 2)
                print("=" * 78)
                print(f"[Remesh] *** DUAL CONTOURING BLOW-UP {_ratio:.1f}x input (normal ~1.5x) -- "
                      f"retrying at resolution {_dc_res} ***")
                print("=" * 78)
            if _ratio > 4.0:
                print(f"[Remesh] *** still {_ratio:.1f}x at resolution {_dc_res} -- this bake's "
                      f"geometry is suspect, check it before trusting the result ***")
            
            new_vertices, new_faces = cumesh.read()
            
            if remove_floaters:
                new_vertices, new_faces = remove_floater2(new_vertices.cpu().numpy(),new_faces.cpu().numpy())
                new_vertices = torch.from_numpy(new_vertices).contiguous().float().cuda()
                new_faces = torch.from_numpy(new_faces).contiguous().int().cuda()
                cumesh.init(new_vertices, new_faces)                    
            
            print(f"After remeshing: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            _tw_v, _tw_f = cumesh.read()
            _hole_tripwire("after remesh", _tw_v.cpu().numpy(), _tw_f.cpu().numpy())
            del _tw_v, _tw_f

            if simplify_method == 'Cumesh':
                cumesh.simplify(target_face_num, verbose=True)
            elif simplify_method == 'Meshlib':
                 # GPU -> CPU -> Meshlib -> CPU -> GPU
                v, f = cumesh.read()
                new_vertices, new_faces = simplify_with_meshlib(v.cpu().numpy(), f.cpu().numpy(), target_face_num)
                cumesh.init(torch.from_numpy(new_vertices).float().cuda(), torch.from_numpy(new_faces).int().cuda())

            print(f"After simplifying: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")            
            _tw_v, _tw_f = cumesh.read()
            _hole_tripwire("after simplify", _tw_v.cpu().numpy(), _tw_f.cpu().numpy())
            del _tw_v, _tw_f
            pbar.update(1)
            
        if fill_holes:
            new_vertices, new_faces = cumesh.read()
            meshlib_mesh = mrmeshnumpy.meshFromFacesVerts(new_faces.detach().clone().cpu().numpy(), new_vertices.detach().clone().cpu().numpy())
            hole_edges = meshlib_mesh.topology.findHoleRepresentiveEdges()
            holes_filled = 0
            
            nb_holes = len(hole_edges)
            print(f"{nb_holes} holes found")

            if nb_holes > 0:
                progress_bar = tqdm(total=nb_holes, desc="Filling holes")
                
                last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
                
                _hf_budget = float(os.environ.get("TRELLIS_HOLEFILL_SECONDS", "90"))
                _hf_t0 = _time_mod.time()
                _hf_stopped = 0
                for i, e in enumerate(hole_edges):
                    # Hard time box. 7,065 holes at ~20/s is ~6 minutes, and a mesh with thousands
                    # of holes is broken anyway -- filling every one does not rescue it, it just
                    # makes the bake unaffordable. Fill what fits in the budget, then say so.
                    if _hf_budget > 0 and (_time_mod.time() - _hf_t0) > _hf_budget:
                        _hf_stopped = nb_holes - i
                        print(f"[HoleFill] *** TIME BUDGET {_hf_budget:.0f}s reached after "
                              f"{i}/{nb_holes} holes -- {_hf_stopped} left unfilled. A mesh with "
                              f"this many holes is suspect; raise TRELLIS_HOLEFILL_SECONDS to "
                              f"force it. ***")
                        break
                    params = mrmeshpy.FillHoleParams()
                    params.metric = mrmeshpy.getUniversalMetric(meshlib_mesh)
                    mrmeshpy.fillHole(meshlib_mesh, e, params)
                    
                    # Calculate current progress
                    current_step = i + 1
                    current_percent = int((current_step / nb_holes) * 100)
                    
                    # Only update the UI if the percentage has moved up
                    if current_percent > last_reported_percent:
                        # Calculate how many holes have been filled since the last UI update
                        # This handles cases where 1% might represent multiple holes
                        if last_reported_percent == -1:
                            # First update
                            progress_bar.update(current_step)
                        else:
                            # Update by the difference since the last check
                            last_step = int((last_reported_percent * nb_holes) / 100)
                            diff = current_step - last_step
                            progress_bar.update(diff)
                        
                        last_reported_percent = current_percent
                            
                progress_bar.close()                 
            
            new_vertices = mrmeshnumpy.getNumpyVerts(meshlib_mesh)
            new_faces = mrmeshnumpy.getNumpyFaces(meshlib_mesh.topology)

            del meshlib_mesh
            gc.collect()
            
            cumesh.init(torch.from_numpy(new_vertices).float().to(coords.device), torch.from_numpy(new_faces).int().to(coords.device))
        
        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            print('Baking colors on vertices...')
            out_vertices, out_faces = cumesh.read()
            out_vertices = out_vertices.cuda()
            out_faces = out_faces.cuda()
            cumesh.compute_vertex_normals()
            out_normals = cumesh.read_vertex_normals()
            
            # Map vertex positions back to original mesh for accurate attribute sampling
            # Use BVH to find the closest point on original mesh surface for more accurate colors
            _, face_id, uvw = bvh.unsigned_distance(out_vertices, return_uvw=True)
            orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]]  # (N_verts, 3, 3)
            mapped_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
            
            # Sample attributes at mapped positions from the voxel grid
            vertex_attrs = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=_safe_grid(mapped_pos, aabb, voxel_size, grid_size, "vertex bake (mapped)"),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = attr_layout['base_color']
            alpha_idx = attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            pbar.update(1)
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals.cpu().numpy()
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = Trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            del cumesh
            gc.collect()
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
            return (textured_mesh, placeholder_texture, placeholder_texture,)
        
        # --- Standard texture baking path ---
        _CHART_GUARD.update(budget=(texture_size * texture_size) // _CHART_MIN_TEXELS,
                            texture_size=texture_size)
        print('Unwrapping ...')        
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )
        pbar.update(1)
        
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()
        out_vmaps = out_vmaps.cuda()
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps]        

        print("Sampling attributes...")
        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        
        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)
        
        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0
        
        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        
        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = bvh.vertices[bvh.faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)        
        
        torch.cuda.synchronize()
        
        # Trilinear sampling from the attribute volume (Color, Material props)
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=_safe_grid(valid_pos, aabb, voxel_size, grid_size, "texture bake"),
            mode='trilinear',
        )
        
        # --- Texture Post-Processing & Material Construction ---
        print("Finalizing mesh...")
        pbar.update(1)
        
        mask = mask.cpu().numpy()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = texture_alpha_mode
        
        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
        if inpainting == 'telea':
            inpainting = cv2.INPAINT_TELEA
        else:
            inpainting = cv2.INPAINT_NS
        
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, inpainting)
        metallic = cv2.inpaint(metallic, mask_inv, 1, inpainting)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, inpainting)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, inpainting)[..., None]
        
        # Create PBR material
        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        baseColorTexture_np = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture_np = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture_np,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture_np,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=double_side_material,
        )        
        
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
        else:
            textured_mesh = Trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False,
                visual=Trimesh.visual.TextureVisuals(uv=uvs_np,material=material)
            )
            
        pbar.update(1)        
        
        del cumesh
        gc.collect()         

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture,)    

class Trellis2Remesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, remesh_project, dual_contouring_resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size        
        
        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device='cuda')

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device='cuda')
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")        
        
        vertices, faces = cumesh.read()
        
        del cumesh
        gc.collect()         
            
        # Build BVH for the current mesh to guide remeshing
        #print(f"Building BVH for current mesh...")
        #bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())
            
        print("Cleaning mesh...")        
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        if dual_contouring_resolution == "Auto":
            resolution = grid_size.max().item()
            print(f"Dual Contouring resolution: {resolution}")
        else:
            resolution = int(dual_contouring_resolution)
        
        print('Performing Dual Contouring ...')
        # Perform Dual Contouring remeshing (rebuilds topology).
        # BLOW-UP FALLBACK: DC normally lands near 1.5x the input face count and was
        # bit-identical across every single-view bake (2,993,958 -> 4,629,724, three runs).
        # Multi-view conditioning raises the variance: bakes produced 27,039,966 faces from
        # 2,784,426 (9.7x) and 23,367,372 from 2,349,148 (9.9x). That path then had thousands
        # of boundary loops, stalled the hole filler for 20+ min, and finished with vertex
        # coordinates around 1e29 -- geometry numerically destroyed, whole bake wasted.
        # Everything here is decimated to ~135k regardless, so on that path a lower-resolution
        # DC is strictly better than a self-destroying one. Normal bakes never retry.
        # Keep the ORIGINAL input so a retry re-remeshes the source, not the blown-up result.
        # The failed attempt MUST be released first: holding the input + a 23M-face result +
        # the new allocation at once OOMs a 16GB card (measured: 15.63 GiB held, 10.16 GiB
        # requested, 15.92 GiB limit). One attempt at a time fits; two do not.
        _in_v, _in_f = vertices, faces
        _in_faces = int(len(faces))
        _dc_res = resolution
        _ratio = 0.0
        _out_v = _out_f = None
        for _dc_try in range(3):
            if _out_v is not None:                     # drop the previous attempt before re-allocating
                del _out_v, _out_f
                _out_v = _out_f = None
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _out_v, _out_f = CuMesh.remeshing.remesh_narrow_band_dc(
                _in_v, _in_f,
                center = center,
                scale = scale * 1.1,
                resolution = _dc_res,
                band = remesh_band,
                project_back = remesh_project, # Snaps vertices back to original surface
                verbose = True,
                # remove_inner_faces not supported in this CuMesh version
                #bvh = bvh,
            )
            _ratio = len(_out_f) / max(_in_faces, 1)
            print(f"[Remesh] dual contouring @{_dc_res}: {_in_faces:,} -> {len(_out_f):,} faces "
                  f"({_ratio:.2f}x input)")
            # Ratio alone is the wrong guard: 2.34x of a 2.8M-face input is 6.5M faces, which
            # sails past a 4x check and then costs ~6 min of hole filling -- for a mesh that gets
            # decimated to ~150k regardless. Bound the ABSOLUTE count too.
            _max_dc = int(os.environ.get("TRELLIS_MAX_DC_FACES", "12000000"))
            _dc_ok, _dc_why = _dc_sane(_out_v, scale)
            print(f"[Remesh] coordinate check: {_dc_why}")
            if not _dc_ok:
                # A destroyed draw, not an over-dense one. This DC is nondeterministic,
                # so the answer is another draw at the SAME resolution -- halving it
                # would throw away detail to fix something that is not a resolution
                # problem.
                print("=" * 78)
                print(f"[Remesh] *** DUAL CONTOURING PRODUCED BROKEN GEOMETRY ({_dc_why}) -- "
                      f"redrawing at the same resolution {_dc_res} ***")
                print("=" * 78)
                continue
            if (_ratio <= 4.0 and len(_out_f) <= _max_dc) or _dc_res <= 256:
                if len(_out_f) > _max_dc:
                    print(f"[Remesh] *** {len(_out_f):,} faces exceeds the {_max_dc:,} budget but "
                          f"resolution is already {_dc_res} -- proceeding, expect a slow bake ***")
                break
            if len(_out_f) > _max_dc and _ratio <= 4.0:
                print(f"[Remesh] *** {len(_out_f):,} faces over the {_max_dc:,} budget "
                      f"({_ratio:.2f}x input) -- retrying lower rather than paying for it ***")
            _dc_res = max(256, _dc_res // 2)
            print("=" * 78)
            print(f"[Remesh] *** DUAL CONTOURING BLOW-UP {_ratio:.1f}x input (normal ~1.5x) -- "
                  f"retrying at resolution {_dc_res} ***")
            print("=" * 78)
        _dc_ok, _dc_why = _dc_sane(_out_v, scale)
        if not _dc_ok:
            raise RuntimeError(
                f"Dual contouring produced unusable geometry on every attempt ({_dc_why}). "
                f"This is the nondeterministic DC draw, not a setting -- re-queue and it will "
                f"very likely succeed. Failing here costs minutes; letting it through aborts "
                f"the ComfyUI process at the texture bake and costs the whole run.")
        vertices, faces = _out_v, _out_f
        del _in_v, _in_f, _out_v, _out_f
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if _ratio > 4.0:
            print(f"[Remesh] *** still {_ratio:.1f}x at resolution {_dc_res} -- geometry is "
                  f"suspect, check this bake before trusting it ***")

        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int() 
            
        print(f"After remeshing: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)
        
class Trellis2ReconstructMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "resolution": ([128,256,512,1024,2048],{"default":512}),             
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, resolution):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        vertices = mesh_copy.vertices.cuda()
        faces = mesh_copy.faces.cuda()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        print('Reconstructing mesh ...')
        vertices, faces = CuMesh.remeshing.reconstruct_mesh_dc(vertices, faces, resolution, verbose=True)
        
        print(f"After reconstruction: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)   

class Trellis2ReconstructMeshWithQuad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "resolution": ([128,256,512,1024,2048],{"default":512}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":False}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        vertices = mesh_copy.vertices.cuda()
        faces = mesh_copy.faces.cuda()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        print('Reconstructing mesh ...')
        vertices, faces = CuMesh.remeshing.reconstruct_mesh_dc_quad(vertices, faces, resolution, verbose=True, remove_inner_faces = remove_inner_faces)
        
        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int()         
        
        print(f"After reconstruction: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)         
        
class Trellis2MeshTexturing:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}), 
                "resolution": ([512,1024,1536],{"default":1024}),
                "texture_size": ("INT",{"default":4096,"min":512,"max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}), 
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),                
            },
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, trimesh, seed, texture_steps, texture_guidance_strength, texture_guidance_rescale, texture_rescale_t, resolution, texture_size, texture_alpha_mode, double_side_material, texture_guidance_interval_start, texture_guidance_interval_end, max_views,bake_on_vertices,use_custom_normals,mesh_cluster_threshold_cone_half_angle_rad, sampler, inpainting,
        verbose, dino_lock, dino_substeps, dino_foundation_cap):
            
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images

        #image = tensor2pil(image)
        
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]                
        
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}

        textured_mesh, baseColorTexture_np, metallicRoughnessTexture_np = pipeline.texture_mesh(mesh=trimesh, 
            image=image_in, 
            seed=seed, 
            tex_slat_sampler_params = tex_slat_sampler_params,
            resolution = resolution,
            texture_size = texture_size,
            texture_alpha_mode = texture_alpha_mode,
            double_side_material = double_side_material,
            max_views = max_views,
            bake_on_vertices = bake_on_vertices,
            use_custom_normals = use_custom_normals,
            mesh_cluster_threshold_cone_half_angle_rad = mesh_cluster_threshold_cone_half_angle_rad,
            sampler = sampler,
            inpainting = inpainting,
            verbose = verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap
        )            

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )
        
class Trellis2MeshTexturingMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}), 
                "resolution": ([512,1024,1536],{"default":1024}),
                "texture_size": ("INT",{"default":4096,"min":512,"max":16384}),
                "texture_alpha_mode": (["OPAQUE","MASK","BLEND"],{"default":"OPAQUE"}),
                "double_side_material": ("BOOLEAN",{"default":False}), 
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "bake_on_vertices": ("BOOLEAN",{"default":False}),
                "use_custom_normals": ("BOOLEAN",{"default":False}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "inpainting": (["telea","ns"],{"default":"telea"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}), 
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),                
            }
        }

    RETURN_TYPES = ("TRIMESH","IMAGE","IMAGE",)
    RETURN_NAMES = ("trimesh","base_color_texture","metallic_roughness_texture",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        pipeline, 
        front_image, 
        trimesh, 
        seed, 
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale, 
        texture_rescale_t, 
        resolution, 
        texture_size, 
        texture_alpha_mode, 
        double_side_material, 
        texture_guidance_interval_start, 
        texture_guidance_interval_end, 
        bake_on_vertices,
        use_custom_normals,
        mesh_cluster_threshold_cone_half_angle_rad,
        front_axis,
        blend_temperature,
        sampler,
        inpainting,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        back_image = None,
        left_image = None,
        right_image = None):
        
        reset_cuda()
        
        # Convert front image tensor to PIL
        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None        
        
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]                
        
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}

        textured_mesh, baseColorTexture_np, metallicRoughnessTexture_np = pipeline.texture_mesh_multiview(mesh=trimesh, 
            front=front_pil,
            back=back_pil,
            left=left_pil,
            right=right_pil,
            seed=seed, 
            tex_slat_sampler_params = tex_slat_sampler_params,
            resolution = resolution,
            texture_size = texture_size,
            texture_alpha_mode = texture_alpha_mode,
            double_side_material = double_side_material,
            bake_on_vertices = bake_on_vertices,
            use_custom_normals = use_custom_normals,
            mesh_cluster_threshold_cone_half_angle_rad = mesh_cluster_threshold_cone_half_angle_rad,
            front_axis = front_axis,
            blend_temperature = blend_temperature,
            sampler = sampler,
            inpainting = inpainting,
            verbose = verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap
        )            

        baseColorTexture = pil2tensor(baseColorTexture_np)
        metallicRoughnessTexture = pil2tensor(metallicRoughnessTexture_np)
        
        return (textured_mesh, baseColorTexture, metallicRoughnessTexture, )        
        
class Trellis2LoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):
        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        trimesh = Trimesh.load(glb_path, force="mesh")
        
        return (trimesh,)  
        
class Trellis2PreProcessImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT",{"default":0,"min":0,"max":1024}),
                "remove_background": ("BOOLEAN",{"default":False}),
                "max_size": ("INT",{"default":2048,"min":512,"max":8192,"step":128}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, image, padding, remove_background, max_size):
        try:   # save the RAW front photo for the in-bake hairline face mask (== offline; BEFORE bg-removal/preprocess,
               # so it matches what _gs_facemask was verified against). Read during the SEAM step. Fresh per character.
            tensor_batch_to_pil_list(image)[0].convert('RGB').save(
                r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_front_photo.png')
        except Exception as _spe:
            print(f"[hairline] could not save _front_photo.png ({_spe!r})")
        if image.ndim == 3:
            image = tensor2pil(image)
            
            if remove_background:
                from rembg import remove
                image = remove(image)
            
            image = self.preprocess_image(image, max_size)
            
            if padding>0:
                border = (int(padding), int(padding), int(padding), int(padding))
                fill_color = self.parse_fill_for_image("0,0,0,255", image)
                image = ImageOps.expand(image,border=border,fill=fill_color)
            
            image = pil2tensor(image)
        elif image.ndim == 4:
            images = convert_tensor_images_to_pil(image)
            tensor_list = []
            for img in images:
                if remove_background:
                    from rembg import remove
                    img = remove(img)
                
                img = self.preprocess_image(img, max_size)
                
                if padding>0:
                    border = (int(padding), int(padding), int(padding), int(padding))
                    fill_color = self.parse_fill_for_image("0,0,0,255", img)
                    img = ImageOps.expand(img,border=border,fill=fill_color)
                
                tensor_list.append(pil2tensor(img))
                
                max_h = max(t.shape[-3] for t in tensor_list)
                max_w = max(t.shape[-2] for t in tensor_list)

                resized_tensors = []

                for t in tensor_list:
                    # Ensure tensor is [C, H, W] for PyTorch's interpolate function
                    # Current shape is likely [H, W, C] or [1, H, W, C]
                    temp_t = t.squeeze() # Get to [H, W, C]
                    temp_t = temp_t.permute(2, 0, 1).unsqueeze(0) # Becomes [1, C, H, W]
                    
                    # 2. Resize to the max dimensions
                    # Using 'bicubic' or 'bilinear' for better quality than 'nearest'
                    temp_t = F.interpolate(temp_t, size=(max_h, max_w), mode='bicubic', align_corners=False)
                    
                    # 3. Convert back to ComfyUI format [H, W, C]
                    temp_t = temp_t.squeeze(0).permute(1, 2, 0)
                    resized_tensors.append(temp_t)                
                
            image = torch.stack(resized_tensors)
        
        return (image,)    

    def parse_fill_for_image(self, fill: str, img):
        values = [int(x.strip()) for x in fill.split(",")]

        if img.mode in ("L", "P"):
            return values[0]

        if img.mode == "RGB":
            return tuple(values[:3])

        if img.mode == "RGBA":
            return tuple(values[:4])

        raise ValueError(f"Unsupported image mode: {img.mode}")         


    def preprocess_image(self, input: Image.Image, max_res) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, max_res / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        # if has_alpha:
            # output = input
        # else:
            # input = input.convert('RGB')
            # if self.low_vram:
                # self.rembg_model.to(self.device)
            # output = self.rembg_model(input)
            # if self.low_vram:
                # self.rembg_model.cpu()
        output = input
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output    

class Trellis2MeshRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "trimesh": ("TRIMESH",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "resolution": ([512,1024,1536],{"default":1024}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                  
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":3.00,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.20,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":3.00,"min":0.00,"max":9.99,"step":0.01}),               
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "generate_texture_slat": ("BOOLEAN", {"default":True}),
                "downsampling":([16,32,64],{"default":16}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, trimesh, image, seed, resolution,
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,        
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        max_num_tokens,
        generate_texture_slat,
        downsampling,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        use_tiled_decoder,
        max_views,
        sampler,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap):

        reset_cuda()

        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images
        
        shape_guidance_interval = [shape_guidance_interval_start,shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]        
        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        mesh = pipeline.refine_mesh(mesh = trimesh, 
                                    image=image_in, 
                                    seed=seed, 
                                    shape_slat_sampler_params = shape_slat_sampler_params, 
                                    tex_slat_sampler_params = tex_slat_sampler_params, 
                                    resolution = resolution, 
                                    max_num_tokens = max_num_tokens, 
                                    generate_texture_slat=generate_texture_slat, 
                                    downsampling=downsampling, 
                                    use_tiled=use_tiled_decoder, 
                                    max_views = max_views, 
                                    sampler = sampler,
                                    verbose = verbose,
                                    dino_lock = dino_lock,
                                    dino_substeps = dino_substeps,
                                    dino_foundation_cap = dino_foundation_cap)[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()        
       
        # Build BVH for the current mesh to guide remeshing
        if generate_texture_slat:
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print('Not building BVH, only used for texturing')
            bvh = None
        
        return (mesh, bvh,)        

class Trellis2PostProcess2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "fill_holes": ("BOOLEAN", {"default":True}),
                "fix_normals": ("BOOLEAN", {"default":False}),
                "fix_face_orientation": ("BOOLEAN", {"default":True}),
                "remove_duplicate_faces": ("BOOLEAN",{"default":True}),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "weld_vertices_digits": ("INT",{"default":4,"min":1,"max":8}),
                "smooth": ("BOOLEAN",{"default":False}),
                "smooth_iterations": ("INT",{"default":10,"min":1,"max":99,"step":1}),
                "subdivide": ("BOOLEAN",{"default":False}),
                "subdivide_iterations": ("INT",{"default":1,"min":1,"max":10}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, fill_holes, fix_normals, fix_face_orientation, remove_duplicate_faces, weld_vertices, weld_vertices_digits,smooth,smooth_iterations,subdivide,subdivide_iterations):
        mesh_copy = copy.deepcopy(mesh)
        
        vertices_np = mesh_copy.vertices.cpu().numpy()
        faces_np = mesh_copy.faces.cpu().numpy()
        
        trimesh = Trimesh.Trimesh(vertices=vertices_np,faces=faces_np)
        
        print(f"Initial mesh: {len(trimesh.faces)} faces")
        #print(f"Is winding consistent? {trimesh.is_winding_consistent}")        
        
        if fix_normals:
            print('Fixing normals ...')
            trimesh.fix_normals()       
            
        if fix_face_orientation:
            if trimesh.is_watertight:
                print('Mesh is watertight, fixing inversion ...')
                Trimesh.repair.fix_inversion(trimesh)
            else:
                print('Mesh is not watertight, cannot fix inversion')

        if remove_duplicate_faces:
            print('Removing duplicate faces ...')
            trimesh.update_faces(trimesh.unique_faces()) 
        
        if fill_holes:
            print('Filling holes ...')
            trimesh.fill_holes()     

        if weld_vertices:
            vertices_count = len(trimesh.vertices)
            trimesh.merge_vertices(digits_vertex=weld_vertices_digits)
            new_vertices_count = len(trimesh.vertices)
            nb_vertices_removed = vertices_count - new_vertices_count            
            faces_count = len(trimesh.faces)
            trimesh.remove_unreferenced_vertices()
            trimesh.update_faces(trimesh.nondegenerate_faces())
            new_faces_count = len(trimesh.faces)
            nb_faces_removed = faces_count - new_faces_count
            print(f"Weld Vertices: Removed {nb_vertices_removed} vertices / {nb_faces_removed} faces")
        
        if smooth:
            print('Smoothing ...')
            Trimesh.smoothing.filter_taubin(trimesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
            
        if subdivide:
            print('Subdividing ...')
            trimesh = trimesh.subdivide_loop(iterations=subdivide_iterations)
        
        new_vertices = torch.from_numpy(trimesh.vertices).float()
        new_faces = torch.from_numpy(trimesh.faces).int()

        print(f"After postprocessing: {len(new_faces)} faces")
        
        mesh_copy.vertices = new_vertices.to(mesh_copy.device)
        mesh_copy.faces = new_faces.to(mesh_copy.device) 
        
        del trimesh
        gc.collect()
                
        return (mesh_copy,)    

class Trellis2OvoxelExportToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "resolution": ([512,1024],{"default":1024}),
                "texture_size": ([512,1024,2048,4096],{"default":2048}),
                "target_face_num": ("INT",{"default":2000000,"min":500,"max":16000000}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, resolution, texture_size, target_face_num):
        mesh_copy = copy.deepcopy(mesh)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_copy.vertices,
            faces=mesh_copy.faces,
            attr_volume=mesh_copy.attrs,
            coords=mesh_copy.coords,
            attr_layout=mesh_copy.layout,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=target_face_num,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
                
        return (glb,)

class Trellis2TrimeshToMeshWithVoxel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "resolution": ([512,1024],{"default":1024}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, resolution):       
        mesh_copy = trimesh.copy()
        
        mvoxel = self.get_voxelmesh_from_trimesh(mesh_copy, resolution)
        
        return (mvoxel,)        
        
    def get_voxelmesh_from_trimesh(self, mesh, resolution):
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        
        coords = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)                
        coords = coords.cpu()

        del voxel_indices
        del dual_vertices
        del intersected
        gc.collect()
            
        pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

        mvoxel = MeshWithVoxel(
                    vertices, faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = coords,
                    attrs = None,
                    voxel_shape = None,
                    layout=pbr_attr_layout
                    )
                    
        return mvoxel
        
class Trellis2Continue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
            },
        }

    RETURN_TYPES = (any, any, )
    RETURN_NAMES = ("output_1", "output_2", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2):        
        return (input_1, input_2,)
        
class Trellis2Continue3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
            },
        }

    RETURN_TYPES = (any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3):        
        return (input_1, input_2, input_3)      

class Trellis2Continue4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4):        
        return (input_1, input_2, input_3, input_4)    

class Trellis2Continue5:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
                "input_5": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any, any,)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4, input_5):        
        return (input_1, input_2, input_3, input_4, input_5)   

class Trellis2Continue6:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
                "input_5": (any,),
                "input_6": (any,),
            },
        }

    RETURN_TYPES = (any, any, any, any, any, any)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5", "output_6")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1, input_2, input_3, input_4, input_5, input_6):        
        return (input_1, input_2, input_3, input_4, input_5, input_6)         
        
class Trellis2MeshWithVoxelToMeshlibMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }

    RETURN_TYPES = ("MESHLIB_MESH", )
    RETURN_NAMES = ("meshlib_mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh):        
        meshlib_mesh = mrmeshnumpy.meshFromFacesVerts(mesh.faces.cpu().numpy(), mesh.vertices.cpu().numpy())                                 
        return (meshlib_mesh,)

class Trellis2FillHolesWithMeshlib:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL", "INT")
    RETURN_NAMES = ("mesh", "holes_filled")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Fill all holes in a mesh using optimal triangulation."

    def process(self, mesh):
        import meshlib.mrmeshpy as mrmeshpy
        
        mesh_copy = copy.deepcopy(mesh)
        mesh = mrmeshnumpy.meshFromFacesVerts(mesh_copy.faces.detach().clone().cpu().numpy(), mesh_copy.vertices.detach().clone().cpu().numpy())
        
        hole_edges = mesh.topology.findHoleRepresentiveEdges()
        holes_filled = 0
        
        nb_holes = len(hole_edges)
        print(f"{nb_holes} holes found")

        # A hole is a LOCAL defect. A boundary loop bigger around than the object itself is not
        # a hole, it is the surface being open, and triangulating one destroys the bake: measured
        # 2026-09-10 on a dual-contouring blow-up, loops with a median perimeter of 0.0046 on a
        # body about 1 unit across and a single loop at 2.7893. Filling it took the mesh from
        # 4,302,582 to 9,243,137 faces and left vertex coordinates at 2.8e11.
        #
        # The bound is the mesh's own bounding-box diagonal -- nothing to tune, and the worst
        # hole any healthy bake produced was 0.1912 against a diagonal near 1.5.
        _hf_per = []
        _hf_diag = 0.0
        _hf_skip = set()
        try:
            _hf_V = mesh_copy.vertices.detach().cpu().numpy()
            _hf_diag = float(np.linalg.norm(_hf_V.max(axis=0) - _hf_V.min(axis=0)))
            _hf_per = [float(mrmeshpy.holePerimeter(mesh.topology, mesh.points, _e))
                       for _e in hole_edges]
            _hf_skip = {_i for _i, _p in enumerate(_hf_per) if _p > _hf_diag}
            if _hf_skip:
                print(f"[HoleFill] *** skipping {len(_hf_skip)} loop(s) bigger around than the "
                      f"mesh itself (bbox diagonal {_hf_diag:.4f}, largest loop "
                      f"{max(_hf_per):.4f}) -- open boundary, not holes; filling them wrecks "
                      f"the geometry ***")
        except Exception as _hfe:
            print(f"[HoleFill] hole-size guard unavailable ({_hfe!r}) - filling everything")
            _hf_skip = set()

        if nb_holes > 0:
            progress_bar = tqdm(total=nb_holes, desc="Filling holes")
            _hf_budget = float(os.environ.get("TRELLIS_HOLEFILL_SECONDS", "90"))
            _hf_t0 = _time_mod.time()
            pbar = ProgressBar(nb_holes)
            
            last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
            
            for i, e in enumerate(hole_edges):                
                if i in _hf_skip:          # bigger around than the whole mesh
                    continue
                # Hard time box: thousands of holes means the mesh is broken, and
                # filling them one at a time costs minutes without rescuing it.
                if _hf_budget > 0 and (_time_mod.time() - _hf_t0) > _hf_budget:
                    print("[HoleFill] *** TIME BUDGET %.0fs reached after %d/%d holes"
                          " -- rest left unfilled (TRELLIS_HOLEFILL_SECONDS to raise)"
                          % (_hf_budget, i, nb_holes))
                    break
                params = mrmeshpy.FillHoleParams()
                params.metric = mrmeshpy.getUniversalMetric(mesh)
                mrmeshpy.fillHole(mesh, e, params)
                
                # Calculate current progress
                current_step = i + 1
                current_percent = int((current_step / nb_holes) * 100)
                
                # Only update the UI if the percentage has moved up
                if current_percent > last_reported_percent:
                    # Calculate how many holes have been filled since the last UI update
                    # This handles cases where 1% might represent multiple holes
                    if last_reported_percent == -1:
                        # First update
                        progress_bar.update(current_step)
                        pbar.update(current_step)
                    else:
                        # Update by the difference since the last check
                        last_step = int((last_reported_percent * nb_holes) / 100)
                        diff = current_step - last_step
                        progress_bar.update(diff)
                        pbar.update(diff)
                    
                    last_reported_percent = current_percent
                        
            progress_bar.close()            
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

        del mesh
        gc.collect()
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy, holes_filled) 
        
class Trellis2FillHolesNicelyWithMeshlib:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL", "INT")
    RETURN_NAMES = ("mesh", "holes_filled")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    DESCRIPTION = "Fill all holes in a mesh using optimal triangulation."

    def process(self, mesh):
        import meshlib.mrmeshpy as mrmeshpy
        
        mesh_copy = copy.deepcopy(mesh)
        mesh = mrmeshnumpy.meshFromFacesVerts(mesh_copy.faces.detach().clone().cpu().numpy(), mesh_copy.vertices.detach().clone().cpu().numpy())
        
        hole_edges = mesh.topology.findHoleRepresentiveEdges()
        holes_filled = 0
        
        nb_holes = len(hole_edges)
        print(f"{nb_holes} holes found")

        # A hole is a LOCAL defect. A boundary loop bigger around than the object itself is not
        # a hole, it is the surface being open, and triangulating one destroys the bake: measured
        # 2026-09-10 on a dual-contouring blow-up, loops with a median perimeter of 0.0046 on a
        # body about 1 unit across and a single loop at 2.7893. Filling it took the mesh from
        # 4,302,582 to 9,243,137 faces and left vertex coordinates at 2.8e11.
        #
        # The bound is the mesh's own bounding-box diagonal -- nothing to tune, and the worst
        # hole any healthy bake produced was 0.1912 against a diagonal near 1.5.
        _hf_per = []
        _hf_diag = 0.0
        _hf_skip = set()
        try:
            _hf_V = mesh_copy.vertices.detach().cpu().numpy()
            _hf_diag = float(np.linalg.norm(_hf_V.max(axis=0) - _hf_V.min(axis=0)))
            _hf_per = [float(mrmeshpy.holePerimeter(mesh.topology, mesh.points, _e))
                       for _e in hole_edges]
            _hf_skip = {_i for _i, _p in enumerate(_hf_per) if _p > _hf_diag}
            if _hf_skip:
                print(f"[HoleFill] *** skipping {len(_hf_skip)} loop(s) bigger around than the "
                      f"mesh itself (bbox diagonal {_hf_diag:.4f}, largest loop "
                      f"{max(_hf_per):.4f}) -- open boundary, not holes; filling them wrecks "
                      f"the geometry ***")
        except Exception as _hfe:
            print(f"[HoleFill] hole-size guard unavailable ({_hfe!r}) - filling everything")
            _hf_skip = set()

        if nb_holes > 0:
            progress_bar = tqdm(total=nb_holes, desc="Filling holes")
            _hf_budget = float(os.environ.get("TRELLIS_HOLEFILL_SECONDS", "90"))
            _hf_t0 = _time_mod.time()
            pbar = ProgressBar(nb_holes)
            
            last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
            
            for i, e in enumerate(hole_edges):          
                if i in _hf_skip:          # bigger around than the whole mesh
                    continue
                # Hard time box: thousands of holes means the mesh is broken, and
                # filling them one at a time costs minutes without rescuing it.
                if _hf_budget > 0 and (_time_mod.time() - _hf_t0) > _hf_budget:
                    print("[HoleFill] *** TIME BUDGET %.0fs reached after %d/%d holes"
                          " -- rest left unfilled (TRELLIS_HOLEFILL_SECONDS to raise)"
                          % (_hf_budget, i, nb_holes))
                    break
                params = mrmeshpy.FillHoleNicelySettings()
                params.triangulateParams.metric = mrmeshpy.getMinAreaMetric(mesh)
                params.smoothCurvature = False
                #  Fill hole represented by `e`
                mrmeshpy.fillHoleNicely(mesh, e, params)
                
                # Calculate current progress
                current_step = i + 1
                current_percent = int((current_step / nb_holes) * 100)
                
                # Only update the UI if the percentage has moved up
                if current_percent > last_reported_percent:
                    # Calculate how many holes have been filled since the last UI update
                    # This handles cases where 1% might represent multiple holes
                    if last_reported_percent == -1:
                        # First update
                        progress_bar.update(current_step)
                        pbar.update(current_step)
                    else:
                        # Update by the difference since the last check
                        last_step = int((last_reported_percent * nb_holes) / 100)
                        diff = current_step - last_step
                        progress_bar.update(diff)
                        pbar.update(diff)
                    
                    last_reported_percent = current_percent
                        
            progress_bar.close()            
        
        new_vertices = mrmeshnumpy.getNumpyVerts(mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

        del mesh
        gc.collect()
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy, holes_filled)         
        
class Trellis2SmoothNormals:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, trimesh):
        new_mesh = trimesh.copy()
        new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)
        
        return (new_mesh,)         

class Trellis2RemeshWithQuad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "remesh_band": ("FLOAT",{"default":1.0}),
                "remesh_project": ("FLOAT",{"default":0.0}),
                "dual_contouring_resolution": (["Auto","128","256","512","1024","2048"],{"default":"Auto"}),
                "remove_floaters": ("BOOLEAN",{"default":True}),
                "remove_inner_faces": ("BOOLEAN",{"default":True})
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, remesh_band, remesh_project, dual_contouring_resolution, remove_floaters, remove_inner_faces):
        reset_cuda()
        
        mesh_copy = copy.deepcopy(mesh)
        
        if remove_floaters:
            mesh_copy = remove_floater(mesh_copy)
        
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        attr_volume = mesh_copy.attrs
        coords = mesh_copy.coords
        attr_layout = mesh_copy.layout
        voxel_size = mesh_copy.voxel_size        
        
        # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device='cuda')

        # Calculate grid dimensions based on AABB and voxel size                
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device='cuda')
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Move data to GPU
        vertices = vertices.cuda()
        faces = faces.cuda()
        
        # Initialize CUDA mesh handler
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        print(f"Current vertices: {cumesh.num_vertices}, faces: {cumesh.num_faces}")
        
        vertices, faces = cumesh.read()
        
        del cumesh
        gc.collect()         
            
        # Build BVH for the current mesh to guide remeshing
        #print(f"Building BVH for current mesh...")
        #bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())
            
        print("Cleaning mesh...")        
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        if dual_contouring_resolution == "Auto":
            resolution = grid_size.max().item()
            print(f"Dual Contouring resolution: {resolution}")
        else:
            resolution = int(dual_contouring_resolution)
        
        print('Performing Dual Contouring ...')
        # Perform Dual Contouring remeshing (rebuilds topology)
        vertices, faces = CuMesh.remeshing.remesh_narrow_band_dc_quad(
            vertices, faces,
            center = center,
            scale = scale * 1.1, # old calculation (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = True,
            remove_inner_faces = remove_inner_faces,
            #bvh = bvh,
        )
        
        if remove_floaters:
            vertices, faces = remove_floater2(vertices.cpu().numpy(),faces.cpu().numpy())
            vertices = torch.from_numpy(vertices).contiguous().float()
            faces = torch.from_numpy(faces).contiguous().int() 
            
        print(f"After remeshing: {len(vertices)} vertices, {len(faces)} faces")                                 
        
        mesh_copy.vertices = vertices.to(mesh_copy.device)
        mesh_copy.faces = faces.to(mesh_copy.device) 
                
        return (mesh_copy,)   

class Trellis2BatchSimplifyMeshAndExport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("STRING",{"default":"2000000,1000000,500000,100000,50000,10000,5000,2500,1000"}),
                "method": (["Cumesh","Meshlib"],{"default":"Cumesh"}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "reorient_vertices":(["None","90 degrees","-90 degrees"],{"default":"90 degrees"}),
                "filename_prefix":("STRING",),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "weld_vertices_digits":("INT",{"default":4,"min":1,"max":8}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("lst_glb_path", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method, fill_holes, reorient_vertices, filename_prefix, file_format, weld_vertices, weld_vertices_digits):
        lst_output_mesh = []
        list_of_faces = parse_string_to_int_list(target_face_num)
        if len(list_of_faces)>0:
            cumesh = CuMesh.CuMesh()
            mesh_copy = copy.deepcopy(mesh)
            
            for target_nbfaces in list_of_faces:
                print(f"Processing at {target_nbfaces} ...")                
                
                vertices = mesh_copy.vertices.detach().clone().cpu().numpy()
                faces = mesh_copy.faces.detach().clone().cpu().numpy()                                
                
                if method=="Cumesh":
                    cumesh.init(torch.from_numpy(vertices).float().cuda(), torch.from_numpy(faces).int().cuda())
                    # options = {
                        # 'method': 'legacy'
                    # }                       
                    cumesh.simplify(target_nbfaces, verbose=True)
                    vertices, faces = cumesh.read()
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                elif method=="Meshlib":
                    vertices, faces = simplify_with_meshlib(vertices, faces, target_nbfaces)
                else:
                    raise Exception("Unknown simplification method")
                
                if fill_holes:
                    import meshlib.mrmeshpy as mrmeshpy

                    mmesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
                    
                    hole_edges = mmesh.topology.findHoleRepresentiveEdges()
                    
                    nb_holes = len(hole_edges)
                    print(f"{nb_holes} holes found")

                    if nb_holes > 0:
                        progress_bar = tqdm(total=nb_holes, desc="Filling holes")
                        _hf_budget = float(os.environ.get("TRELLIS_HOLEFILL_SECONDS", "90"))
                        _hf_t0 = _time_mod.time()
                        
                        last_reported_percent = -1  # Initialize at -1 to ensure 0% triggers an update
                        
                        for i, e in enumerate(hole_edges):
                            # Hard time box: thousands of holes means the mesh is broken, and
                            # filling them one at a time costs minutes without rescuing it.
                            if _hf_budget > 0 and (_time_mod.time() - _hf_t0) > _hf_budget:
                                print("[HoleFill] *** TIME BUDGET %.0fs reached after %d/%d holes"
                                      " -- rest left unfilled (TRELLIS_HOLEFILL_SECONDS to raise)"
                                      % (_hf_budget, i, nb_holes))
                                break
                            params = mrmeshpy.FillHoleParams()
                            params.metric = mrmeshpy.getUniversalMetric(mmesh)
                            mrmeshpy.fillHole(mmesh, e, params)
                            
                            # Calculate current progress
                            current_step = i + 1
                            current_percent = int((current_step / nb_holes) * 100)
                            
                            # Only update the UI if the percentage has moved up
                            if current_percent > last_reported_percent:
                                # Calculate how many holes have been filled since the last UI update
                                # This handles cases where 1% might represent multiple holes
                                if last_reported_percent == -1:
                                    # First update
                                    progress_bar.update(current_step)
                                else:
                                    # Update by the difference since the last check
                                    last_step = int((last_reported_percent * nb_holes) / 100)
                                    diff = current_step - last_step
                                    progress_bar.update(diff)
                                
                                last_reported_percent = current_percent
                                    
                        progress_bar.close()                         
                    
                    vertices = mrmeshnumpy.getNumpyVerts(mmesh)
                    faces = mrmeshnumpy.getNumpyFaces(mmesh.topology)

                    del mmesh
                    gc.collect()
                
                if reorient_vertices == '90 degrees':
                    vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
                elif reorient_vertices == '-90 degrees':
                    vertices[:, 1], vertices[:, 2] = -vertices[:, 2], vertices[:, 1]
                
                trimesh = Trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=False
                )
                
                if weld_vertices:
                    vertices_count = len(trimesh.vertices)
                    trimesh.merge_vertices(digits_vertex=digits)
                    new_vertices_count = len(trimesh.vertices)
                    nb_vertices_removed = vertices_count - new_vertices_count
                    print(f"Weld Vertices: Removed {nb_vertices_removed} vertices")                    
                    

                filename_prefix_with_nbfaces = f"{filename_prefix}_{target_nbfaces}"

                full_output_folder, filename, counter, subfolder, filename_prefix_with_nbfaces = folder_paths.get_save_image_path(filename_prefix_with_nbfaces, folder_paths.get_output_directory())                
                output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
                output_glb_path.parent.mkdir(exist_ok=True)
                
                trimesh.export(output_glb_path, file_type=file_format)
                
                lst_output_mesh.append(str(output_glb_path))

                del trimesh
            
            del cumesh
            del mesh_copy
        
        return (lst_output_mesh,)   

class Trellis2WeldVertices:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_texture": ("BOOLEAN",{"default":True}),
                "merge_normals": ("BOOLEAN",{"default":True}),
                "digits":("INT",{"default":4,"min":1,"max":8}),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, trimesh, merge_texture, merge_normals, digits):
        new_mesh = trimesh.copy()
        vertices_count = len(new_mesh.vertices)
        new_mesh.merge_vertices(merge_tex=merge_texture, merge_norm=merge_normals, digits_vertex=digits, digits_norm=digits, digits_uv=digits)
        new_vertices_count = len(new_mesh.vertices)
        nb_vertices_removed = vertices_count - new_vertices_count
        faces_count = len(new_mesh.faces)
        new_mesh.remove_unreferenced_vertices()
        new_mesh.update_faces(new_mesh.nondegenerate_faces())
        new_faces_count = len(new_mesh.faces)
        nb_faces_removed = faces_count - new_faces_count
        print(f"Weld Vertices: Removed {nb_vertices_removed} vertices / {nb_faces_removed} faces")
        
        return (new_mesh,)   

class Trellis2StringSelector:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings": ("STRING",),
                "index": ("INT",{"default":0,"min":0,"max":1000}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, strings, index):
        if isinstance(strings, list):
            if len(strings) == 0:
                string = ""
            elif len(strings)<=index:
                index = len(strings)-1
                
            string = strings[index]
        elif isinstance(strings, str):
            string = strings
        else:
            raise Exception("string must be a list of a string")            
        
        return (string,)

def _dev_reload_projection():
    """Re-read the projection modules from disk before texturing.

    Editing a projection .py normally needs a ComfyUI restart to take effect, and a restart
    wipes ComfyUI's node cache -- which forces the whole ~7 min geometry/gen_seams chain and
    the ~3 min Qwen pass to re-run just to test a texturing change. Reloading here means a
    texture-side edit needs only a re-queue: everything upstream stays cached and only this
    node re-executes. Dependencies are reloaded before the module that imports them.
    """
    import importlib
    import sys as _s
    for _name in ('camera_utils', 'normal_map_utils', 'uv_island_diag',
                  'face_detection_utils', 'hair_detection_utils_v2',
                  'feature_anchor_warp', 'landmark_warp', 'fill_utils',
                  'texture_projection_multiview'):
        _sfx = '.projection.' + _name
        for _k in [k for k in list(_s.modules) if k.endswith(_sfx)]:
            try:
                importlib.reload(_s.modules[_k])
            except Exception as _e:
                print(f"[dev-reload] {_name}: {_e!r}")


# --- hole tripwire ---------------------------------------------------------------------------
_HOLE_HISTORY = []


def _boundary_edges(V, F):
    """Edges used by exactly one face. Zero on a closed mesh.

    One np.unique on an int64 key rather than unique(axis=0), which is an order of magnitude
    slower on the ~16M edges of a five-million-face mesh.
    """
    import numpy as _np
    e = _np.sort(_np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0].astype(_np.int64) * _np.int64(len(V)) + e[:, 1].astype(_np.int64)
    _, cnt = _np.unique(key, return_counts=True)
    return int((cnt == 1).sum())


def _hole_tripwire(stage, V, F, jump=5.0, reset=False):
    """Count the boundary at `stage` and shout if a step has just torn the mesh open.

    2026-09-11: a bake handed 20,075 holes to the meshlib filler, which burned its 90 s budget
    on 1,742 of them. The same failure is recorded in this file from an earlier bake (2,527 ->
    8,398). Both times a human found it in the log after the damage, and both times the useful
    question -- WHICH step tore the holes -- could not be answered, because the only two counts
    came from different nodes with a dual-contouring remesh and a simplify in between.

    Restarting the server made it go away on the same character with the same cached views and
    the same cached sculpt, so suspect process state rather than the input.

    The threshold is RELATIVE -- a multiple of the smallest count seen this bake -- because a
    mesh legitimately has some boundary and what matters is a step suddenly making far more of
    it. Diagnostic only: never raises. TRELLIS_HOLE_TRIPWIRE=0 turns it off.
    """
    global _HOLE_HISTORY
    import os as _os
    if _os.environ.get("TRELLIS_HOLE_TRIPWIRE", "1") == "0":
        return -1
    try:
        import time as _t
        if reset:
            _HOLE_HISTORY = []
        t0 = _t.time()
        n = _boundary_edges(V, F)
        prev = min([c for _, c in _HOLE_HISTORY], default=None)
        _HOLE_HISTORY.append((stage, n))
        note = ""
        if prev is not None and prev > 0 and n > prev * jump:
            note = "%.0fx the %d seen earlier" % (n / max(prev, 1), prev)
        elif prev == 0 and n > 0:
            note = "the mesh was CLOSED before this step"
        print("[HoleWire] %-18s %8d boundary edge(s)%s   (%.1fs)"
              % (stage, n, ("  <-- " + note) if note else "", _t.time() - t0), flush=True)
        if note:
            print("=" * 78, flush=True)
            print("[HoleWire] *** %s TORE THE MESH OPEN: %s ***" % (stage.upper(), note), flush=True)
            print("[HoleWire] trail: " + " -> ".join("%s %d" % (a, b) for a, b in _HOLE_HISTORY),
                  flush=True)
            print("[HoleWire] This is process state, not the input: the same character with "
                  "the same cached sculpt came out clean on a fresh server. The bake is NOT "
                  "blocked -- the hole fill is time-boxed and the run continues. Judge the "
                  "result on its merits; if it is bad, a restarted server is the known cure, "
                  "and the FIRST bake after a restart is the trustworthy one.", flush=True)
            print("=" * 78, flush=True)
        return n
    except Exception as _e:
        print("[HoleWire] %s skipped (%r)" % (stage, _e), flush=True)
        return -1


def _holefill_report(m, thr, tag='pre'):
    """Print the boundary-loop PERIMETER distribution before filling.

    fill_holes only closes loops with perimeter <= max_hole_perimeter; whatever is left goes to
    FillHolesNicelyWithMeshlib, which triangulates one hole at a time and is slow. Without knowing
    the distribution the threshold is guesswork -- this says exactly what each setting would cost.
    Diagnostic only: never raises.
    """
    try:
        import numpy as _np
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.csgraph import connected_components as _cc
        V = m.vertices.detach().cpu().numpy()
        F = m.faces.detach().cpu().numpy()
        e = _np.sort(_np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uq, cnt = _np.unique(e, axis=0, return_counts=True)
        b = uq[cnt == 1]
        if len(b) == 0:
            print(f"[HoleFill/{tag}] no boundary edges (mesh already closed)", flush=True)
            return
        n = len(V)
        g = _coo((_np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n))
        ncomp, lab = _cc(g, directed=False)
        seg = _np.linalg.norm(V[b[:, 0]] - V[b[:, 1]], axis=1)
        per = _np.bincount(lab[b[:, 0]], weights=seg, minlength=ncomp)
        per = per[per > 0]
        if not len(per):
            return
        q = _np.percentile(per, [50, 75, 90, 95, 99])
        print(f"[HoleFill/{tag}] {len(per)} boundary loops | perimeter med {q[0]:.4f} p75 {q[1]:.4f} "
              f"p90 {q[2]:.4f} p95 {q[3]:.4f} p99 {q[4]:.4f} max {per.max():.4f}", flush=True)
        print(f"[HoleFill/{tag}] threshold {thr}: fills {int((per <= thr).sum())}/{len(per)} "
              f"({100.0 * (per <= thr).mean():.1f}%), leaves {int((per > thr).sum())} for meshlib", flush=True)
        cover = "  ".join(f"{t:g}->{100.0 * (per <= t).mean():.0f}%" for t in
                          (0.03, 0.05, 0.1, 0.2, 0.4, 0.8))
        print(f"[HoleFill/{tag}] coverage by threshold: {cover}", flush=True)
    except Exception as _e:
        print(f"[HoleFill/{tag}] report skipped ({_e!r})", flush=True)


class Trellis2FillHolesWithCuMesh:
    """Fill all holes in a mesh"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "max_permieters": ("FLOAT",{"default":0.030,"min":0.001,"max":99.999,"step":0.001}),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, mesh, max_permieters):
        mesh_copy = copy.deepcopy(mesh)
        _holefill_report(mesh_copy, max_permieters)
        mesh_copy.fill_holes(max_hole_perimeter = max_permieters)
        # VERIFY. The report above only PREDICTS from the perimeter distribution; it never
        # checked the result. One bake printed "fills 2527/2527 (100.0%), leaves 0 for meshlib"
        # and then handed 8,398 holes to the meshlib filler, which took 20+ minutes. Count the
        # boundary loops that actually remain, so the log can never claim a clean fill again.
        _holefill_report(mesh_copy, max_permieters, tag='post')
        _hole_tripwire("after fill_holes",
                       mesh_copy.vertices.detach().cpu().numpy(),
                       mesh_copy.faces.detach().cpu().numpy(), reset=True)
        return (mesh_copy,)         

class Trellis2LaplacianSmoothingWithOpen3d:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "iterations": ("INT",{"default":10, "min":1, "max":100}),
                "method": (["Laplacian", "Taubin"],{"default":"Laplacian"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, iterations, method):
        import open3d
        mesh_copy = copy.deepcopy(mesh)
        vertices = mesh_copy.vertices.cpu().numpy()
        faces = mesh_copy.faces.cpu().numpy().astype(np.int32)
        
        open3d_mesh = open3d.geometry.TriangleMesh()
        open3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        open3d_mesh.triangles = open3d.utility.Vector3iVector(faces)
        
        if method == "Laplacian":
            open3d_mesh = open3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        elif method == "Taubin":
            open3d_mesh = open3d_mesh.filter_smooth_taubin(number_of_iterations=iterations)
            
        open3d_mesh.compute_vertex_normals()
        
        new_vertices = np.asarray(open3d_mesh.vertices)
        new_faces = np.asarray(open3d_mesh.triangles)
        
        mesh_copy.vertices = torch.from_numpy(new_vertices).float().to(mesh_copy.device)
        mesh_copy.faces = torch.from_numpy(new_faces).int().to(mesh_copy.device)
        
        return (mesh_copy,)      

class Trellis2UnWrapTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60.0,"min":0.0,"max":359.9}),
                "mesh_cluster_refine_iterations": ("INT",{"default":0}),
                "mesh_cluster_global_iterations": ("INT",{"default":1}),
                "mesh_cluster_smooth_strength": ("INT",{"default":1}),                
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations, mesh_cluster_global_iterations, mesh_cluster_smooth_strength):
        mesh_cluster_threshold_cone_half_angle_rad = np.radians(mesh_cluster_threshold_cone_half_angle_rad)
        
        mesh_copy = trimesh.copy()
        
        vertices = torch.from_numpy(mesh_copy.vertices).float().cuda()
        faces = torch.from_numpy(mesh_copy.faces).int().cuda()
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)     

        out_vertices, out_faces, out_uvs = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,                
            },
            return_vmaps=False,
            verbose=True,
        )
        
        del cumesh
                
        mesh_copy.vertices = out_vertices.cpu().numpy()
        mesh_copy.faces = out_faces.cpu().numpy()       
        #mesh_copy.visual.uv = out_uvs.cpu().numpy()
        mesh_copy.visual = Trimesh.visual.TextureVisuals(uv=out_uvs.cpu().numpy())
        
        return (mesh_copy,)          
        
class Trellis2MeshWithVoxelCascadeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                
                "low_res_shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "low_res_shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "low_res_shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "low_res_shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "low_res_shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "low_res_shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "low_res_shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                
                "high_res_shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "high_res_shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "high_res_shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "high_res_shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "high_res_shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "high_res_shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "high_res_shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),                                
                "generate_texture_slat": ("BOOLEAN", {"default":True}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),      
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL","BVH", )
    RETURN_NAMES = ("mesh", "bvh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,        
        # low res shape
        low_res_shape_steps, 
        low_res_shape_guidance_strength, 
        low_res_shape_guidance_rescale,
        low_res_shape_rescale_t,
        low_res_shape_sampler,
        low_res_shape_guidance_interval_start,
        low_res_shape_guidance_interval_end,
        # high res shape
        high_res_shape_steps, 
        high_res_shape_guidance_strength, 
        high_res_shape_guidance_rescale,
        high_res_shape_rescale_t,
        high_res_shape_sampler,
        high_res_shape_guidance_interval_start,
        high_res_shape_guidance_interval_end,        
        # texture,
        generate_texture_slat,
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,        
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        # others
        max_num_tokens,
        use_tiled_decoder,
        max_views,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell
        ):
            
        reset_cuda()
        
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images

        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        low_res_shape_guidance_interval = [low_res_shape_guidance_interval_start, low_res_shape_guidance_interval_end]
        high_res_shape_guidance_interval = [high_res_shape_guidance_interval_start, high_res_shape_guidance_interval_end]
        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}        
        low_res_shape_slat_sampler_params = {"steps":low_res_shape_steps,"guidance_strength":low_res_shape_guidance_strength,"guidance_rescale":low_res_shape_guidance_rescale,"guidance_interval":low_res_shape_guidance_interval,"rescale_t":low_res_shape_rescale_t}
        high_res_shape_slat_sampler_params = {"steps":high_res_shape_steps,"guidance_strength":high_res_shape_guidance_strength,"guidance_rescale":high_res_shape_guidance_rescale,"guidance_interval":high_res_shape_guidance_interval,"rescale_t":high_res_shape_rescale_t}       
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
            
        if generate_texture_slat:
            num_steps = 5
        else:
            num_steps = 4

        pbar = ProgressBar(num_steps)
        
        mesh = pipeline.run_cascade(image=image_in, 
                                    seed=seed, 
                                    pipeline_type=pipeline_type, 
                                    sparse_structure_sampler_params = sparse_structure_sampler_params, 
                                    low_res_shape_slat_sampler_params = low_res_shape_slat_sampler_params, 
                                    high_res_shape_slat_sampler_params = high_res_shape_slat_sampler_params,
                                    tex_slat_sampler_params = tex_slat_sampler_params, 
                                    max_num_tokens = max_num_tokens, 
                                    sparse_structure_resolution = sparse_structure_resolution, 
                                    max_views = max_views, 
                                    generate_texture_slat=generate_texture_slat, 
                                    use_tiled=use_tiled_decoder, 
                                    pbar=pbar,
                                    sparse_structure_sampler = sparse_structure_sampler,
                                    low_res_shape_sampler = low_res_shape_sampler,
                                    high_res_shape_sampler = high_res_shape_sampler,
                                    tex_sampler = texture_sampler,
                                    fill_holes = fill_holes,
                                    hole_iterations = hole_iterations,
                                    verbose = verbose,
                                    dino_lock = dino_lock,
                                    dino_substeps = dino_substeps,
                                    hole_fill_algorithm = hole_fill_algorithm,
                                    dino_foundation_cap = dino_foundation_cap,
                                    keep_only_shell = keep_only_shell
                                    )[0]         
        
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()                
        
        if generate_texture_slat:
            # Build BVH for the current mesh to guide remeshing
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()
        else:
            print("Not building BVH : only used for texturing")
            bvh = None
        
        return (mesh,bvh,)      
        
class Trellis2ImageCondGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "max_views": ("INT", {"default": 1, "min": 1, "max": 999}),
            },
        }

    RETURN_TYPES = ("IMAGE_COND", "IMAGE_COND", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("cond_512", "cond_1024", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, max_views,):   
        images = tensor_batch_to_pil_list(image, max_views=max_views)
        image_in = images[0] if len(images) == 1 else images        
        
        if isinstance(image_in, (list, tuple)):
            images = list(image_in)
        else:
            images = [image_in]
            
        pipeline.load_image_cond_model()        
        
        cond_512  = pipeline.get_cond(images, 512, max_views = max_views)        
        cond_1024 = pipeline.get_cond(images, 1024, max_views = max_views)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_image_cond_model()            

        return (cond_512, cond_1024, pipeline,)        

class Trellis2SparseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell
        ):
        
        self.seed_all(seed)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        args = pipeline._pretrained_args
        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.sparse_structure_sampler = getattr(samplers, f"Flow{sparse_sampler_prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])
        pipeline.load_sparse_structure_model()        
        coords = pipeline.sample_sparse_structure(
            image_cond, sparse_structure_resolution,
            1, sparse_structure_sampler_params,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell
        )
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_sparse_structure_model()            

        return (coords, sparse_structure_resolution, pipeline,)
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)         
        
class Trellis2ShapeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "coords": ("COORDS",),
                "resolution": ([512,1024],{"default":1024}),                
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),                
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("shape_slat", "resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, coords, resolution,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}            
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])                    
        
        if resolution == 512:
            pipeline.unload_shape_slat_flow_model_1024()
            pipeline.load_shape_slat_flow_model_512()            
            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_512()
        elif resolution == 1024:
            pipeline.unload_shape_slat_flow_model_512()
            pipeline.load_shape_slat_flow_model_1024()
            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_1024()
        
        return (shape_slat, resolution, pipeline,)      

class Trellis2ShapeCascadeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "shape_slat": ("SHAPE_SLAT",),
                "from_resolution": ("INT",),
                "to_resolution": ([1024,1536],{"default":1024}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT","INT","TRELLIS2PIPELINE","INT",)
    RETURN_NAMES = ("shape_slat","resolution","pipeline","num_tokens")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}                    
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])
        pipeline.load_shape_slat_flow_model_1024()           
        slat, hr_resolution, num_tokens = self.sample(pipeline, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens, image_cond, shape_slat_sampler_params, pipeline.models['shape_slat_flow_model_1024'], verbose, dino_lock, dino_substeps, dino_foundation_cap)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_flow_model_1024()              
        
        return (slat, hr_resolution, pipeline, num_tokens,)         
        
    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution, max_num_tokens, cond, sampler_params, flow_model, verbose, dino_lock, dino_substeps, dino_foundation_cap):
        # Upsample       
        pipeline.load_shape_slat_decoder()
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].to(pipeline.device)
            pipeline.models['shape_slat_decoder'].low_vram = True
        hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].cpu()
            pipeline.models['shape_slat_decoder'].low_vram = False
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_decoder()
        
        hr_resolution = resolution
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
                
        if pipeline.low_vram:
            cond = pipeline._cond_to(cond, pipeline.device)                
        
        coords_dev = coords.to(pipeline.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            cond = pipeline._cond_cpu(cond)
            pipeline._cleanup_cuda()

        return slat, hr_resolution, num_tokens 

class Trellis2TexSlatGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_cond": ("IMAGE_COND",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ([512,1024],{"default":1024}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
        }

    RETURN_TYPES = ("TEXTURE_SLAT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("texture_slat", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_cond, shape_slat, resolution,      
        # shape
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap
        ):

        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        if resolution == 512:
            pipeline.unload_tex_slat_flow_model_1024()
            pipeline.load_tex_slat_flow_model_512()
            tex_slat = pipeline.sample_tex_slat_advanced(
                image_cond, pipeline.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params, texture_sampler,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            if not pipeline.keep_models_loaded:
                pipeline.unload_tex_slat_flow_model_512()
                
        elif resolution == 1024:
            pipeline.unload_tex_slat_flow_model_512()
            pipeline.load_tex_slat_flow_model_1024()
            tex_slat = pipeline.sample_tex_slat_advanced(
                image_cond, pipeline.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params, texture_sampler,
                verbose = verbose,
                dino_lock = dino_lock,
                dino_substeps = dino_substeps,
                dino_foundation_cap = dino_foundation_cap
            )
            
            if not pipeline.keep_models_loaded:
                pipeline.unload_tex_slat_flow_model_1024()
        
        return (tex_slat, pipeline,)      
        
class Trellis2DecodeLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ("INT",),
                "use_tiled_decoder": ("BOOLEAN", {"default":True}),
            },
            "optional": {
                "texture_slat": ("TEXTURE_SLAT",),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "BVH", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("mesh", "bvh", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, shape_slat, resolution, use_tiled_decoder, texture_slat = None):
        mesh = pipeline.decode_latent(shape_slat, texture_slat, resolution, use_tiled=use_tiled_decoder)[0]
        
        if texture_slat == None:
            print("Not building BVH : only used for texturing")
            bvh = None            
        else:
            # Build BVH for the current mesh to guide remeshing
            vertices = mesh.vertices.cuda()
            faces = mesh.faces.cuda()   
            
            print("Building BVH for current mesh...")
            bvh = CuMesh.cuBVH(vertices.detach().clone(), faces.detach().clone())           
            bvh.vertices = vertices.detach().clone()
            bvh.faces = faces.detach().clone()            
        
        
        return (mesh, bvh, pipeline,)    

class Trellis2SimplifyMeshAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "thresh":("FLOAT",{"default":1e-8,"min":1e-12,"max":1e-2,"step":0.000000000001}),
                "lambda_edge_length": ("FLOAT",{"default":0.01,"min":0.00,"max":1.00,"step":0.01}),
                "lambda_skinny": ("FLOAT",{"default":0.001,"min":0.000,"max":0.100,"step":0.001}),
                "lambda_curvature": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_boundary": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_area": ("FLOAT",{"default":0.010,"min":0.000,"max":0.100,"step":0.001}),
                "qem_regularization": ("FLOAT",{"default":1e-8,"min":1e-10,"max":1e-5,"step":0.0000000001}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, thresh, lambda_edge_length, lambda_skinny, lambda_curvature, lambda_boundary, lambda_area, qem_regularization):        
        mesh_copy = copy.deepcopy(mesh)

        options = {
            'method': 'advanced',
            'thresh': thresh,
            'lambda_edge_length': lambda_edge_length,
            'lambda_skinny': lambda_skinny,
            'lambda_curvature': lambda_curvature,
            'lambda_boundary': lambda_boundary,            
            'lambda_area': lambda_area,     
            'qem_regularization': qem_regularization,     
        }         

        mesh_copy.simplify_with_cumesh(target = target_face_num, options = options)
        
        return (mesh_copy,)  

class Trellis2SimplifyTrimeshAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":30000000}),
                "thresh":("FLOAT",{"default":1e-8,"min":1e-12,"max":1e-2,"step":0.000000000001}),
                "lambda_edge_length": ("FLOAT",{"default":0.01,"min":0.00,"max":1.00,"step":0.01}),
                "lambda_skinny": ("FLOAT",{"default":0.001,"min":0.000,"max":0.100,"step":0.001}),
                "lambda_curvature": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_boundary": ("FLOAT",{"default":0.050,"min":0.000,"max":0.500,"step":0.001}),
                "lambda_area": ("FLOAT",{"default":0.010,"min":0.000,"max":0.100,"step":0.001}),
                "qem_regularization": ("FLOAT",{"default":1e-8,"min":1e-10,"max":1e-5,"step":0.0000000001}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, target_face_num, thresh, lambda_edge_length, lambda_skinny, lambda_curvature, lambda_boundary, lambda_area, qem_regularization):
        mesh_copy = copy.deepcopy(trimesh)
        
        cumesh = CuMesh.CuMesh()
        cumesh.init(torch.from_numpy(mesh_copy.vertices).float().cuda(), torch.from_numpy(mesh_copy.faces).int().cuda())
        
        options = {
            'method': 'advanced',
            'thresh': thresh,
            'lambda_edge_length': lambda_edge_length,
            'lambda_skinny': lambda_skinny,
            'lambda_curvature': lambda_curvature,
            'lambda_boundary': lambda_boundary,            
            'lambda_area': lambda_area,     
            'qem_regularization': qem_regularization,     
        }        

        cumesh.simplify(target_face_num, verbose=True, options = options)
        
        new_vertices, new_faces = cumesh.read()
        mesh_copy.vertices = new_vertices.cpu().numpy()
        mesh_copy.faces = new_faces.cpu().numpy()
            
        del cumesh        
        
        return (mesh_copy,)

class Trellis2MultiViewTexturing:
    # ── Re-run when the projection source changes (see _dev_reload_projection) ───
    # ComfyUI caches by node INPUTS and cannot see a .py edit, so a texturing fix would
    # be skipped entirely on a re-queue. Hashing the projection modules' mtimes marks
    # THIS node dirty and leaves the ~7 min geometry chain and ~3 min Qwen pass cached.
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import os as _o
        _d = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "projection")
        _mods = ("texture_projection_multiview.py", "fill_utils.py",
                 "hair_detection_utils_v2.py", "face_detection_utils.py",
                 "feature_anchor_warp.py", "landmark_warp.py",
                 "normal_map_utils.py", "camera_utils.py")
        try:
            return str([(m, _o.path.getmtime(_o.path.join(_d, m)))
                        for m in _mods if _o.path.exists(_o.path.join(_d, m))])
        except Exception:
            return float("nan")

    """
    Apply texture to mesh by projecting multiple view images.
    
    Uses angle-weighted blending: each surface receives texture from all views
    that can "see" it, weighted by how directly the surface faces each camera.
    
    Camera angles (Y-up coordinate system):
    - Azimuth: rotation around Y axis
      - 0° = front (looking in -Z direction)
      - 90° = left (looking in -X direction)  
      - 180° = back (looking in +Z direction)
      - 270° = right (looking in +X direction)
    - Elevation: rotation around X axis
      - 0° = horizontal
      - 90° = top (looking in -Y direction, from above)
      - -90° = bottom (looking in +Y direction, from below)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "blend_texture": ("BOOLEAN", {"default":True}),
                "blend_exponent": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 8.0, "step": 0.5}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.05, "max": 10.0, "step": 0.01}),
                "norm_size": ("FLOAT",{"default":1.15, "min":0.0, "max":9.99, "step":0.01}),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_size": ("INT",{"default":20,"min":0,"max":99999,"step":1}),
                "use_metallic": ("BOOLEAN",{"default":True}),
                "depth_eps": ("FLOAT",{"default":0.0100,"min":0.0001,"max":1.0000,"step":0.0001}),
                "mesh_cluster_threshold_cone_half_angle_rad": ("FLOAT",{"default":60,"min":1,"max":179,"step":1}),
                "u_pixel_offset": ("FLOAT",{"default":0.0,"min":-200.0,"max":200.0,"step":0.5,
                                             "tooltip":"Shift image sampling right (+) or left (-) in pixels. Use to correct eye/feature horizontal misalignment."}),
                "v_pixel_offset": ("FLOAT",{"default":0.0,"min":-200.0,"max":200.0,"step":0.5,
                                             "tooltip":"Shift image sampling down (+) or up (-) in pixels. Use to correct eye/feature vertical misalignment."}),
                "use_cylindrical_uv": ("BOOLEAN", {"default": True, "tooltip": "Use cylindrical UV (U=azimuth, V=height) instead of xatlas island UV. Eliminates 163k micro-island mosaic artifacts. Seam placed at back of character. Automatically sets projection-only mode."}),
                "saturation_boost": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.05,
                                               "tooltip": "Boost texture saturation after projection (>1 = more vivid). Compensates for SV3D's washed-out lighting. 1.0 = no change."}),
            },
            "optional": {
                # Standard views
                "front_image":  ("IMAGE",),   # az=0°
                "back_image":   ("IMAGE",),   # az≈188.6° for SV3D frame 11
                "left_image":   ("IMAGE",),   # az≈102.9° for SV3D frame 6
                "right_image":  ("IMAGE",),   # az≈274.3° for SV3D frame 16
                "top_image":    ("IMAGE",),   # el=90°
                "bottom_image": ("IMAGE",),   # el=-90°
                "front_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "back_weight":  ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "left_weight":  ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "right_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "top_weight":   ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "bottom_weight":("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                # Custom views
                "custom_images": ("IMAGE",),
                "custom_azimuths": ("STRING", {"default": ""}),
                "custom_elevations": ("STRING", {"default": ""}),
                "custom_weights": ("STRING", {"default": ""}),
                "camera_config": ("HY3DCAMERA",),
                # Colour correction – appended so old workflow widget_values are unaffected
                "brightness_correction": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                                                    "tooltip": "Multiply texture brightness after projection. 1.0 = no change. < 1 darkens (was hardcoded 0.88 before — that was making the back too dark)."}),
                # Per-view azimuth overrides – appended AFTER brightness_correction so old widget_values
                # (which end at brightness_correction) are not displaced.
                # SV3D generates 21 frames uniformly over 360°: frame_az = frame_i × 360/21 ≈ 17.14°/frame.
                #
                # RECOMMENDED setup — weighted-blend two adjacent SV3D frames (length=2 on the frame
                # extractor) to land on exactly 90°/180°/270°.  SV3D frame spacing = 360/21 ≈ 17.143°.
                #   Left : frames 5+6  (85.714°/102.857°) → left_blend=0.25  → 75%×fr5 + 25%×fr6 = 90.0° exact
                #   Back : frames 10+11 (171.429°/188.571°) → back_blend=0.50 → 50/50             = 180.0° exact
                #   Right: frames 15+16 (257.143°/274.286°) → right_blend=0.75 → 25%×fr15+75%×fr16 = 270.0° exact
                #
                # Single-frame equivalents (length=1, blend ignored):
                #   Left: start=6 → 102.9°   Back: start=11 → 188.6°   Right: start=16 → 274.3°
                "left_azimuth":  ("FLOAT",{"default": 90.0, "min":0.0,"max":360.0,"step":0.1,
                                           "tooltip":"Azimuth used for the left view projection. "
                                                     "Keep at 90.0 when using length=2 + left_blend=0.25."}),
                "back_azimuth":  ("FLOAT",{"default":180.0, "min":0.0,"max":360.0,"step":0.1,
                                           "tooltip":"Azimuth used for the back view projection. "
                                                     "180.0 is exact with length=2 + back_blend=0.50."}),
                "right_azimuth": ("FLOAT",{"default":270.0, "min":0.0,"max":360.0,"step":0.1,
                                           "tooltip":"Azimuth used for the right view projection. "
                                                     "Keep at 270.0 when using length=2 + right_blend=0.75."}),
                # Weighted frame blend – appended last so old widget_values are unaffected.
                # Controls how two-frame batches (length=2 from the SV3D frame extractor) are mixed.
                # blend=0.0 → 100% first frame,  blend=1.0 → 100% second frame.
                # Defaults give exactly cardinal angles:
                #   left_blend=0.25  → 75%×frame5(85.7°)  + 25%×frame6(102.9°)  = 90.0°
                #   back_blend=0.50  → 50%×frame10(171.4°) + 50%×frame11(188.6°) = 180.0°
                #   right_blend=0.75 → 25%×frame15(257.1°) + 75%×frame16(274.3°) = 270.0°
                "left_blend":  ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Weight of the 2nd frame when left_image is a 2-frame batch. 0.25 → exactly 90°."}),
                "back_blend":  ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Weight of the 2nd frame when back_image is a 2-frame batch. 0.50 → exactly 180°."}),
                "right_blend": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Weight of the 2nd frame when right_image is a 2-frame batch. 0.75 → exactly 270°."}),
                # Debug: paint rescued dark texels a flat colour instead of filling from neighbours.
                # Leave empty for normal operation.  Example values: "0,220,0" (green), "255,0,255" (magenta).
                "debug_color": ("STRING", {"default": "",
                                           "tooltip": "Optional debug colour for Pass 1b dark-pixel rescue. "
                                                      "Format: R,G,B  e.g. '0,220,0' for green. "
                                                      "Leave empty for normal nearest-neighbour fill."}),
                # Pass 1b thresholds — ORDER MATTERS: positions 3-7 must match saved workflow widget_values.
                # pos 3: p1b_dark_thresh  (was always here)
                # pos 4: p1b_bright_thresh (restored to original position 4)
                # pos 5: p1b_sat_thresh    (restored to original position 5)
                # pos 6: p1b_relative_mode (BOOL, replaces old p1b_require_brightness BOOL — same type/default=True)
                # pos 7: p1c_frac          (restored)
                # pos 8: p1c_diff          (restored)
                # pos 9: p1b_bright_fraction (NEW — appended at end, no index shift)
                "p1b_dark_thresh":    ("INT",     {"default": 150, "min": 0,   "max": 255, "step": 5,
                                                   "tooltip": "Pass 1b: rescue UV texels with mean RGB below this. 150 catches seam shadows + near-black."}),
                "p1b_bright_thresh":  ("INT",     {"default": 180, "min": 0,   "max": 255, "step": 5,
                                                   "tooltip": "Pass 1b (absolute mode only): sources must have mean RGB >= this value. "
                                                              "Ignored when p1b_relative_mode is ON."}),
                "p1b_sat_thresh":     ("INT",     {"default": 55,  "min": 0,   "max": 255, "step": 5,
                                                   "tooltip": "Pass 1b: sources must have HSV saturation above this (0-255). Prevents grey pixels being used as fill sources."}),
                "p1b_relative_mode":  ("BOOLEAN", {"default": True,
                                                   "tooltip": "Pass 1b source brightness mode.\n"
                                                              "ON  (relative): floor = median saturated pixel brightness × p1b_bright_fraction — adapts to any skin tone.\n"
                                                              "OFF (absolute): floor = p1b_bright_thresh — fixed value, same behaviour as before."}),
                # Pass 1c thresholds
                "p1c_frac":           ("FLOAT",   {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Pass 1c: texel is a grey outlier if its saturation < FRAC × neighbour median saturation. Higher = more aggressive correction."}),
                "p1c_diff":           ("INT",     {"default": 18,  "min": 0,   "max": 128, "step": 2,
                                                   "tooltip": "Pass 1c: minimum mean per-channel difference vs neighbourhood median to trigger correction."}),
                # NEW — appended at end so existing saved workflows keep their values for all widgets above.
                "p1b_bright_fraction": ("FLOAT",  {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Pass 1b (relative mode): source brightness must be >= this fraction of the character's own median skin brightness.\n"
                                                              "0.6 = source must be at least 60% as bright as the median saturated pixel. Lower to allow darker sources."}),
                "debug_regions":       ("BOOLEAN", {"default": False,
                                                    "tooltip": "Replace output texture with a K-means colour cluster map (flat debug colours).\n"
                                                               "Use to identify which pixels belong to skin / hair / lips / etc.\n"
                                                               "Cluster summary printed to log with average RGB per cluster."}),
                "p1c_k":              ("INT",     {"default": 96, "min": 8, "max": 512, "step": 8,
                                                   "tooltip": "Pass 1c: number of 3-D nearest neighbours used to compute the local colour median.\n"
                                                              "Increase when grey patches are large (the neighbourhood must reach correctly-coloured pixels outside the grey zone).\n"
                                                              "96–128 for arm tops; higher values are slower."}),
                "p1c_iterations":     ("INT",     {"default": 5, "min": 1, "max": 10, "step": 1,
                                                   "tooltip": "Pass 1c: number of correction passes.\n"
                                                              "Each iteration corrects the outer edge of the grey zone; pixels fixed in pass N become valid sources in pass N+1.\n"
                                                              "3–5 passes flood-fills inward through large grey patches. Stops early if nothing changed."}),
                # ── Neck-region filter ────────────────────────────────────────
                # Suppresses hair-edge detections in the neck zone (never hair).
                # Zone = forward-facing (Z > 0), central, near head-body seam.
                "neck_y_min":    ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.55, "step": 0.01,
                                            "tooltip":
                                                "Neck filter — absolute world-Y bottom of the neck zone "
                                                "(character normalised to Y ±0.575; feet at −0.575, top of head at ~+0.525).\n"
                                                "Any forward-facing (Z > 0) detected edge between neck_y_min and neck_y_max is suppressed — "
                                                "the neck is never hair.\n"
                                                "0.30 ≈ top of the shoulders. Raise if stray edges appear below the neck; "
                                                "lower if the bottom of the neck is incorrectly suppressed."}),
                "neck_y_max":    ("FLOAT", {"default": 0.42, "min": 0.0, "max": 0.55, "step": 0.01,
                                            "tooltip":
                                                "Neck filter — absolute world-Y top of the neck zone.\n"
                                                "T_seam_floor (head-body boundary) is typically ~0.372; "
                                                "0.42 extends ~5 cm above it to catch forward-facing strays just inside the head zone.\n"
                                                "Raise (e.g. 0.46) if the very base of the hairline is mis-detected as neck; "
                                                "lower toward 0.37 to shrink the suppression zone."}),
                # ── Cyan hairline edge filter ─────────────────────────────────
                # Controls which Z-Canny edges survive as the cyan hairline layer.
                "cyan_y_margin": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.00, "step": 0.05,
                                            "tooltip":
                                                "Cyan hairline: Y-band floor for Canny edges. "
                                                "floor = T_hair − margin × (T_hair − T_seam_floor).\n"
                                                "Range 0.0 – 1.0. 1.0 = no Y filter (all edges kept). "
                                                "0.95 = floor just 5 % above the nape — lets temple edges go as low as needed. "
                                                "Decrease (e.g. 0.70) to trim more of the lower head zone; risks clipping temple hairline. "
                                                "Increase toward 1.0 if the temple hairline is cut off too early."}),
                "cyan_z_frac":   ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.00, "step": 0.05,
                                            "tooltip":
                                                "Cyan hairline: Z upper-bound. Keeps only Canny edges where world-Z < Z_max × this fraction. "
                                                "The real hairline sits near Z = 0; face-interior UV-seam noise sits at high Z (Z_max ≈ 0.07).\n"
                                                "Range 0.0 – 1.0. 0.42 keeps edges below ~42 % of Z_max (≈ Z < 0.03). "
                                                "Decrease (e.g. 0.30) to remove more face-cap noise; "
                                                "increase (e.g. 0.55) if valid temple edges near the face are lost."}),
                # ── Hair debug fill ───────────────────────────────────────────
                "hair_debug":    ("BOOLEAN", {"default": False,
                                              "tooltip": "Paint detected hair pixels yellow on the output texture.\n"
                                                         "Uses the same 3D-based detection as debug_regions (hair cap + braid) "
                                                         "but applied to the real projected texture, not a debug canvas.\n"
                                                         "Enable to verify hair coverage before wiring up actual hair colour."}),
                # ── Flat skin base colour ─────────────────────────────────────
                "flat_skin":     ("BOOLEAN", {"default": False,
                                              "tooltip": "Replace the projected skin texture with a flat neutral skin tone.\n"
                                                         "Use together with hair_debug to see yellow hair over flat skin "
                                                         "without the projection artefacts distracting from hair coverage."}),
                # ── KAN-9 face-front weighting ────────────────────────────────
                "face_front_only": ("BOOLEAN", {"default": True,
                                              "tooltip": "KAN-9: in the face region (forward-facing, central, upper head) "
                                                         "paint from the SINGLE most-frontal view only, suppressing side/SV3D "
                                                         "views there.\n"
                                                         "Fixes ghosted / double-exposed eyes, nose and mouth caused by "
                                                         "multi-view blending of misaligned features.\n"
                                                         "Note: this stops ghosting; it does not correct a front view that is "
                                                         "itself misaligned (that's a separate alignment fix)."}),
                "face_landmark_warp": ("BOOLEAN", {"default": True,
                                              "tooltip": "KAN-9: per-feature face alignment. Detects 478 MediaPipe FaceLandmarker "
                                                         "points on a render of the mesh face AND on the front image, then warps "
                                                         "(thin-plate spline) the front view's sample positions in the face region "
                                                         "so eyes/nose/mouth land on the mesh's features.\n"
                                                         "Fixes the asymmetric per-feature misalignment a single global transform "
                                                         "cannot. Front view only; face region only (soft mask)."}),
                "dump_stages": ("BOOLEAN", {"default": True,
                                              "tooltip": "KAN-9 debug: save the UV atlas after each post-projection stage "
                                                         "(_stage_0_projected .. _stage_4_final) to the uv_transfer folder, so you "
                                                         "can see exactly which stage alters the face. Diagnostic only."}),
                "fast_skin_fill": ("BOOLEAN", {"default": True,
                                              "tooltip": "Front-only CYLINDRICAL fill: keep the projected front, fill uncovered "
                                                         "back/side texels by interpolating the two still-visible silhouette-edge "
                                                         "colours around the body per height band — photo-derived, height-varying skin "
                                                         "(NOT a flat slab). Skips the expensive 3D KD-tree fill + dark-rescue + "
                                                         "colour-consistency (~2-3 min on a 4k atlas) and the dark-rescue face-flatten. "
                                                         "Turn off for the old KD-tree multi-view fill."}),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("trimesh", "base_color", "metallic_roughness", "normal_map",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(
        self,
        trimesh,
        texture_size,
        blend_texture,
        blend_exponent,
        ortho_scale,
        norm_size,
        fill_holes,
        max_hole_size,
        use_metallic,
        depth_eps,
        mesh_cluster_threshold_cone_half_angle_rad,
        u_pixel_offset=0.0,
        v_pixel_offset=0.0,
        use_cylindrical_uv=True,
        saturation_boost=1.0,
        brightness_correction=1.0,
        baseColorTexture = None,
        front_image=None,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        front_weight=None,
        back_weight=None,
        left_weight=None,
        right_weight=None,
        top_weight=None,
        bottom_weight=None,
        left_azimuth=90.0,
        back_azimuth=180.0,
        right_azimuth=270.0,
        custom_images=None,
        custom_azimuths="",
        custom_elevations="",
        custom_weights="",
        camera_config=None,
        left_blend=0.25,
        back_blend=0.50,
        right_blend=0.75,
        debug_color="",
        p1b_dark_thresh=150,
        p1b_bright_thresh=180,
        p1b_sat_thresh=55,
        p1b_relative_mode=True,
        p1c_frac=0.58,
        p1c_diff=18,
        p1b_bright_fraction=0.6,
        p1c_k=96,
        p1c_iterations=5,
        debug_regions=False,
        neck_y_min=0.30,
        neck_y_max=0.42,
        cyan_y_margin=0.95,
        cyan_z_frac=0.72,
        hair_debug=False,
        flat_skin=False,
        face_front_only=True,
        face_landmark_warp=True,
        dump_stages=True,
        fast_skin_fill=True,
    ):
        _dev_reload_projection()
        from .projection.texture_projection_multiview import texture_mesh_with_multiview

        # Parse debug_color string "R,G,B" → tuple, or None for normal operation
        _debug_color = None
        if debug_color and debug_color.strip():
            try:
                _dc = [int(x.strip()) for x in debug_color.split(",")]
                if len(_dc) == 3:
                    _debug_color = tuple(_dc)
            except Exception:
                print(f"[MultiView] debug_color parse error: {debug_color!r} — ignoring")
        
        reset_cuda()
        
        # Collect views
        images = []
        azimuths = []
        elevations = []
        weights = []
        
        # Standard views with tunable azimuths for left/back/right.
        # SV3D frames are at frame_i × 360/21 ≈ 17.14°/frame — NOT at 90/180/270°.
        # Each tuple: (image, azimuth, elevation, name, weight, blend)
        # blend = weight of the 2nd frame when the image tensor has batch=2.
        # Defaults give exactly cardinal angles via weighted blending of adjacent SV3D frames.
        standard_views = [
            (front_image,  0,             0,   "front",  front_weight, 0.50),
            (back_image,   back_azimuth,  0,   "back",   back_weight,  back_blend),
            (left_image,   left_azimuth,  0,   "left",   left_weight,  left_blend),
            (right_image,  right_azimuth, 0,   "right",  right_weight, right_blend),
            (top_image,    0,             90,  "top",    top_weight,   0.50),
            (bottom_image, 0,            -90,  "bottom", bottom_weight, 0.50),
        ]

        for img, az, el, name, w, blend in standard_views:
            if img is not None:
                images.append(self._tensor_to_pil(img, blend))
                azimuths.append(az)
                elevations.append(el)
                weights.append(w)
                print(f"[MultiView] Added {name} view (az={az}, el={el}, w={w})")

        # Custom views
        if custom_images is not None:
            custom_az_list = self._parse_angles(custom_azimuths)
            custom_el_list = self._parse_angles(custom_elevations)
            custom_w_list = self._parse_angles(custom_weights)
            
            if custom_az_list and custom_el_list:
                num_custom = min(len(custom_az_list), len(custom_el_list), int(custom_images.shape[0]), len(custom_w_list))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(custom_az_list[i])
                    elevations.append(custom_el_list[i])
                    weights.append(custom_w_list[i])
                    print(f"[MultiView] Added custom view {i+1} (az={custom_az_list[i]}, el={custom_el_list[i]})")
            elif camera_config:
                selected_camera_azims = camera_config["selected_camera_azims"]
                selected_camera_elevs = camera_config["selected_camera_elevs"]
                selected_view_weights = camera_config["selected_view_weights"]
                #ortho_scale = camera_config["ortho_scale"]             

                num_custom = min(len(selected_camera_azims), len(selected_camera_elevs), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(selected_camera_azims[i])
                    elevations.append(selected_camera_elevs[i])
                    weights.append(selected_view_weights[i])
                    print(f"[MultiView] Added custom view {i+1} (az={selected_camera_azims[i]}, el={selected_camera_elevs[i]}, w={selected_view_weights[i]})")                
        
        if len(images) == 0:
            raise ValueError("No input images provided! Please connect at least one image.")
        
        print(f"[MultiView] Total views: {len(images)}")
        print(f"[MultiView] Azimuths: {azimuths}")
        print(f"[MultiView] Elevations: {elevations}")

        trimesh_obj, base_color, mr, normal_map = texture_mesh_with_multiview(
            trimesh,
            images,
            azimuths,
            elevations,
            weights,
            texture_size=texture_size,
            mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
            blend_exponent=blend_exponent,
            ortho_scale=ortho_scale,
            blend_texture=blend_texture,
            fill_holes=fill_holes,
            norm_size=norm_size,
            max_hole_size=max_hole_size,
            use_metallic=use_metallic,
            depth_eps=depth_eps,
            u_pixel_offset=u_pixel_offset,
            v_pixel_offset=v_pixel_offset,
            use_cylindrical_uv=use_cylindrical_uv,
            saturation_boost=saturation_boost,
            brightness_correction=brightness_correction,
            debug_color=_debug_color,
            p1b_dark_thresh=int(p1b_dark_thresh),
            p1b_sat_thresh=int(p1b_sat_thresh),
            p1b_bright_fraction=float(p1b_bright_fraction),
            p1c_k=int(p1c_k),
            p1c_iterations=int(p1c_iterations),
            p1b_bright_thresh=int(p1b_bright_thresh),
            p1b_relative_mode=bool(p1b_relative_mode),
            p1c_frac=float(p1c_frac),
            p1c_diff=int(p1c_diff),
            debug_regions=bool(debug_regions),
            neck_y_min=float(neck_y_min),
            neck_y_max=float(neck_y_max),
            cyan_y_margin=float(cyan_y_margin),
            cyan_z_frac=float(cyan_z_frac),
            hair_debug_color=(255, 255, 0) if hair_debug else None,
            skin_base_color=(210, 180, 155) if flat_skin else None,
            face_front_only=bool(face_front_only),
            face_landmark_warp=bool(face_landmark_warp),
            dump_stages=bool(dump_stages),
            fast_skin_fill=bool(fast_skin_fill),
        )

        return (trimesh_obj, pil2tensor(base_color), pil2tensor(mr), pil2tensor(normal_map))

    def _tensor_to_pil(self, tensor, blend=0.5):
        """Convert ComfyUI IMAGE tensor (B,H,W,C) to PIL.

        blend controls how a 2-frame batch is mixed:
          0.0 = 100% first frame, 1.0 = 100% second frame.
        This lets you hit exact azimuths — e.g. SV3D frames 5+6 are at
        85.714° and 102.857°; blend=0.25 gives exactly 90.0°.
        For batches > 2 frames the equal mean is used (blend is ignored)."""
        if len(tensor.shape) == 4:
            if tensor.shape[0] == 2:
                t = tensor.float().cpu()
                blended = t[0] * (1.0 - blend) + t[1] * blend
                arr = (blended.numpy() * 255).clip(0, 255).astype(np.uint8)
                print(f"[MultiView] _tensor_to_pil: weighted blend of 2 frames (blend={blend:.3f})")
            elif tensor.shape[0] > 2:
                arr = (tensor.float().cpu().numpy().mean(axis=0) * 255).clip(0, 255).astype(np.uint8)
                print(f"[MultiView] _tensor_to_pil: equal-mean of {tensor.shape[0]} frames")
            else:
                arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []
            
class Trellis2RenderViewFromMesh:
    """
    Renders the existing base-colour texture of a trimesh from any camera angle
    using the same orthographic projection as the multi-view texturing stage.

    Use this to generate a top-down (el=90) view from the original Trellis mesh
    so you can feed it as top_image to Trellis2MultiViewTexturing.  The Trellis
    mesh has full-coverage texture on every surface including the arm/shoulder
    tops that SV3D never captures at el=0.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh":    ("TRIMESH",),
                "azimuth":    ("FLOAT", {"default": 0.0,  "min": -360.0, "max": 360.0, "step": 1.0,
                                         "tooltip": "Camera azimuth in degrees. 0=front, 90=left, 180=back, 270=right."}),
                "elevation":  ("FLOAT", {"default": 90.0, "min": -90.0,  "max": 90.0,  "step": 1.0,
                                         "tooltip": "Camera elevation. 90=top-down, -90=bottom-up, 0=horizontal."}),
                "ortho_scale":("FLOAT", {"default": 1.15, "min": 0.1,    "max": 5.0,   "step": 0.01,
                                         "tooltip": "Must match the ortho_scale used in the texturing node."}),
                "norm_size":  ("FLOAT", {"default": 1.15, "min": 0.1,    "max": 5.0,   "step": 0.01,
                                         "tooltip": "Must match the norm_size used in the texturing node."}),
                "render_size":("INT",   {"default": 1024, "min": 256,    "max": 4096,  "step": 64}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("rendered_view",)
    FUNCTION      = "process"
    CATEGORY      = "Trellis2Wrapper"
    OUTPUT_NODE   = False

    def process(self, trimesh, azimuth, elevation, ortho_scale, norm_size, render_size):
        from .projection.texture_projection_multiview import render_mesh_from_view
        pil_img = render_mesh_from_view(
            trimesh,
            azimuth    = azimuth,
            elevation  = elevation,
            ortho_scale= ortho_scale,
            norm_size  = norm_size,
            render_size= render_size,
        )
        if pil_img is None:
            raise ValueError("[RenderViewFromMesh] Mesh has no UV/texture — cannot render.")
        img_np = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
        return (torch.from_numpy(img_np).unsqueeze(0),)   # (1, H, W, 3)


class Trellis2ProjectHighPolyToLowPoly:
    """
    Apply texture to mesh by projecting multiple view images.
    
    Uses angle-weighted blending: each surface receives texture from all views
    that can "see" it, weighted by how directly the surface faces each camera.
    
    Camera angles (Y-up coordinate system):
    - Azimuth: rotation around Y axis
      - 0° = front (looking in -Z direction)
      - 90° = left (looking in -X direction)  
      - 180° = back (looking in +Z direction)
      - 270° = right (looking in +X direction)
    - Elevation: rotation around X axis
      - 0° = horizontal
      - 90° = top (looking in -Y direction, from above)
      - -90° = bottom (looking in +Y direction, from below)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_poly_trimesh": ("TRIMESH",),
                "low_poly_trimesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "blend_texture": ("BOOLEAN", {"default":True}),
                "blend_exponent": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 8.0, "step": 0.5}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.05, "max": 10.0, "step": 0.01}),
                "norm_size": ("FLOAT",{"default":1.15, "min":0.0, "max":9.99, "step":0.01}),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_size": ("INT",{"default":20,"min":0,"max":99999,"step":1}),
                "use_metallic": ("BOOLEAN",{"default":True}),
                "depth_eps": ("FLOAT",{"default":0.0100,"min":0.0001,"max":1.0000,"step":0.0001}),
            },
            "optional": {
                # Standard views
                "front_image": ("IMAGE",),   # az=0, el=0
                "back_image": ("IMAGE",),    # az=180, el=0
                "left_image": ("IMAGE",),    # az=90, el=0
                "right_image": ("IMAGE",),   # az=270, el=0
                "top_image": ("IMAGE",),     # az=0, el=90
                "bottom_image": ("IMAGE",),  # az=0, el=-90
                "front_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "back_weight": ("FLOAT",{"default":1.000,"min":0.001,"max":1.000,"step":0.001}),
                "left_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "right_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "top_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                "bottom_weight": ("FLOAT",{"default":0.010,"min":0.001,"max":1.000,"step":0.001}),
                # Custom views
                "custom_images": ("IMAGE",),
                "custom_azimuths": ("STRING", {"default": ""}),
                "custom_elevations": ("STRING", {"default": ""}),
                "custom_weights": ("STRING", {"default": ""}),
                "camera_config": ("HY3DCAMERA",),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("trimesh", "base_color", "metallic_roughness",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def process(
        self,
        high_poly_trimesh,
        low_poly_trimesh,
        texture_size,
        blend_texture,
        blend_exponent,
        ortho_scale,
        norm_size,
        fill_holes,
        max_hole_size,
        use_metallic,
        depth_eps,
        baseColorTexture = None,
        front_image=None,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        front_weight=None,
        back_weight=None,
        left_weight=None,
        right_weight=None,
        top_weight=None,
        bottom_weight=None,
        custom_images=None,
        custom_azimuths="",
        custom_elevations="",
        custom_weights="",
        camera_config = None,
    ):
        _dev_reload_projection()
        from .projection.texture_projection_multiview import texture_mesh_with_multiview
        
        reset_cuda()
        
        # Collect views
        images = []
        azimuths = []
        elevations = []
        weights = []
        
        # Standard views with their camera angles
        standard_views = [
            (front_image, 0, 0, "front", front_weight),
            (back_image, 180, 0, "back", back_weight),
            (left_image, 90, 0, "left", left_weight),
            (right_image, 270, 0, "right", right_weight),
            (top_image, 0, 90, "top", top_weight),
            (bottom_image, 0, -90, "bottom", bottom_weight),
        ]
        
        for img, az, el, name, w in standard_views:
            if img is not None:
                images.append(self._tensor_to_pil(img))
                azimuths.append(az)
                elevations.append(el)
                weights.append(w)
                print(f"[MultiView] Added {name} view (az={az}, el={el}, w={w})")        
                    
        # Custom views
        if custom_images is not None:
            custom_az_list = self._parse_angles(custom_azimuths)
            custom_el_list = self._parse_angles(custom_elevations)
            custom_w_list = self._parse_angles(custom_weights)
            
            if custom_az_list and custom_el_list:
                num_custom = min(len(custom_az_list), len(custom_el_list), int(custom_images.shape[0]), len(custom_w_list))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(custom_az_list[i])
                    elevations.append(custom_el_list[i])
                    weights.append(custom_w_list[i])
                    print(f"[MultiView] Added custom view {i+1} (az={custom_az_list[i]}, el={custom_el_list[i]})")
            elif camera_config:
                selected_camera_azims = camera_config["selected_camera_azims"]
                selected_camera_elevs = camera_config["selected_camera_elevs"]
                selected_view_weights = camera_config["selected_view_weights"]
                #ortho_scale = camera_config["ortho_scale"]             

                num_custom = min(len(selected_camera_azims), len(selected_camera_elevs), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(selected_camera_azims[i])
                    elevations.append(selected_camera_elevs[i])
                    weights.append(selected_view_weights[i])
                    print(f"[MultiView] Added custom view {i+1} (az={selected_camera_azims[i]}, el={selected_camera_elevs[i]}, w={selected_view_weights[i]})")                
        
        if len(images) == 0:
            raise ValueError("No input images provided! Please connect at least one image.")
        
        print(f"[MultiView] Total views: {len(images)}")
        print(f"[MultiView] Azimuths: {azimuths}")
        print(f"[MultiView] Elevations: {elevations}")

        trimesh_obj, base_color, mr = texture_mesh_with_multiview(
            high_poly_trimesh,
            images,
            azimuths,
            elevations,
            weights,
            texture_size=texture_size,
            blend_exponent=blend_exponent,
            ortho_scale=ortho_scale,
            blend_texture=blend_texture,
            fill_holes=fill_holes,
            norm_size=norm_size,
            max_hole_size=max_hole_size,
            use_metallic=use_metallic,
            depth_eps=depth_eps,
            low_poly_mesh=low_poly_trimesh
        )
        
        return (trimesh_obj, pil2tensor(base_color), pil2tensor(mr))
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor (B,H,W,C) to PIL.
        If batch > 1 (e.g. length=2 from the SV3D frame extractor), average all
        frames so the resulting image represents the midpoint angle between them."""
        if len(tensor.shape) == 4:
            if tensor.shape[0] > 1:
                arr = (tensor.float().cpu().numpy().mean(axis=0) * 255).clip(0, 255).astype(np.uint8)
                print(f"[MultiView] _tensor_to_pil: averaged {tensor.shape[0]} frames")
            else:
                arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return [] 

class Trellis2RenderMultiView:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "ortho_scale": ("FLOAT", {"default": 1.1, "min": 0.05, "max": 10.0, "step": 0.01}),
                "blender_exec_path": ("STRING",),
                "azimuths": ("STRING",{"default":"0,90,180,270,0,0"}),
                "elevations": ("STRING",{"default":"0,0,0,0,90,-90"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("images","ortho_scale", "azimuths", "elevations",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True
    
    def process(
        self,
        trimesh,
        render_size,
        ortho_scale,
        blender_exec_path,
        azimuths,
        elevations
    ):
        reset_cuda()
        
        if not hasattr(trimesh.visual, 'material'):
            raise Exception("Trimesh does not have a material")
                    
        custom_az_list = self._parse_angles(azimuths)
        custom_el_list = self._parse_angles(elevations)                    
                    
        if custom_az_list and custom_el_list:
            if len(custom_az_list) != len(custom_el_list):
                raise Exception("azimuths and elevations must have the same amount of values")
                
            textured_maps = self.render_textured_multiview(
                custom_el_list, custom_az_list, ortho_scale, render_size, blender_exec_path, trimesh)
            custom_images = torch.stack(textured_maps, dim=0)

            return (custom_images, ortho_scale, azimuths, elevations,)
        else:
            raise Exception("azimuths and elevations are required")
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []         

    def render_textured_multiview(self, camera_elevs, camera_azims, ortho_scale, resolution, blender_exec_path, mesh):
        from .projection.camera_utils import get_orthographic_projection_matrix
        
        proj = get_orthographic_projection_matrix(
            left=-ortho_scale * 0.5, right=ortho_scale * 0.5,
            bottom=-ortho_scale * 0.5, top=ortho_scale * 0.5,
            near=0.1, far=100
        )        
        textured_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            textured_map = self.render(
                elev, azim, filter_mode='linear', return_type='th', scale=ortho_scale, resolution=resolution, blender_exec_path=blender_exec_path,proj=proj,mesh=mesh)
            textured_maps.append(textured_map)
            
        return textured_maps

    def render(
        self,
        elev,
        azim,
        camera_distance=None,
        center=None,
        resolution=None,
        tex=None,
        keep_alpha=False,
        bgcolor=None,
        filter_mode=None,
        return_type='th',
        scale=1.0,
        blender_exec_path=None,
        proj=None,
        mesh=None,
    ):
        from .projection.camera_utils import get_mv_matrix
        
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=1.1,
            center=center)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        if tex is not None:
            if isinstance(tex, Image.Image):
                tex = torch.tensor(np.array(tex) / 255.0)
            elif isinstance(tex, np.ndarray):
                tex = torch.tensor(tex)
            if tex.dim() == 2:
                tex = tex.unsqueeze(-1)
            tex = tex.float().to(self.device)
        # image = self._render(r_mvp, self.vtx_pos, self.pos_idx, self.vtx_uv, self.uv_idx,
                             # self.tex if tex is None else tex,
                             # self.default_resolution if resolution is None else resolution,
                             # self.max_mip_level, True, filter_mode if filter_mode else self.filter_mode,
                             # elev=elev, azim=azim, camera_distance=camera_distance,scale=scale,blender_exec_path=blender_exec_path)
        image = self.raster_texture(tex, mesh.visual.uv, elev=elev, azim=azim, camera_distance=camera_distance, resolution=resolution, scale=scale, blender_exec_path=blender_exec_path,mesh=mesh)
        mask = (image[..., [-1]] == 1).float()
        if bgcolor is None:
            bgcolor = [0 for _ in range(image.shape[-1] - 1)]
        image = image * mask + (1 - mask) * \
                torch.tensor(bgcolor + [0])
        if keep_alpha == False:
            image = image[..., :-1]
        if return_type == 'np':
            image = image.cpu().numpy()
        elif return_type == 'pl':
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        return image      

    def raster_texture(self, tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto',
                       boundary_mode='wrap', max_mip_level=None, elev=None, azim=None, camera_distance=None, resolution=None, scale=1.0,
                       blender_exec_path=None,mesh=None):
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp_mesh:
            mesh_path = tmp_mesh.name
            tmp_mesh.close()
        
        mesh.export(mesh_path)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            output_path = tmp_out.name
            tmp_out.close()
            
        blender_script = os.path.join(os.path.dirname(__file__), 'projection', 'blender_render.py')
        
        res = resolution[0] if isinstance(resolution, (list, tuple)) else resolution
        
        cmd = [
            blender_exec_path, '-b', '-P', blender_script, '--',
            '--mesh', mesh_path,
            '--output', output_path,
            '--elev', str(elev),
            '--azim', str(azim),
            '--scale', str(scale),
            '--resolution', str(res)
        ]
        
        subprocess.run(cmd, check=True)
        
        image = Image.open(output_path)
        image = torch.tensor(np.array(image) / 255.0).float()
        
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return image          
            
class Trellis2CudaReset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,),
            },
        }

    RETURN_TYPES = (any, )
    RETURN_NAMES = ("output_1", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1):
        reset_cuda()
        return (input_1,)          
        
class Trellis2SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "compress_level": ("INT",{"default":4,"min":1,"max":9,"step":1}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("images_path",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
    SEARCH_ALIASES = ["save", "save image", "export image", "output image", "write image", "download"]

    def save_images(self, images, filename_prefix="ComfyUI", compress_level=4, prompt=None, extra_pnginfo=None):
        from comfy.cli_args import args
        from PIL.PngImagePlugin import PngInfo
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        file_list = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            file_list.append(full_path)
            img.save(full_path, pnginfo=metadata, compress_level=compress_level)
            counter += 1

        if len(file_list)==1:
            file_list = file_list[0]
            
        return (file_list,)
        
class Trellis2VoxelToMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_height_mm": ("FLOAT",{"default":0.0,"min":0.0,"max":500.0,"step":0.1}),
                "sigma": ("FLOAT",{"default":0.0,"min":0.0,"max":9.9,"step":0.1}),
                "coarse_downsample": ("FLOAT",{"default":1.00,"min":1.00,"max":9.00,"step":0.01}),
                "taubin_iterations": ("INT",{"default":0,"min":0,"max":999,"step":1}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_height_mm, sigma, coarse_downsample, taubin_iterations):
        mesh_copy = copy.deepcopy(mesh)
        
        if mesh_copy.coords is None:
            raise Exception("Voxel to Mesh requires Texture Slat")
        else:
            from .blackwell_fix import voxel_to_mesh
            
            trimesh = voxel_to_mesh(mesh_output = mesh_copy, 
                                    target_height_mm = target_height_mm,
                                    sigma = sigma,
                                    coarse_downsample = coarse_downsample,
                                    taubin_iterations = taubin_iterations,
                                    verbose = True)

            mesh_copy.vertices = torch.from_numpy(trimesh.vertices).float()
            mesh_copy.faces = torch.from_numpy(trimesh.faces).int()
        
            del trimesh
        
        return (mesh_copy,)
        
class Trellis2UnloadAllModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_1": (any,)
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output_1",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, input_1):
        print('Unloading all models ...')
        if hasattr(mm, 'current_loaded_models'):
            # Iterate backwards to safely remove items
            for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                loaded_model = mm.current_loaded_models[i]
                
                print(f"[AbsoluteUnload] Force-killing: {loaded_model.model.model.__class__.__name__}")
                
                # Force VRAM unload
                loaded_model.model_unload(1e32)
                
                # Force System RAM unpinning (This is what the standard loop skipped)
                if hasattr(loaded_model.model, 'partially_unload_ram'):
                    loaded_model.model.partially_unload_ram(1e32)
                    
            # Clear ComfyUI's intermediate cross-attention and tensor caches
            if hasattr(mm, 'current_loaded_models'):
                mm.current_loaded_models.clear()            

            import comfy.controlnet
            if hasattr(comfy.controlnet, 'controlnet_loaded_models'):
                comfy.controlnet.controlnet_loaded_models.clear() 
                
        mm.free_memory(memory_required = 1e30, 
                       device = mm.get_torch_device(),
                       ram_required = 1e30)
                       
        print('Clearing cache ...')
        mm.soft_empty_cache()

        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print('Memory cleared')
        
        return (input_1,)    

class Trellis2SparseGeneratorWithReconViaGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, images, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        fill_holes,
        hole_iterations,
        hole_fill_algorithm,
        keep_only_shell
        ):
        
        self.seed_all(seed)
        
        self.load_vggt_model(pipeline)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        args = pipeline._pretrained_args
        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.sparse_structure_sampler = getattr(samplers, f"Flow{sparse_sampler_prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])
        pipeline.load_sparse_structure_vggt_model()
        pipeline.load_sparse_structure_vggt_cond()
        
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        coords = self._run_ss_stage_direct(pipeline = pipeline, 
                                           images = images, 
                                           target_ss_res = sparse_structure_resolution, 
                                           ss_sampler_params = sparse_structure_sampler_params, 
                                           verbose = verbose, 
                                           dino_lock = dino_lock, 
                                           dino_substeps = dino_substeps, 
                                           dino_foundation_cap = dino_foundation_cap, 
                                           fill_holes = fill_holes, 
                                           hole_iterations = hole_iterations, 
                                           hole_fill_algorithm = hole_fill_algorithm, 
                                           keep_only_shell = keep_only_shell)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_sparse_structure_vggt_model()            
            pipeline.unload_sparse_structure_vggt_cond()
            self.unload_vggt_model(pipeline)

        return (coords, sparse_structure_resolution, pipeline)
        
    def load_vggt_model(self, pipeline):
        if pipeline.VGGT_model is None:
            from .vggt.vggt.models.vggt import VGGT
            pipeline.VGGT_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16            
            model_path = os.path.join(folder_paths.models_dir,'recongenvia')
            pipeline.VGGT_model = VGGT.from_pretrained(model_path)
            pipeline.VGGT_model.to('cuda')
            del pipeline.VGGT_model.depth_head
            del pipeline.VGGT_model.track_head
            pipeline.VGGT_model.eval()
                
            self._init_image_cond_model(pipeline)
            
        
    def unload_vggt_model(self, pipeline):
        del pipeline.VGGT_model
        pipeline.VGGT_model = None
        
        del pipeline.models['image_cond_model_vggt']
        pipeline.models['image_cond_model_vggt'] = None
        pipeline.image_cond_model_transform = None          
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()        
        
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    @torch.no_grad()
    def _run_ss_stage_direct(
        self,
        pipeline,
        images,
        target_ss_res: int,
        ss_sampler_params: dict,
        verbose: bool,
        dino_lock: float,
        dino_substeps: int,
        dino_foundation_cap: float,
        fill_holes: bool,
        hole_iterations: int,
        hole_fill_algorithm: str,
        hole_structure: int = 1,
        keep_only_shell: bool = True        
    ) -> torch.Tensor:
        """
        Run only ReconViaGen's sparse structure diffusion stage to obtain coords
        directly, without proceeding to the SLAT/mesh stage.

        Returns:
            coords : (N, 4) int tensor  [batch_idx, x, y, z]  in [0, target_ss_res)
        """           
            
        cuda_device = torch.device('cuda')

        if pipeline.low_vram:
            pipeline.VGGT_model.to(cuda_device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                aggregated_tokens_list, _ = self.vggt_feat(pipeline, images)
            b, n, _, _ = aggregated_tokens_list[0].shape
            image_cond = self.encode_image(pipeline, images).reshape(b, n, -1, 1024)
            ss_cond = self.get_ss_cond(pipeline, image_cond[:, :, 5:], aggregated_tokens_list, 1)

        ss_flow_model = pipeline.models['sparse_structure_flow_vggt_model']
        sampler_params = {**pipeline.sparse_structure_sampler_params, **ss_sampler_params}
        reso = ss_flow_model.resolution
        ss_noise = torch.randn(1, ss_flow_model.in_channels, reso, reso, reso).to(cuda_device)

        with torch.autocast('cuda', dtype=torch.float16):
            ss_latent = pipeline.sparse_structure_sampler.sample(
                ss_flow_model,
                ss_noise,
                **ss_cond,
                **sampler_params,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
            ).samples

        decoder = pipeline.models['sparse_structure_decoder']
        decoded = decoder(ss_latent) > 0
        if target_ss_res != decoded.shape[2]:
            ratio = decoded.shape[2] // target_ss_res
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        
        # Optionally fill holes in the sparse voxel grid using the selected algorithm
        if fill_holes:
            try:
                from scipy.ndimage import binary_closing, label, binary_fill_holes
                arr = decoded.cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[:, 0]
                closed = np.zeros_like(arr)
                for b in range(arr.shape[0]):
                    filled = arr[b].astype(np.bool_)
                    inv = ~filled
                    labeled, num_features = label(inv)
                    border_mask = np.zeros_like(inv)
                    border_mask[0,:,:] = border_mask[-1,:,:] = 1
                    border_mask[:,0,:] = border_mask[:,-1,:] = 1
                    border_mask[:,:,0] = border_mask[:,:,-1] = 1
                    border_labels = np.unique(labeled[border_mask==1])
                    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                    n_holes = np.unique(labeled[holes]).size
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")
                    if hole_fill_algorithm == "morphological_closing":
                        closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                    elif hole_fill_algorithm == "flood_fill":
                        # Robust structure-preserving hole filling:
                        # 1. Morphological closing to connect small gaps
                        # 2. Fill internal holes
                        # 3. Keep only the largest connected component
                        from scipy.ndimage import binary_closing, label, binary_fill_holes
                        # Step 1: Morphological closing (small structure, 1 iter)
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                        # Step 2: Fill internal holes
                        filled = binary_fill_holes(closed1)
                        # Step 3: Keep only the largest connected component
                        labeled, num = label(filled)
                        if num > 0:
                            sizes = np.bincount(labeled.ravel())
                            sizes[0] = 0  # background
                            largest = sizes.argmax()
                            closed[b] = (labeled == largest)
                        else:
                            closed[b] = filled                      
                    elif hole_fill_algorithm == "remove_small_holes":
                        # Remove small holes by area (2D slices)
                        from skimage.morphology import remove_small_holes
                        # Apply per-slice (z axis)
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure**2)
                        closed[b] = temp
                    else:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                        closed[b] = arr[b]
                    # Count holes after filling
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask==1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes-n_holes2} holes filled, {n_holes2} remain after filling.")

                    # Optionally remove deeply interior voxels, keeping surface and near-surface structure
                    if keep_only_shell:
                        from scipy.ndimage import binary_erosion

                        filled = closed[b].astype(np.bool_)
                        before_count = int(filled.sum())
                        struct = np.ones((3, 3, 3), dtype=bool)
                        # Erode twice: only voxels surviving 2 erosion passes are >=2 layers deep
                        # This preserves thin structures (e.g. necks with 3x3 cross-section)
                        eroded = binary_erosion(filled, structure=struct, border_value=0)
                        eroded = binary_erosion(eroded, structure=struct, border_value=0)
                        # Remove only deeply interior voxels (>=2 layers from any surface)
                        shell = filled & ~eroded
                        closed[b] = shell
                        after_count = int(shell.sum())
                        if verbose:
                            print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

                decoded = torch.from_numpy(closed).to(decoded.device)


                # Debug: print tensor info before extracting coordinates
                if verbose:
                    print(f"[Sparse HoleFill] decoded shape: {decoded.shape}, dtype: {decoded.dtype}, device: {decoded.device}")
                    print(f"[Sparse HoleFill] decoded min: {decoded.min().item()}, max: {decoded.max().item()}, unique: {torch.unique(decoded)}")
                # Safety: ensure tensor is contiguous and on CPU for argwhere
                decoded = decoded.contiguous().cpu()

            except ImportError:
                print("[Warning] scipy or skimage not installed, skipping hole filling.")
            except Exception as e:
                print(f"[Warning] Hole filling failed: {e}")

            try:
                coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
            except Exception as e:
                print(f"[Sparse HoleFill] Error in torch.argwhere: {e}")
                raise

            if verbose:
                print(f"[Sparse HoleFill] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist() if coords.numel()>0 else 'empty'}, max: {coords.max(dim=0).values.tolist() if coords.numel()>0 else 'empty'}")

            if coords.numel() == 0:
                raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is empty. Adjust your input, mask, or hole filling parameters.")
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        if pipeline.low_vram:
            pipeline.VGGT_model.to('cpu')
            decoder.to('cpu')
            ss_cond = pipeline._cond_cpu(ss_cond)
            torch.cuda.empty_cache()

        return coords
        
        
    @torch.no_grad()
    def _run_ss_stage(
        self,
        pipeline,
        images,
        target_ss_res: int,
        ss_sampler_params: dict,
        slat_sampler_params: dict,
    ) -> torch.Tensor:
        """
        Generate a rough mesh via vggt_pipeline, then voxelise it into
        surface-only coords at target_ss_res^3 for the downstream shape/tex stages.

        Returns:
            coords : (N, 4) int tensor  [batch_idx, x, y, z]  in [0, target_ss_res)
        """
        vp = self.vggt_pipeline
        # vp.device is dynamic (inferred from model params), so when models are on
        # CPU it returns 'cpu'. Hardcode the target cuda device instead.
        cuda_device = torch.device('cuda')

        if self.low_vram:
            self._vggt_models_to(cuda_device)

        outputs, _, _ = vp.run(
            image=images,
            formats=["mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params=ss_sampler_params,
            slat_sampler_params=slat_sampler_params,
        )
        mesh_result = outputs["mesh"][0]
        coords = self._mesh_to_surface_coords(mesh_result, target_ss_res, cuda_device)

        if self.low_vram:
            self._vggt_models_to('cpu')
            torch.cuda.empty_cache()

        return coords        

    @torch.no_grad()
    def vggt_feat(self, pipeline, image):
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, H, W, C) or (B, C, H, W)"
            # ComfyUI IMAGE tensors are (B, H, W, C); convert to (B, C, H, W)
            if image.shape[-1] in (3, 4):
                image = image.permute(0, 3, 1, 2)
            image = F.interpolate(image, 518, mode='bilinear', align_corners=False)
            image = image.to(pipeline.device)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(pipeline.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pipeline.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, _ = pipeline.VGGT_model.aggregator(image[None])

        return aggregated_tokens_list, image

    def get_ss_cond(self, pipeline, image_cond: torch.Tensor, aggregated_tokens_list: list, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        pipeline.models['sparse_structure_vggt_cond'].to(pipeline.device)
        cond = pipeline.models['sparse_structure_vggt_cond'](aggregated_tokens_list, image_cond)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_slat_cond(self, pipeline, image_cond: torch.Tensor, aggregated_tokens_list: list, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        b, n, _, _ = aggregated_tokens_list[0].shape
        cond = pipeline.models['slat_vggt_cond'](aggregated_tokens_list, image_cond).reshape(b, n, -1, 1024)
        cond = [c.squeeze(1) for c in cond.split(1, dim=1)]
        neg_cond = [torch.zeros_like(c) for c in cond]
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }     

    @torch.no_grad()
    def encode_image(self, pipeline, image, w_layernorm=True) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, H, W, C) or (B, C, H, W)"
            # ComfyUI IMAGE tensors are (B, H, W, C); convert to (B, C, H, W)
            if image.shape[-1] in (3, 4):
                image = image.permute(0, 3, 1, 2)
            image = F.interpolate(image, 518, mode='bilinear', align_corners=False)
            image = image.to(pipeline.device)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(pipeline.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = pipeline.image_cond_model_transform(image).to(pipeline.device)
        pipeline.models['image_cond_model_vggt'].to(pipeline.device)
        features = pipeline.models['image_cond_model_vggt'](image, is_training=True)['x_prenorm']
        if w_layernorm:
            features = F.layer_norm(features, features.shape[-1:])
        return features

    def _init_image_cond_model(self, pipeline, name: str = "dinov2_vitl14_reg"):
        """
        Initialize the image conditioning model.
        """
        try:
            dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), name, source='local',pretrained=True)
        except:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        pipeline.models['image_cond_model_vggt'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pipeline.image_cond_model_transform = transform  

class Trellis2ExtractImagesFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_file": ("STRING",),
                "frames_per_second": ("INT",{"default":1,"min":1,"max":50,"step":1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, video_file, frames_per_second):
        import imageio
        
        vid = imageio.get_reader(video_file, 'ffmpeg')
        fps = vid.get_meta_data()['fps']
        frames = []
        for i, frame in enumerate(vid):
            if i % max(int(fps/frames_per_second), 1) == 0:
                img = Image.fromarray(frame)
                W, H = img.size
                img = img.resize((int(W / H * 1024), 1024))
                frames.append(img)
        vid.close()
        
        tensor_list = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0) for img in frames]
            
        print(f"{len(frames)} frames extracted")
        
        tensor_frames = torch.stack(tensor_list)
        #tensor_frames = tensor_frames.permute(0, 2, 3, 1)
        
        return (tensor_frames,)
        
class Trellis2MaxTokensCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_max_tokens": ("INT",{"default":999999,"min":1,"max":999999,"step":1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("max_tokens",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, default_max_tokens):
        device = mm.get_torch_device()
        try:
            total_vram_bytes = mm.get_total_memory(device)
            total_vram_gb = total_vram_bytes / (1024 ** 3)
            max_tokens = int(2600 * total_vram_gb)
        except Exception as e:
            print(f"Error, cannot get VRAM size : {e}")
            max_tokens = default_max_tokens
        
        return (max_tokens,)  
        
class Trellis2ImageCondMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "front_image": ("IMAGE",)
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },            
        }

    RETURN_TYPES = ("IMAGE_CONDS", "IMAGE_CONDS", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("conds_512", "conds_1024", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, 
        pipeline, 
        front_image, 
        back_image = None,
        left_image = None,
        right_image = None):

        front_pil = tensor2pil(front_image)
        
        # Convert optional view image tensors to PIL
        back_pil = tensor2pil(back_image) if back_image is not None else None
        left_pil = tensor2pil(left_image) if left_image is not None else None
        right_pil = tensor2pil(right_image) if right_image is not None else None           
            
        # Collect views
        views_dict = {'front': front_pil}
        if back_pil is not None: views_dict['back'] = back_pil
        if left_pil is not None: views_dict['left'] = left_pil
        if right_pil is not None: views_dict['right'] = right_pil
        
        views_list = list(views_dict.keys())            
        
        # Calculate conditioning per view
        conds_512 = {}
        conds_1024 = {}
        
        pipeline.load_image_cond_model() 

        for v, img in views_dict.items():
            c512 = pipeline.get_cond([img], 512)
            c1024 = pipeline.get_cond([img], 1024)
            conds_512[v] = c512
            conds_1024[v] = c1024
            
        if not pipeline.keep_models_loaded:
            pipeline.unload_image_cond_model()              

        return (conds_512, conds_1024, views_list, pipeline,)          

class Trellis2SparseMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("COORDS", "INT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, seed, 
        # sparse
        sparse_structure_steps, 
        sparse_structure_guidance_strength, 
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        front_axis,
        blend_temperature
        ):
        
        self.seed_all(seed)
        
        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}                    

        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.load_sparse_structure_model()                
            
        if pipeline.low_vram:
            for v in image_conds:
                image_conds[v] = pipeline._cond_to(image_conds[v], pipeline.device)
                
        # Sample sparse structure latent
        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(1, in_channels, reso, reso, reso).to(pipeline.device)
        
        sampler_class = getattr(samplers, f"Flow{sparse_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )        
        
        sparse_structure_sampler_params = {**pipeline.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
            
        z_s = sampler.sample(
            flow_model,
            noise,
            conds=image_conds,            
            **sparse_structure_sampler_params,            
            views=views_list,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling sparse structure (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()
            
        # Decode sparse structure latent
        decoder = pipeline.models['sparse_structure_decoder']
        if pipeline.low_vram:
            decoder.to(pipeline.device)
            
        # Standard decoding logic from sample_sparse_structure
        decoded = decoder(z_s) > 0
        
        if pipeline.low_vram:
            decoder.cpu()
            pipeline._cleanup_cuda()
            
        # if resolution != decoded.shape[2]:
            # ratio = decoded.shape[2] // resolution
            # decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        if sparse_structure_resolution != decoded.shape[2]:
            if sparse_structure_resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // sparse_structure_resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(sparse_structure_resolution, sparse_structure_resolution, sparse_structure_resolution), mode='nearest') > 0.5            

        # Optionally fill holes in the sparse voxel grid using the selected algorithm
        if fill_holes:
            hole_structure = 1
            try:
                from scipy.ndimage import binary_closing, label, binary_fill_holes
                arr = decoded.cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[:, 0]
                closed = np.zeros_like(arr)
                for b in range(arr.shape[0]):
                    filled = arr[b].astype(np.bool_)
                    inv = ~filled
                    labeled, num_features = label(inv)
                    border_mask = np.zeros_like(inv)
                    border_mask[0,:,:] = border_mask[-1,:,:] = 1
                    border_mask[:,0,:] = border_mask[:,-1,:] = 1
                    border_mask[:,:,0] = border_mask[:,:,-1] = 1
                    border_labels = np.unique(labeled[border_mask==1])
                    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                    n_holes = np.unique(labeled[holes]).size
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")
                    if hole_fill_algorithm == "morphological_closing":
                        closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                    elif hole_fill_algorithm == "flood_fill":
                        # Robust structure-preserving hole filling:
                        # 1. Morphological closing to connect small gaps
                        # 2. Fill internal holes
                        # 3. Keep only the largest connected component
                        from scipy.ndimage import binary_closing, label, binary_fill_holes
                        # Step 1: Morphological closing (small structure, 1 iter)
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                        # Step 2: Fill internal holes
                        filled = binary_fill_holes(closed1)
                        # Step 3: Keep only the largest connected component
                        labeled, num = label(filled)
                        if num > 0:
                            sizes = np.bincount(labeled.ravel())
                            sizes[0] = 0  # background
                            largest = sizes.argmax()
                            closed[b] = (labeled == largest)
                        else:
                            closed[b] = filled                      
                    elif hole_fill_algorithm == "remove_small_holes":
                        # Remove small holes by area (2D slices)
                        from skimage.morphology import remove_small_holes
                        # Apply per-slice (z axis)
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure**2)
                        closed[b] = temp
                    else:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                        closed[b] = arr[b]
                    # Count holes after filling
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask==1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes-n_holes2} holes filled, {n_holes2} remain after filling.")

                    # Optionally remove deeply interior voxels, keeping surface and near-surface structure
                    if keep_only_shell:
                        from scipy.ndimage import binary_erosion

                        filled = closed[b].astype(np.bool_)
                        before_count = int(filled.sum())
                        struct = np.ones((3, 3, 3), dtype=bool)
                        # Erode twice: only voxels surviving 2 erosion passes are >=2 layers deep
                        # This preserves thin structures (e.g. necks with 3x3 cross-section)
                        eroded = binary_erosion(filled, structure=struct, border_value=0)
                        eroded = binary_erosion(eroded, structure=struct, border_value=0)
                        # Remove only deeply interior voxels (>=2 layers from any surface)
                        shell = filled & ~eroded
                        closed[b] = shell
                        after_count = int(shell.sum())
                        if verbose:
                            print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

                decoded = torch.from_numpy(closed).to(decoded.device)


                # Debug: print tensor info before extracting coordinates
                if verbose:
                    print(f"[Sparse HoleFill] decoded shape: {decoded.shape}, dtype: {decoded.dtype}, device: {decoded.device}")
                    print(f"[Sparse HoleFill] decoded min: {decoded.min().item()}, max: {decoded.max().item()}, unique: {torch.unique(decoded)}")
                # Safety: ensure tensor is contiguous and on CPU for argwhere
                decoded = decoded.contiguous().cpu()

            except ImportError:
                print("[Warning] scipy or skimage not installed, skipping hole filling.")
            except Exception as e:
                print(f"[Warning] Hole filling failed: {e}")

            try:
                coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
            except Exception as e:
                print(f"[Sparse HoleFill] Error in torch.argwhere: {e}")
                raise

            if verbose:
                print(f"[Sparse HoleFill] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist() if coords.numel()>0 else 'empty'}, max: {coords.max(dim=0).values.tolist() if coords.numel()>0 else 'empty'}")

            if coords.numel() == 0:
                raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is empty. Adjust your input, mask, or hole filling parameters.")
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        coords = coords.cpu()

        del decoded
        del z_s
        if pipeline.low_vram:
            for v in image_conds:
                image_conds[v] = pipeline._cond_cpu(image_conds[v])
            pipeline._cleanup_cuda()

        return (coords, sparse_structure_resolution, views_list, pipeline,)
        
    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   

class Trellis2ShapeMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "coords": ("COORDS",),
                "resolution": ([512,1024],{"default":1024}),                
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),  
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT", "INT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("shape_slat", "resolution", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, coords, resolution,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}            
        
        if resolution == 512:
             pipeline.load_shape_slat_flow_model_512()
             shape_slat = self.sample(
                pipeline, shape_sampler,
                image_conds, views_list,
                pipeline.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
             )
             if not pipeline.keep_models_loaded:
                 pipeline.unload_shape_slat_flow_model_512()
                 
        elif resolution == 1024:
             pipeline.load_shape_slat_flow_model_1024()
             shape_slat = self.sample(
                pipeline, shape_sampler,
                image_conds, views_list,
                pipeline.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
             )
             if not pipeline.keep_models_loaded:
                 pipeline.unload_shape_slat_flow_model_1024()
        
        return (shape_slat, resolution, views_list, pipeline,)
        
    def sample(
        self,
        pipeline,
        shape_sampler,
        conds: dict,
        views: list,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        verbose: bool = False,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)

        coords_dev = coords.to(pipeline.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        
        # sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            # sigma_min=1e-5,
            # resolution=flow_model.resolution,
        # )
        sampler_class = getattr(samplers, f"Flow{shape_sampler}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )        
        
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
            
        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,            
            **sampler_params,            
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock = dino_lock,
            dino_substeps = dino_substeps,
            dino_foundation_cap = dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()

        return slat        
        
class Trellis2ShapeCascadeMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "shape_slat": ("SHAPE_SLAT",),
                "from_resolution": ("INT",),
                "to_resolution": ([1024,1536],{"default":1024}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),                
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                   
            },
        }

    RETURN_TYPES = ("SHAPE_SLAT","INT","VIEWS_LIST", "TRELLIS2PIPELINE","INT",)
    RETURN_NAMES = ("shape_slat","resolution","views_list", "pipeline","num_tokens")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens,      
        # shape
        shape_steps, 
        shape_guidance_strength, 
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):
            
        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]        
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}                    
        
        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        
        pipeline.load_shape_slat_flow_model_1024()         
        flow_model = pipeline.models['shape_slat_flow_model_1024']
        
        sampler_class = getattr(samplers, f"Flow{shape_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )
        
        pipeline.shape_slat_sampler = sampler

        slat, hr_resolution, num_tokens = self.sample(pipeline, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens, image_conds, shape_slat_sampler_params, flow_model, verbose, dino_lock, dino_substeps, dino_foundation_cap, views_list, front_axis, blend_temperature)
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_flow_model_1024()              
        
        return (slat, hr_resolution, views_list, pipeline, num_tokens,)         
        
    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution, max_num_tokens, conds, sampler_params, flow_model, verbose, dino_lock, dino_substeps, dino_foundation_cap, views, front_axis, blend_temperature):
        # Upsample       
        pipeline.load_shape_slat_decoder()
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].to(pipeline.device)
            pipeline.models['shape_slat_decoder'].low_vram = True
        hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].cpu()
            pipeline.models['shape_slat_decoder'].low_vram = False
        
        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_decoder()
        
        hr_resolution = resolution
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
                
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)               
        
        coords_dev = coords.to(pipeline.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            noise,
            conds=conds,            
            **sampler_params,            
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (MultiView HR)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()                                

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()

        return slat, hr_resolution, num_tokens         
        
class Trellis2TexSlatMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image_conds": ("IMAGE_CONDS",),
                "views_list": ("VIEWS_LIST",),
                "shape_slat": ("SHAPE_SLAT",),
                "resolution": ([512,1024],{"default":1024}),                
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "texture_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "texture_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),         
                "texture_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),                                                               
                "texture_guidance_interval_start": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "texture_guidance_interval_end": ("FLOAT",{"default":0.90,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "front_axis": (["z", "x"], {"default": "z"}),
                "blend_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),                    
            },
        }

    RETURN_TYPES = ("TEXTURE_SLAT", "VIEWS_LIST", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("texture_slat", "views_list", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image_conds, views_list, shape_slat, resolution,      
        # shape
        texture_steps, 
        texture_guidance_strength, 
        texture_guidance_rescale,
        texture_rescale_t,
        texture_sampler,
        texture_guidance_interval_start,
        texture_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        front_axis,
        blend_temperature
        ):

        texture_guidance_interval = [texture_guidance_interval_start,texture_guidance_interval_end]
        tex_slat_sampler_params = {"steps":texture_steps,"guidance_strength":texture_guidance_strength,"guidance_rescale":texture_guidance_rescale,"guidance_interval":texture_guidance_interval,"rescale_t":texture_rescale_t}
        
        if resolution == 512:
            pipeline.load_tex_slat_flow_model_512()
            flow_model = pipeline.models['tex_slat_flow_model_512']
        else:
            pipeline.load_tex_slat_flow_model_1024()
            flow_model = pipeline.models['tex_slat_flow_model_1024']
        
        tex_slat = self.sample(
            pipeline,
            texture_sampler,
            image_conds, views_list,
            shape_slat=shape_slat, 
            flow_model=flow_model,
            sampler_params=tex_slat_sampler_params,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap
        )  
         
        if not pipeline.keep_models_loaded:
            if resolution == 512:
                pipeline.unload_tex_slat_flow_model_512()
            else:
                pipeline.unload_tex_slat_flow_model_1024()
        
        return (tex_slat, views_list, pipeline,) 

    def sample(
        self,
        pipeline,
        sampler,
        conds: dict,
        views: list,
        shape_slat: SparseTensor,
        flow_model,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        verbose: bool = False,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        """
        Sample structured latent for texture with multi-view blending.
        """
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_to(conds[v], pipeline.device)

        # Normalize shape slat for conditioning
        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        #coords = shape_slat.coords
        #coords_dev = coords.to(pipeline.device)
        
        # Calculate noise channels: total input - concat cond channels
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise_channels = in_channels - shape_slat.feats.shape[1]
        
        # noise = SparseTensor(
            # feats=torch.randn(coords.shape[0], noise_channels, device=pipeline.device),
            # coords=coords_dev,
        # )
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(pipeline.device))
        
        sampler_params = {**pipeline.tex_slat_sampler_params, **sampler_params}
        
        # sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            # sigma_min=1e-5,
            # resolution=flow_model.resolution,
        # )
        tex_sampler_prefix = pipeline.GetSamplerName(sampler)
        
        sampler_class = getattr(samplers, f"Flow{tex_sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )          
        
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
            
        slat = sampler.sample(
            flow_model,
            noise,
            conds=conds,
            **sampler_params,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,            
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples
        
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()

        std = torch.tensor(pipeline.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        #del coords_dev
        if pipeline.low_vram:
            for v in conds:
                conds[v] = pipeline._cond_cpu(conds[v])
            pipeline._cleanup_cuda()
            
        return slat        
        
class Trellis2AlignmentPreview:
    """
    Renders the mesh silhouette overlaid on a source image using the exact same
    auto-alignment math as Trellis2MultiViewTexturing.  Run this first to tune
    u_pixel_offset / v_pixel_offset without waiting for the full projection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh":   ("TRIMESH",),
                "image":     ("IMAGE",),
            },
            "optional": {
                "azimuth":        ("FLOAT",   {"default": 0.0,   "min": -360.0, "max": 360.0, "step": 1.0,
                                               "tooltip": "Camera azimuth in degrees. 0=front, 90=left, 180=back, 270=right."}),
                "elevation":      ("FLOAT",   {"default": 0.0,   "min": -90.0,  "max": 90.0,  "step": 1.0,
                                               "tooltip": "Camera elevation in degrees. 0=horizontal, 90=top-down."}),
                "ortho_scale":    ("FLOAT",   {"default": 1.15,  "min": 0.1,    "max": 10.0,  "step": 0.05,
                                               "tooltip": "Must match the ortho_scale in Trellis2MultiViewTexturing."}),
                "norm_size":      ("FLOAT",   {"default": 1.15,  "min": 0.1,    "max": 10.0,  "step": 0.05,
                                               "tooltip": "Must match norm_size in Trellis2MultiViewTexturing."}),
                "x_pixel_offset": ("FLOAT",   {"default": 0.0,   "min": -2048.0,"max": 2048.0,"step": 1.0,
                                               "tooltip": "Horizontal pixel offset. Positive = mesh shifts RIGHT. Negative = mesh shifts LEFT."}),
                "y_pixel_offset": ("FLOAT",   {"default": 0.0,   "min": -2048.0,"max": 2048.0,"step": 1.0,
                                               "tooltip": "Vertical pixel offset. Positive = mesh shifts DOWN. Negative = mesh shifts UP."}),
                "overlay_alpha":  ("FLOAT",   {"default": 0.30,  "min": 0.0,    "max": 1.0,   "step": 0.05,
                                               "tooltip": "Opacity of the semi-transparent fill inside the mesh silhouette."}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("preview",)
    FUNCTION      = "process"
    CATEGORY      = "Trellis2Wrapper"

    def process(self, trimesh, image,
                azimuth=0.0, elevation=0.0,
                ortho_scale=1.15, norm_size=1.15,
                x_pixel_offset=0.0, y_pixel_offset=0.0,
                overlay_alpha=0.30):
        from .projection.texture_projection_multiview import render_alignment_preview

        img_pil = tensor2pil(image)

        result = render_alignment_preview(
            mesh           = trimesh,
            image          = img_pil,
            azimuth        = float(azimuth),
            elevation      = float(elevation),
            ortho_scale    = float(ortho_scale),
            norm_size      = float(norm_size),
            u_pixel_offset = float(x_pixel_offset),
            v_pixel_offset = float(y_pixel_offset),
            overlay_alpha  = float(overlay_alpha),
        )

        return (pil2tensor(result),)


class Trellis2FaceAlignPreview:
    """
    KAN-9 fast per-feature alignment check.  Renders the mesh's geometric relief
    (orange edges = nose/eyes/lips/jaw) onto the source photo using the exact same
    auto-alignment math as the texturing node, with the photo's own features in
    cyan.  Orange-on-cyan = aligned; any offset is the misalignment that smears
    colour during the real bake.  Runs in well under a second — use this instead
    of the full pipeline to tune u/v offsets or diagnose scale/proportion drift.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh":   ("TRIMESH",),
                "image":     ("IMAGE",),
            },
            "optional": {
                "azimuth":        ("FLOAT",   {"default": 0.0,   "min": -360.0, "max": 360.0, "step": 1.0,
                                               "tooltip": "Camera azimuth. 0=front, 90=left, 180=back, 270=right. Use 0 for the front face check."}),
                "elevation":      ("FLOAT",   {"default": 0.0,   "min": -90.0,  "max": 90.0,  "step": 1.0}),
                "ortho_scale":    ("FLOAT",   {"default": 1.15,  "min": 0.1,    "max": 10.0,  "step": 0.05,
                                               "tooltip": "MUST match the ortho_scale used in Trellis2MultiViewTexturing."}),
                "norm_size":      ("FLOAT",   {"default": 1.15,  "min": 0.1,    "max": 10.0,  "step": 0.05,
                                               "tooltip": "MUST match norm_size used in Trellis2MultiViewTexturing."}),
                "blend":          ("FLOAT",   {"default": 0.0,   "min": 0.0,    "max": 1.0,   "step": 0.05,
                                               "tooltip": "Fill the shaded mesh relief over the photo (0=just edges, 0.5=relief visible). Edges alone are usually clearest."}),
                "show_mesh_edges":   ("BOOLEAN", {"default": True,
                                               "tooltip": "Orange = the MESH's geometric feature edges (where the bake will place colour)."}),
                "show_source_edges": ("BOOLEAN", {"default": True,
                                               "tooltip": "Cyan = the PHOTO's feature edges (where colour should come from)."}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("preview",)
    FUNCTION      = "process"
    CATEGORY      = "Trellis2Wrapper"

    def process(self, trimesh, image,
                azimuth=0.0, elevation=0.0,
                ortho_scale=1.15, norm_size=1.15,
                blend=0.0, show_mesh_edges=True, show_source_edges=True):
        from .projection.texture_projection_multiview import render_face_projection_preview

        img_pil = tensor2pil(image)

        result = render_face_projection_preview(
            mesh              = trimesh,
            image             = img_pil,
            azimuth           = float(azimuth),
            elevation         = float(elevation),
            ortho_scale       = float(ortho_scale),
            norm_size         = float(norm_size),
            blend             = float(blend),
            show_mesh_edges   = bool(show_mesh_edges),
            show_source_edges = bool(show_source_edges),
        )

        return (pil2tensor(result),)


NODE_CLASS_MAPPINGS = {
    "Trellis2LoadModel": Trellis2LoadModel,
    "Trellis2MeshWithVoxelGenerator": Trellis2MeshWithVoxelGenerator,
    "Trellis2LoadImageWithTransparency": Trellis2LoadImageWithTransparency,
    "Trellis2SimplifyMesh": Trellis2SimplifyMesh,
    "Trellis2MeshWithVoxelToTrimesh": Trellis2MeshWithVoxelToTrimesh,
    "Trellis2ExportMesh": Trellis2ExportMesh,
    "Trellis2PostProcessMesh": Trellis2PostProcessMesh,
    "Trellis2UnWrapAndRasterizer": Trellis2UnWrapAndRasterizer,
    "Trellis2MeshWithVoxelAdvancedGenerator": Trellis2MeshWithVoxelAdvancedGenerator,
    "Trellis2PostProcessAndUnWrapAndRasterizer": Trellis2PostProcessAndUnWrapAndRasterizer,
    "Trellis2Remesh": Trellis2Remesh,
    "Trellis2MeshTexturing": Trellis2MeshTexturing,
    "Trellis2LoadMesh": Trellis2LoadMesh,
    "Trellis2PreProcessImage": Trellis2PreProcessImage,
    "Trellis2MeshRefiner": Trellis2MeshRefiner,
    "Trellis2PostProcess2": Trellis2PostProcess2,
    "Trellis2OvoxelExportToGLB": Trellis2OvoxelExportToGLB,
    "Trellis2TrimeshToMeshWithVoxel": Trellis2TrimeshToMeshWithVoxel,
    "Trellis2SimplifyTrimesh": Trellis2SimplifyTrimesh,
    "Trellis2Continue": Trellis2Continue,
    "Trellis2ProgressiveSimplify": Trellis2ProgressiveSimplify,
    "Trellis2ReconstructMesh": Trellis2ReconstructMesh,
    "Trellis2MeshWithVoxelToMeshlibMesh": Trellis2MeshWithVoxelToMeshlibMesh,
    "Trellis2FillHolesWithMeshlib": Trellis2FillHolesWithMeshlib,
    "Trellis2SmoothNormals": Trellis2SmoothNormals,
    "Trellis2RemeshWithQuad": Trellis2RemeshWithQuad,
    "Trellis2BatchSimplifyMeshAndExport": Trellis2BatchSimplifyMeshAndExport,
    "Trellis2MeshWithVoxelMultiViewGenerator": Trellis2MeshWithVoxelMultiViewGenerator,
    "Trellis2MeshTexturingMultiView": Trellis2MeshTexturingMultiView,
    "Trellis2WeldVertices": Trellis2WeldVertices,
    "Trellis2ReconstructMeshWithQuad": Trellis2ReconstructMeshWithQuad,
    "Trellis2StringSelector": Trellis2StringSelector,
    "Trellis2FillHolesWithCuMesh": Trellis2FillHolesWithCuMesh,
    "Trellis2LaplacianSmoothingWithOpen3d": Trellis2LaplacianSmoothingWithOpen3d,
    "Trellis2UnWrapTrimesh": Trellis2UnWrapTrimesh,
    "Trellis2MeshWithVoxelCascadeGenerator": Trellis2MeshWithVoxelCascadeGenerator,
    "Trellis2ImageCondGenerator": Trellis2ImageCondGenerator,
    "Trellis2SparseGenerator": Trellis2SparseGenerator,
    "Trellis2ShapeGenerator": Trellis2ShapeGenerator,
    "Trellis2ShapeCascadeGenerator": Trellis2ShapeCascadeGenerator,
    "Trellis2TexSlatGenerator": Trellis2TexSlatGenerator,
    "Trellis2DecodeLatents": Trellis2DecodeLatents,
    "Trellis2SimplifyMeshAdvanced": Trellis2SimplifyMeshAdvanced,
    "Trellis2SimplifyTrimeshAdvanced": Trellis2SimplifyTrimeshAdvanced,
    "Trellis2MultiViewTexturing": Trellis2MultiViewTexturing,
    "Trellis2Continue3": Trellis2Continue3,
    "Trellis2Continue4": Trellis2Continue4,
    "Trellis2Continue5": Trellis2Continue5,
    "Trellis2Continue6": Trellis2Continue6,
    "Trellis2CudaReset": Trellis2CudaReset,
    "Trellis2ProjectHighPolyToLowPoly": Trellis2ProjectHighPolyToLowPoly,
    "Trellis2RenderMultiView": Trellis2RenderMultiView,
    "Trellis2SaveImage": Trellis2SaveImage,
    "Trellis2VoxelToMesh": Trellis2VoxelToMesh,
    "Trellis2UnloadAllModels": Trellis2UnloadAllModels,
    "Trellis2SparseGeneratorWithReconViaGen": Trellis2SparseGeneratorWithReconViaGen,
    "Trellis2ExtractImagesFromVideo": Trellis2ExtractImagesFromVideo,
    "Trellis2MaxTokensCalculator": Trellis2MaxTokensCalculator,
    "Trellis2FillHolesNicelyWithMeshlib": Trellis2FillHolesNicelyWithMeshlib,
    "Trellis2SparseMultiViewGenerator": Trellis2SparseMultiViewGenerator,
    "Trellis2ImageCondMultiViewGenerator": Trellis2ImageCondMultiViewGenerator,
    "Trellis2ShapeMultiViewGenerator": Trellis2ShapeMultiViewGenerator,
    "Trellis2ShapeCascadeMultiViewGenerator": Trellis2ShapeCascadeMultiViewGenerator,
    "Trellis2TexSlatMultiViewGenerator": Trellis2TexSlatMultiViewGenerator,
    "Trellis2RenderViewFromMesh": Trellis2RenderViewFromMesh,
    "Trellis2AlignmentPreview": Trellis2AlignmentPreview,
    "Trellis2FaceAlignPreview": Trellis2FaceAlignPreview,
    }
    

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadModel": "Trellis2 - LoadModel",
    "Trellis2MeshWithVoxelGenerator": "Trellis2 - Mesh With Voxel Generator",
    "Trellis2LoadImageWithTransparency": "Trellis2 - Load Image with Transparency",
    "Trellis2SimplifyMesh": "Trellis2 - Simplify Mesh",
    "Trellis2MeshWithVoxelToTrimesh": "Trellis2 - Mesh With Voxel To Trimesh",
    "Trellis2ExportMesh": "Trellis2 - Export Mesh",
    "Trellis2PostProcessMesh": "Trellis2 - PostProcess Mesh (using Cumesh)",
    "Trellis2UnWrapAndRasterizer": "Trellis2 - UV Unwrap and Rasterize",
    "Trellis2MeshWithVoxelAdvancedGenerator": "Trellis2 - Mesh With Voxel Advanced Generator",
    "Trellis2PostProcessAndUnWrapAndRasterizer": "Trellis2 - Post Process/UnWrap and Rasterize",
    "Trellis2Remesh": "Trellis2 - Remesh",
    "Trellis2MeshTexturing": "Trellis2 - Mesh Texturing",
    "Trellis2LoadMesh": "Trellis2 - Load Mesh",
    "Trellis2PreProcessImage": "Trellis2 - PreProcess Image",
    "Trellis2MeshRefiner": "Trellis2 - Mesh Refiner",
    "Trellis2PostProcess2": "Trellis2 - PostProcess Mesh (using Trimesh)",
    "Trellis2OvoxelExportToGLB": "Trellis2 - Ovoxel Export to GLB",
    "Trellis2TrimeshToMeshWithVoxel": "Trellis2 - Trimesh to Mesh with Voxel",
    "Trellis2SimplifyTrimesh": "Trellis2 - Simplify Trimesh",
    "Trellis2Continue": "Trellis2 - Continue",
    "Trellis2ProgressiveSimplify": "Trellis2 - Progressive Simplify",
    "Trellis2ReconstructMesh": "Trellis2 - Reconstruct Mesh",
    "Trellis2MeshWithVoxelToMeshlibMesh": "Trellis2 - Mesh with Voxel to Meshlib Mesh",
    "Trellis2FillHolesWithMeshlib": "Trellis2 - Fill Holes with Meshlib",
    "Trellis2SmoothNormals": "Trellis2 - Smooth Normals",
    "Trellis2RemeshWithQuad": "Trellis2 - Remesh With Quad",
    "Trellis2BatchSimplifyMeshAndExport": "Trellis2 - Batch Simplify Mesh And Export",
    "Trellis2MeshWithVoxelMultiViewGenerator": "Trellis2 - Mesh With Voxel Multi-View Generator",
    "Trellis2MeshTexturingMultiView": "Trellis2 - Mesh Texturing Multi-View",
    "Trellis2WeldVertices": "Trellis2 - Weld Vertices",
    "Trellis2ReconstructMeshWithQuad": "Trellis2 - Reconstruct Mesh With Quad",
    "Trellis2StringSelector": "Trellis2 - String Selector",
    "Trellis2FillHolesWithCuMesh": "Trellis2 - Fill Holes with CuMesh",
    "Trellis2LaplacianSmoothingWithOpen3d": "Trellis2 - Laplacian Smoothing (using open3d)",
    "Trellis2UnWrapTrimesh": "Trellis2 - UnWrap Trimesh",
    "Trellis2MeshWithVoxelCascadeGenerator": "Trellis2 - Mesh With Voxel Cascade Generator",
    "Trellis2ImageCondGenerator": "Trellis2 - ImageCond Generator",
    "Trellis2SparseGenerator": "Trellis2 - Sparse Generator",
    "Trellis2ShapeGenerator": "Trellis2 - Shape Generator",
    "Trellis2ShapeCascadeGenerator": "Trellis2 - Shape Cascade Generator",
    "Trellis2TexSlatGenerator": "Trellis2 - Tex Slat Generator",
    "Trellis2DecodeLatents": "Trellis2 - Decode Latents",
    "Trellis2SimplifyMeshAdvanced": "Trellis2 - Simplify Mesh Advanced",
    "Trellis2SimplifyTrimeshAdvanced": "Trellis2 - Simplify Trimesh Advanced",
    "Trellis2MultiViewTexturing": "Trellis2 - Projection MultiView Texturing",
    "Trellis2Continue3": "Trellis2 - Continue 3",
    "Trellis2Continue4": "Trellis2 - Continue 4",
    "Trellis2Continue5": "Trellis2 - Continue 5",
    "Trellis2Continue6": "Trellis2 - Continue 6",
    "Trellis2CudaReset": "Trellis2 - Cuda Reset",
    "Trellis2ProjectHighPolyToLowPoly": "Trellis2 - Projection HighPoly To LowPoly",
    "Trellis2RenderMultiView": "Trellis2 - Render MultiView",
    "Trellis2SaveImage": "Trellis2 - Save Image",
    "Trellis2VoxelToMesh": "Trellis2 - Voxel to Mesh",
    "Trellis2UnloadAllModels": "Trellis2 - Unload All ComfyUI Models",
    "Trellis2SparseGeneratorWithReconViaGen": "Trellis2 - Sparse Generator with ReconViaGen",
    "Trellis2ExtractImagesFromVideo": "Trellis2 - Extract Images from Video",
    "Trellis2MaxTokensCalculator": "Trellis2 - Max Tokens Calculator",
    "Trellis2FillHolesNicelyWithMeshlib": "Trellis2 - Fill Holes Nicely With Meshlib",
    "Trellis2SparseMultiViewGenerator": "Trellis2 - Sparse MultiView Generator",
    "Trellis2ImageCondMultiViewGenerator": "Trellis2 - ImageCond MultiView Generator",
    "Trellis2ShapeMultiViewGenerator": "Trellis2 - Shape MultiView Generator",
    "Trellis2ShapeCascadeMultiViewGenerator": "Trellis2 - Shape Cascade MultiView Generator",
    "Trellis2TexSlatMultiViewGenerator": "Trellis2 - Tex Slat MultiView Generator",
    "Trellis2RenderViewFromMesh": "Trellis2 - Render View From Mesh",
    "Trellis2AlignmentPreview":  "Trellis2 - Alignment Preview",
    "Trellis2FaceAlignPreview":  "Trellis2 - Face Align Preview (fast, KAN-9)",
    }
