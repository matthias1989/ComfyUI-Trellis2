"""BRIDGE map step (python_embeded).
Reads gen_seams' exported face centroids (_gs_cen.npy, gen space), runs the REAL
hairline detector (_faceear_attempted) on the fixture, maps each gen face to its
nearest atlas texel, and writes per-face hair (_gs_ishair.npy) for gen_seams.

gen space:  x=lateral  y=depth(front=min)  z=height
tex_pos  :  X=lateral  Y=height            Z=depth(front=max)   -> depth flipped
"""
import numpy as np, os, io, contextlib, sys, importlib.util
from scipy.spatial import cKDTree
_proj = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
_uv   = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
sys.path.insert(0, _proj)

# real detector -> texel hair (tex_pos space)
d = np.load(os.path.join(_proj, 'hair_detection_fixture.npz')); TEX = int(d['texture_size'][0])
spec = importlib.util.spec_from_file_location('fea', os.path.join(_proj, 'hair_detection_utils_v2_faceear_attempted.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
uvm = d['uv_mask'].astype(bool)
hdu = m.HairDetectionUtils(TEX, uvm, d['tex_pos'], d['tex_normals'],
                           _norm_verts=d['norm_verts'], _norm_faces=d['norm_faces'],
                           _norm_uvs=d['norm_uvs'], trimesh=1)
with contextlib.redirect_stdout(io.StringIO()):
    mask = hdu.get_hair_mask().astype(bool)        # full hair region (cap border + coverage)
print(f"[map] using full hair mask: {int(mask.sum()):,} texels")
tp = d['tex_pos'][uvm]; th = mask[uvm]
tree = cKDTree(tp)

# gen centroids
cen_path = os.path.join(_uv, '_gs_cen.npy')
if not os.path.exists(cen_path):
    print("ERROR: _gs_cen.npy missing — run gen_seams with GS_EXPORT_CEN=1 first"); sys.exit(1)
cen = np.load(cen_path).astype(np.float64)
gx, gy, gz = cen[:,0], cen[:,1], cen[:,2]

def map_axis(g, src, dst, flip):
    s1, s99 = np.percentile(src, [1, 99]); t = (g - s1) / (s99 - s1)
    if flip: t = 1.0 - t
    d1, d99 = np.percentile(dst, [1, 99]); return d1 + t * (d99 - d1)

P = np.column_stack([map_axis(gx, gx, tp[:,0], False),   # lateral
                     map_axis(gz, gz, tp[:,1], False),   # height
                     map_axis(gy, gy, tp[:,2], True)])   # depth (flip)
dist, idx = tree.query(P, k=1, workers=-1)
gh = th[idx]
np.save(os.path.join(_uv, '_gs_ishair.npy'), gh)
print(f"[map] gen faces={len(gh):,}  hair={int(gh.sum()):,} ({100*gh.mean():.1f}%)  "
      f"meanNN={dist.mean():.4f}  src_hair_texels={100*th.mean():.1f}%")
