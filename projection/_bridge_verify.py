"""Verify (source=_faceear_attempted) pushed through the bridge, measured in GEN space.
This is what last_seams will show. Decide if it's good before plumbing."""
import numpy as np, os, io, contextlib, sys, importlib.util
from scipy.spatial import cKDTree
_dir = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
_uv  = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
sys.path.insert(0, _dir)

d = np.load(os.path.join(_dir,'hair_detection_fixture.npz')); TEX=int(d['texture_size'][0])
uvm = d['uv_mask'].astype(bool)
spec = importlib.util.spec_from_file_location('fea', os.path.join(_dir,'hair_detection_utils_v2_faceear_attempted.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
hdu = m.HairDetectionUtils(TEX, uvm, d['tex_pos'], d['tex_normals'],
                           _norm_verts=d['norm_verts'], _norm_faces=d['norm_faces'],
                           _norm_uvs=d['norm_uvs'], trimesh=1)
with contextlib.redirect_stdout(io.StringIO()):
    mask = hdu.get_hair_mask().astype(bool)
tp = d['tex_pos'][uvm]; th = mask[uvm]
tree = cKDTree(tp)

v = np.load(os.path.join(_uv,'_genseams_viz.npz'))
cen = v['co'][v['tris']].mean(axis=1).astype(np.float64)
gx,gy,gz = cen[:,0],cen[:,1],cen[:,2]
def map_axis(g,src,dst,flip):
    s1,s99=np.percentile(src,[1,99]); t=(g-s1)/(s99-s1)
    if flip: t=1-t
    d1,d99=np.percentile(dst,[1,99]); return d1+t*(d99-d1)
Pq = np.column_stack([map_axis(gx,gx,tp[:,0],False),
                      map_axis(gz,gz,tp[:,1],False),
                      map_axis(gy,gy,tp[:,2],True)])
_,idx = tree.query(Pq,k=1,workers=-1); bh = th[idx]   # bridged hair per gen tri

# gen-space regions (x=lat y=depth(front-) z=height)
aX=np.abs(gx)
crown  = gz>0.46
drape  = (gy>0.03)&(gz>0.15)&(gz<0.45)&(aX<0.08)
sscalp = (aX>0.05)&(aX<0.10)&(gz>0.36)&(gz<0.46)&(np.abs(gy)<0.04)
lowside= (aX>0.04)&(aX<0.095)&(gz>0.18)&(gz<0.38)&(gy<0.0)
face   = (gy<-0.02)&(gz>0.33)&(gz<0.46)&(aX<0.06)
ear    = (aX>0.04)&(aX<0.10)&(gz>0.42)&(gz<0.47)&(gy>-0.03)&(gy<0.04)
def c(h,r): return 100*h[r].mean() if r.sum() else float('nan')
print("BRIDGED (_faceear_attempted -> gen space):")
print(f"  crown {c(bh,crown):.0f}  drape {c(bh,drape):.0f}  sscalp {c(bh,sscalp):.0f}  "
      f"lowside {c(bh,lowside):.0f} | FACE {c(bh,face):.0f}  EAR {c(bh,ear):.0f}")
print(f"  total hair {100*bh.mean():.1f}%   mapped-hair x-extent "
      f"[{gx[bh].min():.2f},{gx[bh].max():.2f}]  z-extent [{gz[bh].min():.2f},{gz[bh].max():.2f}]")
# tight ear-cut effect (height+depth box, NOT |X|)
ecut = (aX>0.035)&(gz>0.415)&(gz<0.475)&(gy>-0.035)&(gy<0.045)
bh2 = bh & ~ecut
print("AFTER tight ear-cut (z[0.415,0.475] & depth y[-0.035,0.045] & |x|>0.035):")
print(f"  crown {c(bh2,crown):.0f}  drape {c(bh2,drape):.0f}  sscalp {c(bh2,sscalp):.0f}  "
      f"lowside {c(bh2,lowside):.0f} | FACE {c(bh2,face):.0f}  EAR {c(bh2,ear):.0f}")
