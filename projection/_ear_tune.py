"""Tune a fixed-|X| ear cut on top of _faceear_attempted (the best-coverage base).
Find the scalp<->ear lateral gap, then pick the threshold that drops EAR without
killing side-scalp / lowside (the big missing zone below+behind the ear)."""
import numpy as np, os, io, contextlib, sys, importlib.util
_dir = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
sys.path.insert(0, _dir)
d = np.load(os.path.join(_dir, 'hair_detection_fixture.npz')); TEX = int(d['texture_size'][0])
uvm = d['uv_mask'].astype(bool); P = d['tex_pos'][uvm]
X, Y, Z = P[:,0], P[:,1], P[:,2]; aX = np.abs(X)

spec = importlib.util.spec_from_file_location('fea', os.path.join(_dir,'hair_detection_utils_v2_faceear_attempted.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
hdu = m.HairDetectionUtils(TEX, uvm, d['tex_pos'], d['tex_normals'],
                           _norm_verts=d['norm_verts'], _norm_faces=d['norm_faces'],
                           _norm_uvs=d['norm_uvs'], trimesh=1)
with contextlib.redirect_stdout(io.StringIO()):
    base = hdu.get_hair_mask().astype(bool)
h0 = base[uvm]

# lateral structure of hair in the ear height band
band = (Y > 0.39) & (Y < 0.46)
hb = h0 & band
print("HAIR |X| histogram in ear-band Y[0.39,0.46] (where is scalp vs ear?):")
edges = np.arange(0.04, 0.18, 0.01)
hist,_ = np.histogram(aX[hb], bins=edges)
for i,c in enumerate(hist):
    print(f"   |X| {edges[i]:.2f}-{edges[i+1]:.2f}: {c:5d}  {'#'*int(c/max(hist.max(),1)*40)}")

crown   = Y > 0.46
drape   = (Z < -0.04) & (Y > 0.15) & (Y < 0.45) & (aX < 0.07)
sscalp  = (aX > 0.055) & (aX < 0.098) & (Y > 0.36) & (Y < 0.46) & (np.abs(Z) < 0.04)
lowside = (aX > 0.04)  & (aX < 0.095) & (Y > 0.18) & (Y < 0.38) & (Z < 0.0)
ear     = (aX > 0.105) & (Y > 0.37)   & (Y < 0.47) & (np.abs(Z) < 0.06)
face    = (Z > 0.04)   & (Y > 0.33)   & (Y < 0.46) & (aX < 0.06)
def cov(h,r): return 100*h[r].mean() if r.sum() else float('nan')

print(f"\n{'cut':<14}{'crown':>6}{'drape':>6}{'sscalp':>7}{'lowside':>8}|{'EAR':>5}{'FACE':>5}")
print(f"{'(base)':<14}{cov(h0,crown):>6.0f}{cov(h0,drape):>6.0f}{cov(h0,sscalp):>7.0f}"
      f"{cov(h0,lowside):>8.0f}|{cov(h0,ear):>5.0f}{cov(h0,face):>5.0f}")
for thr in [0.095, 0.100, 0.105, 0.110, 0.120]:
    cut = (aX > thr) & (Y > 0.34) & (Y < 0.48)
    h = h0 & ~cut
    print(f"|X|>{thr:.3f}     {cov(h,crown):>6.0f}{cov(h,drape):>6.0f}{cov(h,sscalp):>7.0f}"
          f"{cov(h,lowside):>8.0f}|{cov(h,ear):>5.0f}{cov(h,face):>5.0f}")
print("\nwant: crown/drape/sscalp/lowside HIGH ; EAR ~0 ; FACE 0")
