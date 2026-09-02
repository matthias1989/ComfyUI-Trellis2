"""
Bridge test: map the REAL hair detector's per-texel hair result (tex_pos space)
into gen_seams' coordinate space, and verify the mapping is sane.

gen_seams space (co):   col0=lateral(x)  col1=depth(y, front=min)  col2=height(z)
tex_pos space:          col0=lateral(X)  col1=height(Y)            col2=depth(Z, front=max)
"""
import numpy as np, os, io, contextlib, sys
_dir = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
_uv  = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'
sys.path.insert(0, _dir)
from scipy.spatial import cKDTree

# ── 1. Real detector → texel hair (tex_pos space) ──────────────────────────────
d = np.load(os.path.join(_dir, 'hair_detection_fixture.npz'))
TEX = int(d['texture_size'][0])
from hair_detection_utils_v2 import HairDetectionUtils
hdu = HairDetectionUtils(TEX, d['uv_mask'].astype(bool), d['tex_pos'], d['tex_normals'],
                         _norm_verts=d['norm_verts'], _norm_faces=d['norm_faces'],
                         _norm_uvs=d['norm_uvs'], trimesh=1)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mask = hdu.get_hair_mask().astype(bool)
uvm = d['uv_mask'].astype(bool)
texel_pos  = d['tex_pos'][uvm]            # (Npix,3) tex_pos space
texel_hair = mask[uvm]                    # (Npix,) bool
print(f"[1] texels={texel_pos.shape[0]:,}  hair_texels={int(texel_hair.sum()):,} "
      f"({100*texel_hair.mean():.1f}%)")
print(f"    tex_pos ranges  X[{texel_pos[:,0].min():.3f},{texel_pos[:,0].max():.3f}] "
      f"Y[{texel_pos[:,1].min():.3f},{texel_pos[:,1].max():.3f}] "
      f"Z[{texel_pos[:,2].min():.3f},{texel_pos[:,2].max():.3f}]")

# ── 2. gen_seams faces (its space) from the viz npz ───────────────────────────
v = np.load(os.path.join(_uv, '_genseams_viz.npz'))
co = v['co'].astype(np.float64); tris = v['tris']; tris_hair = v['tris_hair'].astype(bool)
cen = co[tris].mean(axis=1)               # (Ntri,3) gen space
gx, gy, gz = cen[:,0], cen[:,1], cen[:,2]
print(f"[2] gen tris={cen.shape[0]:,}  gen_own_hair={int(tris_hair.sum()):,} "
      f"({100*tris_hair.mean():.1f}%)")
print(f"    gen ranges  x[{gx.min():.3f},{gx.max():.3f}] "
      f"y[{gy.min():.3f},{gy.max():.3f}] z[{gz.min():.3f},{gz.max():.3f}]")

# ── 3. Transform gen → tex_pos (per-axis, percentile-fit, depth flipped) ──────
def map_axis(g, src, dst, flip):
    s1, s99 = np.percentile(src, [1, 99]); t = (g - s1) / (s99 - s1)
    if flip: t = 1.0 - t
    d1, d99 = np.percentile(dst, [1, 99]); return d1 + t * (d99 - d1)

# lateral: gen x -> tex X ; height: gen z -> tex Y ; depth: gen y -> tex Z (flip)
def build(flip_lat, flip_dep):
    tX = map_axis(gx, gx, texel_pos[:,0], flip_lat)
    tY = map_axis(gz, gz, texel_pos[:,1], False)
    tZ = map_axis(gy, gy, texel_pos[:,2], flip_dep)
    return np.column_stack([tX, tY, tZ])

tree = cKDTree(texel_pos)
for flip_lat in (False, True):
    for flip_dep in (False, True):
        P = build(flip_lat, flip_dep)
        dist, idx = tree.query(P, k=1, workers=-1)
        gen_hair = texel_hair[idx]
        # agreement with gen's OWN shell hair on the back drape (both should be hair)
        drape = (gz > 0.34) & (gy > 0.05)
        face  = (gz > 0.30) & (gz < 0.45) & (gy < -0.02) & (np.abs(gx) < 0.05)
        print(f"[3] flip_lat={flip_lat} flip_dep={flip_dep}  "
              f"mapped_hair={100*gen_hair.mean():.1f}%  "
              f"meanNN={dist.mean():.4f}  "
              f"drape_hair={100*gen_hair[drape].mean():.0f}%  "
              f"face_hair={100*gen_hair[face].mean():.0f}%")

print("\n--- detail for flip_lat=False flip_dep=True (expected correct) ---")
P = build(False, True); dist, idx = tree.query(P, k=1, workers=-1); gh = texel_hair[idx]
# spatial extent of mapped hair (gen space) — should be head + back drape
hx, hy, hz = gx[gh], gy[gh], gz[gh]
print(f"mapped-hair extent  x[{hx.min():.2f},{hx.max():.2f}] "
      f"y[{hy.min():.2f},{hy.max():.2f}] z[{hz.min():.2f},{hz.max():.2f}]")
# how much mapped-hair is BELOW the head (z<0.30) = should be only the drape/nape
low = gh & (gz < 0.30)
print(f"mapped-hair below z=0.30: {100*low.sum()/max(gh.sum(),1):.0f}% of hair "
      f"(drape/nape ok; arms/legs bad — check x,y)")
if low.sum():
    print(f"   of that low hair:  |x|>0.20 (arms?) {100*(np.abs(gx[low])>0.20).mean():.0f}%  "
          f"z<-0.10 (legs?) {100*(gz[low]<-0.10).mean():.0f}%")
