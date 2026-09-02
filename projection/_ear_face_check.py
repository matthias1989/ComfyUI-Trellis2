"""Confirm the ear(|X|~0.15) vs scalp(|X|<0.10) gap exists in FACE space,
in both tex_pos space (fixture) and gen_seams space (viz), so a fixed lateral
cut separates ear from scalp cleanly."""
import numpy as np, os
_dir = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
_uv  = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer'

def hist(aX, lab):
    edges = np.arange(0.04, 0.20, 0.01); h,_ = np.histogram(aX, bins=edges)
    mx = max(h.max(),1)
    print(f"  {lab}:")
    for i,c in enumerate(h):
        print(f"     |X| {edges[i]:.2f}-{edges[i+1]:.2f}: {c:6d} {'#'*int(c/mx*44)}")

# FACE space (tex_pos): centroid |X| in ear-height band
d = np.load(os.path.join(_dir,'hair_detection_fixture.npz'))
C = d['norm_verts'][d['norm_faces']].mean(axis=1)   # face centroids, tex_pos space
X,Yh,Z = C[:,0],C[:,1],C[:,2]
band = (Yh>0.39)&(Yh<0.46)
print("FACE centroids (tex_pos space)  X=lateral Y=height Z=depth")
hist(np.abs(X[band]), "all faces, ear-band Y[0.39,0.46]")

# gen_seams space (viz): tri centroid |x| in ear-height band (z=height there)
v = np.load(os.path.join(_uv,'_genseams_viz.npz'))
cen = v['co'][v['tris']].mean(axis=1)                # gen space: x=lat y=depth z=height
gx,gy,gz = cen[:,0],cen[:,1],cen[:,2]
bandg = (gz>0.39)&(gz<0.46)
print("\nGEN centroids (gen_seams space)  x=lateral y=depth z=height")
hist(np.abs(gx[bandg]), "all tris, ear-band z[0.39,0.46]")
print(f"\ntex_pos head |X| p99 (band): {np.percentile(np.abs(X[band]),99):.3f}   "
      f"gen head |x| p99 (band): {np.percentile(np.abs(gx[bandg]),99):.3f}")
