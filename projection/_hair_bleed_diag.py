"""KAN-8 hair-BLEED diagnostic.  Renders the current hair mask over the whole
upper body (front + both sides) so we can see where hair leaks onto neck / chin
/ shoulders, and quantifies bleed in defined no-hair zones.

Yellow = hair mask.  Gray = not hair.  Outputs _bleed_front/_bleed_sideR/L.png."""
import numpy as np, os, io, contextlib, sys
import cv2
_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _dir)
d = np.load(os.path.join(_dir, "hair_detection_fixture.npz"))
TEX = int(d["texture_size"][0])
from hair_detection_utils_v2 import HairDetectionUtils
_kw = dict(_norm_verts=d["norm_verts"], _norm_faces=d["norm_faces"],
           _norm_uvs=d["norm_uvs"], trimesh=1) if "norm_verts" in d else {}
hdu = HairDetectionUtils(TEX, d["uv_mask"].astype(bool), d["tex_pos"], d["tex_normals"], **_kw)
with contextlib.redirect_stdout(io.StringIO()):
    mask = hdu.get_hair_mask()

ys, xs = np.where(d["uv_mask"].astype(bool))
P = d["tex_pos"][ys, xs]; px, py, pz = P[:,0], P[:,1], P[:,2]
cov = mask[ys, xs]
ymax, ymin = py.max(), py.min()

def render(H_axis, V_axis, sel, fname, title, h0, h1, v0, v1):
    if sel.sum()==0: print(f"{title}: none"); return
    A=H_axis[sel]; B=V_axis[sel]; C=cov[sel]
    # Equal world-units-per-pixel on both axes so nothing is stretched.
    span = max(h1-h0, v1-v0)
    hc=(h0+h1)/2; vc=(v0+v1)/2
    h0,h1 = hc-span/2, hc+span/2; v0,v1 = vc-span/2, vc+span/2
    H=W=560
    col=np.clip((A-h0)/(h1-h0)*(W-1),0,W-1).astype(int)
    row=np.clip((v1-B)/(v1-v0)*(H-1),0,H-1).astype(int)
    img=np.full((H,W,3),35,np.uint8)
    img[row[~C],col[~C]]=(150,150,150)
    img[row[C],col[C]]=(0,220,220)
    cv2.putText(img,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.imwrite(os.path.join(_dir,fname),img)
    print(f"{title}: texels={int(sel.sum()):,} hair={int(C.sum()):,} -> {fname}")

# FRONT: X horiz, Y vert (upper body)
upper = py > -0.15
render(px, py, upper, "_bleed_front.png", "FRONT (X-Y) hair mask", -0.30,0.30, -0.15, ymax+0.02)
# SIDE R: Z horiz (front=right), Y vert, right half
render(pz, py, upper & (px>0.02), "_bleed_sideR.png", "R SIDE (Z-Y) front=right", -0.10,0.10, -0.15, ymax+0.02)
render(pz, py, upper & (px<-0.02), "_bleed_sideL.png", "L SIDE (Z-Y) front=right", -0.10,0.10, -0.15, ymax+0.02)

# ── Quantify bleed in no-hair zones ──────────────────────────────────────────
print("\n[bleed zones] hair texels that are almost certainly WRONG:")
# Front neck/chin: forward-facing, below jaw, central
fz = {
 "front-neck/chin (z>0.02, py[0.30,0.42], |X|<0.10)": (pz>0.02)&(py>=0.30)&(py<0.42)&(np.abs(px)<0.10),
 "front-lower-neck (z>0.02, py[0.20,0.30], |X|<0.12)": (pz>0.02)&(py>=0.20)&(py<0.30)&(np.abs(px)<0.12),
 "shoulders (py[0.10,0.30], |X|>0.12)": (py>=0.10)&(py<0.30)&(np.abs(px)>0.12),
 "jaw-sides (z>0.0, py[0.34,0.42], |X|[0.06,0.12])": (pz>0.0)&(py>=0.34)&(py<0.42)&(np.abs(px)>=0.06)&(np.abs(px)<0.12),
}
for name,m in fz.items():
    n=int(m.sum()); h=int(cov[m].sum())
    print(f"   {name}: {h:,}/{n:,} hair ({100*h/max(n,1):.0f}%)")

# What SHOULD stay (back hair / ponytail) — sanity it's still covered
keep = (pz<-0.02)&(py>-0.15)&(np.abs(px)<0.12)
print(f"\n[keep] back hair/ponytail (z<-0.02, |X|<0.12): {int(cov[keep].sum()):,}/{int(keep.sum()):,} hair "
      f"({100*cov[keep].mean():.0f}%) — must stay high")
