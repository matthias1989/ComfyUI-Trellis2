"""Render the ear region per-face, coloured by mask status, to see exactly what
is yellow (hair) on/around the ear and why.

Per face (in the ear neighborhood) we sample its UV centroid in the atlas:
  YELLOW = in hair mask (painted hair)
  TAN    = in ear override mask (painted skin)
  GRAY   = neither (projection skin)
Rendered as a side view (pz horizontal front=right, py vertical) for each side."""
import numpy as np, os, io, contextlib, sys
import cv2
_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _dir)
d = np.load(os.path.join(_dir, "hair_detection_fixture.npz"))
TEX = int(d["texture_size"][0])
from hair_detection_utils_v2 import HairDetectionUtils
hdu = HairDetectionUtils(TEX, d["uv_mask"].astype(bool), d["tex_pos"], d["tex_normals"],
    _norm_verts=d["norm_verts"], _norm_faces=d["norm_faces"], _norm_uvs=d["norm_uvs"], trimesh=1)
with contextlib.redirect_stdout(io.StringIO()):
    mask = hdu.get_hair_mask()
ear_uv = hdu.get_ear_uv_mask()

V = d["norm_verts"].astype(np.float64); F = d["norm_faces"].astype(np.int64)
UV = d["norm_uvs"].astype(np.float64)
cent = V[F].mean(1); cx, cy, cz = cent[:,0], cent[:,1], cent[:,2]
uvc = UV[F].mean(1)                       # face UV centroid
pcol = np.clip(uvc[:,0]*TEX, 0, TEX-1).astype(int)
prow = np.clip(uvc[:,1]*TEX, 0, TEX-1).astype(int)   # v*TEX (matches rasterizer)
is_hair = mask[prow, pcol]
is_ear  = ear_uv[prow, pcol] if ear_uv is not None else np.zeros(len(F), bool)

ymax = cy.max(); ymin = cy.min(); yr = ymax-ymin
T_head = ymax-0.28*yr; T_hair = ymax-0.20*(ymax-T_head); T_sf = T_head+0.60*(T_hair-T_head)

def render(side_sel, fname, title):
    sel = side_sel & (cy > 0.39)
    if sel.sum()==0:
        print(f"{title}: none"); return
    Z=cz[sel]; Y=cy[sel]; hh=is_hair[sel]; ee=is_ear[sel]
    H=W=520; z0,z1=-0.10,0.10; y0,y1=0.39,ymax+0.01
    col=np.clip((Z-z0)/(z1-z0)*(W-1),0,W-1).astype(int)
    row=np.clip((y1-Y)/(y1-y0)*(H-1),0,H-1).astype(int)
    img=np.full((H,W,3),30,np.uint8)
    img[row[~hh & ~ee], col[~hh & ~ee]] = (160,160,160)   # gray neither
    img[row[ee], col[ee]] = (150,190,225)                 # tan-ish (BGR) ear override
    img[row[hh & ~ee], col[hh & ~ee]] = (0,220,220)       # yellow hair
    cv2.line(img,(W//2,0),(W//2,H-1),(80,80,80),1)
    cv2.putText(img,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.imwrite(os.path.join(_dir,fname),img)
    n=int(sel.sum())
    print(f"{title}: faces={n}  hair={int(hh.sum())} ear-skin={int(ee.sum())} neither={int((~hh&~ee).sum())} -> {fname}")

render(cx>0.03,  "_ear_R.png", "R EAR side (front=right)")
render(cx<-0.03, "_ear_L.png", "L EAR side (front=right)")

# Quantify: faces that are HAIR and lateral & in ear Y-band (the yellow-on-ear)
ear_y = (cy>0.42)&(cy<0.485)
for thr in [0.04,0.05,0.06,0.07]:
    m = ear_y & (np.abs(cx)>thr) & is_hair
    print(f"  hair faces |X|>{thr:.2f} in ear Y[0.42,0.485]: {int(m.sum())}")
