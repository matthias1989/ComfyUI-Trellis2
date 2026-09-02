"""KAN-8 hair-coverage diagnostic.  Loads the fixture, runs get_hair_mask, and
renders synthetic front/back ortho views of the head/hair zone coloured by mask
coverage so we can SEE the back-center seam band and the forehead gap.

Yellow = covered by hair mask.  Red = head/hair-zone texel NOT covered (gap).
Outputs _cov_back.png and _cov_front.png next to this file.  Text stats too."""
import numpy as np, os, sys, io, contextlib
import cv2

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
fx = os.path.join(_dir, "hair_detection_fixture.npz")
d = np.load(fx)
tex_pos = d["tex_pos"]; tex_normals = d["tex_normals"]
uv_mask = d["uv_mask"].astype(bool); TEX = int(d["texture_size"][0])

from hair_detection_utils_v2 import HairDetectionUtils
_kw = {}
if "norm_verts" in d:
    _kw = dict(_norm_verts=d["norm_verts"], _norm_faces=d["norm_faces"],
               _norm_uvs=d["norm_uvs"], trimesh=1)  # trimesh non-None enables ear path
    print("[diag] per-face data present -> ear subtraction ACTIVE (matches pipeline)")
else:
    print("[diag] no per-face data -> ear subtraction SKIPPED (re-run pipeline to capture)")
hdu = HairDetectionUtils(TEX, uv_mask, tex_pos, tex_normals, **_kw)
with contextlib.redirect_stdout(io.StringIO()):
    mask = hdu.get_hair_mask()

ys, xs = np.where(uv_mask)
P = tex_pos[ys, xs]
px, py, pz = P[:, 0], P[:, 1], P[:, 2]
covered = mask[ys, xs]

ymax = py.max(); ymin = py.min(); yr = ymax - ymin
T_head = ymax - 0.28 * yr
T_hair = ymax - 0.20 * (ymax - T_head)
T_sf = T_head + 0.60 * (T_hair - T_head)
print(f"T_sf={T_sf:.3f} T_hair={T_hair:.3f}  hair_mask_texels={int(mask.sum()):,}")

# Head/hair zone = above seam_floor.  "Should arguably be hair" reference is hard;
# instead just show coverage of ALL head-zone texels split front/back.
head = py > T_sf
print(f"head-zone texels={int(head.sum()):,}  covered={int(covered[head].sum()):,} "
      f"({100*covered[head].mean():.1f}%)")

def render(sel, fname, title):
    """Ortho project selected texels onto X(horiz)-Y(vert); colour by coverage."""
    if sel.sum() == 0:
        print(f"  {title}: no texels"); return
    X = px[sel]; Y = py[sel]; C = covered[sel]
    H = W = 480
    x0, x1 = -0.16, 0.16
    y0, y1 = T_sf - 0.02, ymax + 0.02
    col = np.clip(((X - x0) / (x1 - x0) * (W - 1)), 0, W - 1).astype(int)
    row = np.clip(((y1 - Y) / (y1 - y0) * (H - 1)), 0, H - 1).astype(int)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # paint not-covered first (red), then covered (yellow) so yellow wins ties
    img[row[~C], col[~C]] = (0, 0, 220)        # red  (BGR) = gap
    img[row[C], col[C]] = (0, 220, 220)        # yellow = covered
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(_dir, fname), img)
    n = int(sel.sum()); ncov = int(C.sum())
    print(f"  {title}: texels={n:,} covered={ncov:,} ({100*ncov/n:.1f}%)  -> {fname}")

# Back of head (pz<0) and front (pz>=0), central strip only (|x|<0.16)
central = np.abs(px) < 0.16
render(head & central & (pz < 0.0),  "_cov_back.png",  "BACK (pz<0)")
render(head & central & (pz >= 0.0), "_cov_front.png", "FRONT (pz>=0)")

def render_side(sel, fname, title):
    """Side view of lateral texels: pz(horiz, front=right) vs py(vert)."""
    if sel.sum() == 0:
        print(f"  {title}: no texels"); return
    Z = pz[sel]; Y = py[sel]; C = covered[sel]
    H = W = 480
    z0, z1 = -0.10, 0.10
    y0, y1 = T_sf - 0.02, ymax + 0.02
    col = np.clip(((Z - z0) / (z1 - z0) * (W - 1)), 0, W - 1).astype(int)
    row = np.clip(((y1 - Y) / (y1 - y0) * (H - 1)), 0, H - 1).astype(int)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[row[~C], col[~C]] = (0, 0, 220)
    img[row[C], col[C]] = (0, 220, 220)
    cv2.line(img, (W//2, 0), (W//2, H-1), (90, 90, 90), 1)   # pz=0 line
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(_dir, fname), img)
    print(f"  {title}: texels={int(sel.sum()):,} covered={100*C.mean():.1f}%  -> {fname}")

# Lateral (temple/over-ear) side views, each side.  Front (pz>0) is to the RIGHT.
render_side(head & (px > 0.05),  "_cov_side_R.png", "R SIDE |x|>0.05 (front=right)")
render_side(head & (px < -0.05), "_cov_side_L.png", "L SIDE |x|>0.05 (front=right)")

# Lateral coverage by pz (how far forward onto the side the cap reaches)
print("\n[temple] lateral |X|>0.05 coverage by pz (forward = higher pz):")
lat = head & (np.abs(px) > 0.05)
for zlo, zhi in [(-0.02,0.0),(0.0,0.02),(0.02,0.04),(0.04,0.06),(0.06,0.09)]:
    m = lat & (pz >= zlo) & (pz < zhi)
    if m.sum() == 0: continue
    print(f"   pz=[{zlo:+.2f},{zhi:+.2f}) texels={int(m.sum()):6d}  covered={100*covered[m].mean():5.1f}%")

# Cheek/jaw bleed check: central-front face should STAY uncovered
print("\n[bleed-check] central front face (|X|<0.04, pz>0.03) — should stay LOW:")
face = head & (np.abs(px) < 0.04) & (pz > 0.03)
print(f"   texels={int(face.sum()):,}  covered={100*covered[face].mean():.1f}%")

# Quantify the back-center band: uncovered back texels by |x| bin
print("\n[band] BACK head-zone (pz<0) coverage by |X| bin:")
back = head & (pz < 0.0)
for xlo, xhi in [(0.00,0.01),(0.01,0.02),(0.02,0.03),(0.03,0.05),(0.05,0.08),(0.08,0.12)]:
    m = back & (np.abs(px) >= xlo) & (np.abs(px) < xhi)
    if m.sum() == 0: continue
    print(f"   |X|=[{xlo:.2f},{xhi:.2f}) texels={int(m.sum()):6d}  "
          f"covered={100*covered[m].mean():5.1f}%")

# Quantify forehead reach: front-top texels (pz>0) coverage by py near T_hair
print("\n[forehead] FRONT (pz>0.02) coverage by py near hairline:")
fr = head & (pz > 0.02)
for ylo, yhi in [(T_hair-0.02,T_hair),(T_hair,T_hair+0.02),
                 (T_hair+0.02,T_hair+0.04),(T_hair+0.04,ymax)]:
    m = fr & (py >= ylo) & (py < yhi)
    if m.sum() == 0: continue
    print(f"   py=[{ylo:.3f},{yhi:.3f}) texels={int(m.sum()):6d}  "
          f"covered={100*covered[m].mean():5.1f}%  pz_max={pz[m].max():.3f}")
