"""KAN-8 ear-island analysis.  Uses the enriched fixture (norm_verts/faces/uvs)
to test whether the ear forms its own UV island (from the gen_seams ear cut) and
whether subtracting that whole island would cleanly remove the ear globe+tip
without eating over-ear scalp.

Compares against the current |X|-threshold ear detection."""
import numpy as np, os, sys
from collections import defaultdict

_dir = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(_dir, "hair_detection_fixture.npz"))
if "norm_verts" not in d:
    print("fixture lacks per-face data"); sys.exit(1)
V = d["norm_verts"].astype(np.float64)   # (Nv,3)
F = d["norm_faces"].astype(np.int64)     # (Nf,3)
UV = d["norm_uvs"].astype(np.float64)    # (Nv,2)
print(f"verts={len(V):,} faces={len(F):,} uvs={len(UV):,}")

cent = V[F].mean(axis=1)                  # (Nf,3) face centroids
cx, cy, cz = cent[:,0], cent[:,1], cent[:,2]

# Thresholds (match _compute_thresholds, head zone)
ymax, ymin = cy.max(), cy.min(); yr = ymax-ymin
T_head = ymax - 0.28*yr
T_hair = ymax - 0.20*(ymax-T_head)
T_sf   = T_head + 0.60*(T_hair-T_head)
in_head = (cy > T_sf) & (cy < T_hair + 0.04)
head_x_max = float(np.abs(cx[in_head]).max())
print(f"T_sf={T_sf:.3f} T_hair={T_hair:.3f} head_x_max={head_x_max:.3f}")

# Current detection: |X| > head_x_max*0.54
cur_thr = head_x_max*0.54
cur_ear = in_head & (np.abs(cx) > cur_thr)
print(f"\n[current |X|>{cur_thr:.3f}] ear_faces={int(cur_ear.sum()):,}")
print(f"   their X:[{np.abs(cx[cur_ear]).min():.3f},{np.abs(cx[cur_ear]).max():.3f}]")

# ── UV-island connected components (faces sharing a UV vertex) ────────────────
# Build union-find over faces via shared UV-vertex index (UV seams split verts,
# so UV-connectivity == island membership).
parent = np.arange(len(F))
def find(a):
    while parent[a]!=a:
        parent[a]=parent[parent[a]]; a=parent[a]
    return a
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
uvvert_faces = defaultdict(list)
for fi, tri in enumerate(F):
    for vi in tri:
        uvvert_faces[vi].append(fi)
for vi, fs in uvvert_faces.items():
    for k in range(1,len(fs)):
        union(fs[0], fs[k])
roots = np.array([find(i) for i in range(len(F))])
uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
print(f"\nUV islands: {len(uniq)}  sizes(top12)={sorted(counts.tolist(),reverse=True)[:12]}")

# For each island, lateral stats in head zone → find the ear island(s)
print("\n[islands] those with lateral faces in head zone (candidate ears):")
ear_island_ids = []
for k, root in enumerate(uniq):
    fmask = roots==root
    fh = fmask & in_head
    if fh.sum() < 5: continue
    axm = np.abs(cx[fh])
    frac_lat = float((axm > cur_thr).mean())
    # An ear island: small, mostly lateral, in head zone
    if frac_lat > 0.5 and counts[k] < 8000:
        ear_island_ids.append(root)
        print(f"   island root={root} faces={int(counts[k]):,} headfaces={int(fh.sum()):,} "
              f"|X|=[{axm.min():.3f},{axm.max():.3f}] fracLat={frac_lat:.2f} "
              f"x_sign={'R' if cx[fh].mean()>0 else 'L'}")

ear_island_face = np.isin(roots, ear_island_ids)
print(f"\n[island-subtract] total ear-island faces={int(ear_island_face.sum()):,}")
if ear_island_face.sum() > 0:
    em = ear_island_face
    print(f"   3D bbox X:[{cx[em].min():+.3f},{cx[em].max():+.3f}] "
          f"Y:[{cy[em].min():+.3f},{cy[em].max():+.3f}] "
          f"Z:[{cz[em].min():+.3f},{cz[em].max():+.3f}]")
    print(f"   |X| range: [{np.abs(cx[em]).min():.3f},{np.abs(cx[em]).max():.3f}]  "
          f"(current threshold misses |X|<{cur_thr:.3f})")
    # how many ear-island faces are BELOW the current |X| threshold (the globe)
    missed = em & (np.abs(cx) <= cur_thr)
    print(f"   ear-island faces the |X| threshold MISSES (the globe): {int(missed.sum()):,}")
    if missed.sum() > 0:
        print(f"      missed |X|:[{np.abs(cx[missed]).min():.3f},{np.abs(cx[missed]).max():.3f}]  "
              f"Z:[{cz[missed].min():+.3f},{cz[missed].max():+.3f}]")
