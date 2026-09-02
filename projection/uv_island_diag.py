"""
KAN-8 UV-island diagnostic.

Quantifies, from a hair_detection_fixture .npz (tex_pos / tex_normals / uv_mask /
texture_size), whether the head/ear geometry is cleanly separated in the UV
atlas.  Used to A/B different `mesh_cluster_threshold_cone_half_angle_rad`
(cone half-angle) settings while tuning the unwrap.

Run:
    python uv_island_diag.py [path_to_fixture.npz]

Defaults to ./hair_detection_fixture.npz.  Compare two runs by pointing it at
each fixture in turn.

Decisive metric for the ear fix:
  * head_x_max  — max |X| of any head-zone texel in the atlas.  If ear faces
    fold over the cheek in a shared chart, the ear surface is overwritten and
    head_x_max collapses toward the cheek width.  A successful split RAISES
    head_x_max toward the true ear extent and puts texels at high |X| in the
    ear-height slices (frac|X|>0.10 goes from ~0 to >0).
"""
import sys, os
import numpy as np


def analyse(path):
    d = np.load(path)
    analyse_arrays(d["tex_pos"], d["uv_mask"].astype(bool),
                   int(d["texture_size"][0]), label=os.path.basename(path))


def analyse_arrays(tex_pos, uv_mask, TEX, label=""):
    """Print KAN-8 UV-island metrics for in-memory arrays.

    tex_pos : (H,W,3) float  world position per texel
    uv_mask : (H,W) bool      True inside UV islands
    TEX     : int             atlas resolution
    label   : str             header tag (e.g. the cone angle of this run)
    """
    H, W = uv_mask.shape
    ys, xs = np.where(uv_mask)
    P = tex_pos[ys, xs]
    px, py, pz = P[:, 0], P[:, 1], P[:, 2]
    N = len(ys)
    model_diag = float(np.linalg.norm(P.max(0) - P.min(0)))

    print(f"\n############ uv_island_diag {label} ############")
    print(f"tex_size={TEX}  island_texels={N:,} ({100*N/(H*W):.1f}% of atlas)")
    print(f"world  X:[{px.min():+.3f},{px.max():+.3f}]  "
          f"Y:[{py.min():+.3f},{py.max():+.3f}]  "
          f"Z:[{pz.min():+.3f},{pz.max():+.3f}]  model_diag={model_diag:.3f}")

    v = ys / TEX; u = xs / TEX

    # ── UV layout: cylindrical vs island ─────────────────────────────────────
    cc = np.corrcoef(v, py)[0, 1]
    print(f"\n[layout] corr(atlas_row, world_Y) = {cc:+.3f}  "
          f"({'CYLINDRICAL' if cc > 0.9 else 'ISLAND/xatlas'})")

    # ── per-tile 3D spread (interleaving) ────────────────────────────────────
    print("[coherence] per-tile 3D-diag / model-diag (low => coherent islands):")
    for TILE in (16, 32):
        bi = (ys // TILE) * (W // TILE + 1) + (xs // TILE)
        order = np.argsort(bi); bi_s = bi[order]; Ps = P[order]
        uniq, start = np.unique(bi_s, return_index=True)
        diags = []
        for k in range(len(uniq)):
            a = start[k]; b = start[k + 1] if k + 1 < len(uniq) else len(bi_s)
            if b - a < 4:
                continue
            blk = Ps[a:b]
            diags.append(np.linalg.norm(blk.max(0) - blk.min(0)))
        diags = np.array(diags) / model_diag
        print(f"   TILE={TILE:2d}px blocks={len(diags):6d}  "
              f"p50={np.median(diags):.3f} p90={np.percentile(diags,90):.3f} "
              f"p99={np.percentile(diags,99):.3f} max={diags.max():.3f}")

    # ── thresholds (mirror hair_detection_utils_v2._compute_thresholds) ──────
    ymax, ymin = float(py.max()), float(py.min()); yr = ymax - ymin
    T_head = ymax - 0.28 * yr
    T_hair = ymax - 0.20 * (ymax - T_head)
    T_sf = T_head + 0.60 * (T_hair - T_head)
    head = py > T_sf
    head_x_max = float(np.abs(px[head]).max())
    print(f"\n[head] T_sf={T_sf:.3f} T_hair={T_hair:.3f}  "
          f"head_x_max={head_x_max:.3f}  head_texels={int(head.sum()):,}")
    print("[ears] lateral texels in head zone:")
    for frac in (0.54, 0.65, 0.80):
        thr = head_x_max * frac
        n = int((head & (np.abs(px) > thr)).sum())
        print(f"   |X|>{thr:.3f} (x{frac}): {n:,} "
              f"({100*n/max(1,head.sum()):.2f}% of head)")
    print("[ears] per-Y-slice lateral extent "
          "(frac|X|>0.10 rising = ear surface now present):")
    for ylo, yhi in [(T_sf, T_sf + 0.02), (T_sf + 0.02, T_sf + 0.04),
                     (T_sf + 0.04, T_hair), (T_hair, ymax)]:
        m = (py >= ylo) & (py < yhi)
        if m.sum() < 20:
            print(f"   py=[{ylo:.3f},{yhi:.3f}) px={int(m.sum())}"); continue
        axm = np.abs(px[m])
        print(f"   py=[{ylo:.3f},{yhi:.3f}) px={int(m.sum()):6d}  "
              f"|X|max={axm.max():.3f}  |X|p99={np.percentile(axm,99):.3f}  "
              f"frac|X|>0.10={np.mean(axm > 0.10):.3f}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "hair_detection_fixture.npz")
    analyse(p)
