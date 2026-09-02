"""Compare hair-coverage of every saved detector version against the fixture.
tex_pos space: X=lateral(+-0.476)  Y=height(+-0.525)  Z=depth(front=+, +-0.09)"""
import numpy as np, os, io, contextlib, sys, importlib.util
_dir = r'C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection'
sys.path.insert(0, _dir)
d = np.load(os.path.join(_dir, 'hair_detection_fixture.npz')); TEX = int(d['texture_size'][0])
uvm = d['uv_mask'].astype(bool); P = d['tex_pos'][uvm]
X, Y, Z = P[:,0], P[:,1], P[:,2]

# regions (want hair):
crown   = Y > 0.46
bdrape  = (Z < -0.04) & (Y > 0.15) & (Y < 0.45) & (np.abs(X) < 0.07)
sscalp  = (np.abs(X) > 0.055) & (np.abs(X) < 0.092) & (Y > 0.36) & (Y < 0.46) & (np.abs(Z) < 0.035)
# regions (want skin = low hair):
fbleed  = (Z > 0.04) & (Y > 0.33) & (Y < 0.46) & (np.abs(X) < 0.06)
ebleed  = (np.abs(X) > 0.060) & (np.abs(X) < 0.095) & (Y > 0.385) & (Y < 0.455) & (np.abs(Z) < 0.030)

def cov(hair, reg): return 100*hair[reg].mean() if reg.sum() else float('nan')

cands = ['hair_detection_utils_v2.py','hair_detection_utils_v2_backup.py',
         'hair_detection_utils_v2_haircovered.py','hair_detection_utils_v2_haircovered_ears_attempted.py',
         'hair_detection_utils_v2_faceear_attempted.py','hair_detection_utils_v2_earsworking.py']
print(f"{'version':<48} {'hair%':>6} {'crown':>6} {'bdrape':>7} {'sscalp':>7} | {'FACE':>5} {'EAR':>5}")
print('-'*92)
for c in cands:
    fp = os.path.join(_dir, c)
    if not os.path.exists(fp): print(f"{c:<48} MISSING"); continue
    try:
        spec = importlib.util.spec_from_file_location('m_'+c.replace('.','_'), fp)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        hdu = m.HairDetectionUtils(TEX, uvm, d['tex_pos'], d['tex_normals'],
                                   _norm_verts=d['norm_verts'], _norm_faces=d['norm_faces'],
                                   _norm_uvs=d['norm_uvs'], trimesh=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mask = hdu.get_hair_mask().astype(bool)
        h = mask[uvm]
        print(f"{c:<48} {100*h.mean():>5.1f} {cov(h,crown):>5.0f} {cov(h,bdrape):>6.0f} "
              f"{cov(h,sscalp):>6.0f} | {cov(h,fbleed):>4.0f} {cov(h,ebleed):>4.0f}")
    except Exception as e:
        print(f"{c:<48} ERR {type(e).__name__}: {str(e)[:40]}")
print('-'*92)
print("want: crown/bdrape/sscalp HIGH ; FACE/EAR LOW")
