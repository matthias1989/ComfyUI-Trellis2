"""Test the OUTER-SHELL hair detector: a face is hair if there's another mesh
surface just behind it along its inward normal (hair drapes over body).
Renders hair(yellow)/skin(gray) front+side so we can judge the separation."""
import numpy as np, os, sys
import cv2
from scipy.spatial import cKDTree
_dir=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(_dir,'hair_detection_fixture.npz'))
V=d['norm_verts'].astype(np.float64); F=d['norm_faces'].astype(np.int64)
C=V[F].mean(1); cx,cy,cz=C[:,0],C[:,1],C[:,2]
# outward face normals (assume mesh normals point outward; orient by centroid-from-center)
e1=V[F[:,1]]-V[F[:,0]]; e2=V[F[:,2]]-V[F[:,0]]
N=np.cross(e1,e2); N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-9)
# Consistent winding → all normals point the same way (in or out).  Orient OUT
# by checking the crown faces (top of head) should point +Y.
crown=cy>np.percentile(cy,98)
if N[crown,1].mean()<0: N=-N
tree=cKDTree(C)
def shell(deltas,R):
    """Hair = a body surface sits CLOSE behind the face along its inward normal.
    Probe several depths so varying hair-body gaps (tight at back, looser at the
    crown) are all caught."""
    hit=np.zeros(len(F),bool); ar=np.arange(len(F))
    for delta in deltas:
        P=C-delta*N
        dist,idx=tree.query(P,k=1)
        behind=((C[idx]-C)*N).sum(1) < -0.005
        hit |= (dist<R)&behind&(idx!=ar)
    return hit
raw=shell([0.012,0.020,0.030,0.040],0.013)
# Coarse pre-exclude clearly-not-hair zones (arms far lateral, legs).
cand=raw & (np.abs(cx)<0.16) & (cy>0.05)

# Connectivity: keep only shell-hair that forms ONE connected sheet reaching the
# scalp.  Arm-underside / breast / crotch hits are separate islands.
from collections import defaultdict
Sl=np.where(cand)[0]; Sset=set(Sl.tolist())
parent={int(f):int(f) for f in Sl}
def find(a):
    while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
    return a
vf=defaultdict(list)
for f in Sl.tolist():
    for v in F[f].tolist(): vf[v].append(f)
for v,fs in vf.items():
    for k in range(1,len(fs)):
        ra,rb=find(fs[0]),find(fs[k])
        if ra!=rb: parent[ra]=rb
roots=np.array([find(int(f)) for f in Sl])
from collections import Counter
sz=Counter(roots.tolist())
print('shell-hair connected components (size desc, top 12):')
for r,c in sz.most_common(12):
    fs=Sl[roots==r]
    print(f'  root={r} faces={c:5d} X=[{cx[fs].min():+.2f},{cx[fs].max():+.2f}] '
          f'Y=[{cy[fs].min():+.2f},{cy[fs].max():+.2f}] Z=[{cz[fs].min():+.3f},{cz[fs].max():+.3f}]')
# Keep big CENTRAL components (hair drape/nape/back-head); drop lateral
# (shoulders) and forward (face/nose) and small (ears) false-positive islands.
keep_roots=[]
for r,c in sz.items():
    if c<1500: continue
    fs=Sl[roots==r]
    if abs(float(cx[fs].mean()))<0.07 and float(cz[fs].mean())<0.0:  # central + back-ish
        keep_roots.append(r)
hair=np.zeros(len(F),bool)
hair[Sl[np.isin(roots,keep_roots)]]=True
print(f'kept {len(keep_roots)} central-back components')

# Ear exclusion (belt-and-suspenders): lateral head-zone, side-facing
ears=(cy>0.41)&(cy<0.49)&(np.abs(cx)>0.055)&(cz>-0.03)
hair &= ~ears
print(f'hair faces: {int(hair.sum()):,}/{len(F):,}  ({100*hair.mean():.1f}%)')
for lbl,sel in [('head py>0.40',cy>0.40),('neck py[0.34,0.40]',(cy>0.34)&(cy<0.40)),
                ('upper-back/chest py[0.10,0.34]',(cy>0.10)&(cy<0.34)),
                ('lower py<0.10',cy<0.10)]:
    if sel.sum(): print(f'  {lbl}: hair {100*hair[sel].mean():.0f}% of {int(sel.sum())} faces')

def render(A,B,fname,title,a0,a1,b0,b1):
    span=max(a1-a0,b1-b0); ac=(a0+a1)/2; bc=(b0+b1)/2
    a0,a1,b0,b1=ac-span/2,ac+span/2,bc-span/2,bc+span/2
    H=W=600
    sel=cy>-0.10
    col=np.clip((A[sel]-a0)/(a1-a0)*(W-1),0,W-1).astype(int)
    row=np.clip((b1-B[sel])/(b1-b0)*(H-1),0,H-1).astype(int)
    hh=hair[sel]
    img=np.full((H,W,3),30,np.uint8)
    img[row[~hh],col[~hh]]=(150,150,150)
    img[row[hh],col[hh]]=(0,220,220)
    cv2.putText(img,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.imwrite(os.path.join(_dir,fname),img)
render(cz,cy,'_shell_side.png','SIDE shell-hair (front=right)',-0.10,0.10,-0.10,0.55)
render(cx,cy,'_shell_front.png','FRONT shell-hair',-0.30,0.30,-0.10,0.55)
print('rendered _shell_side.png _shell_front.png')
