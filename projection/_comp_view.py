"""Visualize mesh connected components + test the layering signal (hair = outer
shell over body).  Renders side & front, faces colored by component."""
import numpy as np, os, sys
import cv2
import scipy.sparse as sp, scipy.sparse.csgraph as cg
_dir=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(_dir,'hair_detection_fixture.npz'))
V=d['norm_verts'].astype(float); F=d['norm_faces'].astype(int)
cent=V[F].mean(1); cx,cy,cz=cent[:,0],cent[:,1],cent[:,2]
e0=F[:,[0,1,2]].ravel(); e1=F[:,[1,2,0]].ravel()
g=sp.coo_matrix((np.ones(len(e0),np.uint8),(e0,e1)),shape=(len(V),len(V))).tocsr()
nc,vl=cg.connected_components(g,directed=False); fcomp=vl[F[:,0]]
uniq,counts=np.unique(fcomp,return_counts=True)
order=uniq[np.argsort(counts)[::-1]]
# distinct colors (BGR)
COLORS=[(0,220,220),(0,140,255),(255,120,0),(0,0,230),(0,230,0),(230,0,230),
        (200,200,200),(120,60,200),(60,200,200),(200,120,60)]
cmap={int(r):COLORS[i%len(COLORS)] for i,r in enumerate(order)}

def render(A,B,sel,fname,title,a0,a1,b0,b1):
    span=max(a1-a0,b1-b0); ac=(a0+a1)/2; bc=(b0+b1)/2
    a0,a1=ac-span/2,ac+span/2; b0,b1=bc-span/2,bc+span/2
    H=W=600
    col=np.clip((A[sel]-a0)/(a1-a0)*(W-1),0,W-1).astype(int)
    row=np.clip((b1-B[sel])/(b1-b0)*(H-1),0,H-1).astype(int)
    img=np.full((H,W,3),30,np.uint8)
    cols=np.array([cmap[int(c)] for c in fcomp[sel]])
    img[row,col]=cols
    cv2.putText(img,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.imwrite(os.path.join(_dir,fname),img)

up=cy>-0.10
render(cz,cy,up,'_comp_side.png','SIDE by component (front=right)',-0.10,0.10,-0.10,0.55)
render(cx,cy,up,'_comp_front.png','FRONT by component',-0.30,0.30,-0.10,0.55)
print('components (size desc) with color + bbox:')
for r in order[:8]:
    m=fcomp==r; c=cmap[int(r)]
    print(f'  comp {int(r):3d} faces={int(m.sum()):6d} colorBGR={c} '
          f'X=[{cx[m].min():+.2f},{cx[m].max():+.2f}] Y=[{cy[m].min():+.2f},{cy[m].max():+.2f}] '
          f'Z=[{cz[m].min():+.3f},{cz[m].max():+.3f}]')

# ── Layering test: for back faces, is there a body face IN FRONT of it (less
# negative z) at the same (X,Y)?  If yes -> this back face is the outer hair shell.
print('\\n[layering] back faces (z<-0.03) — fraction with another face >0.015 in front at same XY:')
back=np.where((cz<-0.03)&(cy>0.05)&(cy<0.40))[0]
# grid by (X,Y) cells
gx=np.round(cx/0.01).astype(int); gy=np.round(cy/0.01).astype(int)
from collections import defaultdict
cellz=defaultdict(list)
allsel=(cy>0.0)&(cy<0.45)
for i in np.where(allsel)[0]:
    cellz[(gx[i],gy[i])].append(cz[i])
cnt_outer=0; cnt_single=0
for i in back:
    zs=cellz[(gx[i],gy[i])]
    if any(z > cz[i]+0.015 for z in zs): cnt_outer+=1
    else: cnt_single+=1
print(f'  back faces={len(back)}  outer-shell(has body in front)={cnt_outer}  single-layer={cnt_single}')
print(f'  -> {100*cnt_outer/max(1,len(back)):.0f}% of deep-back faces are an OUTER shell (hair over body)')
