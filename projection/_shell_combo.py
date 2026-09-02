"""Combined hair detector:
  SHELL (inward ray hits body within SHELL_D)  -> reliable: drape, frame, ear-clean
  ROUGH (local normal variation)               -> the textured crown shell misses
hair = the connected component(s) of (shell | rough) that CONTAIN shell faces and
are large.  Isolated rough (ear, nose, lips) has no shell + is broken from the
hairline by smooth skin -> excluded.  Arms removed by central-column gate.
Render red(hair)-over-gray."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
from mathutils.bvhtree import BVHTree
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data
co=np.array([list(v.co) for v in me.vertices], dtype=float)
faces=[tuple(p.vertices) for p in me.polygons]
fn=np.array([list(p.normal) for p in me.polygons], dtype=float)
fc=np.array([list(p.center) for p in me.polygons], dtype=float)
nf=len(faces)
print("x range",round(co[:,0].min(),3),round(co[:,0].max(),3),
      " z range",round(co[:,2].min(),3),round(co[:,2].max(),3),
      " y range",round(co[:,1].min(),3),round(co[:,1].max(),3))
# ---- SHELL ----
bvh=BVHTree.FromPolygons([tuple(v) for v in co.tolist()], faces, all_triangles=False, epsilon=0.0)
diag=float(np.linalg.norm(co.max(0)-co.min(0)))
EPS=diag*0.0010; SHELL=diag*0.030
shell=np.zeros(nf,bool)
for i in range(nf):
    n=Vector(fn[i]); c=Vector(fc[i])
    hit=bvh.ray_cast(c - n*EPS, -n, SHELL*2.5)
    if hit[0] is not None and hit[2]!=i and hit[3]>EPS*1.5 and hit[3]<SHELL: shell[i]=True
print("shell faces", int(shell.sum()))
# ---- ROUGH ----
bm=bmesh.new(); bm.from_mesh(me)
pairs=[(e.link_faces[0].index,e.link_faces[1].index) for e in bm.edges if len(e.link_faces)==2]
bm.free()
pairs=np.array(pairs,dtype=np.int64)
rij=1.0-np.sum(fn[pairs[:,0]]*fn[pairs[:,1]],axis=1)
rough=np.zeros(nf); cnt=np.zeros(nf)
np.add.at(rough,pairs[:,0],rij); np.add.at(rough,pairs[:,1],rij)
np.add.at(cnt,pairs[:,0],1.0); np.add.at(cnt,pairs[:,1],1.0)
rough=rough/np.maximum(cnt,1)
for _ in range(3):
    acc=np.zeros(nf); c2=np.zeros(nf)
    np.add.at(acc,pairs[:,0],rough[pairs[:,1]]); np.add.at(acc,pairs[:,1],rough[pairs[:,0]])
    np.add.at(c2,pairs[:,0],1.0); np.add.at(c2,pairs[:,1],1.0)
    rough=0.5*rough+0.5*acc/np.maximum(c2,1)
is_rough = rough > 0.009
# ---- central-column gate (kill arms/legs) ----
x=fc[:,0]; z=fc[:,2]
zone = (np.abs(x) < 0.16) & (z > -0.10)
cand = (shell | is_rough) & zone
print("rough faces",int(is_rough.sum())," cand faces",int(cand.sum()))
# ---- connected components on cand; keep comps with shell support + size ----
par=list(range(nf))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for a,b in pairs.tolist():
    if cand[a] and cand[b]:
        ra,rb=find(a),find(b)
        if ra!=rb: par[ra]=rb
roots=np.array([find(f) for f in range(nf)])
shell_in_comp={}; size={}
for f in range(nf):
    if cand[f]:
        r=roots[f]; size[r]=size.get(r,0)+1
        if shell[f]: shell_in_comp[r]=shell_in_comp.get(r,0)+1
keep={r for r in size if size[r]>=1500 and shell_in_comp.get(r,0)>=80}
hair = cand & np.isin(roots, list(keep))
print("FINAL hair faces", int(hair.sum()), " comps kept", len(keep))
np.save(os.path.join(D,"_combo_hair.npy"), hair)
# ---- render ----
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
m2=bpy.data.meshes.new("m"); m2.from_pydata(co.tolist(),[],[list(f) for f in faces]); m2.update()
ob2=bpy.data.objects.new("m",m2); bpy.context.collection.objects.link(ob2)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.80,0.80,0.80,1)
mh=bpy.data.materials.new("h"); mh.diffuse_color=(0.92,0.28,0.26,1)
m2.materials.append(mb); m2.materials.append(mh)
hl=hair.tolist()
for i,p in enumerate(m2.polygons): p.material_index=1 if hl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'
sc.display.shading.color_type='MATERIAL'; sc.display.shading.show_cavity=True
sc.render.resolution_x=860; sc.render.resolution_y=940; sc.render.image_settings.file_format='PNG'
zmin,zmax=co[:,2].min(),co[:,2].max(); xmin,xmax=co[:,0].min(),co[:,0].max()
cx=(xmin+xmax)/2; cy=(co[:,1].min()+co[:,1].max())/2
zlo=zmin+0.70*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.42
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((1,0.05,0.06),"_combo_side.png")
shot((-0.55,-1,0.10),"_combo_q.png")
shot((0,-1,0.10),"_combo_front.png")
shot((0.2,-0.3,1.0),"_combo_top.png")
print("DONE_COMBO")
