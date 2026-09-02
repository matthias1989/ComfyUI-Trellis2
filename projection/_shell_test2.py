"""Prove the SHELL/LAYERING signal separates hair from ear/face better than
roughness.  For each face cast a short ray INWARD (-normal); if it hits another
surface within SHELL_D, there's a body layer behind it -> hair.  The ear/cheek/
nose have nothing behind (solid body) so the first inward hit is the FAR wall
(large distance) -> skin.  Render red(hair)-over-gray to eyeball it."""
import bpy, numpy as np, os
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
bvh=BVHTree.FromPolygons([tuple(v) for v in co.tolist()], faces, all_triangles=False, epsilon=0.0)
diag=float(np.linalg.norm(co.max(0)-co.min(0)))
EPS=diag*0.0010
SHELL=diag*0.030      # max hair->body gap (tune); ~3% of bbox diagonal
print("nf",nf,"diag",round(diag,3),"EPS",round(EPS,4),"SHELL",round(SHELL,4))
shell=np.zeros(nf,bool); dist=np.full(nf,-1.0)
for i in range(nf):
    n=Vector(fn[i]); c=Vector(fc[i])
    hit=bvh.ray_cast(c - n*EPS, -n, SHELL*2.5)
    if hit[0] is not None and hit[2]!=i and hit[3]>EPS*1.5:
        dist[i]=hit[3]
        if hit[3] < SHELL: shell[i]=True
print("raw shell faces", int(shell.sum()), "/", nf)
# keep only large connected components (drop tiny ear/feature specks)
pairs=[]
import bmesh
bm=bmesh.new(); bm.from_mesh(me)
for e in bm.edges:
    if len(e.link_faces)==2: pairs.append((e.link_faces[0].index,e.link_faces[1].index))
bm.free()
pairs=np.array(pairs,dtype=np.int64)
par=list(range(nf))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for a,b in pairs.tolist():
    if shell[a] and shell[b]:
        ra,rb=find(a),find(b)
        if ra!=rb: par[ra]=rb
roots=np.array([find(f) for f in range(nf)])
u,inv,cnt=np.unique(roots,return_inverse=True,return_counts=True)
hair=shell & (cnt[inv] >= 1500)
print("after size-filter", int(hair.sum()))
np.save(os.path.join(D,"_shell_hair.npy"), hair)
# ---- render red over gray ----
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
zlo=zmin+0.74*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.42
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((1,0.05,0.06),"_shell_side.png")
shot((-0.55,-1,0.10),"_shell_q.png")
shot((0,-1,0.10),"_shell_front.png")
print("DONE_SHELL")
