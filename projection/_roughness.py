"""Test: is HAIR separable from SKIN by SURFACE ROUGHNESS?
Hair has strand/clump detail (bumpy) -> high local normal variation.
Skin (face) is smooth -> low.  This is position/normal/depth-independent, so it
should work for ANY character.  Render a roughness heatmap to verify."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me=obj.data
bm=bmesh.new(); bm.from_mesh(me); bm.normal_update()
bm.faces.ensure_lookup_table()
nf=len(bm.faces)
co=np.array([list(v.co) for v in bm.verts])
faces=[[v.index for v in f.verts] for f in bm.faces]
fn=np.array([list(f.normal) for f in bm.faces])
# centroid per face (for camera + region report)
fc=np.array([list(f.calc_center_median()) for f in bm.faces])
pairs=[]
for e in bm.edges:
    if len(e.link_faces)==2:
        pairs.append((e.link_faces[0].index, e.link_faces[1].index))
bm.free()
pairs=np.array(pairs, dtype=np.int64)
# per-edge roughness = 1 - dot(normals); per-face = mean over incident edges
rij=1.0-np.sum(fn[pairs[:,0]]*fn[pairs[:,1]],axis=1)
rough=np.zeros(nf); cnt=np.zeros(nf)
np.add.at(rough,pairs[:,0],rij); np.add.at(rough,pairs[:,1],rij)
np.add.at(cnt,pairs[:,0],1.0); np.add.at(cnt,pairs[:,1],1.0)
rough=rough/np.maximum(cnt,1)
# smooth over neighbours a few passes -> coherent regions
for _ in range(4):
    acc=np.zeros(nf); c2=np.zeros(nf)
    np.add.at(acc,pairs[:,0],rough[pairs[:,1]]); np.add.at(acc,pairs[:,1],rough[pairs[:,0]])
    np.add.at(c2,pairs[:,0],1.0); np.add.at(c2,pairs[:,1],1.0)
    rough=0.5*rough+0.5*acc/np.maximum(c2,1)
ps=np.percentile(rough,[10,25,50,75,90,95,99])
print("roughness percentiles:", [round(float(x),4) for x in ps])
# region check (gen-ish coords: this mesh is native, use bbox-relative)
z=fc[:,2]; y=fc[:,1]; x=fc[:,0]
zr=(z-z.min())/(z.max()-z.min())
def m(mask): return round(float(rough[mask].mean()),4) if mask.sum() else 0
# crude regions on native mesh: top head vs mid face (front)
fwd = y < np.percentile(y,25)   # front-facing-ish (lower Y = front, guessing)
print("mean rough: top25%z", m(zr>0.75), " midface(frontish, z .55-.75)", m((zr>0.55)&(zr<0.75)&fwd))

# heatmap render: 6 buckets blue->red across [p25,p95]
lo,hi=ps[1],ps[4]
b=np.clip(((rough-lo)/(hi-lo)*5),0,5).astype(int)
cols=[(0.2,0.3,0.8),(0.2,0.7,0.8),(0.4,0.8,0.4),(0.9,0.85,0.2),(0.9,0.5,0.15),(0.95,0.1,0.1)]
for o in list(bpy.data.objects):
    if o.name!=obj.name: bpy.data.objects.remove(o,do_unlink=True)
m2=bpy.data.meshes.new("hm"); m2.from_pydata(co.tolist(),[],faces); m2.update()
ob2=bpy.data.objects.new("hm",m2); bpy.context.collection.objects.link(ob2)
for c in cols:
    mat=bpy.data.materials.new("c"); mat.diffuse_color=(c[0],c[1],c[2],1); m2.materials.append(mat)
bl=b.tolist()
for i,p in enumerate(m2.polygons): p.material_index=bl[i]
try: obj.hide_render=True
except: pass
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=860; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); zmin,zmax=co[:,2].min(),co[:,2].max(); ymin,ymax=co[:,1].min(),co[:,1].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2; zlo=zmin+0.74*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.42
cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd2=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd2.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
for dv,nm in [((0,-1,0.12),"_rough_a.png"),((1,0.1,0.12),"_rough_b.png"),((-0.6,-1,0.12),"_rough_c.png")]:
    shot(dv,nm)
print("DONE_ROUGH")
