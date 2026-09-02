"""Find the hairline as a GEOMETRIC crease directly on the mesh (no projection,
no bridge).  The hair sits proud of the skin -> a sharp dihedral ridge at the
hairline.  Detect sharp edges, mark adjacent faces, render to compare with the
user's viewport."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
D = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"

obj = bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
me = obj.data
bm = bmesh.new(); bm.from_mesh(me); bm.normal_update()
nF = len(bm.faces); nV = len(bm.verts)
co = np.array([list(v.co) for v in bm.verts], dtype=np.float32)
print(f"mesh verts={nV} faces={nF}")

# signed dihedral per edge; abs = sharpness, sign<0 = concave (crease under overhang)
ang = np.zeros(len(bm.edges), dtype=np.float32)
bm.edges.ensure_lookup_table()
for e in bm.edges:
    if len(e.link_faces) == 2:
        try: ang[e.index] = e.calc_face_angle_signed()
        except Exception: ang[e.index] = 0.0
absd = np.degrees(np.abs(ang))
print("dihedral-abs percentiles (deg):", [round(float(np.percentile(absd,p)),1) for p in (50,75,90,95,99)])

# face -> sharp if any edge sharp; render at a few thresholds
bm.faces.ensure_lookup_table()
def face_sharp(thr_deg, concave_only=False):
    fs = np.zeros(nF, dtype=bool)
    for f in bm.faces:
        for e in f.edges:
            a = ang[e.index]
            if (np.degrees(abs(a)) > thr_deg) and ((a < 0) or not concave_only):
                fs[f.index] = True; break
    return fs

# build tri list once
tris=[]; tri2face=[]
for f in bm.faces:
    vs=[v.index for v in f.verts]
    for k in range(1,len(vs)-1):
        tris.append((vs[0],vs[k],vs[k+1])); tri2face.append(f.index)
tris=np.array(tris,dtype=np.int32); tri2face=np.array(tri2face)

for thr in (25,35,45):
    fs=face_sharp(thr)
    zc=co[:,2]; head=zc> (zc.min()+0.74*(zc.max()-zc.min()))
    print(f"thr={thr}deg  sharp_faces={int(fs.sum())}  (in head-z: {int((fs).sum())})")

# render with thr=30 concave creases highlighted
fs = face_sharp(30, concave_only=False)
ts = fs[tri2face]
for o in list(bpy.data.objects):
    if o.type=='MESH' and o.name!=obj.name: bpy.data.objects.remove(o,do_unlink=True)
m2=bpy.data.meshes.new("vz"); m2.from_pydata(co.tolist(),[],tris.tolist()); m2.update()
ob2=bpy.data.objects.new("vz",m2); bpy.context.collection.objects.link(ob2)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.78,0.72,0.66,1)
mh=bpy.data.materials.new("h"); mh.diffuse_color=(0.95,0.1,0.1,1)
m2.materials.append(mb); m2.materials.append(mh)
tsl=ts.tolist()
for i,p in enumerate(m2.polygons): p.material_index=1 if tsl[i] else 0
try: obj.hide_render=True
except Exception: pass
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=900; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); zmin,zmax=co[:,2].min(),co[:,2].max(); ymin,ymax=co[:,1].min(),co[:,1].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
zlo=zmin+0.80*(zmax-zmin); zc2=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.42
cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc2))+dvv*4
    fwd=(Vector((cx,cy,zc2))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Z').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0.4,-1,0.05),"_geo_front.png")
shot((1,0.2,0.05),"_geo_side.png")
shot((0.2,1,0.05),"_geo_back.png")
print("DONE_GEO")
