"""Back + back-3/4 head render: gray body (rebuilt from viz) + red seam edges, to
see the back-of-head / behind-ear hairline."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
bm=bmesh.new(); bm.from_mesh(obj.data)
seam_e=[(e.verts[0].index,e.verts[1].index) for e in bm.edges if e.seam]
bm.free()
v=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=v['co'].astype(float); tris=v['tris']
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
mb=bpy.data.meshes.new("body"); mb.from_pydata(co.tolist(),[],tris.tolist()); mb.update()
ob=bpy.data.objects.new("body",mb); bpy.context.collection.objects.link(ob)
gm=bpy.data.materials.new("g"); gm.diffuse_color=(0.78,0.78,0.78,1); mb.materials.append(gm)
ctr=co.mean(0); cob=ctr+(co-ctr)*1.010
ms=bpy.data.meshes.new("seam"); ms.from_pydata(cob.tolist(), seam_e, []); ms.update()
os_=bpy.data.objects.new("seam",ms); bpy.context.collection.objects.link(os_)
bpy.context.view_layer.objects.active=os_; os_.select_set(True)
bpy.ops.object.convert(target='CURVE'); os_.data.bevel_depth=0.0026; os_.data.bevel_resolution=2
rm=bpy.data.materials.new("r"); rm.diffuse_color=(1.0,0.05,0.05,1); os_.data.materials.append(rm)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=900; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
zmin,zmax=co[:,2].min(),co[:,2].max(); xmin,xmax=co[:,0].min(),co[:,0].max()
cx=(xmin+xmax)/2; cy=(co[:,1].min()+co[:,1].max())/2
zlo=zmin+0.78*(zmax-zmin); zc=(zlo+zmax)/2
span=(zmax-zlo)*0.66
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0,1,0.08),"_back_b.png")
shot((-0.8,0.9,0.10),"_back_q.png")
print("DONE_BACK")
