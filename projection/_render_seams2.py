"""Render the marked seam edges (red) over the gray mesh.  Body is rebuilt from the
viz (gen space) so it actually displays; seam edges come from last_seams.blend
(same vertex indexing) positioned with the viz coords."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
bm=bmesh.new(); bm.from_mesh(obj.data)
seam_e=[(e.verts[0].index,e.verts[1].index) for e in bm.edges if e.seam]
bm.free()
print("seam edges:",len(seam_e))
v=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=v['co'].astype(float); tris=v['tris']
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
mb=bpy.data.meshes.new("body"); mb.from_pydata(co.tolist(),[],tris.tolist()); mb.update()
ob=bpy.data.objects.new("body",mb); bpy.context.collection.objects.link(ob)
gm=bpy.data.materials.new("g"); gm.diffuse_color=(0.80,0.80,0.80,1); mb.materials.append(gm)
ctr=co.mean(0); cob=ctr+(co-ctr)*1.012
ms=bpy.data.meshes.new("seam"); ms.from_pydata(cob.tolist(), seam_e, []); ms.update()
os_=bpy.data.objects.new("seam",ms); bpy.context.collection.objects.link(os_)
bpy.context.view_layer.objects.active=os_; os_.select_set(True)
bpy.ops.object.convert(target='CURVE'); os_.data.bevel_depth=0.0035; os_.data.bevel_resolution=1
rm=bpy.data.materials.new("r"); rm.diffuse_color=(1.0,0.05,0.05,1); os_.data.materials.append(rm)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=860; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); ymin,ymax=co[:,1].min(),co[:,1].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2; zlo=zmin+0.72*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.40
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((-0.55,-1,0.12),"_sm2_q.png"); shot((1,0.05,0.12),"_sm2_side.png"); shot((0,-1,0.12),"_sm2_front.png")
print("DONE_SEAMS2")
