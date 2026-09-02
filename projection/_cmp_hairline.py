"""Render the real mesh SHADED (studio light, so the hair-mass shape & visible
hairline show) with my current hair classification tinted red. Lets me compare
where the visible hairline is vs where my seam boundary sits."""
import bpy, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
d=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=d['co'].astype(float); tris=d['tris']; th=d['tris_hair']
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.collection.objects.link(ob)
mb=bpy.data.materials.new("body"); mb.diffuse_color=(0.82,0.82,0.82,1)
mh=bpy.data.materials.new("hair"); mh.diffuse_color=(0.90,0.35,0.32,1)
me.materials.append(mb); me.materials.append(mh)
thl=th.tolist()
for i,p in enumerate(me.polygons): p.material_index=1 if thl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'
sc.display.shading.light='STUDIO'           # shaded -> reveals the hair-mass shape
sc.display.shading.color_type='MATERIAL'
sc.display.shading.show_cavity=True
sc.render.resolution_x=820; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); ymin,ymax=co[:,1].min(),co[:,1].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
zlo=zmin+0.78*(zmax-zmin); zc=(zlo+zmax)/2          # head zoom
span=max(zmax-zlo,(xmax-xmin))*0.34
cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0,-1,0.12),"_cmp_front.png")
shot((0.9,-0.5,0.12),"_cmp_q.png")
shot((1,0,0.12),"_cmp_side.png")
print("DONE_CMP")
