import bpy, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
d=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=d['co'].astype(float); tris=d['tris']; th=d['tris_hair']
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(), [], tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.collection.objects.link(ob)
mb=bpy.data.materials.new("body"); mb.diffuse_color=(0.80,0.72,0.66,1)
mh=bpy.data.materials.new("hair"); mh.diffuse_color=(0.92,0.86,0.12,1)
me.materials.append(mb); me.materials.append(mh)
thl=th.tolist()
for i,p in enumerate(me.polygons): p.material_index = 1 if thl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=820; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); ymin,ymax=co[:,1].min(),co[:,1].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
zlo=zmin+0.55*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo, xmax-xmin)*0.60
cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0,-1,0),"_seam_front.png")
shot((0, 1,0),"_seam_back.png")
shot((1, 0,0),"_seam_right.png")
shot((-1,0,0),"_seam_left.png")
print("DONE_VIZ bbox z", round(zmin,2), round(zmax,2))
