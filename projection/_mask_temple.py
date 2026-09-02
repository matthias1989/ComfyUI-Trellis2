"""Render the is_hair MASK (red=hair, gray=skin) from the viz, zoomed tight on the
temple/in-front-of-ear, so we can SEE the exact hair/skin pattern (fingers/holes)."""
import bpy, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
v=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=v['co'].astype(float); tris=v['tris']; h=v['tris_hair'].astype(bool)
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.collection.objects.link(ob)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.80,0.80,0.80,1)
mh=bpy.data.materials.new("h"); mh.diffuse_color=(0.92,0.25,0.22,1)
me.materials.append(mb); me.materials.append(mh)
hl=h.tolist()
for i,p in enumerate(me.polygons): p.material_index=1 if hl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'
sc.display.shading.color_type='MATERIAL'; sc.display.shading.show_cavity=True
sc.render.resolution_x=950; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
# tight on the right temple/ear
cx,cy,cz = 0.11, -0.01, 0.385
span=0.16
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,cz))+dvv*4
    fwd=(Vector((cx,cy,cz))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((1,0.05,0.05),"_mask_side.png")
shot((0.5,-1,0.10),"_mask_q.png")
print("DONE_MASK")
