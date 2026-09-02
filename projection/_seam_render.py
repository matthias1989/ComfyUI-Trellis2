"""Occlusion-correct render of the SEAM hair labelling: build the head mesh from
_genseams_viz.npz (co + tris + tris_hair), colour hair=blue, proposed-carve=red,
skin=gray, and render front + front-left so we see what is ACTUALLY visible."""
import bpy, os, numpy as np
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
vz=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=vz['co'].astype(np.float64); tris=vz['tris']; th=vz['tris_hair'].astype(bool)
# proposed carve (same formula as the scatter): forward-DOWN, lateral, not the very top
v0,v1,v2=co[tris[:,0]],co[tris[:,1]],co[tris[:,2]]
fc=(v0+v1+v2)/3.0
fn=np.cross(v1-v0,v2-v0); fn/=(np.linalg.norm(fn,axis=1,keepdims=True)+1e-9)
fx,fz=fc[:,0],fc[:,2]; ny,nz=fn[:,1],fn[:,2]
carve = th&(ny<-0.20)&(nz<-0.10)&(fz<0.47)&(np.abs(fx)>0.045)
# build mesh
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("seam"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
obj=bpy.data.objects.new("seam",me); bpy.context.scene.collection.objects.link(obj)
mg=bpy.data.materials.new("g"); mg.diffuse_color=(0.78,0.74,0.70,1)   # skin gray
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.12,0.45,0.88,1)   # hair blue
mr=bpy.data.materials.new("r"); mr.diffuse_color=(0.88,0.13,0.13,1)   # carve red
me.materials.append(mg); me.materials.append(mb); me.materials.append(mr)
mi=np.where(carve,2,np.where(th,1,0)).astype(np.int32)
me.polygons.foreach_set("material_index", mi)
me.update()
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=900; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
# frame the head/upper body
zmn,zmx=co[:,2].min(),co[:,2].max(); xmn,xmx=co[:,0].min(),co[:,0].max()
cx=(xmn+xmx)/2; cy=(co[:,1].min()+co[:,1].max())/2; zlo=0.20; zc=(zlo+zmx)/2
span=max(zmx-zlo,(xmx-xmn))*0.55
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0,-1,0.0),"_seam_front.png")        # pure front (front faces point -Y)
shot((-0.55,-1,0.0),"_seam_frontleft.png")# front-left 3/4 (how you usually view)
print("DONE_SEAMRENDER carve",int(carve.sum()),"hair",int(th.sum()))
