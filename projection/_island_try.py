"""Fast island-boundary iteration: compute hair region GEOMETRICALLY (head/drape
minus the face oval the user drew in green) and render from the same angle so we
converge to the green target. No gen_seams / UV run needed.
gen coords: x=lateral(+-0.46)  y=depth(front<0, +-0.08)  z=height(-0.41..0.50)"""
import bpy, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
v=np.load(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\_genseams_viz.npz")
co=v['co'].astype(float); tris=v['tris']
cen=co[tris].mean(axis=1); x=cen[:,0]; y=cen[:,1]; z=cen[:,2]; ax=np.abs(x)

# ---- TUNABLE boundary (approx the green) -------------------------------------
FACE  = (y < -0.008) & (ax < 0.062) & (z > 0.295) & (z < 0.455)   # face oval (skin)
NECK  = (y < -0.012) & (ax < 0.055) & (z > 0.10)  & (z < 0.300)   # front of neck (skin)
TORSOF= (y < -0.004) & (z < 0.10)                                 # front chest/torso (skin)
SCALP = (z > 0.295) & ~FACE                                       # head minus face
DRAPE = (z <= 0.295) & (z > -0.05) & (y > -0.02) & (ax < 0.15) & ~NECK   # hanging hair (back/side)
is_hair = (SCALP | DRAPE) & ~TORSOF
# -----------------------------------------------------------------------------
print(f"hair tris={int(is_hair.sum())}/{len(is_hair)}")

for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
me=bpy.data.meshes.new("m"); me.from_pydata(co.tolist(),[],tris.tolist()); me.update()
ob=bpy.data.objects.new("m",me); bpy.context.collection.objects.link(ob)
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.82,0.82,0.82,1)
mh=bpy.data.materials.new("h"); mh.diffuse_color=(0.90,0.32,0.30,1)
me.materials.append(mb); me.materials.append(mh)
hl=is_hair.tolist()
for i,p in enumerate(me.polygons): p.material_index=1 if hl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'
sc.display.shading.color_type='MATERIAL'; sc.display.shading.show_cavity=True
sc.render.resolution_x=850; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); ymin,ymax=co[:,1].min(),co[:,1].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2; zlo=zmin+0.74*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.40
cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((1,0.05,0.05),"_isl_sideR.png")
shot((-1,0.05,0.05),"_isl_sideL.png")
shot((-0.55,-1,0.12),"_isl_q.png")
print("DONE_ISL")
