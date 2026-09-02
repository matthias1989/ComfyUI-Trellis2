"""Render the ACTUAL marked seam edges (bright red tubes) over the gray mesh,
so we can see what the seams look like (like the user's viewport) and verify the
arm ring closes + the hairline follows the hair."""
import bpy, bmesh, numpy as np, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
obj.hide_render=False; obj.hide_viewport=False
me=obj.data
bm=bmesh.new(); bm.from_mesh(me)
verts=np.array([list(v.co) for v in bm.verts], dtype=float)
seam_e=[(e.verts[0].index,e.verts[1].index) for e in bm.edges if e.seam]
bm.free()
print("seam edges:",len(seam_e))
ctr=verts.mean(0)
co_out=ctr+(verts-ctr)*1.012     # push seams slightly proud of the surface
sm=bpy.data.meshes.new("seams"); sm.from_pydata(co_out.tolist(), seam_e, [])
so=bpy.data.objects.new("seams",sm); bpy.context.collection.objects.link(so)
bpy.context.view_layer.objects.active=so; so.select_set(True)
bpy.ops.object.convert(target='CURVE')
so.data.bevel_depth=0.0038; so.data.bevel_resolution=1
matr=bpy.data.materials.new("r"); matr.diffuse_color=(1.0,0.04,0.04,1)
so.data.materials.append(matr)
matb=bpy.data.materials.new("b"); matb.diffuse_color=(0.80,0.80,0.80,1)
me.materials.clear(); me.materials.append(matb)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=860; sc.render.resolution_y=900; sc.render.image_settings.file_format='PNG'
xmin,xmax=verts[:,0].min(),verts[:,0].max(); ymin,ymax=verts[:,1].min(),verts[:,1].max(); zmin,zmax=verts[:,2].min(),verts[:,2].max()
cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
def cam_to(zlo_frac, name_dirs, span_mul):
    zlo=zmin+zlo_frac*(zmax-zmin); zc=(zlo+zmax)/2
    span=max(zmax-zlo,(xmax-xmin))*span_mul
    cam_d=bpy.data.cameras.new("c"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
    cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam); sc.camera=cam
    for dv,nm in name_dirs:
        dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
        fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
        sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam,do_unlink=True)
# head zoom (profiles)
cam_to(0.74,[((1,0.05,0.12),"_sm_headR.png"),((-1,0.05,0.12),"_sm_headL.png"),((0,-1,0.12),"_sm_front.png")],0.40)
# wider (arms / drape)
cam_to(0.45,[((0.6,-0.7,0.10),"_sm_armq.png")],0.62)
print("DONE_SEAMS")
