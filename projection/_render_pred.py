import bpy, os, math
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
OBJ=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.obj"
TEX=os.path.join(D,"_pred_real.png")
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.obj_import(filepath=OBJ)
obj=next(o for o in bpy.data.objects if o.type=='MESH')
mat=bpy.data.materials.new("P"); mat.use_nodes=True
b=mat.node_tree.nodes.get("Principled BSDF")
img=bpy.data.images.load(TEX); tx=mat.node_tree.nodes.new("ShaderNodeTexImage"); tx.image=img
mat.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
obj.data.materials.clear(); obj.data.materials.append(mat)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
sc.render.resolution_x=820; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
cs=[obj.matrix_world@Vector(c) for c in obj.bound_box]
xs=[v.x for v in cs]; ys=[v.y for v in cs]; zs=[v.z for v in cs]
xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
zlo=zmin+0.66*(zmax-zmin); zc=(zlo+zmax)/2          # head + neck + shoulders
span=max(zmax-zlo, xmax-xmin)*0.58
cam_d=bpy.data.cameras.new("C"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("C",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dirv,name):
    dv=Vector(dirv).normalized(); cam.location=Vector((cx,cy,zc))+dv*4.0
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,name); bpy.ops.render.render(write_still=True)
shot((0,-1,0),"_pred_front.png")
shot((0,1,0),"_pred_back.png")
shot((1,0,0),"_pred_right.png")
shot((-1,0,0),"_pred_left.png")
print("DONE")
