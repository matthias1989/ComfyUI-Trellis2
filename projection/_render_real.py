import bpy, os, math
from mathutils import Vector

D   = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
OBJ = r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.obj"
TEX = os.path.join(D, "_newtex.png")

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.obj_import(filepath=OBJ)
obj = next(o for o in bpy.data.objects if o.type == 'MESH')

mat = bpy.data.materials.new("RealTex"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
img = bpy.data.images.load(TEX)
tex = mat.node_tree.nodes.new("ShaderNodeTexImage"); tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
obj.data.materials.clear(); obj.data.materials.append(mat)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'FLAT'
scene.display.shading.color_type = 'TEXTURE'
scene.render.resolution_x = 1100; scene.render.resolution_y = 1100
scene.render.image_settings.file_format = 'PNG'

cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
xs=[v.x for v in cs]; ys=[v.y for v in cs]; zs=[v.z for v in cs]
xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
zlo=zmin+0.80*(zmax-zmin); zc=(zlo+zmax)/2     # TIGHT on the face
span=(zmax-zlo)*0.60
cam_d=bpy.data.cameras.new("C"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("C",cam_d); scene.collection.objects.link(cam); scene.camera=cam

def shot(dirv, name):
    d=Vector(dirv).normalized()
    cam.location=Vector((cx,cy,zc))+d*4.0
    fwd=(Vector((cx,cy,zc))-cam.location).normalized()
    cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=os.path.join(D,name)
    bpy.ops.render.render(write_still=True)

shot(( 0,-1,0), "_rr_A.png")
shot(( 0, 1,0), "_rr_B.png")
shot(( 1, 0,0), "_rr_C.png")
shot((-1, 0,0), "_rr_D.png")
print("DONE_RENDER")
