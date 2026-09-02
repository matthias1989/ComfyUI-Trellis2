import bpy, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
FBX=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00185_\run_20260606_035259\character_posed_00185_.fbx"
ALB=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00185_\run_20260606_035259\character_posed_00185_.fbm\character_posed_00185__albedo.png"
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.fbx(filepath=FBX)
meshes=[o for o in bpy.data.objects if o.type=='MESH']
obj=max(meshes, key=lambda o: len(o.data.vertices))
# force material to the albedo, flat
mat=bpy.data.materials.new("ALB"); mat.use_nodes=True
b=mat.node_tree.nodes.get("Principled BSDF")
img=bpy.data.images.load(ALB)
tx=mat.node_tree.nodes.new("ShaderNodeTexImage"); tx.image=img
mat.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
for m in meshes:
    m.data.materials.clear(); m.data.materials.append(mat)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
sc.render.resolution_x=800; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
# bbox over all meshes
import math
allcs=[]
for m in meshes:
    allcs += [m.matrix_world@Vector(c) for c in m.bound_box]
xs=[v.x for v in allcs]; ys=[v.y for v in allcs]; zs=[v.z for v in allcs]
xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
# figure which axis is up = largest extent
ext={'x':xmax-xmin,'y':ymax-ymin,'z':zmax-zmin}
up=max(ext,key=ext.get)
cx,cy,cz=(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2
print("EXT",ext,"UP",up,"bbox z",round(zmin,3),round(zmax,3))
if up=='z':
    hi=zmax; lo=zmin+0.60*(zmax-zmin); ctr=Vector((cx,cy,(lo+hi)/2)); upvec='Y'; dirs=[(0,-1,0),(1,0,0)]
elif up=='y':
    hi=ymax; lo=ymin+0.60*(ymax-ymin); ctr=Vector((cx,(lo+hi)/2,cz)); upvec='Z'; dirs=[(0,0,-1),(1,0,0)]
else:
    hi=xmax; lo=xmin+0.60*(xmax-xmin); ctr=Vector(((lo+hi)/2,cy,cz)); upvec='Z'; dirs=[(0,-1,0),(0,0,1)]
span=max(ext.values())*0.42
cam_d=bpy.data.cameras.new("C"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("C",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dirv,name):
    d=Vector(dirv).normalized(); cam.location=ctr+d*5.0
    fwd=(ctr-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z',upvec).to_euler()
    sc.render.filepath=os.path.join(D,name); bpy.ops.render.render(write_still=True)
shot(dirs[0],"_fbx_front.png")
shot(dirs[1],"_fbx_side.png")
print("DONE_FBX")
