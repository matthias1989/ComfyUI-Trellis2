import bpy, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
base=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00185_\run_20260607_005314"
FBX=os.path.join(base,"character_posed_00185_.fbx")
ALB=os.path.join(base,"character_posed_00185_.fbm","character_posed_00185__albedo.png")
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.fbx(filepath=FBX)
meshes=[o for o in bpy.data.objects if o.type=='MESH']
mat=bpy.data.materials.new("A"); mat.use_nodes=True
b=mat.node_tree.nodes.get("Principled BSDF"); img=bpy.data.images.load(ALB)
tx=mat.node_tree.nodes.new("ShaderNodeTexImage"); tx.image=img
mat.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
for m in meshes: m.data.materials.clear(); m.data.materials.append(mat)
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
sc.render.resolution_x=900; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
allc=[]
for m in meshes: allc+=[m.matrix_world@Vector(c) for c in m.bound_box]
xs=[v.x for v in allc]; ys=[v.y for v in allc]; zs=[v.z for v in allc]
xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
ext={'x':xmax-xmin,'y':ymax-ymin,'z':zmax-zmin}; up=max(ext,key=ext.get)
cx,cy,cz=(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2
hi=max(zmax,ymax,xmax)
if up=='z': lo=zmin+0.74*(zmax-zmin); ctr=Vector((cx,cy,(lo+zmax)/2)); upv='Y'; dirs={'front':(0,-1,0),'q':(0.7,-0.7,0),'side':(1,0,0)}
elif up=='y': lo=ymin+0.74*(ymax-ymin); ctr=Vector((cx,(lo+ymax)/2,cz)); upv='Z'; dirs={'front':(0,0,-1),'q':(0.7,0,-0.7),'side':(1,0,0)}
else: lo=xmin+0.74*(xmax-xmin); ctr=Vector(((lo+xmax)/2,cy,cz)); upv='Z'; dirs={'front':(0,-1,0),'q':(0,-0.7,0.7),'side':(0,0,1)}
span=(hi-lo)*0.62
cd=bpy.data.cameras.new("C"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("C",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    d=Vector(dv).normalized(); cam.location=ctr+d*5
    fwd=(ctr-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z',upv).to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
print("UP",up,"ext",ext)
shot(dirs['front'],"_fbx2_front.png"); shot(dirs['q'],"_fbx2_q.png"); shot(dirs['side'],"_fbx2_side.png")
print("DONE")
