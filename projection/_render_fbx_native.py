import bpy, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
FBX=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\characters\character_posed_00185_\run_20260606_035259\character_posed_00185_.fbx"
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.fbx(filepath=FBX)
meshes=[o for o in bpy.data.objects if o.type=='MESH']

# Report material + texture wiring exactly as imported
info=[]
for m in meshes:
    for slot in m.material_slots:
        mat=slot.material
        if not mat: continue
        imgs=[]
        if mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type=='TEX_IMAGE' and n.image:
                    # what is this image plugged into?
                    tgt=[]
                    for o2 in n.outputs:
                        for l in o2.links:
                            tgt.append(f"{l.to_node.name}.{l.to_socket.name}")
                    imgs.append({"img":n.image.name,"file":bpy.path.abspath(n.image.filepath),"->":tgt})
        info.append({"mesh":m.name,"mat":mat.name,"verts":len(m.data.vertices),
                     "uv_layers":[u.name for u in m.data.uv_layers],"imgs":imgs})

# render with NATIVE materials (whatever the FBX defines), flat texture view
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='FLAT'; sc.display.shading.color_type='TEXTURE'
sc.render.resolution_x=800; sc.render.resolution_y=1000; sc.render.image_settings.file_format='PNG'
allcs=[]
for m in meshes: allcs+=[m.matrix_world@Vector(c) for c in m.bound_box]
xs=[v.x for v in allcs]; ys=[v.y for v in allcs]; zs=[v.z for v in allcs]
xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
sc.render.resolution_x=1000; sc.render.resolution_y=1000
cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
# TIGHT crop on head+neck: z from 0.30 to top
hz=zmax; lz=zmin+0.78*(zmax-zmin); ctr=Vector((cx,cy,(lz+hz)/2))
span=(hz-lz)*0.62
cam_d=bpy.data.cameras.new("C"); cam_d.type='ORTHO'; cam_d.ortho_scale=span*2
cam=bpy.data.objects.new("C",cam_d); sc.collection.objects.link(cam); sc.camera=cam
def shot(dirv,name):
    d=Vector(dirv).normalized(); cam.location=ctr+d*5.0
    fwd=(ctr-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,name); bpy.ops.render.render(write_still=True)
shot((0,-1,0),"_fbxn_front.png")       # front
shot((0.7,-0.7,0),"_fbxn_q.png")       # 3/4 right-front (jaw/ear/neck)
shot((1,0,0),"_fbxn_side.png")         # right profile
import json
print("MATINFO="+json.dumps(info))
print("DONE")
