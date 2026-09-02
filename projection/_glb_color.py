"""Import last_seams.glb (UV matches the baked basecolor) + apply char_basecolor,
render the head in colour to confirm we can read hair vs skin per-face."""
import bpy, os, glob
from mathutils import Vector
import numpy as np
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
GLB=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.glb"
# pick the basecolor closest in time to the glb
cands=sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_*.png"))
TEX=cands[-1] if cands else None
print("using basecolor:", TEX)
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
bpy.ops.import_scene.gltf(filepath=GLB)
obj=max([o for o in bpy.data.objects if o.type=='MESH'], key=lambda o: len(o.data.vertices))
me=obj.data
mat=bpy.data.materials.new("col"); mat.use_nodes=True
nt=mat.node_tree; nt.nodes.clear()
out=nt.nodes.new("ShaderNodeOutputMaterial"); em=nt.nodes.new("ShaderNodeEmission")
tex=nt.nodes.new("ShaderNodeTexImage"); uv=nt.nodes.new("ShaderNodeUVMap")
uv.uv_map="UVMap"
tex.image=bpy.data.images.load(TEX)
nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
nt.links.new(tex.outputs['Color'], em.inputs['Color'])
nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
me.materials.clear(); me.materials.append(mat)
sc=bpy.context.scene
sc.render.engine='CYCLES'; sc.cycles.samples=4
sc.render.resolution_x=950; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
co=np.array([ (obj.matrix_world @ v.co)[:] for v in me.vertices])
xmin,xmax=co[:,0].min(),co[:,0].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(co[:,1].min()+co[:,1].max())/2; zlo=zmin+0.72*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.5
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0.5,-1,0.10),"_glb_q.png")
shot((1,0.05,0.06),"_glb_side.png")
print("DONE_GLBCOLOR")
