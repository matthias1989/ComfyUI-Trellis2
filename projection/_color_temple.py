"""Render geometry_0 with the basecolor texture so we can SEE hair (dark) vs skin
(tan) on the actual geometry around the temple/ear -- settles whether that patch is
hair or skin."""
import bpy, os
from mathutils import Vector
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
TEX=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_00357_.png"
obj=bpy.data.objects.get('geometry_0') or next(o for o in bpy.data.objects if o.type=='MESH')
# remove other objects so framing is clean
for o in list(bpy.data.objects):
    if o is not obj: bpy.data.objects.remove(o,do_unlink=True)
me=obj.data
mat=bpy.data.materials.new("col"); mat.use_nodes=True
nt=mat.node_tree; nt.nodes.clear()
out=nt.nodes.new("ShaderNodeOutputMaterial")
em=nt.nodes.new("ShaderNodeEmission")
tex=nt.nodes.new("ShaderNodeTexImage")
try: tex.image=bpy.data.images.load(TEX)
except Exception as e: print("IMG LOAD FAIL", e)
nt.links.new(tex.outputs['Color'], em.inputs['Color'])
nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
me.materials.clear(); me.materials.append(mat)
sc=bpy.context.scene
sc.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
try: sc.render.engine='CYCLES'
except: pass
sc.render.resolution_x=950; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
co=[obj.matrix_world @ v.co for v in me.vertices]
import numpy as np
A=np.array([[c.x,c.y,c.z] for c in co])
cx,cy,cz=0.11,-0.02,0.385; span=0.17
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,cz))+dvv*4
    fwd=(Vector((cx,cy,cz))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((1,0.04,0.05),"_color_side.png")
shot((0.5,-1,0.10),"_color_q.png")
print("DONE_COLOR")
