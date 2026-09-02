"""COLOUR-based hair detector: sample the baked basecolor per face (via the GLB's
matching UV) and classify hair (blonde = yellowish) vs skin (tan).  This is what the
eye uses.  Render the mask to verify it follows the actual hair, then save per-face
centroid + hair flag for gen_seams to consume via a KDTree bridge."""
import bpy, os, glob
from mathutils import Vector
import numpy as np
D=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Trellis2\projection"
GLB=r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\uv_transfer\last_seams.glb"
TEX=sorted(glob.glob(r"C:\applications\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\output\3D\char_basecolor_*.png"))[-1]
print("basecolor:", TEX)
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
bpy.ops.import_scene.gltf(filepath=GLB)
obj=max([o for o in bpy.data.objects if o.type=='MESH'], key=lambda o: len(o.data.vertices))
me=obj.data
img=bpy.data.images.load(TEX); W,H=img.size
px=np.array(img.pixels[:], dtype=np.float32).reshape(H,W,4)  # row 0 = bottom
nf=len(me.polygons)
# per-face UV centroid
uvl=me.uv_layers.active.data
# world coords, then normalize EXACTLY like gen_seams (so centroids match its space)
mw=obj.matrix_world
cow=np.array([list(mw @ v.co) for v in me.vertices], dtype=float)
_zmn,_zmx=float(cow[:,2].min()),float(cow[:,2].max())
_xmn,_xmx=float(cow[:,0].min()),float(cow[:,0].max())
_zs=0.91/(_zmx-_zmn); _zo=-0.41-_zmn*_zs; _xs=0.92/(_xmx-_xmn); _xc=(_xmn+_xmx)/2.0
con=np.empty_like(cow)
con[:,2]=cow[:,2]*_zs+_zo; con[:,0]=(cow[:,0]-_xc)*_xs; con[:,1]=cow[:,1]*_zs
co=cow   # world coords for camera framing
fc=np.array([con[list(p.vertices)].mean(0) for p in me.polygons], dtype=float)  # NORMALIZED centroids
print("normalized centroid bbox:", [round(float(x),3) for x in con.min(0)], [round(float(x),3) for x in con.max(0)])
cols=np.zeros((nf,3),dtype=np.float32)
for i,p in enumerate(me.polygons):
    u=v=0.0
    for li in p.loop_indices:
        uv=uvl[li].uv; u+=uv[0]; v+=uv[1]
    n=len(p.loop_indices); u/=n; v/=n
    xpix=int(np.clip(u,0,1)*(W-1)); ypix=int(np.clip(v,0,1)*(H-1))
    cols[i]=px[ypix,xpix,:3]
R,Gc,B=cols[:,0],cols[:,1],cols[:,2]
_lum = 0.299*R + 0.587*Gc + 0.114*B
# The projection washes the blonde hue down to near-skin, but hair ALBEDO stays DARKER
# than face skin in the basecolor.  Classify by luminance (+ a yellow tilt where present).
_yellow = (_lum < 0.80) | ((B < 0.945*Gc) & (Gc > 0.15))   # darker OR yellowish = HAIR
_skin   = (_lum > 0.85) & (B > 0.965*Gc)                    # bright + non-yellow = SKIN
print("color stats: mean RGB", [round(float(x),3) for x in cols.mean(0)],
      " lum[min/med/max]", [round(float(x),3) for x in (np.min(_lum),np.median(_lum),np.max(_lum))],
      " hair", int(_yellow.sum()), " skin", int(_skin.sum()), "/", nf)
np.savez(os.path.join(D,"_color_hair.npz"), centroids=fc.astype('float32'),
         hair=_yellow, skin=_skin, rgb=cols)
# render the colour-classified mask: red=hair, gray=skin
for m in list(me.materials): pass
mb=bpy.data.materials.new("b"); mb.diffuse_color=(0.8,0.8,0.8,1)
mh=bpy.data.materials.new("h"); mh.diffuse_color=(0.92,0.25,0.22,1)
me.materials.clear(); me.materials.append(mb); me.materials.append(mh)
hl=_yellow.tolist()
for i,p in enumerate(me.polygons): p.material_index=1 if hl[i] else 0
sc=bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'
sc.render.resolution_x=950; sc.render.resolution_y=950; sc.render.image_settings.file_format='PNG'
xmin,xmax=co[:,0].min(),co[:,0].max(); zmin,zmax=co[:,2].min(),co[:,2].max()
cx=(xmin+xmax)/2; cy=(co[:,1].min()+co[:,1].max())/2; zlo=zmin+0.70*(zmax-zmin); zc=(zlo+zmax)/2
span=max(zmax-zlo,(xmax-xmin))*0.5
cd=bpy.data.cameras.new("c"); cd.type='ORTHO'; cd.ortho_scale=span*2
cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam); sc.camera=cam
def shot(dv,nm):
    dvv=Vector(dv).normalized(); cam.location=Vector((cx,cy,zc))+dvv*4
    fwd=(Vector((cx,cy,zc))-cam.location).normalized(); cam.rotation_euler=fwd.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(D,nm); bpy.ops.render.render(write_still=True)
shot((0.5,-1,0.10),"_colmask_q.png")
shot((1,0.05,0.06),"_colmask_side.png")
print("DONE_COLORDETECT")
