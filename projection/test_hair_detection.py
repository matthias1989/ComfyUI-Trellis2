"""
Standalone test for HairDetectionUtils — no ComfyUI / Blender / GPU needed.

Usage:
    python test_hair_detection.py

Prerequisites:
    Run ComfyUI once with debug_regions=True to generate hair_detection_fixture.npz
    (saved automatically next to this file on first run).

Output:
    hair_detection_debug.png  — debug canvas, same image you see in Blender
"""

import sys
import os
import io
import numpy as np
import cv2

# Force UTF-8 stdout so → and other unicode in print statements don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Make sure projection/ is importable
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

FIXTURE = os.path.join(_dir, 'hair_detection_fixture.npz')
OUTPUT  = os.path.join(_dir, 'hair_detection_debug.png')


def main():
    if not os.path.exists(FIXTURE):
        print(f"[ERROR] Fixture not found: {FIXTURE}")
        print("  → Run ComfyUI once with debug_regions=True to generate it.")
        sys.exit(1)

    print(f"[Test] Loading fixture: {FIXTURE}")
    data         = np.load(FIXTURE)
    tex_pos      = data['tex_pos']
    tex_normals  = data['tex_normals']
    uv_mask      = data['uv_mask'].astype(bool)
    texture_size = int(data['texture_size'][0])
    print(f"[Test] texture_size={texture_size}  "
          f"pos={tex_pos.shape}  normals={tex_normals.shape}  "
          f"uv_mask={uv_mask.sum():,} px")

    from hair_detection_utils_v2 import HairDetectionUtils

    hdu    = HairDetectionUtils(texture_size, uv_mask, tex_pos, tex_normals)
    canvas = hdu.run()

    # Save as PNG (RGB → BGR for OpenCV)
    cv2.imwrite(OUTPUT, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[Test] Saved → {OUTPUT}")

    # Quick sanity: count pixels per detection colour
    cyan    = np.all(canvas == [  0, 255, 255], axis=2).sum()
    yellow  = np.all(canvas == [255, 255,   0], axis=2).sum()
    green   = np.all(canvas == [  0, 200,   0], axis=2).sum()
    green2  = np.all(canvas == [  0, 220,   0], axis=2).sum()
    magenta = np.all(canvas == [220,   0, 220], axis=2).sum()
    print(f"[Test] cyan={cyan:,}  yellow={yellow:,}  "
          f"green={green+green2:,}  magenta={magenta:,}")


if __name__ == '__main__':
    main()
