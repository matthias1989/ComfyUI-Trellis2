# Apply Blackwell GPU compatibility patches before any TRELLIS2 imports.
# This is a no-op on non-Blackwell GPUs, so it's safe to always call.
try:
    from .blackwell_fix import patch_all as _bf_patch_all
    _bf_patch_all()
except Exception as _bf_err:
    print(f"[ComfyUI-Trellis2] blackwell_fix.patch_all() skipped: {_bf_err}")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]