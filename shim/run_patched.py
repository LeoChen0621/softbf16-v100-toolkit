"""Launch ComfyUI with softbf16 patches applied to the running process.

    SOFTBF16_OPS=1 python run_patched.py --port 8188 <other main.py flags>

SOFTBF16_OPS=0 keeps ComfyUI on native bf16 (only the sm70 dtype gate is
relaxed) — that is the "what bf16 costs today" reference arm.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.environ.get("SOFTBF16_COMFY", r"D:\ComfyUI")

sys.path.insert(0, HERE)
os.chdir(COMFY)
sys.path.insert(0, COMFY)

# ComfyUI parses its CLI at import time, so rewrite argv before touching comfy.
sys.argv = [os.path.join(COMFY, "main.py")] + sys.argv[1:]

import comfy_shim  # noqa: E402

comfy_shim.enable(ops=os.environ.get("SOFTBF16_OPS", "1") == "1")

runpy.run_path(sys.argv[0], run_name="__main__")
