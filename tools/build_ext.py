"""Compile (or load from cache) the softbf16 CUDA extension for this package.

Run under a vcvars64 environment; the installer does that for you.
"""
import os
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "vendor" / "softbf16" / "src"
LIBS = PKG / "cudalibs"

os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(PKG / "ext_cache"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ["LIB"] = str(LIBS / "lib") + ";" + os.environ.get("LIB", "")
os.environ["PATH"] = str(Path(sys.executable).parent / "Scripts") + ";" + os.environ.get("PATH", "")

sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

major, minor = torch.cuda.get_device_capability()
if major != 7:
    print("WARNING: this kernel targets sm_70 (V100); this GPU is sm_%d%d" % (major, minor))

t0 = time.time()
ext = load(
    name="rowcol_scaled_gemm_bf16_sm%d%d" % (major, minor),
    sources=[str(SRC / "kernels" / "rowcol_scaled_gemm_bf16.cu")],
    extra_include_paths=[str(LIBS / "nvidia" / n / "include")
                         for n in ("cublas", "cusparse", "cusolver")],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["cublas.lib"],
    verbose=False,
)
print("extension ready in %.1fs -> %s" % (time.time() - t0, ext.__file__))
