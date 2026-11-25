"""Drop-in shim for bf16 training on V100 (sm70).

    import train_shim; train_shim.enable()
    # ... build model / start trainer ...

Every bf16 F.linear with a bf16 weight goes through softbf16's scaled-fp16
tensor-core GEMM instead of torch's fp32-emulated bf16 path. We measured
fwd+bwd at 1.31x fp16 (vs 6.97x for native bf16) on the z-image layer shapes,
which maps to ~19.8 s/step -> ~3.7 s/step for the MOSS LoRA setup.

Keep optimizer state in fp32; only the matmuls change.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def _src_dir():
    for rel in ("vendor/softbf16/src", "Soft-BF16-V100-main/src"):
        p = ROOT / rel
        if (p / "bf16_online.py").exists():
            return p
    raise SystemExit("softbf16 source not found under %s" % ROOT)


SRC = _src_dir()
LIBS = ROOT / "cudalibs"

_enabled = False


def enable(verbose=True):
    global _enabled
    if _enabled:
        return
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "ext_cache"))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    os.environ["LIB"] = str(LIBS / "lib") + ";" + os.environ.get("LIB", "")
    os.environ["PATH"] = str(Path(sys.executable).parent / "Scripts") + ";" + os.environ.get("PATH", "")
    sys.path.insert(0, str(SRC))

    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load

    ext = load(
        name="rowcol_scaled_gemm_bf16_sm70",
        sources=[str(SRC / "kernels" / "rowcol_scaled_gemm_bf16.cu")],
        extra_include_paths=[str(LIBS / "nvidia" / n / "include")
                             for n in ("cublas", "cusparse", "cusolver")],
        extra_cuda_cflags=["-O3"], extra_ldflags=["cublas.lib"], verbose=False,
    )
    import bf16_online
    bf16_online._ext = ext

    orig = F.linear
    hits = {"n": 0}

    def linear(input, weight, bias=None):
        if (input.dtype is torch.bfloat16 and input.is_cuda and input.dim() >= 2
                and isinstance(weight, torch.Tensor) and weight.dim() == 2
                and weight.dtype is torch.bfloat16
                and weight.shape[1] == input.shape[-1]
                and weight.device == input.device):
            hits["n"] += 1
            return bf16_online.linear(input, weight, bias)
        return orig(input, weight, bias)

    F.linear = linear
    _enabled = True
    if verbose:
        print("[train_shim] bf16 F.linear routed through softbf16", file=sys.stderr, flush=True)
    return hits
