"""Post-install self test: load the kernel, run one bf16 GEMM, check accuracy.

Exit code 0 on success.
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

major, minor = torch.cuda.get_device_capability()
ext = load(
    name="rowcol_scaled_gemm_bf16_sm%d%d" % (major, minor),
    sources=[str(SRC / "kernels" / "rowcol_scaled_gemm_bf16.cu")],
    extra_include_paths=[str(LIBS / "nvidia" / n / "include")
                         for n in ("cublas", "cusparse", "cusolver")],
    extra_cuda_cflags=["-O3"], extra_ldflags=["cublas.lib"], verbose=False)

import bf16_online  # noqa: E402
bf16_online._ext = ext

print("GPU:", torch.cuda.get_device_name(), "cc", (major, minor))

# accuracy on a small matrix against fp32
m, k, n = 512, 1024, 512
a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
ref = a.float() @ w.float().t()
out = bf16_online.linear(a, w)
rel = ((out.float() - ref).norm() / ref.norm()).item()
print("accuracy: rel err vs fp32 = %.3e (bf16 rounding is ~1e-3)" % rel)
if rel > 5e-3:
    print("FAIL: accuracy worse than expected")
    sys.exit(1)

# throughput on 8192^3
m = k = n = 8192
a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
for _ in range(2):
    bf16_online.linear(a, w)
torch.cuda.synchronize()
t0 = time.perf_counter()
iters = 5
for _ in range(iters):
    bf16_online.linear(a, w)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters
tf = 2 * m * k * n / dt / 1e12
print("8192^3 bf16 GEMM: %.2f ms  %.1f TFLOPS" % (dt * 1e3, tf))

fp16 = F.linear(a.half(), w.half())
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    F.linear(a.half(), w.half())
torch.cuda.synchronize()
dt16 = (time.perf_counter() - t0) / iters
print("8192^3 fp16 GEMM: %.2f ms  %.1f TFLOPS  (softbf16 = %.0f%% of cuBLAS fp16)" % (
    dt16 * 1e3, 2 * m * k * n / dt16 / 1e12, 100 * dt16 / dt))

if tf < 40:
    print("FAIL: throughput lower than expected (check clocks / power limit)")
    sys.exit(1)
print("SELFTEST OK")
