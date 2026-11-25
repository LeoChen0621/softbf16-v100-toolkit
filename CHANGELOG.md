# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/), versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-11-25

First public release. Everything below was measured on a single
Tesla V100-PCIE-32GB (sm_70, TCC), Windows 11, torch 2.7.0.dev+cu128,
ComfyUI 0.43.0, with the power limit restored to the factory 250 W.

### Added

- **Kernel build pipeline** (`install.ps1`, `tools/build_ext.py`): fetches the
  cuBLAS/cuSPARSE/cuSOLVER headers from the PyPI win_amd64 wheels (no full CUDA
  Toolkit needed), generates `cublas.lib` from torch's bundled DLL, and compiles
  the softbf16 extension into a local cache.
- **ComfyUI integration** (`shim/comfy_shim.py`, `shim/run_patched.py`): loads the
  kernel, relaxes ComfyUI's sm70 bf16 gate, and routes bf16 `F.linear`,
  `F.conv*`, SDPA and `xformers.ops.memory_efficient_attention` through it.
  ComfyUI itself is never modified.
- **Faster bf16 attention** (`shim/fast_attn.py`): keeps the overflow-range check
  but skips the per-call power-of-two rescale when the tensors sit inside fp16's
  safe band. 10.39 ms → 7.99 ms per call at B=1/H=30/S=4096/D=128, bit-identical
  output.
- **Training shim** (`shim/train_shim.py`): one-line `enable()` for bf16 training.
- **Upstream integration port** (`shim/fork_ops.py`): the ComfyUI-V100
  `ops.py` / `quant_ops.py` / `sd.py` patches as monkeypatches.
- **Tools**: `fp8_convert.py` (plain fp8 cast, half the VRAM), `free_vram.py`
  (`/free` is queued, `--now` forces it), `selftest.py`.
- **Docs**: `docs/benchmarks.zh-CN.md` with the full measurement log, including
  the negative results.

### Measurements

| Item | Result |
|---|---|
| 8192³ bf16 GEMM | 70.3 TFLOPS (native bf16 10.0, cuBLAS fp16 82) |
| bf16 accuracy | identical to torch bf16 (same max error) |
| z-image bf16, 1024², 8 steps | fp16 10.5 s → bf16+toolkit 14.9 s, native bf16 unusable |
| training fwd+bwd (real layer shapes) | native bf16 6.97x fp16 → toolkit 1.31x |
| training loop (2 blocks, AdamW) | 209.4 → 72.6 ms/step, loss delta 3.8e-8 |
| fp8 weights | same speed as fp16, half the VRAM (HunyuanVideo 15.5 → 7.8 GB) |
| power limit 200 W → 250 W | cuBLAS 79.9 → 93.0 TFLOPS (+16%) |

### Known limitations

- In inference the toolkit is **slower than plain fp16** (1.42x on z-image).
  It only pays off when bf16 numerics are required (fp16 overflow/clipping).
- MiniMax H3 does not benefit: its unet is `int8_convrot`, so no bf16 GEMM is
  involved, and the extra packed buffers push the 32 GB card into OOM.
- Only `linear` / `conv` / `attention` are covered; elementwise, norms and
  softmax are untouched (they are bandwidth-bound, not tensor-core work).
- Vendored kernel is AGPL-3.0; this repository is distributed under the same
  license.
