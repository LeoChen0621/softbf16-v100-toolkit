"""Apply softbf16 to a stock ComfyUI process without touching comfy/ sources.

enable() must run before ComfyUI starts sampling. It:
  * builds/loads the CUDA extension from the workspace copy (cached),
  * routes bf16 F.linear through softbf16 (bf16/fp8 weights, bf16 acts),
  * routes bf16 conv through fp32 cuDNN (sm70 has no bf16 conv kernel),
  * routes bf16 SDPA through scaled fp16 on the EFFICIENT backend.
fp16/fp32 tensors are untouched.
"""
import os
import sys
import time
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
_ATTN_SEEN = {"n": 0, "x": 0}


def _log(msg):
    print("[softbf16-shim] %s" % msg, file=sys.stderr, flush=True)


def _classify(name):
    n = name.lower()
    if any(k in n for k in ("gemm", "xmma", "cutlass", "cublas", "wmma", "sgemm")):
        return "gemm"
    if any(k in n for k in ("attention", "softmax", "fmha", "flash", "swiglu")):
        return "attention/softmax"
    if "conv" in n:
        return "conv"
    if any(k in n for k in ("elementwise", "reduce", "norm", "cat", "copy", "cast")):
        return "elementwise/copy"
    return "other"


def _install_profiler():
    import torch
    import comfy.samplers as cs

    orig = cs.sample

    def profiled(*a, **k):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = orig(*a, **k)
        cats = {}
        for e in prof.key_averages():
            t = e.self_device_time_total
            if t > 0:
                c = _classify(e.key)
                cats[c] = cats.get(c, 0) + t
        total = sum(cats.values()) or 1
        _log("[profile] sampling cuda total %.1f ms" % (total / 1000.0))
        for c, t in sorted(cats.items(), key=lambda x: -x[1]):
            _log("[profile]   %-18s %8.1f ms  %5.1f%%" % (c, t / 1000.0, 100.0 * t / total))
        top = sorted((e for e in prof.key_averages() if e.self_device_time_total > 0),
                     key=lambda e: -e.self_device_time_total)[:12]
        for e in top:
            _log("[profile] top %7.1f ms %5.1f%%  %s" % (
                e.self_device_time_total / 1000.0,
                100.0 * e.self_device_time_total / total,
                e.key[:110]))
        return out

    cs.sample = profiled
    _log("sampler profiler enabled")


def enable(ops=True):
    """ops=False only relaxes ComfyUI's sm70 bf16 gate (reference arm)."""
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "ext_cache"))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    os.environ["LIB"] = str(LIBS / "lib") + ";" + os.environ.get("LIB", "")
    # torch's JIT needs the ninja.exe that ships with the pip package
    os.environ["PATH"] = str(Path(sys.executable).parent / "Scripts") + ";" + os.environ.get("PATH", "")
    sys.path.insert(0, str(SRC))

    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load

    t0 = time.time()
    ext = load(
        name="rowcol_scaled_gemm_bf16_sm70",
        sources=[str(SRC / "kernels" / "rowcol_scaled_gemm_bf16.cu")],
        extra_include_paths=[str(LIBS / "nvidia" / n / "include")
                             for n in ("cublas", "cusparse", "cusolver")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["cublas.lib"],
        verbose=False,
    )
    import bf16_online
    import bf16_conv
    import bf16_attn
    import fast_attn
    bf16_online._ext = ext
    _log("extension ready in %.1fs" % (time.time() - t0))

    # ComfyUI 0.43 decides fp16 for sm70, so a bf16 checkpoint is silently cast to
    # fp16 and none of the kernels below would ever see bf16. Relax the gate the
    # same way ComfyUI-V100 does, so weights and compute stay bf16.
    if os.environ.get("SOFTBF16_BF16", "1") != "1":
        _log("bf16 gate left alone (SOFTBF16_BF16=0)")
        if os.environ.get("SOFTBF16_PROFILE") == "1":
            _install_profiler()
        return
    import comfy.model_management as mm
    mm.should_use_bf16 = lambda *a, **k: True
    _log("should_use_bf16 -> True (sm70 bf16 gate relaxed)")
    if not ops:
        _log("ops disabled: reference arm on native bf16")
        if os.environ.get("SOFTBF16_PROFILE") == "1":
            _install_profiler()
        return

    # --- linear ---
    fork_mode = os.environ.get("SOFTBF16_FORK") == "1"
    if fork_mode:
        # ComfyUI-V100's own integration: softbf16_ops classes + fp8 layout op.
        import fork_ops
        fork_ops.apply(bf16_online)
        _log("fork ops mode: module-level softbf16_ops (no F.linear patch)")
        _orig_linear = F.linear

        def _softbf16_linear(input, weight, bias=None):
            return _orig_linear(input, weight, bias)

    else:
        _orig_linear = F.linear
        _seen = {"n": 0, "hit": 0}

        def _softbf16_linear(input, weight, bias=None):
            # nn.Parameter is a Tensor subclass and comfy passes it straight through,
            # so accept any Tensor that is not a QuantizedTensor wrapper.
            if (input.dtype is torch.bfloat16 and input.is_cuda and input.dim() >= 2
                    and isinstance(weight, torch.Tensor) and weight.dim() == 2
                    and getattr(weight, "_qdata", None) is None
                    and weight.dtype in (torch.bfloat16, torch.float8_e4m3fn)
                    and weight.shape[1] == input.shape[-1]
                    and weight.device == input.device):
                _seen["hit"] += 1
                if _seen["hit"] == 1:
                    _log("first softbf16 linear: input=%s weight=%s shape=%s" % (
                        input.dtype, weight.dtype, tuple(input.shape[-2:]) + tuple(weight.shape)))
                return bf16_online.linear(input, weight, bias)
            _seen["n"] += 1
            if _seen["n"] == 1:
                _log("first plain linear: input=%s weight=%s" % (input.dtype, weight.dtype))
            return _orig_linear(input, weight, bias)

    F.linear = _softbf16_linear

    # --- conv ---
    # Route every bf16 conv to fp32: cuDNN has no bf16 kernel on sm_70, and the
    # native path both fails and is slower than fp32 operands.
    bf16_conv._MIN_MACS_PASS = 0.0
    bf16_conv.install()

    # --- attention ---
    # Only bf16 is detoured: sm_70 has no bf16 SDPA kernel. Every other dtype
    # (fp32 text encoder, fp16 VAE) keeps ComfyUI's own backend dispatch.
    _orig_sdpa = F.scaled_dot_product_attention

    def _softbf16_attention(query, key, value, *args, **kwargs):
        if os.environ.get("SOFTBF16_ATTN_LOG") == "1" and _ATTN_SEEN["n"] < 8:
            _ATTN_SEEN["n"] += 1
            _log("[attn] q=%s dt=%s mode=%s" % (tuple(query.shape), query.dtype, bf16_attn.get_mode()))
        if query.dtype is not torch.bfloat16:
            return _orig_sdpa(query, key, value, *args, **kwargs)
        mask = kwargs.get("attn_mask")
        if mask is not None and mask.dtype != torch.bool and mask.dtype != torch.float16:
            kwargs["attn_mask"] = mask.half()
        return fast_attn.attention(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _softbf16_attention

    # ComfyUI's unet attention goes through xformers when it is installed, and
    # xformers has no bf16 kernel on sm_70 ("bf16 is only supported on A100+").
    # Route that call through the same scaled fp16 path, in xformers' layout.
    try:
        import xformers.ops as xops
        _orig_mea = xops.memory_efficient_attention

        def _softbf16_mea(q, k, v, *args, **kwargs):
            if os.environ.get("SOFTBF16_ATTN_LOG") == "1" and _ATTN_SEEN["x"] < 8:
                _ATTN_SEEN["x"] += 1
                _log("[attn-x] q=%s dt=%s" % (tuple(q.shape), q.dtype))
            if q.dtype is not torch.bfloat16:
                return _orig_mea(q, k, v, *args, **kwargs)
            if q.dim() == 3:  # VAE attention: no head axis
                m3 = kwargs.get("attn_bias")
                if m3 is not None and m3.dtype != torch.bool:
                    m3 = m3.half()
                return fast_attn.attention(q.unsqueeze(1), k.unsqueeze(1),
                                           v.unsqueeze(1), attn_mask=m3,
                                           scale=kwargs.get("scale")).squeeze(1)
            bias = kwargs.get("attn_bias")
            if bias is not None and bias.dtype != torch.bool:
                kwargs["attn_bias"] = None
                mask = bias.half()
            else:
                mask = bias
            out = fast_attn.attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                      v.permute(0, 2, 1, 3), attn_mask=mask,
                                      scale=kwargs.get("scale"))
            return out.permute(0, 2, 1, 3)

        xops.memory_efficient_attention = _softbf16_mea
        _log("patched xformers.ops.memory_efficient_attention for bf16")
    except Exception as e:  # xformers absent is fine
        _log("xformers patch skipped: %s" % e)

    _log("patched F.linear / F.conv* / F.scaled_dot_product_attention (attn mode=%s)"
         % bf16_attn.get_mode())
