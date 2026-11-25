"""Port of ComfyUI-V100 (branch v100) ops.py / quant_ops.py / sd.py patches.

Applied as monkeypatches so comfy/ stays untouched; set SOFTBF16_FORK=1.
  * ops.py        -> softbf16_ops (bf16/fp8 weights), selected by pick_operations
  * quant_ops.py  -> fp8 QuantizedTensor linear fed to the kernel without dequant
  * sd.py         -> explicit cast dtype wins (env SOFTBF16_CAST=fp16|bf16)
"""
import logging
import os

import torch

import comfy.ops as ops
import comfy.quant_ops as qo
from comfy.ops import CastBiasWeightContext

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _log(msg):
    print("[fork-ops] %s" % msg, flush=True)


def apply(bf16_online):
    class softbf16_ops(ops.manual_cast):
        class Linear(ops.manual_cast.Linear):
            def forward_comfy_cast_weights(self, input):
                w = self.weight
                if (isinstance(w, torch.Tensor) and w.dtype in _FP8
                        and input.dtype == torch.bfloat16
                        and not isinstance(w, qo.QuantizedTensor)
                        and len(self.weight_function) == 0 and len(self.bias_function) == 0):
                    # W8A16: keep the 1-byte weight, pack it to scaled fp16 in the kernel
                    with CastBiasWeightContext(self, input, dtype=w.dtype,
                                               bias_dtype=input.dtype,
                                               offloadable=True) as (weight, bias):
                        return bf16_online.linear(input, weight, bias)
                with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                    return bf16_online.linear(input, weight, bias)

        class Conv1d(ops.manual_cast.Conv1d):
            def forward_comfy_cast_weights(self, input):
                with CastBiasWeightContext(self, input, dtype=torch.float32,
                                           bias_dtype=torch.float32, offloadable=True) as (weight, bias):
                    return self._conv_forward(input.float(), weight, bias).to(input.dtype)

        class Conv2d(ops.manual_cast.Conv2d):
            def forward_comfy_cast_weights(self, input):
                with CastBiasWeightContext(self, input, dtype=torch.float32,
                                           bias_dtype=torch.float32, offloadable=True) as (weight, bias):
                    return self._conv_forward(input.float(), weight, bias).to(input.dtype)

        class Conv3d(ops.manual_cast.Conv3d):
            def forward_comfy_cast_weights(self, input, autopad=None):
                with CastBiasWeightContext(self, input, dtype=torch.float32,
                                           bias_dtype=torch.float32, offloadable=True) as (weight, bias):
                    return self._conv_forward(input.float(), weight, bias, autopad=autopad).to(input.dtype)

    ops.softbf16_ops = softbf16_ops

    _orig_pick = ops.pick_operations

    def pick_operations(weight_dtype, compute_dtype, *args, **kwargs):
        wd = getattr(weight_dtype, "dtype", weight_dtype)
        if (compute_dtype in (None, torch.bfloat16)
                and wd in (torch.bfloat16,) + _FP8):
            logging.info("Using softbf16 ops")
            return softbf16_ops
        return _orig_pick(weight_dtype, compute_dtype, *args, **kwargs)

    ops.pick_operations = pick_operations
    _log("ops.py: softbf16_ops registered")

    # ---- quant_ops.py: fp8 QuantizedTensor without per-call dequant ----
    try:
        from comfy_kitchen.tensor.fp8 import _handle_fp8_linear

        def _softbf16_fp8_linear(qt, args, kwargs):
            input_tensor, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else None
            scale = getattr(weight._params, "scale", None)
            if (isinstance(weight, qo.QuantizedTensor)
                    and not isinstance(input_tensor, qo.QuantizedTensor)
                    and input_tensor.dtype in (torch.bfloat16, torch.float32)
                    and not getattr(weight._params, "transposed", False)
                    and weight._qdata.dtype == torch.float8_e4m3fn
                    and scale is not None and scale.numel() == 1):
                out = bf16_online.linear(input_tensor, weight._qdata).float() * scale.float()
                if bias is not None:
                    out = out + bias.float()
                return out.to(input_tensor.dtype)
            return _handle_fp8_linear(qt, args, kwargs)

        qo.register_layout_op(torch.ops.aten.linear.default,
                              qo.TensorCoreFP8E4M3Layout)(_softbf16_fp8_linear)
        _log("quant_ops.py: fp8 W8A16 handler registered")
    except Exception as e:
        _log("quant_ops patch skipped: %s" % e)

    # ---- sd.py: an explicit cast dtype wins over the model's declared dtypes ----
    forced = os.environ.get("SOFTBF16_CAST")
    if forced in ("fp16", "bf16", "fp32"):
        import comfy.model_management as mm
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[forced]
        orig = mm.unet_manual_cast

        def unet_manual_cast(weight_dtype, device, supported_dtypes=None):
            if supported_dtypes is None or dt in supported_dtypes:
                return dt
            return orig(weight_dtype, device, supported_dtypes)

        mm.unet_manual_cast = unet_manual_cast
        _log("sd.py: manual cast pinned to %s" % forced)
