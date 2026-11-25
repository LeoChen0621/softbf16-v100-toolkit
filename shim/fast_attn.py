"""Cheaper bf16 attention on sm70 than bf16_attn's default 'fp16_scaled'.

bf16_attn always pays: 3x aminmax -> host sync -> 3 scaling multiplies ->
fp16 cast -> fp32 output -> divide -> bf16 cast. For DiT activations the
magnitudes sit far inside fp16's range, so all that scaling is dead work.

This wrapper keeps the range check (3 aminmax, which is the safety net) and,
when the tensors are already in a safe band, skips the scaling entirely:
three casts in, one cast out.

Set SOFTBF16_ATTN_FAST=1 to skip even the check (bit-identical to
bf16_attn's 'fp16' mode).
"""
import math
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import bf16_attn

# Inside this band fp16 holds bf16 values with headroom to spare.
_SAFE_LO = 2.0 ** -10
_SAFE_HI = 2.0 ** 10
_ORIG = F.scaled_dot_product_attention


def _fast(q, k, v, *args, **kwargs):
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = _ORIG(q.half(), k.half(), v.half(), *args, **kwargs)
    return out.to(torch.bfloat16)


def attention(query, key, value, *args, **kwargs):
    if query.dtype is not torch.bfloat16:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return _ORIG(query, key, value, *args, **kwargs)
    if os.environ.get("SOFTBF16_ATTN_FAST") == "1":
        return _fast(query, key, value, *args, **kwargs)
    mx = torch.stack([torch.aminmax(t)[1] for t in (query, key, value)]).tolist()
    if all(math.isfinite(m) and _SAFE_LO <= m <= _SAFE_HI for m in mx):
        return _fast(query, key, value, *args, **kwargs)
    return bf16_attn.attention(query, key, value, *args, **kwargs)


def install():
    F.scaled_dot_product_attention = attention
