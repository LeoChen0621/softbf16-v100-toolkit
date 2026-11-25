"""Cast a safetensors model to plain fp8_e4m3fn (no scale keys).

Matches the format ComfyUI produces with UNETLoader(weight_dtype=fp8_e4m3fn) +
UNetSave, which we verified runs on V100 at fp16 speed with half the VRAM.
Only float tensors with >= 65536 elements are cast; norms/embeddings stay as-is.

    python fp8_convert.py <src.safetensors> <dst.safetensors>
"""
import sys
import time

import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]

t0 = time.time()
sd = load_file(src, device="cpu")
print("loaded %d tensors in %.1fs (%.1f GB)" % (
    len(sd), time.time() - t0, sum(v.numel() * v.element_size() for v in sd.values()) / 1024 ** 3),
    flush=True)

out = {}
ncast = 0
sample = []
for k, v in sd.items():
    if v.is_floating_point() and v.numel() >= 65536:
        q = v.to(torch.float8_e4m3fn)
        out[k] = q
        ncast += 1
        if len(sample) < 3:
            deq = q.to(torch.float32)
            ref = v.to(torch.float32)
            sample.append((k, (deq - ref).abs().max().item(),
                           ((deq - ref).norm() / ref.norm()).item(),
                           ref.abs().max().item()))
    else:
        out[k] = v

print("cast %d tensors, kept %d" % (ncast, len(out) - ncast), flush=True)
for k, mx, rel, amax in sample:
    print("  %-58s max|err| %.5f  rel %.2e  (|w|max %.4f)" % (k[:58], mx, rel, amax), flush=True)

t0 = time.time()
save_file(out, dst, metadata={"format": "pt"})
print("saved %s in %.1fs" % (dst, time.time() - t0), flush=True)
