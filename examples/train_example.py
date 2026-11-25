"""Minimal training example: bf16 matmuls through the softbf16 kernel.

Run under a vcvars64 shell (the extension needs MSVC + ninja):

    python train_example.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shim"))
import train_shim  # noqa: E402

train_shim.enable()

model = nn.Sequential(
    nn.Linear(3840, 10240, bias=False),
    nn.SiLU(),
    nn.Linear(10240, 3840, bias=False),
).cuda().to(torch.bfloat16)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(1024, 3840, device="cuda", dtype=torch.bfloat16) * 0.1
target = torch.randn(1024, 3840, device="cuda", dtype=torch.bfloat16) * 0.1

for step in range(5):
    loss = (model(x).float() - target.float()).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("step %d  loss %.5f" % (step, loss.item()))
