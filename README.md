# softbf16-v100-toolkit

让 **Tesla V100（sm_70）** 在 Windows + ComfyUI 上跑 **bf16 / fp8** 权重：把 bf16 的 GEMM 映射到 fp16 张量核（逐行/逐列 2 的幂缩放，fp32 累加），而不是退回 fp32。

基于 [Silvet / Soft-BF16-V100](https://github.com/S-i-l-v-e-t/Soft-BF16-V100) 的内核（AGPL-3.0），加上一层 ComfyUI 适配、训练适配和几个运维工具。

---

## 它是什么 / 不是什么

**是**：让 V100 上"必须用 bf16"的模型（fp16 会溢出/削波/出黑帧）以接近 fp16 的速度运行。
**不是**：V100 加速器。V100 没有 bf16/fp8/fp4 张量核，任何"模拟"都不可能超过 fp16 的上限。

实测（这台机器，250 W 功耗墙）：

| 场景 | 结果 |
|---|---|
| 8192³ bf16 GEMM | **70.3 TFLOPS**（原生 bf16 10.0，cuBLAS fp16 82） |
| bf16 精度 | 与 torch bf16 逐位一致（max\|err\| 完全相同） |
| z-image bf16 端到端 | fp16 10.5 s → bf16+本工具 **14.9 s**（1.42x），原生 bf16 跑不起来 |
| 训练 fwd+bwd（真实层形状） | 原生 bf16 6.97x fp16 → **本工具 1.31x** |
| 真实训练循环验证 | 209 → 73 ms/step（**2.89x**），loss 同轨迹 |
| fp8 权重 | 与 fp16 速度持平，显存减半 |

**结论**：推理侧比 fp16 慢 35~42%，只在"fp16 不可用"时用；**训练侧是唯一能拿数量级收益的地方**（MOSS LoRA 19.8 → 约 3.7 s/step）。

---

## 已验证环境

| 项 | 值 |
|---|---|
| GPU | Tesla V100-PCIE-32GB（sm_70，TCC） |
| OS | Windows 11 |
| Python / torch | 3.12 / torch 2.7.0.dev+cu128（CUDA 12.8） |
| ComfyUI | 0.43.0 |
| MSVC | Visual Studio Build Tools 2022（C++ 工作负载） |
| 驱动 | 582.78 |

CUDA 工具链只需要 `nvcc`；cuBLAS/cuSPARSE/cuSOLVER 的头文件由安装脚本从 PyPI 的 win_amd64 轮子里取，**不需要装完整 CUDA Toolkit**。

---

## 安装

```powershell
cd softbf16-v100-toolkit
.\install.ps1                        # 默认 Python=D:\ComfyUI\python\python.exe
```

可选参数：

| 参数 | 说明 |
|---|---|
| `-Python <path>` | 指定解释器 |
| `-Comfy <path>` | ComfyUI 目录（只用于提示，不改它） |
| `-CudaDir <dir>` | 复用已有的 `cudalibs`（离线安装） |
| `-NoBuild` | 只铺文件，不编译 |

安装做四件事：取 CUDA 头 → 用 `dumpbin`+`lib` 生成 `cublas.lib` → 编译内核到 `ext_cache` → 自检（8192³ GEMM 精度 + 吞吐，低于 40 TFLOPS 视为失败）。

首次编译约 **1–2 分钟**，之后走缓存。

---

## 用法 1：ComfyUI 跑 bf16 权重

```powershell
$env:SOFTBF16_COMFY = "D:\ComfyUI"
D:\ComfyUI\python\python.exe <toolkit>\shim\run_patched.py `
    --listen --preview-method auto --disable-cuda-malloc --bf16-unet
```

- 启动日志应出现 `[softbf16-shim] extension ready`、`should_use_bf16 -> True`、`patched ...`；
- **必须带 `--bf16-unet`**：ComfyUI 0.43 在 V100 上默认把 bf16 权重静默转 fp16（日志 `model weight dtype torch.float16`），那样内核根本不会被调用；
- 实例仍监听 8188，和原版参数一样，随时换回。

**重要：只对 bf16 权重用。** fp8 模型要锁 `--fp16-unet` 且**不要**开本工具——否则 dtype 被拉到 bf16，单步会慢 8.6 倍。

回滚：换回原来的启动脚本即可，本工具没有改动 ComfyUI 的任何文件。

---

## 用法 2：训练（收益最大的场景）

```python
import sys
sys.path.insert(0, r"<toolkit>\shim")
import train_shim
train_shim.enable()      # 之后所有 bf16 F.linear 走缩放 fp16 张量核
```

保持优化器状态为 fp32；只改 matmul，不改数值语义（输出与真 bf16 一致）。

实测：两个 z-image 形状 block 的真实训练循环 209.4 → **72.6 ms/step**（2.89x），loss 差 3.8e-8。

---

## 用法 3：fp8 转换（省显存，不提速）

```powershell
python <toolkit>\tools\fp8_convert.py <src.safetensors> <dst.safetensors>
```

纯 cast（无 scale 键），与 ComfyUI `UNetSave(fp8_e4m3fn)` 格式一致；V100 上按 fp16 计算，**速度不变、显存减半**（HunyuanVideo 15.5 → 7.8 GB 已验证）。

---

## 运维工具

| 工具 | 用途 |
|---|---|
| `tools/free_vram.py <port> [--now]` | 释放 ComfyUI 驻留显存（`/free` 是排队语义，`--now` 立刻生效） |
| `tools/selftest.py` | 检查内核 + 吞吐 |

**显存驻留值得管**：同一任务在驻留 16.8 GB 的实例上要 24.3 s，重启后 10.4 s。切换模型族前跑一次 `free_vram.py --now`。

---

## 目录结构

```
softbf16-v100-toolkit/
  install.ps1 / uninstall.ps1
  shim/
    comfy_shim.py     # 加载内核 + 补 F.linear / F.conv* / SDPA / xformers
    fast_attn.py      # bf16 attention：保留范围检查，安全带内跳过缩放（-23%）
    fork_ops.py       # ComfyUI-V100 三个补丁的 monkeypatch 版（SOFTBF16_FORK=1）
    run_patched.py    # ComfyUI 启动包装器
    train_shim.py     # 训练用 drop-in
  tools/
    build_ext.py / selftest.py / make_cublas_def.py
    fp8_convert.py / free_vram.py
  vendor/softbf16/    # 上游内核源码（AGPL-3.0，未修改）
  cudalibs/ ext_cache/  # 安装时生成
```

环境变量：`SOFTBF16_COMFY`（ComfyUI 目录）、`SOFTBF16_OPS=0`（只放宽 dtype 不补算子，参考组）、`SOFTBF16_FORK=1`（用 fork 的 ops 集成）、`SOFTBF16_ATTN_FAST=1`（跳过 attention 范围检查）、`SOFTBF16_CAST=fp16|bf16|fp32`。

---

## 常见问题

**Q：`where cl` 失败 / `Ninja is required`？**
必须在 `vcvars64` 环境里启动。`install.ps1` 和自带的启动方式都会处理；手动跑 Python 时请先 `call vcvars64.bat`。

**Q：加了工具反而更慢？**
这是预期——它只让 bf16 追上 fp16 的一部分。如果你的模型本来能跑 fp16，不要用。

**Q：H3 能用吗？**
本机 H3 是 `int8_convrot` + nvfp4 文本编码器，5 个热点里没有一个是 bf16 GEMM，实测速度持平且多吃 2–3 GB 显存（会 OOM）。官方 bf16 权重约 40 GB，单卡装不下。

**Q：为什么 8192³ 只有 70 TFLOPS，规格写 112？**
这张卡被 200 W 功耗墙压到 1080 MHz；`nvidia-smi -pl 250` 后 cuBLAS 79.9→93.0 TFLOPS（+16%），是当前最便宜的提速。

**Q：许可？**
内核（`vendor/softbf16/`）是 **AGPL-3.0**，本工具的 shim 与它链接，按 AGPL-3.0 分发。商用/闭源分发前请自行评估。

---

## 致谢

- [Soft-BF16-V100](https://github.com/S-i-l-v-e-t/Soft-BF16-V100)（内核）
- [ComfyUI-V100](https://github.com/S-i-l-v-e-t/ComfyUI-V100)（集成思路，`fork_ops.py` 是它的 monkeypatch 版）
- [raylight-V100](https://github.com/S-i-l-v-e-t/raylight-V100)（多卡，未包含）
