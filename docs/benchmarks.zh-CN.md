# softbf16 在 V100 上的实测报告

> 2026-11-24 · Tesla V100-PCIE-32GB（sm70，TCC）· HP Z820 · Windows 11 · ComfyUI 0.43.0 · torch 2.7.0.dev20250307+cu128
> 目的：判断 https://github.com/S-i-l-v-e-t/Soft-BF16-V100 对现有 V100 出图/视频流程是否有实际价值

## 一、结论

1. **70 TFLOPS 属实**：8192³ bf16 GEMM 实测 **70.3 TFLOPS**，是 cuBLAS fp16（72.6）的 97%，是 V100 原生 bf16 路径（10.0）的 **7.0 倍**。
2. **它是真 bf16，不是降级 fp16**：与 fp32 参考的误差和 torch bf16 输出逐位相同（max 0.49977 / rel 1.658e-03）。
3. **端到端有用，但没到 "bf16 = fp16 速度"**：z-image turbo（bf16 权重，1024²，8 步）
   - fp16 计算（现状默认）：**10.4 s**
   - 原生 bf16：**44.5 s**（5.55 s/it）
   - bf16 + softbf16：**15.6 s**（1.72 s/it，含 VAE 解码）
   → 比原生 bf16 快 **2.9 倍**，只比 fp16 慢 **约 1.4 倍**，且数值是 bf16。
4. **意外发现：ComfyUI 0.43 在 V100 上根本不会用 bf16**。日志 `model weight dtype torch.float16`——即使加 `--bf16-unet`，权重也会被 manual_cast 降成 fp16。所以现在所有 "bf16 模型" 实际都在偷偷跑 fp16，这正是 ACE-Step 削波、H3 黑帧这类问题的来源。
5. **强行跑 bf16 会直接崩**：xformers 报 `No operator found ... bf16 is only supported on A100+`。softbf16 必须同时接管 attention（我们补了 xformers patch）才能跑通。

## 二、GEMM 微基准（8192³ / 8192×4096×4096 / 4096×3072×3072）

| 形状 | torch fp32 | torch bf16 | torch fp16 | softbf16 bf16 | softbf16 + fp8 权重 |
|---|---|---|---|---|---|
| 8192³ | 13.3 | 10.0 | 72.6 | **70.3** | 69.9 |
| 8192×4096×4096 | 13.5 | 10.0 | 67.6 | 64.6 | 65.1 |
| 4096×3072×3072 | 12.7 | 9.9 | 69.1 | 55.4 | 55.1 |

单位 TFLOPS；V100-PCIE fp16 张量核峰值为 112 TFLOPS。
小矩阵收益会降（55 TFLOPS），因为打包开销占比上升；`m*k*n < 2e9` 会直接走 fp32 兜底。

## 三、精度（1024×2048×1024，对 fp32 参考）

| | max\|err\| | 相对误差 |
|---|---|---|
| torch bf16 | 0.49977 | 1.658e-03 |
| softbf16 bf16 | 0.49977 | 1.658e-03 |
| torch fp16 | 0.06258 | 2.073e-04 |
| softbf16 + fp8 权重 | 1.13596 | 2.344e-03 |

softbf16 的精度就是 bf16 的精度（比 fp16 差一档），换来的是动态范围——bf16 的 8 位指数正好装得下 fp16 装不下的激活值。

## 四、Attention（bf16，B=1 H=8 S=2048 D=64）

| 路径 | 耗时 |
|---|---|
| F.sdpa bf16（math 兜底） | 2.216 ms |
| bf16_attn 缩放 fp16 | **1.017 ms** |
| F.sdpa fp16（理想值） | 0.529 ms |

## 五、端到端 A/B（z-image turbo，1024²，8 步，res_multistep，cfg 1）

| Arm | 采样 | 墙钟 | 说明 |
|---|---|---|---|
| A. fp16（`--fp16-unet`，现状） | 1.16 s/it | 10.4–11.9 s | 权重被静默转 fp16 |
| B. 原生 bf16（放宽 dtype 门限） | 5.55 s/it | 44.5–45.2 s | 不补 attention 会直接报错 |
| C. bf16 + softbf16 | 1.72 s/it | 14.6–15.6 s | 全流程跑通，出图正常 |
| C′. 重建扩展 + fast_attn 后复测 | 1.64 s/it | **14.94 s** | 同一轮里 fp16 为 **10.52 s**（1.42x） |

> C′ 是同一次重建、同一台机器上背靠背测的：bf16+softbf16 14.94 s / fp16 10.52 s。比修复前的 ~15.6 s 少 0.6 s，与 attention 微基准预测的 0.55 s 吻合；相对原生 bf16（~45 s）仍是 3.0 倍。

样例：[bf16+softbf16 出图](sample_bf16_softbf16.png) · [fp16 出图](sample_fp16.png)

**为什么 C 仍比 A 慢 35%**：GEMM 已不是瓶颈（70 TFLOPS），剩下的是
- 每次 linear 调用都重新打包权重（bf16 → 缩放 fp16），没有缓存——同一形状下 softbf16 比 cuBLAS fp16 慢约 25%（1.40 ms vs 1.12 ms @4096×3072×3072）；
- attention 每层做 aminmax + 一次 host 同步，bf16 路径比 fp16 慢约 2 倍；
- 逐元素/Norm 不是原因：实测 bf16 与 fp16 的 SiLU/LayerNorm 带宽完全相同（约 750 GB/s，见表），和 tensor core 无关。

## 五之二、端到端 A/B（MiniMax H3，640×352 / 56 帧 / 4 步 turbo）

| Arm | 冷启（含 34GB 模型加载） | 热跑 | 采样 | 备注 |
|---|---|---|---|---|
| A. fp16 + `minimax_h3_fp16_fix.py`（现状） | 200.9 s | 45.2 s | ≈8.4 s/it | 可连续跑 |
| B. bf16 + softbf16（去掉 `--fp16-unet`） | 146–151 s | — | 8.37 s/it（首跑）/ 6.77 s/it（重载后） | **第二条任务 OOM** |

- 出片正常：bf16 视频 1.1 MB，与 fp16 版同量级，无黑帧。
- 采样速度两者基本持平，bf16 甚至略快——因为它省掉了 fp16 补丁里"残差流走 fp32"的开销。
- 但 **bf16 臂显存峰值 30.53 / 31.86 GB，只剩 43 MB**，第二条连续任务直接在 `SamplerCustomAdvanced` OOM；fp16 臂同样位置能连跑。`expandable_segments:True` 也没救回来。
- 原因：H3 的 unet 是 `int8_convrot`，反量化成 bf16 之后 softbf16 还要再打包一份 fp16 副本（A 与 W），额外 2–3 GB 峰值——而 640×352 本来就已经贴着 31 GB 上限。

**结论：H3 不值得为 softbf16 改（限定条件：我们这台 32GB V100 + `int8_convrot` 权重 + fp16 计算）。** 速度没有净赚，显存余量却被吃光；真要用得把 `MiniMaxLowVRAMAttention.head_chunks` 从 5 调到 10 或降一档分辨率，得不偿失。

> 范围说明：这个结论不是"H3 这个模型不可救"。H3 这条链路的 5 个热点——int8 unet（反量化后是 fp16 GEMM）、LowVRAMAttention（fp16 SDPA）、qwen3vl **nvfp4** 文本编码器（软件反量化，softbf16 不支持 fp4）、fp16 视频 VAE、fp32 音频 VAE——**没有一个是 bf16 GEMM 回退**，所以结构上就没有它可以加速的位置，只剩打包带来的显存成本。
> 反过来，如果 H3 是**官方 bf16 权重**，它的线性层会全部落在原生 bf16 慢路径上，softbf16 就有 3–4 倍的空间；但 20B 的 bf16 权重约 40 GB，单张 32GB V100 装不下（必须分片/offload），得先解决显存。真正可行的中间态是 **H3 fp8 (e4m3) + softbf16 的 W8A16**：1 byte/参数、权重不反量化到 fp16，这正是作者 fork 里 `quant_ops.py` 打的补丁；本地没有 fp8 版 H3，所以这条没能实测。


## 五之三、物理上限：这张卡其实被功耗墙压着

用 8192³ fp16 GEMM 持续压测，同时采样时钟：

| 状态 | SM 时钟 | 功耗 | 实测 |
|---|---|---|---|
| 空闲 | 1230 MHz | 34 W | — |
| 持续 GEMM | **1072–1087 MHz** | **203 W（上限 200 W）** | cuBLAS **79.6–79.9 TFLOPS** |

- 规格表的 112 TFLOPS 是 1380 MHz 下的值；实际满载只有 ~1080 MHz，**当频峰值约 87.7 TFLOPS**。
- cuBLAS 79.9 / 87.7 = **91%**——已经很满。
- 之前短时测得 72.6 TFLOPS，是因为还没跑热、频率没掉下来；这也解释了测量抖动。
- 所以"V100 没有 `cp.async` 导致只能到 65%"不成立：65% 的分母用错了。缺 `cp.async`/`ldmatrix`/`mbarrier` 影响的是"接近峰值的难度"，不是把上限砍到 65%。

## 五之四、采样时间都花在哪（fp16 基线，8 步，torch.profiler）

单次冷启动采样 CUDA 总计 13.79 s：

| 类别 | 时间 | 占比 |
|---|---|---|
| 纯 GEMM（cutlass/volta s884gemm） | 4.81 s | 34.9% |
| **Memcpy HtoD（pageable）** | **4.64 s** | **33.7%** |
| attention（fmha_cutlassF sm70） | 2.23 s | 16.2% |
| elementwise / copy / reduce | 2.10 s | 15.3% |

- 那 4.64 s 的 H2D 是**冷启动时 11.5 GB 权重的分页上传**，只在首跑出现——这就是"第一次 41 s、后面 11 s"的原因，不是每步都有。
- 热跑的 1.16 s/it（约 9.3 s / 8 步）大致是：GEMM ≈55%、attention ≈20%、elementwise/copy ≈20%。
- 注意 profiler 本身有 40% 左右开销，比例只看量级。

## 五之五、边界条件（回应"TensorCore 不是许愿机"）

成立的部分：
- Tensor core 只吃 matmul。SiLU/LayerNorm/softmax 是纯带宽活，实测 bf16 与 fp16 完全同速（约 750 GB/s），换任何代数都一样，A100/H100 也不进 TC。
- 所以"靠 TC 把 V100 端到端提速 100%"是不可能的；Amdahl 上限定死了。

不准确的部分：
- **"GEMM/GEMV 的 ASIC"**：TC 是可编程的 mma 协处理器，不是 ASIC；而 **GEMV 恰恰是 TC 最没用武之地的形状**（batch=1 是带宽瓶颈），两者不能并称。
- **"没 async memcpy 所以只能 65%"**：见五之三，真实瓶颈是 200 W 功耗墙，当频峰值只有 87.7 TFLOPS，cuBLAS 已经 91%。
- **"Ampere TC 也搞不定 SiLU/Norm"**：结论对，但因果说反了——不是 TC 不行，是这件事本来就不归 TC 管。

对 softbf16 的影响：不改变定位（bf16 权重的 GEMM 解药），但确实说明它不是"V100 加速器"；而且提醒我们，这张卡所有"利用率"都应该按 ~1080 MHz / 87.7 TFLOPS 来算。



## 五之六、Attention 还有没有空间（V100 30 头 / S=4096 / D=128）

z-image 真实形状是 xformers 布局 `(B=1, S=4096, H=30, D=128)`（外加 4192 的 text+image 拼接版）。同形状 fp16 下：

| 后端 | 耗时 | 算力 |
|---|---|---|
| sdpa efficient（cutlass 32x128） | 7.69 ms | 33.5 TFLOPS |
| xformers mem_efficient | 7.78 ms | 33.1 TFLOPS |
| sdpa math | 11.80 ms | 21.8 TFLOPS |
| sdpa efficient + 2/4/8 段 query 切分 | 7.73 / 7.74 / 8.00 ms | 33.3 / 33.3 / 32.2 |

- 三个可用后端基本同速（sdpa 与 xformers 走的是同一族 cutlass kernel），**切分 query 没有收益**，说明不是 occupancy 问题。
- 从 S=2048 到 8192，算力一直平在 32–33 TFLOPS（≈ 满频峰值的 38%），是**饱和**而不是"分块选错"。
- 分块本身没有暴露的调参入口：PyTorch 不提供，xformers 也只在内部按 D/dtype 选配置，最终落到同一个 `fmha_cutlassF_f16_aligned_32x128_rf_sm70`。所以"32x128 未必最优"在原理上可能成立，但**没有可切换的路径**；要更快只能自己写 sm70 的 FA kernel（无 `cp.async`/`ldmatrix`，D=128 寄存器压力大），乐观 1.2–1.4 倍，而 attention 只占 ~20%，端到端最多 4–7%。

真正有空间的是 **bf16 路径的缩放开销**（同形状 B=1/H=30/S=4096/D=128）：

| 路径 | 耗时 | 相对 fp16 |
|---|---|---|
| fp16 sdpa efficient | 7.71 ms | 1.00x |
| bf16 sdpa 默认（math 兜底） | 41.45 ms | 5.4x |
| bf16_attn `fp16_scaled`（默认） | 10.08 ms | 1.31x |
| bf16_attn `fp16`（不缩放） | 7.89 ms | 1.02x |
| bf16_attn `fp32` | 25.60 ms | 3.3x |

- 缩放版比"直接转 fp16"多 2.2 ms/次（29%），成本来自每次调用的 3 次 aminmax + 一次 host 同步（`.tolist()`）。
- 按 30 层 × 8 步算，这部分约 0.5 s/张（z-image bf16 全流程 15.6 s 里）。换成 `mode=fp16` 能省掉，但会失去溢出保护（H3 那种宽动态范围就不行）。

**已修**：[`shim/fast_attn.py`](softbf16_spike/shim/fast_attn.py) 保留 3 次 aminmax 做安全检查，但当 q/k/v 都落在 fp16 安全带（2⁻¹⁰…2¹⁰）时**跳过所有缩放**：三次 cast 进、一次 cast 出，不再产生 fp32 中间量与除法。

| 路径 | 耗时 | 相对默认 |
|---|---|---|
| bf16_attn 默认（fp16_scaled） | 10.39 ms | 1.00x |
| bf16_attn `mode=fp16`（不检查） | 8.20 ms | 0.79x |
| **fast_attn（检查 + 免缩放）** | **7.99 ms** | **0.77x** |

与默认路径的差异：`max|err| 0.00000`、相对 4.3e-10（2 的幂缩放本来就是精确的，安全带内跳过不改变结果）。端到端实测（重建扩展后背靠背）：**bf16+softbf16 14.94 s vs fp16 10.52 s**，比修复前的 ~15.6 s 少 0.6 s，和微基准预测的 0.55 s 一致。`SOFTBF16_ATTN_FAST=1` 可进一步跳过检查（等同 `mode=fp16`，只在确定模型动态范围安全时用）。

## 五之七、fp8 权重路径仿真：上限就是"比 fp16 慢 11%"

用 z-image 的真实线性层形状（从 safetensors 头读出：6.15B 参数在 2D 权重里，97.8% 集中在 4 种形状），M=4096（对应 1024² 的 token 数），逐层实测再按层数汇总：

| 路径 | 每步 GEMM | 等效算力 | 相对 fp16 | 端到端估算 |
|---|---|---|---|---|
| fp16（基准） | 615.8 ms | 81.7 TFLOPS | 1.00x | 1.14 s/step |
| fp8 → 反量化到 **fp16** 再 GEMM | 771.2 ms | 65.2 | 1.25x | 1.29 s/step |
| fp8 → 反量化到 **bf16** 再 GEMM | 5281 ms | 9.5 | 8.6x | **5.87 s/step** |
| softbf16 W8A16（bf16 激活，fp8 权重） | 813 ms | 61.9 | 1.32x | 1.40 s/step |
| softbf16 bf16（bf16 权重） | 823 ms | 61.1 | 1.34x | 1.41 s/step |

（fp16 基准反推 1.14 s/step，与实测 1.16 对得上，说明这个形状模型可信。）

结论：

1. **V100 没有 fp8 张量核，所以 fp8 的速度上限就是"反量化 + fp16 GEMM"，也就是比纯 fp16 慢 11%**。没有任何配置能让 fp8 比 fp16 快；它唯一的收益是**显存减半**（z-image 11.5 GB → 5.7 GB）。
2. **关键坑在反量化的目标 dtype**：落到 fp16 只慢 11%，落到 bf16 直接慢 8.6 倍（单步 5.87 s）。这正解释了你们报告里"fp8 在 V100 上 20 s/步"的现象——不是 fp8 本身贵，是 compute dtype 被选成了 bf16。
3. **softbf16 的 W8A16 在速度上不如"反量化到 fp16"**（1.40 vs 1.29 s/step），因为它必须用 bf16 激活，GEMM 只能跑 62 TFLOPS，省下的反量化不够补 GEMM 的损失。它只在"必须保留 bf16 动态范围"时才有意义。
4. 因此：**fp8 模型要锁死 fp16 计算**（`--fp16-unet`，并且不要让 `should_use_bf16` 把它拉回 bf16）。我们的 shim 会放宽 bf16 门限——**不要对 fp8 模型启用它**，否则正好踩中 5.87 s/step 那条路。

## 五之八、移植 fork 的三个补丁 + z-image fp8 实测

把 ComfyUI-V100 的 `ops.py`（`softbf16_ops`）/ `quant_ops.py`（fp8 W8A16 handler）/ `sd.py`（显式 dtype 优先）改写成可回滚的 monkeypatch：[`shim/fork_ops.py`](softbf16_spike/shim/fork_ops.py)，`SOFTBF16_FORK=1` 启用，不碰 `comfy/` 原文件。验证：启动日志出现 `ops.py: softbf16_ops registered` / `quant_ops.py: fp8 W8A16 handler registered`，且出图正常。

同一 prompt、1024²、8 步，权重换成 `z-image-turbo-fp8-e4m3fn.safetensors`（5.73 GB）：

| Arm | 配置 | 冷启 | 热跑 |
|---|---|---|---|
| A | fp8 权重 + **fp16** 计算（现状） | 54.4 s | **11.07 s** |
| B | fp8 权重 + bf16 计算 + fork 的 W8A16 | 43.8 s | **14.83 s**（1.34x） |
| C | fp8 权重 + bf16 计算，不打补丁 | — | **跑不起来**（xformers 无 bf16 kernel） |

结论：

1. **移植本身成功**，W8A16 路径出图正常，但**比"fp8 权重 + fp16 计算"慢 34%**——它必须用 bf16 激活，GEMM 只有 62 TFLOPS，省下的反量化补不回来。
2. **fp8 的真实收益是显存，不是速度**：A 的 11.07 s 与 bf16 权重版（11.1 s）完全持平，说明这台机器上 ComfyUI 的逐层反量化几乎免费（5.7 GB→11.5 GB 的转换被缓存/流水吸收）。这比五之七的仿真更乐观——仿真给的是 +11% 的**上界**，真实环境里没有付出。
3. C 再次确认：没有 softbf16 时，bf16 计算在这套栈上**直接不可用**（不是慢，是报错）。

所以 fp8 模型的正确姿势是：**锁定 fp16 计算**（`--fp16-unet`），拿到一半显存、速度不变；fork 的 W8A16 只在"必须保留 bf16 动态范围"时才值得，别为了速度去用它。

## 五之九、三项落地：fp8 转换、显存驻留、H3 head_chunks

### 1. fp8 转换（HunyuanVideo 1.5）

脚本 [`ab/fp8_convert.py`](softbf16_spike/ab/fp8_convert.py)：只把 ≥65536 元素的浮点张量转成 `float8_e4m3fn`，不写 scale 键——与 ComfyUI `UNetSave(weight_dtype=fp8_e4m3fn)` 的格式一致，V100 上按 fp16 计算。

- 15.51 GB → **7.76 GB**，转换耗时 16.2 s，逐张量相对误差 ~2.7%（e4m3 的正常水平，|w|max < 0.6，远未饱和）
- A/B（640×352 / 33 帧 / 6 步 / cfg 6，同一 seed）：

| 顺序 | arm | 冷跑 | 热跑 | 文件 |
|---|---|---|---|---|
| 第一次 | fp16 | 111.7 s | — | 15.51 GB |
| | fp8 | — | 71.9 s | 7.76 GB |
| 反序 | fp8 | 92.1 s | — | 7.76 GB |
| | fp16 | — | 69.7 s | 15.51 GB |

**两种顺序都是"后跑的那个快"**，说明差异来自缓存/加载，不是 dtype。真实结论：**fp8 与 fp16 速度持平（热跑约 70 s），收益是显存减半**——留给更高分辨率/更长帧数的余量。样例帧：[sample_hv15_fp8_frame.png](softbf16_spike/ab/sample_hv15_fp8_frame.png)。

### 2. 管住显存驻留

ComfyUI 的 `/free` 是**排队语义**：请求只设标志，下一次 prompt 开始时才真正卸载（所以立刻查 VRAM 看不到变化）。工具 [`ab/free_vram.py`](softbf16_spike/ab/free_vram.py)：

```powershell
D:\ComfyUI\python\python.exe .\softbf16_spike\ab\free_vram.py 8188          # 排到下一次任务
D:\ComfyUI\python\python.exe .\softbf16_spike\ab\free_vram.py 8188 --now    # 立刻生效
```

实测可把 25.6 GB 驻留清到 0.42 GB。用法：切换模型族或批处理前调用；**同一模型连跑时不要调**（会触发 80–130 s 重载）。

### 3. H3 head_chunks 调优

640×352 / 56 帧 / 4 步，同一 seed：

| head_chunks | 墙钟 | 显存峰值 |
|---|---|---|
| 5（原值） | 179.5 s（含首次加载） | 30.64 GB |
| 8 | 44.2 s | 31.36 GB |
| **10** | **40.7 s** | 31.36 GB |
| 12 | 40.7 s | 31.36 GB |

**建议改成 10**：比 8 快约 8%，显存峰值不变；12 无进一步收益。改法：工作流里 `MiniMaxLowVRAMAttention.head_chunks = 10`。

## 五之十、训练侧：唯一的数量级收益（已实测）

用 z-image 的真实层形状（97.8% 的 2D 参数，M=4096）跑 **fwd+bwd**，三种路径：

| 路径 | 每步 170 个 GEMM | 相对 fp16 |
|---|---|---|
| fp16 | 2131 ms | 1.00x |
| 原生 bf16 | 14835 ms | **6.96x** |
| **bf16 + softbf16（带 backward）** | **2784 ms** | **1.31x** |

按你们 MOSS LoRA 的实测基准（bf16 19.8 s/step）折算：

```
19.8 s/step × (2784 / 14835) = 3.7 s/step
```

也就是 **19.8 → 约 3.7 s/step，5.3 倍**，同时保留 bf16 的动态范围（fp16 2.5 s/step 的数值受限问题不再需要回避）。这是这套内核在 V100 上唯一能拿到数量级收益的场景——推理侧它永远追不上 fp16。

落地要点：训练框架的 `nn.Linear` 调用的是 `F.linear`，用 shim 里那套 `F.linear` 补丁即可；优化器状态保持 fp32。

## 五之十一、收尾四项

1. **Flux dev fp8**：`flux1-dev-fp16.safetensors`（22.17 GB，BF16）→ `flux1-dev-fp8_plain.safetensors`（**11.09 GB**），转换 44 s，逐张量相对误差 3–5%。注意它是纯 cast（无 scale），画质略低于已有的 16 GB `fp8_scaled` 版，只在显存吃紧时用。

2. **训练 shim**：[`shim/train_shim.py`](softbf16_spike/shim/train_shim.py)，`import train_shim; train_shim.enable()` 即可把 bf16 `F.linear` 接到内核。用两个 z-image 形状的 block 跑真实训练循环验证：

| | ms/step | loss（首→末） |
|---|---|---|
| 原生 bf16 | 209.4 | 0.02234 → 0.01370 |
| bf16 + shim | **72.6** | 0.02234 → 0.01370 |

**2.89 倍**，loss 差 3.8e-8（同轨迹）。比微基准的 5.3 倍低，因为小模型里优化器/SiLU/切片的占比更高；真实 MOSS LoRA 的 GEMM 占比更高，会更接近 5 倍。

3. **权重打包预打包的收益**（实测上限，未集成）：把权重在加载时打包一次、推理时只打包激活：

| 形状 | 现路径 | 预打包 | 收益 |
|---|---|---|---|
| 4096×3840×10240 | 4.89 ms | 4.34 ms | **11.3%** |
| 4096×3840×11520 | 5.24 ms | 4.85 ms | 7.4% |
| 4096×10240×3840 | 4.90 ms | 4.42 ms | 9.9% |

折算到 bf16 推理全流程约 −5%（14.94 → ~14.2 s）。**未实现**：需要在模型加载时把 bf16 权重替换成"打包好的 fp16 + 每列 scale"（显存等价），会侵入加载流程，风险大于收益，先记录方案。

4. **`run_comfy.py --free`**：跑完自动释放显存（已验证），批处理切换模型族时可直接加这个参数。

## 六、接入方式（本次做法，未改 comfy 源码）

- `shim/comfy_shim.py`：编译/加载扩展 → 放宽 `should_use_bf16`（否则 ComfyUI 永远选 fp16）→ 补 `F.linear`、`F.conv*`（转 fp32）、`F.scaled_dot_product_attention`（bf16 走缩放 fp16）→ 补 `xformers.ops.memory_efficient_attention`（含 3D VAE 情况）。
- `shim/fork_ops.py`：ComfyUI-V100 三个补丁的 monkeypatch 版，`SOFTBF16_FORK=1` 启用。
- `shim/run_patched.py`：启动包装器，`SOFTBF16_OPS=0` 只放宽 dtype 不补算子（参考组）。
- `ab/start_armB.bat`：在 `vcvars64` 环境里启动（Windows 必须，torch JIT 需要 cl.exe + ninja）。
- CUDA 头文件：从 PyPI 解压 `nvidia-cublas/cusparse/cusolver-cu12` 的 win_amd64 轮子到 `cudalibs/`（只解压不安装）；用 `dumpbin /exports` + `lib /def` 从 torch 自带 `cublas64_12.dll` 生成 `cublas.lib`。

## 七、风险与限制

- **AGPL-3.0**，商用/闭源分发注意。
- 项目非常新（2026-11-16 首提交），kernel 内有手调阈值（repair 上限按尺寸线性插值），建议先隔离验证。
- Windows 不在作者验证范围；需要完整 CUDA 开发头 + MSVC + ninja，且必须在 vcvars 环境启动。
- 只覆盖 linear / conv / attention；其他 bf16 算子（LayerNorm、softmax、逐元素）仍走 PyTorch 慢路径。
- fp8 权重需要 per-tensor scale 乘在输出上（作者的 ComfyUI-V100 在 `quant_ops.py` 里这么做），不能直接把带 scale 的 fp8 权重丢进去。
- 小矩阵（<2e9 FLOPs）走 fp32 兜底，VAE/小投影层不会变快。

## 八、操作层面的顺带发现

测试前那台 ComfyUI 服务器已连续运行一天、显存里驻留 16.8 GB 模型；同样跑 fp16 z-image 要 **24.3 s**。重启（清空驻留）后只要 **10.4 s**。**跑大活前重启 ComfyUI 能白捡 2 倍**，比换内核便宜得多。

## 九、建议

1. 需要 bf16 数值、且权重本身是 bf16 的模型（z-image 这类）：**值得接**，2.9 倍起步；代价是比 fp16 慢约 1.35 倍。
2. **H3 不推荐**：速度持平、显存爆掉（见五之二）。
3. 已经能用 fp16 跑好的模型（Wan fp16、Qwen fp8）：**不必接**。
4. 真正可能吃到大头的是 **bf16 LoRA 训练**（MOSS 记录里 bf16 19.8 s/step vs fp16 2.5 s/step）：这个库带 backward，理论上能把 8 倍差距压到 1.5 倍内，但要用 ai-toolkit 单独验证。
5. 若要继续压推理开销：给权重打包加缓存（同一权重重复调用可复用打包结果），预计再省 10–20%。

## 十、环境状态

- 测试用 ComfyUI 实例已全部关闭；8188 端口已按原命令行恢复：`python main.py --listen --preview-method auto --disable-cuda-malloc --fp16-unet`。
- 未向 ComfyUI 的 python 环境安装任何包（只 `pip download` 后解压到工作区）。
- 测试产物：`D:\ComfyUI\output\abtest\`（12 张 z-image 1024²）、`D:\ComfyUI\output\video\ab_h3_*.mp4`（5 条 H3 640×352），可随时删除。
