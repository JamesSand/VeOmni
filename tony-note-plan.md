# 实现计划：在 VeOmni 中实现量化感知蒸馏（QAD）

> 本文是 `tony-note.md` 的配套文档。它将 Tony 提出的 QAD 四要素
>（自蒸馏、伪量化、STE、选择性冻结）映射到 VeOmni 代码库中的具体改动。
> 所有 file:line 引用均基于 `zhizhou-dev-qad` worktree 的 `eb5e9c4` 提交。

## 0. TL;DR — 改动的整体形态

QAD 是**全新功能（greenfield）**：全仓库 grep `quant/fake_quant/STE/int4/int8/fp8/bitsandbytes/gptq/awq`
未发现*任何*权重量化基础设施（仅有的匹配是 VQGAN codebook 和 dtype 字节表）。
所以我们是新增代码，不是重构。

**部署目标固定为 W4A4 NVFP4（TRT-LLM 推理路径）**。训练支持三种模式：
`w4`（只量化 weight）/ `w4a4`（weight + activation）/ `a4`（只量化 activation），
三者最终都导出成同一个 w4a4 NVFP4 格式 —— scale 策略与各模式的差异见 §1.5。

四个要素能干净地映射到现有的扩展点上：

| 要素（tony-note 章节） | 在 VeOmni 中的落点 |
|---|---|
| 伪量化 + STE（§4、§5） | 新建 `veomni/quantize/` 包：`FakeQuantLinear` 包装层 + STE `autograd.Function`。风格上参考 `veomni/ops/` 的可插拔 backend 模式。 |
| 自蒸馏损失（§3） | 新建 `TextQADTrainer`，克隆 **DPO trainer** 的结构 —— teacher 在 `no_grad` 下前向，student 在 fwd-context 下前向，合并成一个 loss 再 `backward()`。 |
| 选择性冻结（§6） | 扩展 `_freeze_model_module()`（`veomni/trainer/base.py:418`）—— 这个现成的冻结钩子恰好运行在 parallelize + optimizer *之前*。`build_optimizer` 本身已经按 `requires_grad` 过滤参数。 |
| 配置入口 | 新建 `QADConfig` dataclass 挂到 `TrainingArguments` 上，仿照 `OptimizerConfig`/`OpsImplementationConfig`。自动映射到 YAML 的 `train.qad.*`。 |
| 训练入口 | `tasks/train_text_qad.py` —— 8 行代码，克隆 `tasks/train_text_dpo.py`。 |

**DPO trainer（`veomni/trainer/text_dpo_trainer.py`）是整个功能最好的模板**：
它已经解决了"第二个参考模型、冻结、FSDP 同构切分、no_grad 下运行、
合并成 loss、只对 policy 反向传播"这一整套问题。

---

## 1. 伪量化 + STE — `veomni/quantize/`（新包）

### 1.1 文件结构

```
veomni/quantize/
├── __init__.py          # 公开 API：wrap_linears_for_qad()、QADQuantConfig
├── config.py            # QAD 量化配置 dataclass（训练模式 + NVFP4 固定参数，见 §4）
├── fake_quant.py        # NVFP4 quantize/dequantize 数学（两级 scale，见 §1.3）
├── calibrate.py         # activation global-scale 校准（EMA absmax + 跨 rank 规约，见 §1.5）
├── ste.py               # 带 clamp 掩码的直通估计器 autograd.Function
└── modules.py           # FakeQuantLinear(nn.Linear) 包装层
```

### 1.2 `ste.py` — 直通估计器（tony-note §5）

一个 `torch.autograd.Function`：

- **forward：** 计算 `W_q = quantize(W, scale, zp, format)`，返回 `dequantize(W_q, ...)`。
- **backward：** 梯度直通（`∂Q/∂W ≈ 1`），**对被 clamp 到可表示范围之外的元素置零**
  （§5 "Handling the clamp region"）。forward 时把"在范围内"的布尔掩码存入 `ctx`。

这是梯度接触量化器的*唯一*位置；其余一切都是普通浮点计算。

### 1.3 `fake_quant.py` — NVFP4 quantize/dequantize 数学

部署目标是 **NVFP4**（TRT-LLM 的 w4a4 路径），格式本身规定死了 scale 结构，
这里没有自由设计项。必须**与部署推理 kernel 位级一致（bit-exact）**
（tony-note §8 首要陷阱）。

NVFP4 = FP4 E2M1（可表示值 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}，max=6）+
**两级 scale**，weight 和 activation 共用同一套结构：

```
第一级：per-block scale   —— 沿收缩维（K）每 16 个元素一组，存 FP8 E4M3
第二级：per-tensor global —— 整个 tensor 一个 FP32 标量
```

量化公式（modelopt / TRT-LLM 约定）：

```
s_global     = amax(|X|) / (448 × 6)                  # 448 = E4M3 max，FP32
s_block_e4m3 = cast_e4m3(block_amax / 6 / s_global)   # 注意：scale 本身被量化！
X_fp4        = round_e2m1(X / (s_block_e4m3 × s_global))，clamp 到 ±6
Q(X)         = X_fp4 × s_block_e4m3 × s_global
```

两个容易漏掉的 bit-exact 细节：
- **block scale 本身要过 E4M3 量化**（double quantization）——
  伪量化里不模拟这一步，就和推理 kernel 对不上。
- **舍入模式**（`round_e2m1` / `cast_e4m3`）必须与部署 kernel 一致。

**参考实现 / 单测基准：** 训练环境里已有 NVIDIA Model-Optimizer（modelopt），
其 NVFP4 fake-quant（`NVFP4_DEFAULT_CFG` 那套 quantizer）实现的正是上述语义，
且 TRT-LLM 消费的就是 modelopt 导出的 checkpoint。`fake_quant.py` 的单测直接与
modelopt 的 quantizer **逐值对拍**；也应评估训练路径直接调用 modelopt 的
quant 函数而非自行重写（见 §7 阶段 1）。

### 1.4 `modules.py` — `FakeQuantLinear(nn.Linear)`

包装线性层，使 `forward(x)` 计算 `x @ STE(quantize(self.weight))ᵀ + bias`
（`w4a4`/`a4` 模式下先对 `x` 做 activation 伪量化，见 §1.5）。
`self.weight` 保留为**高精度潜权重（latent weight）**（可训练的累加器，
§5 "Why this works"）。weight 的两级 scale 每次 forward 现算、不落盘；
activation 的 global scale（`w4a4`/`a4` 模式）是 buffer 而非 parameter
（NVFP4 无 zero-point）。

**关键时序约束（meta 初始化）。** 默认 FSDP2 路径下 `init_device="meta"` ⇒
`empty_init=True`（`veomni/models/auto.py:239-242`），因此
**`build_foundation_model` 返回时真实权重并不存在** —— 权重要到
`build_parallelize_model`（`base.py:503`）阶段才通过 broadcast/DTensor 加载。所以：

- **模块替换**（nn.Linear → FakeQuantLinear）在 **meta 树**上、parallelize 之前完成 ——
  安全，因为这只是结构性操作。
- **scale 与真实权重的时序**：weight 的两级 scale 每次 forward 从当前权重现算
  （§1.5），不存在"包装时初始化 scale"的问题；activation 的 global scale 校准
  （`w4a4`/`a4` 模式）必须放在 parallelize **之后**（权重已加载）、训练开始
  **之前**，作为独立的校准阶段运行。在 meta 路径上**不要**在包装时读取权重。

**FSDP2 交互（constraints §5、§16）。** `parallelize_model_fsdp2` 对每个 transformer
block 调用 `fully_shard()`，所以 `FakeQuantLinear.weight` 在运行时是切分后的 DTensor。
量化器必须作用于 *FSDP 在 forward 中呈现的权重*（做 matmul 时已经 gather 完整）。
保持参数名 `weight` 不变还能保证 **DCP checkpoint key 稳定**（constraint §16 ——
重命名参数会导致 checkpoint 无法加载）。用 `pytest tests/checkpoints/` 验证。

### 1.5 三种训练模式下的 scale 策略

支持三种训练模式（`train.qad.mode`），最终都部署成 w4a4 NVFP4：

- `"w4"` —— 只对 weight 做伪量化训练；
- `"w4a4"` —— weight + activation 都做伪量化训练；
- `"a4"` —— 只对 activation 做伪量化训练。

**Weight scale（`w4`/`w4a4`）：两级都每次 forward 从当前潜权重重算，且 detach。**

- `s_global^W = amax(W)/(448×6)`；`s_block^W` 每 16 个 in-features 元素一组，
  算完立刻过 E4M3。
- 为什么动态重算而不是校准后冻结：(a) 我们的导出路径是对最终潜权重做 PTQ
  （amax 从最终权重算）—— 训练每步同规则重算 ⇒ 训练/导出自洽；(b) 潜权重
  在训练中漂移，scale 必须跟着走，否则 clamp 掩码会越掩越多（tony-note §8
  的失败模式）。注意（源码实证）：modelopt 自己的 QAT 流程走的是另一条自洽
  路线 —— weight global amax 在 `mtq.quantize` 校准时冻结、QAT-export 时直接
  读这个冻结值（`per-block` 才从最终权重重算）。两条路线各自自洽，不能混用：
  我们的 checkpoint 必须配我们的导出脚本（或对最终权重做全新 PTQ 校准）。
- 为什么 detach：scale 是 amax 的函数，让梯度流过 max 会把单个元素的梯度
  耦合到整个 block，破坏 STE 稳定性。标准 QAT 实践是把 scale 当 forward 常量。
- FSDP2 交互：block 沿 in-features 切，FSDP 通常 Shard(0)（out-features），
  每个 shard 内 16 元素块完整，per-block scale 可纯本地计算。

**Activation scale（`w4a4`/`a4`）：混合式 —— NVFP4 与普通 INT8 的关键差别。**
TRT-LLM 的 NVFP4 activation kernel 的工作方式决定了训练怎么做：

- **per-block E4M3 scale：动态。** kernel 运行时对每个 token 的每 16 个元素
  现算。训练时 forward 里照做，无需校准。
- **per-tensor global scale（FP32）：静态，必须校准。** kernel 需要提前拿到
  这个值（它决定 E4M3 block scale 的编码范围）。校准阶段（`calibrate.py`）只
  收集每个量化点的全局 amax（EMA absmax 跑 `calib_steps` 个 batch）后冻结；
  DP/SP 下统计量必须跨 rank all-reduce max，保证所有 rank 用同一个 grid
  （constraints §5–§8 "no shard sees a different grid"）。

**三模式汇总：**

| | `w4`：只训 W | `w4a4`：W+A 都训 | `a4`：只训 A |
|---|---|---|---|
| 训练中 weight | NVFP4 伪量化，两级 scale 每步重算 | 同左 | 全精度，不量化 |
| 训练中 activation | 不量化 | block 动态现算；global 校准后冻结 | 同左 |
| activation 校准时机 | 训练**后**（导出前），最终权重 + weight fake-quant 打开 | 训练**前**，`W₀` + weight fake-quant 打开 | 训练**前**，全精度权重下；训练后用 weight-PTQ 后的模型**重校准** |
| 导出时 weight | 从最终潜权重重算 scale（与训练同规则，自动一致） | 同左 | **PTQ**：从全精度权重一次性算（训练没见过此误差） |
| 可训练参数 | 量化线性层潜权重 | 同左 | 被包装层的权重（全精度训练，无 weight STE） |
| 与部署的关系 | 少模拟 activation 误差 | **唯一 bit-exact 模式** | 少模拟 weight 误差（≈ A-QAD + W-PTQ） |

> 重要：三种模式最终都部署成 w4a4，所以 `w4` 和 `a4` 的训练 loss 只模拟了部署
> 误差的一部分 —— 它们是消融/课程式训练模式，**最终评测必须统一在完整 w4a4
> 仿真（或真实 TRT-LLM 路径）下进行**，不能只比各自训练模式下的 loss。

各模式的坑：

- `w4`：导出前的 activation 校准要在**最终**权重上做 —— 训练把权重挪过了，
  `W₀` 上校准的 amax 不再准确。
- `w4a4`：顺序 = 先开 weight fake-quant、在 `W₀` 上校准 activation global →
  冻结 → 开训。训练中 weight 两级 scale 每步动、activation global 不动、
  activation block 每步现算。
- `a4`：训练时 activation 来自全精度权重产出的分布，部署时上游权重是 4-bit，
  分布会偏移 → 导出前必须用 weight-PTQ 后的模型重校准 activation global。
  它承担了 weight 的全部 PTQ 误差，预期是三者中恢复效果最差的。

**activation 伪量化插入点：** 与 serving kernel 一致 —— 在被量化 GEMM 的输入处
（`FakeQuantLinear.forward` 内先对 `x` 做 activation fake-quant，再做 weight
fake-quant 的 matmul）。STE（含 clamp 掩码）同时挂在两个量化节点上。

---

## 2. Trainer — `veomni/trainer/text_qad_trainer.py`（新建）

克隆 `text_dpo_trainer.py`。它已经使用组合模式
（`self.base = BaseTrainer.__new__(BaseTrainer)`）并手动调用各 builder。

### 2.1 Teacher 模型 —— 推荐方案 A（独立冻结 teacher）

**方案 A（默认，DPO 直接照搬）：独立的冻结 teacher。** 把 `_build_reference_model()`
（`text_dpo_trainer.py:133-170`）原样复制为 `_build_teacher_model()`：
用**相同的 weights_path** 再 `build_foundation_model` 一份，
`requires_grad_(False)`，FSDP 同构切分，`MixedPrecisionConfig(enable=False)`，
关闭 grad checkpointing，`.eval()`。代价是显存里多一份冻结权重 ——
但没有梯度、没有 optimizer state（DPO 的 reference model 已证明这个成本可接受；
必要时可叠加 offload）。这是唯一忠实于 tony-note §2 目标
`Distill(z_S, z_T)`（`z_T` 来自**冻结的 `W₀`**）的实现。

> 注意：不要用"单模型 + 量化器开关"的共享权重方式实现 teacher ——
> teacher 会随 student 的潜权重一起漂移，目标退化为自洽性
> `KL(f(W) || f(Q(W)))`，且存在 `W` 落到量化网格上 loss 归零、
> 模型质量却不受任何锚定的退化解。teacher 必须来自冻结的 `W₀`。

**方案 B（折中）：离线预计算 teacher logits。** teacher 是冻结的，
所以 `z_T` 对每条训练数据是常量 —— 可以在训练前用推理批处理算好
top-k logits 存盘，训练时当作数据集字段读入。零 teacher 显存、零 teacher
训练时算力，代价是一次离线推理 + 存储 + top-k 截断误差。数据集固定、
多次实验复用同一 teacher 时特别划算。可作为方案 A 之后的后续优化。

推荐 A 作为默认；B 留作后续的显存/算力优化。

### 2.2 重写 `forward_backward_step`（核心，tony-note §7）

严格镜像 `text_dpo_trainer.py:273-313`：

```
micro_batch = self.base.preforward(micro_batch)
with torch.no_grad():                       # teacher：全精度、无梯度（§3）
    z_T = self.teacher_model(micro_batch)    # 方案 A：冻结的 W₀ 副本
with self.base.model_fwd_context, set_batch_invariant_mode(...):
    z_S = student_forward(micro_batch)       # 开启量化器；backward 走 STE
loss, loss_dict = qad_distill_loss(z_S, z_T, tau, alpha)   # §3 KL（+可选 CE）
with self.base.model_bwd_context, set_batch_invariant_mode(...):
    loss.backward()                          # 梯度经 STE 到达潜权重 W
return loss, loss_dict
```

`train_step` / `train()` 直接从 DPO 复制（`:333-407`）不改 ——
`optimizer.step()` 本来就只更新 `requires_grad` 的参数（见 §3）。

### 2.3 蒸馏损失（tony-note §3）

`L = τ² · KL(softmax(z_T/τ) || softmax(z_S/τ))`，可选 `+ (1-α)·CE(z_S, y)`。

**实现路径：复用仓库现成的分块 top-k 蒸馏 kernel，全程不物化 `[B,L,V]`。**
`veomni/ops/kernels/cross_entropy/chunk_topk_distill.py` 已经提供
`chunk_topk_distill_function`：分块融合 lm_head 投影 + teacher top-k 前向 KL，
带闭式 backward（梯度流向 hidden 和 lm_head 权重）、`IGNORE_INDEX` 位置输出恒 0
（问题 5 的掩码由 kernel 内部完成）、`temperature` 参数（即 τ）、SP 感知，
且**已接进模型 forward**：student 前向传
`return_log_probs=True + teacher_topk_ids + teacher_topk_log_probs` kwargs，
从 `outputs.fused_linear_aux.distillation_losses` 拿逐位置 KL
（有完整测试 `tests/ops/test_chunk_topk_distill.py`，与 verl 语义对齐）。

所以 loss 是 **top-k 前向 KL**（teacher 的 top-K 词表子集 + mass 指标），
不是全词表 KL —— K 可配（默认 ~128），这是标准的蒸馏近似，且换来两侧
显存都只有 `chunk × V` / `L × K` 的峰值。QAD 侧只需新写一个小工具：
**teacher top-k 提取**（no_grad 下分块 `hidden @ lm_head.T → log_softmax(τ) →
topk(K)`，~20 行，无需 backward）。`student_mass`/`teacher_mass` 顺带成为
wandb 指标（top-k 覆盖率，检验 K 是否够大）。

---

## 3. 选择性冻结（tony-note §6）

扩展 `_freeze_model_module()`（`veomni/trainer/base.py:418-421`）——
它的执行时机正好：**在 `_build_model`（`:293`）之后、parallelize（`:305`）和
optimizer（`:307`）之前**，此处设置的 flag 会被下游全部尊重。

QAD trainer 冻结钩子内的步骤：
1. `wrap_linears_for_qad(self.model, qad_config)` —— 把目标线性层（attention
   Q/K/V/O + MLP 矩阵，tony-note §4 "Where fake quant goes"）替换为 `FakeQuantLinear`。
2. `self.model.requires_grad_(False)`，然后**仅**重新打开 `FakeQuantLinear`
   潜权重的梯度。其余一切（embedding、norm、bias、未量化的 LM head）保持冻结 ——
   tony-note §6 "Typical split"。`a4` 模式下同样只打开被包装层的权重 ——
   它们全精度训练、无 weight STE，靠调整权重来补偿激活量化误差（§1.5）。
3. `pretty_print_trainable_parameters(self.model)`（已导入，`base.py:420`）核对结果。

optimizer 无需改动：`build_optimizer`（`veomni/optim/optimizer.py:298-314`）
构建 param group 时按 `p.requires_grad` 过滤，Muon / ExtraParallel-FSDP2
路径同理（`:230,506-557`）。冻结会自动省掉被冻结参数的 optimizer state
（tony-note §6 "Efficiency"）。

---

## 4. 配置入口 — `QADConfig` dataclass

在 `veomni/arguments/arguments_types.py` 中新增 `@dataclass QADConfig`，挂到
`TrainingArguments`（`:436-698`）上：`qad: QADConfig = field(default_factory=QADConfig)`，
与 `optimizer: OptimizerConfig`（`:554`）完全同构。从 `arguments/__init__.py` 重新导出。
递归解析器（`parser.py:64-115`）会自动暴露 YAML 的 `train.qad.*` 和 CLI 的
`--train.qad.*` —— **解析器零改动**。

字段（NVFP4 目标下大幅简化 —— 格式参数是定死的，不暴露自由度）：
- `enable: bool`
- `mode: Literal["w4","w4a4","a4"]`（§1.5 的三种训练模式，默认 `"w4a4"`）
- `calib_steps: int`（activation global scale 校准的 batch 数，`w4a4`/`a4` 用）
- `target_modules: list[str]`（要包装哪些线性层）
- `tau: float`、`alpha: float`（蒸馏温度 / 硬标签混合系数）
- `teacher_mode: Literal["separate","offline"]`（§2.1 的方案 A / B，默认 `"separate"`）

NVFP4 的格式参数（E2M1、block_size=16、block scale E4M3、global scale FP32）
是**常量，不进配置** —— 暴露成配置项只会制造与部署格式不一致的机会。

如果 trainer 需要带类型的根参数类，再加一个 `VeOmniQADArguments(VeOmniArguments)`
（DPO 在 `text_dpo_trainer.py:77-81` 就是这样加 `dpo_config` 的）——
或者直接读 `args.train.qad` 也行。

---

## 5. 训练入口 + 配置示例

- `tasks/train_text_qad.py` —— 克隆 `tasks/train_text_dpo.py` 的 8 行：
  `parse_args(VeOmniQADArguments)` → `TextQADTrainer(args)` → `trainer.train()`。
- `configs/text/<model>_qad.yaml` —— 克隆一份 text 配置，把 `model.model_path`
  指向全精度 checkpoint（teacher 和 student 都从它初始化，tony-note §3），
  加一个 `train.qad` 配置块，并使用**小学习率 + warmup**（tony-note §7 "Learning rate"）。

---

## 6. Checkpoint 与导出（tony-note §9）

- **训练 checkpoint：** DCP 保存**潜权重** + 量化 buffer（activation global
  scale 等）—— 保持参数名稳定以使 DCP key 匹配（constraint §16）。训练期间不做导出。
- **导出（独立的最终步骤）：** 写一个 `scripts/qad/export_quantized.py`，把潜权重
  物化为 TRT-LLM 可消费的 NVFP4 checkpoint（FP4 打包权重 + E4M3 block scales +
  FP32 global scales），优先直接复用 modelopt 的导出管线。按 §1.5 的模式差异：
  `w4` 导出前先在最终权重上做 activation 校准；`a4` 先做 weight PTQ 再重校准
  activation。不在训练 PR 范围内，作为后续跟进。现有的 DeepSeek-V3
  `scripts/deepseek_v3/fp8_cast_bf16.py` 可作为独立转换脚本的先例。

---

## 7. 分阶段 / PR 划分

小而独立可评审的 PR（遵循 CLAUDE.md 的 commit flow，一个 PR 一个关注点）：

1. **`veomni/quantize/` 核心** —— NVFP4 伪量化数学（两级 scale + block scale 的
   E4M3 双重量化）+ STE + `FakeQuantLinear`，单测断言 (a) 与 modelopt 的 NVFP4
   quantizer **逐值对拍**的位级一致性、(b) STE 的 clamp 掩码行为。此阶段同时评估
   "直接调 modelopt 的 quant 函数 vs 自行实现"。不接 trainer。
   跑 `pytest tests/` + 新增 `tests/quantize/`。
2. **`QADConfig` + arguments 接线** —— dataclass、重新导出、一个解析测试。
3. **`TextQADTrainer` + 冻结 + 训练入口** —— 克隆 DPO 的 trainer、冻结钩子、
   `tasks/train_text_qad.py`、先接通 `w4` 模式；蒸馏 loss 走现成的
   `chunk_topk_distill_function`（§2.3），新写 teacher top-k 提取工具。在
   `tests/toy_config/` 上做 E2E 冒烟（约束：trainer 改动跑 `tests/e2e/`）。
   同时补上 **evaluation**（VeOmni 的 `EvaluateCallback._evaluate` 是空壳，
   `evaluate_callback.py:37-39` 只有 `pass`）：从 `data.eval_path` 建 eval
   dataloader，`model.eval()` + `no_grad` 循环，all-reduce 后 log `eval/*`：
   - `eval/student_ppl` —— 量化 student 的 PPL（student forward 本来就开着
     fake quant，天然是 w4a4 仿真评测）；PPL 只算 `labels != IGNORE_INDEX`
     的 response 位置，与训练掩码一致；
   - `eval/kl_to_teacher` —— eval 集上对 teacher 的 KL（蒸馏泛化）；
   - `eval/latent_fp_ppl`（低频可选）—— 关掉量化器测潜权重全精度 PPL，
     监控潜权重偏离 `W₀` 的程度；
   - `eval/teacher_ppl` —— 静态基线，训练开始时测一次；
   - naive-PTQ 参照线 —— 训练前对 `W₀` 直接 PTQ 测一次 PPL，作为 step-0
     记录（tony-note §9 三点对比的下界）。

   eval 的数据准备也是本阶段交付物：`scripts/qad/make_eval_split.py` ——
   指定 held-out shard、从中固定抽取 ~2000 条（固定 seed）、写出带版本号的
   eval 子集文件 + 训练侧排除该 shard 的文件列表（§9 问题 6 的切分与
   噪声预算依据）。
4. **activation 路径** —— block 动态 scale + global scale 校准阶段
   （`calibrate.py`，含跨 rank 规约）+ 接通 `w4a4` / `a4` 模式（§1.5）。
5. **（后续）导出脚本** —— `scripts/qad/`，含各模式的 activation 校准要求（§1.5）。

（原计划的"分块 KL kernel"阶段已取消：`chunk_topk_distill_function` 现成可用，
见 §2.3；teacher top-k 提取并入阶段 3。）

## 8. 风险 / 需要验证的点（来自 tony-note §8 + VeOmni constraints）

- **伪量化位级一致**（§8）：第一正确性关卡 —— 与 modelopt 的 NVFP4 quantizer
  逐值对拍，特别是 block scale 的 E4M3 双重量化和两处舍入模式（§1.3）。
- **消融模式的评测口径**（§1.5）：`w4` / `a4` 的训练 loss 只反映部分部署误差，
  三种模式的最终对比必须统一在完整 w4a4 仿真（或真实 TRT-LLM 路径）下做。
- **meta 初始化下的时序**（§1.4）：weight scale 每步现算无初始化问题；activation
  global scale 校准必须在 parallelize 之后、训练之前，不能在包装时读权重。
- **训练精度 vs 模拟格式**（§8）：累加计算跑在 bf16/fp32，同时数值遵循低比特网格；
  别让框架自身的降精度悄悄扰动模拟格式。与 `set_batch_invariant_mode` 和
  `MixedPrecisionConfig` 有交互。
- **分布式一致性**（§8 + constraints §5–§8）：teacher 和 student 必须切分一致
  （方案 A 复用 DPO 的做法：从 policy 的 parallel plan 复制配置后同构切分）；
  scale/分组的计算方式必须与 FSDP/EP 切分一致。
- **需盯防的失败模式**（§8）：loss 不跟随 teacher（student 上量化器没生效 /
  teacher 上误开了量化器）、梯度消失（用了真实取整梯度而非 STE，或 scale 太紧导致
  clamp 掩码掩掉几乎所有元素）、scale 漂移（学习率太高）。
- 提交前过 **Ruff / 注释仅英文 / PR 标题格式**（constraints §18–§20）；
  按 CLAUDE.md commit flow 跑 `/veomni-review`。

---

## 9. 落地目标：Qwen3-8B + reasonmix-sftmix-combined（8×H100）

首个实际训练运行的模型/数据/机器，以及针对性排查出的问题。

- **模型**：`model-and-data/models/Qwen3-8B`（`model_type: qwen3`，VeOmni 原生支持，
  `configs/text/qwen3.yaml` 可作模板；hidden 4096 / 36 层 / vocab **151936**）。
- **数据**：`model-and-data/datasets/reasonmix-sftmix-combined-Qwen3-8B` ——
  **Qwen3-8B 自己用 vLLM 生成的回复**（temperature 0.6，max_tokens 8192），对 QAD
  自蒸馏是理想的 on-policy 数据。用 `tokenized_default-Qwen3-8B` 子集
  （275 个 parquet，300 万条，`input_ids`/`labels`/`attention_mask`，labels 已按
  `IGNORE_INDEX=-100` 掩掉 prompt span）。raw 子集有 670 万条（tokenized 是滞后快照）。
- **机器**：8×H100 80GB。粗算 `teacher_mode=separate` 可行：student fp32
  参数+梯度+Adam ≈ 16GB/卡（FSDP2 8 卡分片），teacher bf16 冻结 ≈ 2GB/卡，
  开 gradient checkpointing 后显存大头是 logits（见问题 3）。

### 排查出的问题

1. **缺 `pretokenized` 数据类型（必须新增）。** `data_type` Literal
   （`arguments_types.py:1156`）只有 plaintext/conversation/classification/dpo，
   全部从文本重新 tokenize，无法直接消费预 tokenized parquet。新增
   `@DATA_TRANSFORM_REGISTRY.register("pretokenized")` transform（~30 行：
   list→tensor、dtype 转换 —— labels int64 / input_ids int32 / attention_mask int8 ——
   超长截断）+ 扩展 Literal。**不要**走 raw 子集 + `conversation` 重 tokenize 的路：
   VeOmni 的 `chatml` 模板与生成时的 Qwen3 官方模板（含 `<think>` 处理）不保证
   逐 token 一致，会引入模板漂移。
2. **`max_seq_len` 开到 8192。** 生成上限 8192、平均回复 ~3400 token、~8% 顶格；
   qwen3.yaml 默认 2048 会截断大部分 reasoning trace。
3. **logits 显存问题已由现成 kernel 解决。** vocab 151936 × seq 8192 ⇒ 单条
   序列 fp32 logits ≈ 5GB —— 但 §2.3 的 `chunk_topk_distill_function` 路径两侧
   都不物化 `[L,V]`（student 分块投影、teacher 只留 top-K），seq 8192 直接可跑，
   无需过渡方案。
4. **H100 跑不了真实 NVFP4 kernel 评测。** QAD 训练无碍（伪量化是 bf16/fp32 仿真，
   modelopt 的 NVFP4 fake-quant 与逐值对拍在 Hopper 上可跑），但 TRT-LLM 的真实
   w4a4 NVFP4 推理需要 **Blackwell** —— §1.5 的"真实 TRT-LLM 路径评测"要另找
   B200 节点，本机只能做伪量化仿真评测。
5. **KL 的位置掩码（设计决策）。** 数据是 prompt+response 打包的，蒸馏 KL 默认只算
   `labels != IGNORE_INDEX` 的 response 位置（与 SFT 掩码一致），在 loss 实现中
   显式处理；packed 边界遵循 constraint §10/§12。
6. **eval 数据：按 shard 级从训练数据切 held-out。** 从 275 个 tokenized parquet
   中拿出 **1 个 shard**（实测每 shard 10,924 条，~3800 万 token，占数据 0.36%；
   每条 = 一个 prompt + Qwen3-8B 的完整回复）作为 `data.eval_path`，训练集用
   其余 274 个。训练中评测不用整个 shard —— 取该 shard 内**固定的 ~2000 条子集**
   （~700 万 token）以控制每次 eval 的耗时。要求：(a) shard 级切分（行级抽样会
   把同一次生成 run 的近邻样本泄漏进两侧）；(b) eval 子集**固定且带版本**——
   全精度基线 / naive PTQ / QAD 三个对比点必须用同一份；(c) 后续离线评测
   （导出模型上跑 lm-eval 下游任务）作为 out-of-distribution 补充，
   in-distribution PPL 单独看有盲区。

   **为什么 2000 条够（噪声预算）：** 有限 eval 集上的 PPL 是带抽样噪声的估计，
   噪声随 token 数按 1/√N 收敛。每 token log-loss 的标准差 ~2 nat 量级 ⇒
   700 万 token 下均值标准误 ≈ 2/√(7×10⁶) ≈ 0.0008 nat，折到 PPL 约 **±0.1%**；
   而 w4a4 下 PTQ 退化与 QAD 追回量是**百分之几到几十**的量级 —— 尺子精度比
   被测效应细两个数量级，所以三点排序、gap 大小、训练中 PPL 趋势都可信。
   经典 90/10 切分在 300 万样本量级纯属浪费训练数据：eval 集大小由"PPL 要多稳"
   决定，不由比例决定。**能分辨的下限**：小于噪声量级的差异（如两个超参变体
   只差 0.05%）2000 条分不出来，需加大 eval 集或多次测量。另外 (b) 的固定
   eval 集实为**配对测量** —— 抽样噪声在模型间做差时大部分抵消，模型间*差异*
   的有效噪声比 ±0.1% 更小，这是"固定且带版本"直接提高对比分辨率的原因，
   不只是工程洁癖。

---

## 10. 部署：单节点 8×H100 训练 + k8s 多 pod 并行消融

**一个 QAD 实验 = 一个 8×H100 节点，不需要多机。** 冒烟实测（Qwen3-8B，w4，
seq 8192，teacher+student 双模型）：显存峰值 **26.7GB / 80GB**，余量巨大。
算力上 QAD 每步 ≈ teacher 前向 + student 前向反向 ≈ 普通 SFT 的 ~1.3 倍，
一个 epoch（~100 亿 token）8 卡约 2 天量级 —— 单节点完全可行。

**默认路径：直接在当前 dev pod（`zhizhousha-dev-8gpu-ssh`）跑：**

```
torchrun --nproc_per_node=8 tasks/train_text_qad.py configs/text/qwen3_qad.yaml
```

**k8s 多 pod 的正确用法：三个模式消融并行，一个 pod 一个实验。**
基建在 `~/workspace/low-precision-project/k8s-from-h100-pod/`（dev pod 内置
`kubectl` + kubeconfig 直连 research-common H100 集群 API，已验证连通）。
共享 home PVC（`home-zhizhousha`）让每个新 pod 内的 worktree、venv、模型、
数据集路径与 dev pod 完全一致，零同步。参照 `manifests/example-1gpu-pod.yaml`
改成 8 GPU 单 pod manifest，开 2–3 个 pod 分别跑 `w4` / `w4a4` / `a4`
（改 `--train.qad.mode` 和 `--train.checkpoint.output_dir` 即可），
消融周转时间从串行 ~6 天缩到 ~2 天。

（目录里的 `qad-train-2node.yaml` 双节点 manifest 是把单个实验拆到 16 卡的
加速选项 —— 显存和正确性上都不需要，仅在赶单个实验的墙钟时间时才用。）

**注意点：**

- **triton cache 竞争（实测踩坑）**：共享文件系统 home（wekafs/PVC）上多 rank
  并发编译同一 triton kernel 会在 `~/.triton/cache` 撞车（读到写了一半的
  .cubin → FileNotFoundError，概率性）。`tasks/train_text_qad.py` 已按
  LOCAL_RANK 设置本地盘 `TRITON_CACHE_DIR`；k8s 双 pod 共享 PVC 时同样必须。
- NGC 容器已知坑（modelopt 被容器内置版本 shadow、triton 缺 ptxas/cuda.h ——
  之前在 Model-Optimizer 的 .venv 里修过）：进新 pod 后先确认共享 PVC 上的
  VeOmni .venv 与 25.11 镜像兼容，并跑一次 modelopt import 冒烟；
- manifest 是 `sleep infinity` + 手动 exec，训练崩了**不会自动重启** ——
  长训练要么包一层 supervisor 脚本，要么后续改成 Job + `restartPolicy`；
- kubeconfig 是集群凭据，已在 `.gitignore`，不要提交或外发。

---

## 11. 导出 w4 / w4a4 checkpoint 至 w4a4 部署格式 + 统一口径评测

### 输入（两条已训 checkpoint，训练已停）

| checkpoint | DCP 路径 | 训练量 | 训练中最后 eval | 口径 |
|---|---|---|---|---|
| **w4a4-QAD** | `qad-runs/qwen3-8b-w4a4-v3/checkpoints/global_step_400` | 400 步 | 1.3872（恢复 29%） | w4a4 |
| **w4-QAD** | `qad-runs/qwen3-8b-w4-v3/checkpoints/global_step_400` | 400 步 | 1.3575（恢复 34%） | **w4a16** |

**两条统一用 step-400 checkpoint,训练量严格一致**,对比公平。w4 的
step-800（1.3529,恢复 43%）另测一版作为"更多训练量"的参考点。注意 w4
的恢复率是 w4a16 口径，换到 w4a4 部署口径必然回吐一部分（激活误差它没
训过）。

### 导出流程（已在冒烟 checkpoint 上端到端验证）

每条 checkpoint 走两阶段：

```
Stage A（VeOmni venv）:
  merge_dcp_to_hf.py --load-dir <DCP> --model-assets-dir <run>/model_assets
    → bf16 潜权重 HF checkpoint

Stage B（Model-Optimizer venv）:
  export_nvfp4.py --hf-latent <A输出> --calib-data <qad-split>/train
    → strip act_global_amax buffer → mtq.quantize(NVFP4, calib) → TRT-LLM 格式
```

**两条 checkpoint 的关键差异在激活 scale 来源：**

- **w4a4-QAD**：训练前校准的激活 amax 就存在 DCP 的 `act_global_amax`
  buffer 里 —— 导出加 `--reuse-trained-act-amax`，把**训练仿真时的激活
  grid 原样带进部署**（最高保真：部署的就是 loss 优化过的那个格）。
  weight scale 从最终潜权重重算（与训练每步同规则，自洽）。
- **w4-QAD**：训练全程未碰激活，buffer 全零 —— 走默认 fresh calibration
  路径（modelopt 在最终权重上跑 128 条校准样本）。即该产物 =
  "权重 QAD + 激活 PTQ"。**不要**加 `--reuse-trained-act-amax`（脚本会
  因无有效值报错，这是有意的 guard）。

### 统一 w4a4 口径评测（fake-quant 仿真，本机 H100 可做）

新脚本 `scripts/qad/eval_w4a4.py`：加载 HF 潜权重 → `wrap_linears_for_qad
(mode="w4a4")` → 校准激活（64 batch 训练数据；w4a4-QAD 另测一版注入训练
amax 的）→ 在固定 eval-500 上测 PPL。产出四点对比表：

| 对比点 | 含义 |
|---|---|
| teacher（1.3241） | 全精度上界 |
| PTQ-W₀（1.4125，已有） | naive PTQ 下界 |
| **w4a4-QAD @400** | 端到端 QAD |
| **w4-QAD @800 + A-PTQ** | 权重 QAD、激活 PTQ 的混合路线 |

这张表回答核心问题：**激活量化到底需不需要参与训练**（若 w4-QAD+A-PTQ
逼近 w4a4-QAD，训练可以省掉激活仿真的开销和 a4 模式暴露的敏感性）。

真实 kernel 验证（TRT-LLM on Blackwell）仍是后续 —— 本机只能仿真口径。

### 执行顺序

1. Stage A × 2（两条并行，~5 分钟）
2. Stage B × 2（w4a4 带 --reuse-trained-act-amax；w4 不带）
3. 写 eval_w4a4.py → 四点评测 → 汇总表
4. a4-v4 跑完后同样导出 + 纳入对比（第五点）

### 结果（2026-07-02，统一 w4a4 仿真口径，eval-500 / 173 万 token）

| 对比点 | PPL | vs teacher gap | 恢复率 |
|---|---|---|---|
| teacher（全精度上界） | 1.3240 | — | — |
| PTQ-W₀（下界） | 1.4126 | 0.0886 | 0% |
| **w4a4-QAD @400** | 1.3870 | 0.0630 | **28.9%** |
| **w4-QAD @400 + A-PTQ** | 1.3862 | 0.0622 | **29.8%** |
| w4-QAD @800 + A-PTQ（2× 训练量参考） | 1.3803 | 0.0563 | 36.5% |
| a4-QAD @800 + W-PTQ（训练激活 grid） | 1.3958 | 0.0718 | 19.0% |

a4 补充结论：同为 800 步，a4 路线（19.0%）的恢复效率约为 w4 路线
（36.5%）的一半 —— 且它在自己的 w16a4 训练口径下只恢复了 ~3%
（横盘至步 800 才首次跌破起点），其 w4a4 口径下的 19% 主要说明
"补偿激活误差学到的权重调整"与"补偿权重量化误差"有部分共通性,
但作为训练策略全面劣于 w4/w4a4。三模式最终排序：
**w4-QAD ≈ w4a4-QAD ≫ a4-QAD**。

交叉验证：teacher/PTQ 点与训练中 veomni 路径的独立实现一致到第 4 位
（1.3240/1.4126 vs 1.3241/1.4125）。两条 checkpoint 的 TRT-LLM 格式
NVFP4 产物均已导出（`export/{w4a4-qad-400,w4-qad-400,w4-qad-800}/nvfp4_ckpt`）。

**核心结论：激活量化不需要参与训练。** 同训练量（400 步）下，
w4-QAD+激活PTQ（1.3862）与端到端 w4a4-QAD（1.3870）之差 0.0008，
在 eval 噪声（±0.0014）以内 —— 统计上不可区分。与 a4 消融互相印证
（纯激活侧训练不仅无增益还易恶化）：激活量化误差基本不是训练可补偿
的成分，权重侧 QAD 才是恢复的全部来源。工程含义：训练管线可以只做
w4（免校准阶段、免激活仿真开销、避开 a4 暴露的 lr 敏感性），激活量化
交给部署时 PTQ；省出的算力投入更多训练步数收益更大（w4@800 → 36.5%）。
