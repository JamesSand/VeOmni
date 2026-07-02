# QAD 实现与实验工作汇总（2026-07-02）

> 一天内完成：VeOmni 上的 NVFP4 量化感知蒸馏（QAD）全链路实现 → 三模式训练消融
> → 部署格式导出 → 统一口径评测。设计文档见 `tony-note-plan.md`（§1–§11）。
> 分支：`zhizhou-dev-qad`（15 个本地 commit，待 push）。

## 一句话结果

Qwen3-8B 自蒸馏 QAD 在 400 步（~2.1 亿 token）内追回 naive PTQ 约 29% 的
质量损失（w4@800 步达 36.5%），并得到一个改变训练配方的结论：
**激活量化不需要参与训练** —— 纯 w4 训练 + 部署时激活 PTQ 与端到端 w4a4
训练效果统计上不可区分。

## 1. 实现的代码（全部已 commit）

| 模块 | 内容 |
|---|---|
| `veomni/quantize/` | NVFP4 伪量化核心：E2M1 + per-16 E4M3 block scale（双重量化）+ FP32 global scale，与 modelopt kernel 语义逐步对齐；带 clamp 掩码的 STE；`FakeQuantLinear`（w4/w4a4/a4 三模式，meta-init 安全、DCP key 稳定）；激活 amax 校准（跨 rank MAX 规约） |
| `veomni/trainer/text_qad_trainer.py` | DPO 式组合 trainer：冻结全精度 teacher（FSDP2 同构分片）+ 伪量化 student；蒸馏 loss 复用现成 `chunk_topk_distill_function`（top-128 前向 KL，两侧均不物化 `[L,V]` logits）；选择性冻结；激活校准阶段；held-out 评测（student/teacher PPL、KL、step-0 即 PTQ 基线）；wandb 指标 |
| `veomni/arguments/` | `QADConfig`（`train.qad.*`：mode/tau/alpha/teacher_topk/calib_steps 等） |
| `veomni/data/` | `pretokenized` data_type（直接消费离线 tokenize 的 parquet，避免模板漂移）；datasets 2.x 读 datasets≥4 parquet 的 `List` 特性兼容 shim |
| `scripts/qad/` | `make_eval_split.py`（shard 级 held-out + 固定 2000 条 eval 子集）；`export_nvfp4.py`（DCP→HF→modelopt PTQ→TRT-LLM 格式，支持注入训练校准的激活 grid）；`eval_w4a4.py`（统一 w4a4 口径评测） |
| `tasks/train_text_qad.py` + `configs/text/qwen3_qad.yaml` | 训练入口与 Qwen3-8B 配置 |
| 测试 | 55+ 个：NVFP4 规范测试（RNE 边界/双重量化/守卫）、modelopt 逐值对拍（6/6）、教师对齐锚定测试（错位一格 KL 暴涨 6 个量级）、参数解析、回归测试 |

## 2. 发现并修复的 bug

1. **上游 VeOmni 静默零梯度 bug（最重要）**：`chunk_logprobs`/`chunk_topk_distill`
   两个 kernel 的 backward 用 saved tensor 的 `requires_grad` 判断是否计算
   dhidden —— 对真实模型 backbone 激活恒为 False → **lm_head 以外所有参数
   梯度精确为 0**，而 kernel 单测和上游 e2e 测试（只断言 lm_head）全绿。
   通过"eval 逐位不变 + grad_norm 精确 0"发现，逐层二分定位，改用
   `ctx.needs_input_grad` 修复并补回归测试。**DPO 生产路径同样受影响，
   建议向上游提 PR**（commit `d01da08`）。
2. wekafs 共享 home 上多 rank triton kernel 缓存竞争（概率性 FileNotFoundError）
   → per-rank 本地 `TRITON_CACHE_DIR`（中心化在 `veomni/__init__.py`）。
3. `data.train_size` 默认 1e7 token 把训练静默截成 20 步 → config 显式设置。
4. datasets 2.21 无法读 datasets≥4 写的 parquet（`List` 特性）→ `_FEATURE_TYPES` shim。
5. 两轮 code review（每轮独立 subagent）共 21 条 issue 全部修复，含分布式
   挂死隐患、DTensor 统计崩溃、梯度累积归一化、HF 导出污染等。

## 3. 实验（8×H100 dev pod + 2 个 k8s pod 并行）

数据：`reasonmix-sftmix-combined-Qwen3-8B`（Qwen3-8B 自生成，on-policy），
274 shard 训练 / 固定 500 条 eval（~173 万 token，噪声 ±0.1%）。
统一配置：batch 64×8192、top-128 KL、τ=1、α=1（纯蒸馏）。

| 实验 | lr | 停止步数 | 各自口径 eval 轨迹 |
|---|---|---|---|
| w4a4-v3 | 2e-5 | 400（后 resume 至 800） | 1.4125 → 1.3872（恢复 29%） |
| w4-v3 | 2e-5 | 800 | 1.3746 → 1.3554@200 → 1.3529@800（43%，未收敛） |
| a4-v3 | 2e-5 | 400（废弃） | 恶化 -37%，判定 lr 过大 |
| a4-v4 | 5e-6 | 832 | 横盘至 800 首次微降（1.3575→1.3564） |

a4 消融发现：无 STE 格点缓冲的模式（全精度权重直接生效）对 lr 极端敏感，
恢复效率约为 w4 路线的一半。

## 4. 统一 w4a4 口径评测（六点，同一脚本/同一 eval 集）

| 对比点 | PPL | 恢复率 |
|---|---|---|
| teacher（bf16 上界） | 1.3240 | — |
| naive PTQ-W₀（下界） | 1.4126 | 0% |
| **w4-QAD @400 + 激活PTQ** | **1.3862** | **29.8%** |
| **w4a4-QAD @400** | 1.3870 | 28.9% |
| w4-QAD @800 + 激活PTQ | 1.3803 | 36.5% |
| a4-QAD @800 + 权重PTQ | 1.3958 | 19.0% |

**核心结论**：同预算下 w4-QAD+激活PTQ ≈ w4a4-QAD（差 0.0008 < 噪声
±0.0014）→ 激活量化无需参与训练；恢复排序 **w4 ≈ w4a4 ≫ a4**；
推荐生产配方 = 纯 w4 训练 + 部署时激活 PTQ，算力全投步数。
交叉验证：teacher/PTQ 点与训练路径独立实现一致到小数点后 4 位。

## 5. 产物

- **HuggingFace（private, togethercomputer org）**：
  - https://huggingface.co/togethercomputer/Qwen3-8B-VeOmni-QAD-w4-step400-NVFP4
  - https://huggingface.co/togethercomputer/Qwen3-8B-VeOmni-QAD-w4a4-step400-NVFP4
- **本地 NVFP4 导出**（TRT-LLM/SGLang on Blackwell 可加载）：
  `qad-runs/export/{w4a4-qad-400,w4-qad-400,w4-qad-800,a4-qad-800}/nvfp4_ckpt`
- DCP checkpoint（各 run 每 200 步）、bf16 潜权重 HF 版、wandb 曲线
  （`VeOmni-QAD` 项目 7 条 run）
- 数据切分：`model-and-data/datasets/qad-split-Qwen3-8B/`（含 SPLIT.json 版本清单）

## 6. 进行中 / 待办

- [ ] w4a4 resume 400→800 步训练中（`train_resume800.log`，~3h，完成后可导出对齐 w4@800 的对比点）
- [ ] `git push origin zhizhou-dev-qad`（15 commits，凭据在 VSCode 侧）
- [ ] HF 上 3 个误传的旧命名 repo 待确认删除（`Qwen3-8B-QAD-w4-step400-bf16` 等）
- [ ] Blackwell 节点真 kernel 验证（TRT-LLM/SGLang 加载 NVFP4 产物）
- [ ] 零梯度 kernel 修复是否向上游 VeOmni 提 PR
- [ ] w4 若跑满 3000 步预期恢复率更高（800 步时 43% 且未收敛）
