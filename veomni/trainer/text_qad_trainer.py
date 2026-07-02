# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization-aware distillation (QAD) trainer.

Self-distillation of an NVFP4 fake-quantized student against its own frozen
full-precision checkpoint (the teacher), recovering the quality lost to w4a4
quantization. Structure mirrors ``TextDPOTrainer`` (frozen second model, both
FSDP2-sharded identically); the distillation loss rides the existing chunked
top-k forward-KL kernel (``chunk_topk_distill_function``) through the model's
``return_log_probs=True`` + ``teacher_topk_*`` forward kwargs, so neither
model ever materializes a ``[L, vocab]`` logits tensor with gradients.
"""

import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import PreTrainedModel

from ..arguments import MixedPrecisionConfig, VeOmniArguments
from ..data import build_data_transform
from ..data.data_collator import add_flash_attention_kwargs_from_position_ids
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..quantize import (
    QADQuantConfig,
    finalize_calibration,
    iter_fake_quant_linears,
    start_calibration,
    wrap_linears_for_qad,
)
from ..utils import helper, logging
from ..utils.constants import IGNORE_INDEX
from ..utils.device import synchronize
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer, VeOmniIter


logger = logging.get_logger(__name__)

_NON_MODEL_KEYS = {"labels"}


@torch.no_grad()
def aligned_topk_from_logits(
    logits: torch.Tensor,
    topk: int,
    temperature: float,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher logits → label-aligned per-position top-k ids and log-probs.

    ``chunk_topk_distill_function`` expects teacher tensors aligned to the
    label time-step: after its internal causal shift, position ``t`` pairs
    ``(student_hidden[t-1], label[t], teacher[t])``, so ``teacher[t]`` must be
    the teacher's predictive distribution for token ``t`` — i.e. computed
    from teacher logits at position ``t-1``. We therefore right-shift the
    top-k tensors by one position; slot 0 is a dummy the kernel drops.
    (Pinned by the identical-teacher KL≈0 test in tests/quantize.)

    log_softmax runs in fp32 chunk-by-chunk so the transient is
    ``chunk_size × vocab`` instead of ``L × vocab``.
    """
    if logits.shape[1] < 2:
        raise ValueError(f"Sequence too short for distillation: L={logits.shape[1]}")

    ids_chunks: List[torch.Tensor] = []
    logps_chunks: List[torch.Tensor] = []
    source = logits[:, :-1]  # position t predicts token t+1
    for start in range(0, source.shape[1], chunk_size):
        chunk = source[:, start : start + chunk_size].float()
        log_probs = torch.log_softmax(chunk / temperature, dim=-1)
        top_logps, top_ids = log_probs.topk(topk, dim=-1)
        ids_chunks.append(top_ids)
        logps_chunks.append(top_logps)

    ids = torch.cat(ids_chunks, dim=1)
    logps = torch.cat(logps_chunks, dim=1)
    # Right-shift into label alignment; slot 0 is dropped by the kernel.
    ids = torch.nn.functional.pad(ids, (0, 0, 1, 0), value=0)
    logps = torch.nn.functional.pad(logps, (0, 0, 1, 0), value=0.0)
    return ids, logps


@torch.no_grad()
def compute_teacher_topk(
    teacher: PreTrainedModel,
    micro_batch: Dict[str, torch.Tensor],
    topk: int,
    temperature: float,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher forward (full precision, no grad) → label-aligned top-k targets."""
    model_inputs = {k: v for k, v in micro_batch.items() if k not in _NON_MODEL_KEYS}
    logits = teacher(**model_inputs, use_cache=False).logits  # [B, L, V], no grad
    return aligned_topk_from_logits(logits, topk=topk, temperature=temperature, chunk_size=chunk_size)


class TextQADTrainer:
    """QAD trainer composing BaseTrainer with a frozen full-precision teacher."""

    base: BaseTrainer
    teacher_model: PreTrainedModel

    def __init__(self, args: VeOmniArguments):
        qad = args.train.qad
        if not qad.enable:
            raise ValueError("TextQADTrainer requires train.qad.enable=true.")
        if qad.teacher_mode != "separate":
            raise NotImplementedError(f"teacher_mode={qad.teacher_mode!r} is not implemented yet.")
        if bool(args.model.lora_config):
            raise NotImplementedError(
                "TextQADTrainer does not support LoRA: QAD trains the latent weights of the "
                "quantized linears directly (and the HF-LoRA checkpoint callback would crash)."
            )
        if qad.teacher_model_path is not None and qad.teacher_model_path != args.model.model_path:
            raise NotImplementedError(
                "Only self-distillation is supported: the teacher is loaded with the student's "
                "config_path and its top-k vocab ids must match the student's vocabulary. "
                f"Got teacher_model_path={qad.teacher_model_path!r} != model_path={args.model.model_path!r}."
            )
        if args.train.checkpoint.save_hf_weights:
            raise ValueError(
                "train.checkpoint.save_hf_weights must be false for QAD: FakeQuantLinear's "
                "act_global_amax buffers would leak into the exported HF safetensors. Deployment "
                "artifacts come from the NVFP4 export script (latent DCP checkpoint -> PTQ)."
            )
        if args.data.data_type != "pretokenized":
            logger.warning_rank0(
                f"QAD is typically trained on pretokenized teacher generations; got data_type={args.data.data_type}."
            )

        self.quant_config = QADQuantConfig(mode=qad.mode, target_modules=list(qad.target_modules))

        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        if get_parallel_state().sp_enabled:
            raise NotImplementedError(
                "TextQADTrainer does not support sequence parallel yet; teacher top-k tensors "
                "would need SP-aware slicing. Disable ulysses/cp for QAD."
            )
        self.base._build_model()
        self._wrap_and_freeze()
        self.base._build_model_assets()

        self._build_data_transform()
        self.base._build_dataset()
        self.base._build_collate_fn()
        self.base._build_dataloader()

        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()

        self._build_teacher_model()
        # Calibration rebuilds the dataloader, so callbacks (EnvironMeter
        # holds a dataloader reference) initialize after it.
        self._calibrate_activations()
        self.base._init_callbacks()

        self._load_eval_samples()
        self._teacher_eval_nll: Optional[float] = None

    # ----------------------------- build steps -----------------------------

    def _wrap_and_freeze(self):
        """Swap target linears for FakeQuantLinear (meta-safe) and freeze the rest.

        Runs before parallelize/optimizer so FSDP2 shards the swapped tree and
        ``build_optimizer``'s requires_grad filter sees the final flags. Only
        the wrapped layers' latent weights train — in ``a4`` mode too, where
        they compensate activation error at full precision (no weight STE).
        """
        wrapped = wrap_linears_for_qad(self.base.model, self.quant_config)
        logger.info_rank0(f"QAD: wrapped {wrapped} linears (mode={self.quant_config.mode})")

        self.base.model.requires_grad_(False)
        for _, module in iter_fake_quant_linears(self.base.model):
            module.weight.requires_grad_(True)
        pretty_print_trainable_parameters(self.base.model)

    def _build_data_transform(self):
        args: VeOmniArguments = self.base.args
        self.base.data_transform = build_data_transform(
            args.data.data_type,
            tokenizer=self.base.tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )

    def _build_teacher_model(self):
        """Frozen full-precision teacher, FSDP2-sharded like the student (DPO pattern).

        The teacher never sees the fake quantizer and never receives
        gradients; bf16, no mixed precision, no gradient checkpointing.
        """
        args: VeOmniArguments = self.base.args
        teacher_path = args.train.qad.teacher_model_path or args.model.model_path
        logger.info_rank0(f"Building frozen QAD teacher from {teacher_path}")

        self.teacher_model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=teacher_path,
            torch_dtype="bfloat16",
            init_device=args.train.init_device,
            ops_implementation=args.model.ops_implementation,
        )
        self.teacher_model.requires_grad_(False)

        cpu_load_param_name = None
        if hasattr(self.base.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.base.model.get_parallel_plan(), "cpu_load_param_name", None)

        self.teacher_model = build_parallelize_model(
            self.teacher_model,
            init_device=args.train.init_device,
            weights_path=teacher_path,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=MixedPrecisionConfig(enable=False),
            enable_gradient_checkpointing=False,
            basic_modules=list(
                set(getattr(self.teacher_model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=False,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
        )
        self.teacher_model.eval()
        helper.print_device_mem_info("VRAM usage after building QAD teacher")

    def _calibrate_activations(self):
        """Freeze the activation global scales (w4a4/a4) before training.

        Runs ``calib_steps`` batches through the student with weight fake
        quant in its training state (w4a4 calibrates under quantized weights
        so statistics reflect the served model — tony-note-plan §1.5), then
        max-reduces the amax across ranks and rebuilds the dataloader so
        training sees the full stream.
        """
        args: VeOmniArguments = self.base.args
        if not self.quant_config.quantize_activations:
            return

        logger.info_rank0(f"QAD: calibrating activation scales over {args.train.qad.calib_steps} batches")
        start_calibration(self.base.model)
        self.base.model.eval()
        calib_iter = iter(self.base.train_dataloader)
        with torch.no_grad():
            for _ in range(args.train.qad.calib_steps):
                try:
                    micro_batches = next(calib_iter)
                    has_batch = torch.ones((), device=self.base.device)
                except StopIteration:
                    micro_batches = []
                    has_batch = torch.zeros((), device=self.base.device)
                # FSDP2 forwards are collective (param all-gathers): if ANY
                # rank ran out of data, all ranks must stop together or the
                # remaining ranks deadlock waiting for the empty one.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
                if has_batch.item() == 0:
                    logger.warning_rank0("QAD: calibration stream exhausted early on some rank; stopping.")
                    break
                for micro_batch in micro_batches:
                    micro_batch = self.base.preforward(micro_batch)
                    model_inputs = {k: v for k, v in micro_batch.items() if k not in _NON_MODEL_KEYS}
                    self.base.model(**model_inputs, use_cache=False)
        del calib_iter
        count = finalize_calibration(self.base.model, process_group=None)
        self.base.model.train()
        logger.info_rank0(f"QAD: calibrated {count} activation scales")
        # Rebuild so training starts from a fresh stream (calibration consumed batches).
        self.base._build_dataloader()

    # ----------------------------- evaluation -----------------------------

    def _load_eval_samples(self):
        """Load the fixed eval subset (scripts/qad/make_eval_split.py), rank-sliced.

        Each DP rank evaluates ``samples[rank::world_size]``; sums are
        all-reduced. Sequences are evaluated one per forward (no packing), so
        per-sample NLL/token counts are exact.
        """
        args: VeOmniArguments = self.base.args
        self._eval_samples: List[Dict[str, torch.Tensor]] = []
        if not args.data.eval_path:
            logger.warning_rank0("QAD: data.eval_path not set — in-training evaluation disabled.")
            return

        import pyarrow.parquet as pq

        files = sorted(glob.glob(os.path.join(args.data.eval_path, "*.parquet")))
        if not files:
            raise ValueError(f"QAD eval_path {args.data.eval_path} contains no parquet files.")
        table = pq.read_table(files)
        rank = args.train.global_rank
        world = args.train.world_size
        max_len = args.data.max_seq_len
        for i in range(rank, table.num_rows, world):
            input_ids = torch.tensor(table.column("input_ids")[i].as_py()[:max_len], dtype=torch.long)
            labels = torch.tensor(table.column("labels")[i].as_py()[:max_len], dtype=torch.long)
            if (labels[1:] != IGNORE_INDEX).sum() == 0:
                continue
            self._eval_samples.append({"input_ids": input_ids, "labels": labels})
        logger.info_rank0(f"QAD: loaded eval subset from {args.data.eval_path} ({table.num_rows} rows total)")

    def _eval_batch(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = sample["input_ids"].unsqueeze(0).to(self.base.device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
        }
        add_flash_attention_kwargs_from_position_ids(batch)
        batch["labels"] = sample["labels"].unsqueeze(0).to(self.base.device)
        return batch

    @torch.no_grad()
    def evaluate(self):
        """Held-out metrics: student PPL (w4a4-sim), teacher PPL, KL-to-teacher.

        Evaluated at step 0 before any update, ``eval/student_ppl`` IS the
        naive-PTQ baseline (quantized W0) — the lower bound of tony-note §9's
        three-point comparison; ``eval/teacher_ppl`` is the upper bound.
        """
        args: VeOmniArguments = self.base.args
        if not self._eval_samples:
            return
        qad = args.train.qad
        self.base.model.eval()

        # FSDP2 forwards are collective — every rank must run the same
        # number of forwards. Ranks with fewer samples repeat their first one
        # with zero weight (the wrap-around pass keeps collectives aligned
        # without double counting).
        max_count = torch.tensor(len(self._eval_samples), device=self.base.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
        max_count = int(max_count.item())

        need_teacher_nll = self._teacher_eval_nll is None
        sums = torch.zeros(4, dtype=torch.float64, device=self.base.device)  # s_nll, t_nll, kl, tokens
        for idx in range(max_count):
            is_padding = idx >= len(self._eval_samples)
            sample = self._eval_samples[0] if is_padding else self._eval_samples[idx]
            weight = 0.0 if is_padding else 1.0
            batch = self._eval_batch(sample)
            labels = batch.pop("labels")
            n_valid = (labels[..., 1:] != IGNORE_INDEX).sum()

            student_out = self.base.model(**batch, labels=labels, use_cache=False)
            sums[0] += student_out.loss.double() * n_valid * weight

            if need_teacher_nll:
                teacher_out = self.teacher_model(**batch, labels=labels, use_cache=False)
                sums[1] += teacher_out.loss.double() * n_valid * weight

            topk_ids, topk_logps = compute_teacher_topk(
                self.teacher_model, batch, topk=qad.teacher_topk, temperature=qad.tau
            )
            distill_out = self.base.model(
                **batch,
                labels=labels,
                use_cache=False,
                return_log_probs=True,
                temperature=qad.tau,
                teacher_topk_ids=topk_ids,
                teacher_topk_log_probs=topk_logps,
            )
            sums[2] += distill_out.fused_linear_aux.distillation_losses.double().sum() * weight
            sums[3] += n_valid * weight

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

        tokens = sums[3].clamp(min=1)
        student_ppl = torch.exp(sums[0] / tokens).item()
        kl_to_teacher = (sums[2] / tokens).item()
        if need_teacher_nll:
            self._teacher_eval_nll = (sums[1] / tokens).item()
        teacher_ppl = float(torch.exp(torch.tensor(self._teacher_eval_nll)))

        metrics = {
            "eval/student_ppl": student_ppl,
            "eval/teacher_ppl": teacher_ppl,
            "eval/kl_to_teacher": kl_to_teacher,
            "eval/ppl_gap": student_ppl - teacher_ppl,
        }
        logger.info_rank0(f"QAD eval @ step {self.base.state.global_step}: {metrics}")
        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(metrics, step=self.base.state.global_step)

        self.base.model.train()

    # ----------------------------- training -----------------------------

    def qad_loss(self, aux: Any, step_token_count: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate per-position kernel outputs into the QAD objective.

        ``L = alpha * tau^2 * KL_topk + (1 - alpha) * CE``. Sums are
        normalized by the valid-token count of the WHOLE optimizer step (all
        gradient-accumulation micro batches), so accumulated gradients — and
        the logged per-step metrics — are invariant to the number of
        accumulation steps (per-micro-batch means would scale the effective
        LR with world size; cf. base's ``mean_global_loss``). The kernel
        already zeroes IGNORE_INDEX positions; the tau^2 factor keeps
        gradient magnitude comparable across temperatures (tony-note §3).
        When ``tau != 1`` the CE term is also temperature-scaled (kernel
        limitation); the default ``alpha=1`` (pure distillation) is
        unaffected.
        """
        args: VeOmniArguments = self.base.args
        tau = args.train.qad.tau
        alpha = args.train.qad.alpha

        distill_loss = (tau**2) * aux.distillation_losses.sum() / step_token_count
        ce_loss = -aux.log_probs.sum() / step_token_count
        # Wire the graph only through active terms: at alpha=1.0 even a
        # 0-weighted CE term would force the kernel's scatter backward.
        if alpha >= 1.0:
            loss = distill_loss
        elif alpha <= 0.0:
            loss = ce_loss
        else:
            loss = alpha * distill_loss + (1.0 - alpha) * ce_loss

        loss_dict = {
            "kl_loss": distill_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "student_mass": (aux.student_mass.sum() / step_token_count).detach(),
            "teacher_mass": (aux.teacher_mass.sum() / step_token_count).detach(),
        }
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: Dict[str, torch.Tensor], step_token_count: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        args: VeOmniArguments = self.base.args
        qad = args.train.qad

        micro_batch = self.base.preforward(micro_batch)

        teacher_topk_ids, teacher_topk_log_probs = compute_teacher_topk(
            self.teacher_model, micro_batch, topk=qad.teacher_topk, temperature=qad.tau
        )

        with self.base.model_fwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            outputs = self.base.model(
                **micro_batch,
                use_cache=False,
                return_log_probs=True,
                temperature=qad.tau,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_log_probs=teacher_topk_log_probs,
            )

        loss, loss_dict = self.qad_loss(outputs.fused_linear_aux, step_token_count)

        with self.base.model_bwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            loss.backward()

        del micro_batch
        return loss, loss_dict

    def _weight_overflow_ratio(self) -> float:
        """Mean pre-round overflow across wrapped layers (weight-health signal)."""
        ratios = [module.weight_overflow_ratio() for _, module in iter_fake_quant_linears(self.base.model)]
        return sum(ratios) / len(ratios) if ratios else 0.0

    def train_step(self, data_iterator: Any) -> None:
        args: VeOmniArguments = self.base.args
        self.base.state.global_step += 1

        interval = args.train.qad.clamp_stats_interval
        collect_this_step = interval > 0 and self.base.state.global_step % interval == 0

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.base.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)

        # Valid-token count of the whole step for accumulation-invariant
        # normalization (labels are still on CPU here; cheap sum).
        step_token_count = (
            torch.stack([(mb["labels"][..., 1:] != IGNORE_INDEX).sum() for mb in micro_batches])
            .sum()
            .clamp(min=1)
            .to(self.base.device)
        )

        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base._configure_hsdp_allreduce(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch, step_token_count)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)

        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        if collect_this_step and self.quant_config.quantize_weights:
            total_loss_dict["weight_overflow_ratio"] = self._weight_overflow_ratio()

        self.base.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

        eval_steps = args.train.eval_steps
        if eval_steps and self.base.state.global_step % eval_steps == 0:
            self.evaluate()

    def train(self):
        args: VeOmniArguments = self.base.args
        self.base.on_train_begin()
        if self.base.state.global_step == 0:
            # Step-0 eval: student_ppl here is the naive-PTQ baseline (W0
            # quantized, untrained) and teacher_ppl the full-precision bound.
            self.evaluate()
        logger.info(
            f"Rank{args.train.local_rank} Start QAD training (mode={args.train.qad.mode}). "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch

            self.base.on_epoch_begin()

            self.base.data_iterator = VeOmniIter(
                self.base.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.base.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.base.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        self.base.destroy_distributed()
