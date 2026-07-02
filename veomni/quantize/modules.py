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

"""``nn.Linear`` wrapper applying NVFP4 fake quantization for QAD training."""

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QADQuantConfig
from .fake_quant import NVFP4_BLOCK_SIZE, nvfp4_global_amax, nvfp4_weight_overflow_ratio
from .ste import nvfp4_fake_quant_ste


class FakeQuantLinear(nn.Linear):
    """Linear layer whose forward runs through NVFP4 fake quantization.

    ``self.weight`` stays the high-precision latent weight (the trainable
    accumulator of QAD); weight scales are recomputed from it on every
    forward with the deployment rule, so training and export see the same
    grid at all times. For activations the per-block scales are dynamic while
    the per-tensor global scale comes from the ``act_global_amax`` buffer,
    which calibration must populate before ``act_quant_enabled`` forwards run.

    ``weight_quant_enabled`` / ``act_quant_enabled`` are plain attributes so a
    trainer can toggle them (e.g. to measure the latent full-precision model).
    Weight-health statistics come from :meth:`weight_overflow_ratio`, computed
    on demand at logging steps (never on the forward hot path).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"in_features={self.in_features} is not a multiple of the NVFP4 block size {NVFP4_BLOCK_SIZE}"
            )
        self.weight_quant_enabled: bool = False
        self.act_quant_enabled: bool = False
        self.act_calibrating: bool = False
        # fp32 scalar; 0 means "not calibrated yet". Persistent so DCP
        # checkpoints carry the calibrated activation scale across resumes.
        # NOTE (DCP key change): wrapping adds this key to the state dict, so
        # a pre-QAD DCP checkpoint cannot resume into a QAD run (strict load
        # fails loudly) — start QAD runs from HF weights instead.
        self.register_buffer("act_global_amax", torch.zeros((), dtype=torch.float32))
        self._act_amax_checked: bool = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quantize_weights: bool,
        quantize_activations: bool,
    ) -> "FakeQuantLinear":
        """Wrap an existing linear, reusing its parameter objects.

        Reusing (not copying) ``weight``/``bias`` keeps meta-device tensors
        valid, preserves DCP checkpoint keys, and keeps any ParallelPlan
        references to the original parameters intact.
        """
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
            dtype=linear.weight.dtype,
        )
        module.weight = linear.weight
        if linear.bias is not None:
            module.bias = linear.bias
        module.act_global_amax = module.act_global_amax.to(
            linear.weight.device if linear.weight.device.type != "meta" else "cpu"
        )
        module.weight_quant_enabled = quantize_weights
        module.act_quant_enabled = quantize_activations
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_calibrating:
            with torch.no_grad():
                batch_amax = nvfp4_global_amax(x)
                self.act_global_amax.copy_(torch.maximum(self.act_global_amax, batch_amax))
        elif self.act_quant_enabled:
            if not self._act_amax_checked:
                # One host sync on the first quantized forward only. Checks
                # finite-and-positive rather than just nonzero: the
                # weights_path=None materialization flow (to_empty +
                # init_weights) leaves custom buffers as garbage, which a
                # zero-sentinel check would silently accept.
                amax_value = self.act_global_amax.item()
                if not (amax_value > 0.0 and torch.isfinite(self.act_global_amax).item()):
                    raise RuntimeError(
                        f"Activation quantization is enabled but act_global_amax={amax_value} is not a "
                        "calibrated positive finite value. Run calibration (veomni.quantize.calibrate) "
                        "before training."
                    )
                self._act_amax_checked = True
            x = nvfp4_fake_quant_ste(x, self.act_global_amax)

        weight = nvfp4_fake_quant_ste(self.weight) if self.weight_quant_enabled else self.weight
        return F.linear(x, weight, self.bias)

    def weight_overflow_ratio(self) -> float:
        """Pre-round overflow fraction of the current weight (see fake_quant docs).

        Depends only on the weight, so it is computed on demand at logging
        steps instead of accumulating on the forward hot path (which would
        also double-count under gradient checkpointing recompute).

        Outside forward, FSDP2 params are sharded DTensors — gather the full
        tensor first (collective: every rank must call this in the same
        order, which the trainer's iterate-all-layers loop guarantees).
        """
        weight = self.weight
        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.full_tensor()
        return nvfp4_weight_overflow_ratio(weight)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weight_quant={self.weight_quant_enabled}, act_quant={self.act_quant_enabled}"


def wrap_linears_for_qad(model: nn.Module, config: QADQuantConfig) -> int:
    """Replace target ``nn.Linear`` modules with :class:`FakeQuantLinear`.

    Matches modules whose name's last component is in ``config.target_modules``.
    Safe on meta-device models (structural only — parameters are reused, not
    copied). Returns the number of wrapped modules.
    """
    targets = set(config.target_modules)
    wrapped = 0
    already_wrapped = 0
    skipped_subclasses = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name not in targets or not isinstance(child, nn.Linear):
                continue
            if isinstance(child, FakeQuantLinear):
                already_wrapped += 1
                continue
            # A plain type check, not isinstance: silently replacing an
            # nn.Linear subclass would drop its overridden behavior.
            if type(child) is not nn.Linear:
                skipped_subclasses.append(f"{parent_name}.{child_name} ({type(child).__name__})")
                continue
            setattr(
                parent,
                child_name,
                FakeQuantLinear.from_linear(
                    child,
                    quantize_weights=config.quantize_weights,
                    quantize_activations=config.quantize_activations,
                ),
            )
            wrapped += 1
    if skipped_subclasses:
        raise ValueError(
            f"wrap_linears_for_qad matched {len(skipped_subclasses)} nn.Linear SUBCLASS modules it "
            f"refuses to silently replace: {skipped_subclasses[:5]}. Handle these types explicitly."
        )
    if wrapped == 0:
        if already_wrapped > 0:
            raise ValueError(f"Model is already wrapped for QAD ({already_wrapped} FakeQuantLinear modules found).")
        raise ValueError(
            f"wrap_linears_for_qad matched no modules; target_modules={sorted(targets)} "
            "does not name any nn.Linear children of this model."
        )
    return wrapped


def iter_fake_quant_linears(model: nn.Module) -> Iterator[Tuple[str, FakeQuantLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinear):
            yield name, module


def set_quantizer_enabled(
    model: nn.Module,
    weight_quant: Optional[bool] = None,
    act_quant: Optional[bool] = None,
) -> None:
    """Toggle quantizers on every :class:`FakeQuantLinear` (``None`` = leave as is)."""
    for _, module in iter_fake_quant_linears(model):
        if weight_quant is not None:
            module.weight_quant_enabled = weight_quant
        if act_quant is not None:
            module.act_quant_enabled = act_quant
