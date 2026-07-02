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

"""Activation global-scale calibration for NVFP4 QAD (w4a4 / a4 modes).

NVFP4 activation quantization is hybrid: per-16-element block scales are
computed dynamically by the inference kernel, but the per-tensor FP32 global
scale must be known ahead of time — it is the one statistic calibration
collects. Matching modelopt's default max calibrator, we track the running
``max`` of per-batch amax (not an EMA).

Usage::

    start_calibration(model)
    with torch.no_grad():
        for batch in calib_batches:
            model(**batch)
    finalize_calibration(model, process_group=dist.group.WORLD)

``finalize_calibration`` all-reduces every collected amax with MAX across
ranks so all shards freeze the same grid (DP/SP ranks see different tokens).
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .modules import iter_fake_quant_linears


def start_calibration(model: nn.Module) -> None:
    """Put every :class:`FakeQuantLinear` into activation-amax collection mode.

    During calibration the activation quantizer is bypassed; weight fake quant
    stays in whatever state the training mode configured (w4a4 calibrates with
    weight quantization active so the statistics reflect the served model).
    """
    for _, module in iter_fake_quant_linears(model):
        module.act_calibrating = True
        module.act_global_amax.zero_()


def finalize_calibration(model: nn.Module, process_group: Optional[dist.ProcessGroup] = None) -> int:
    """Freeze collected amax values and leave calibration mode.

    Returns the number of calibrated modules. Raises if any module saw no
    data (its amax is still zero), which would silently produce a degenerate
    quantization grid.
    """
    modules = [m for _, m in iter_fake_quant_linears(model)]
    if not modules:
        raise ValueError("finalize_calibration found no FakeQuantLinear modules.")

    if dist.is_available() and dist.is_initialized():
        # One stacked all-reduce instead of one per layer. Assumes every rank
        # holds the same module list (true for FSDP-replicated structure;
        # pipeline parallel would violate this and is not supported here).
        # Buffers may live on CPU under fsdp offload — move to the
        # communication device for the collective.
        from ..utils.device import get_device_type

        stacked = torch.stack([m.act_global_amax for m in modules]).to(get_device_type())
        dist.all_reduce(stacked, op=dist.ReduceOp.MAX, group=process_group)
        for module, amax in zip(modules, stacked.unbind()):
            module.act_global_amax.copy_(amax)

    uncalibrated = [name for name, m in iter_fake_quant_linears(model) if m.act_global_amax.item() == 0.0]
    if uncalibrated:
        raise RuntimeError(
            f"{len(uncalibrated)} FakeQuantLinear modules collected no activation statistics "
            f"during calibration (first few: {uncalibrated[:5]}). Run more calibration batches."
        )

    for module in modules:
        module.act_calibrating = False
    return len(modules)
