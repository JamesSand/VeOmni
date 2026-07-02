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

"""Straight-through estimator for NVFP4 fake quantization.

Forward: the true NVFP4 quantize→dequantize value (:mod:`.fake_quant`).
Backward: identity for elements inside the representable range
(``|x| <= global_amax``), zero for elements beyond it — nudging an
already-saturated element further out cannot change the output, so its
gradient is meaningless and only destabilizes training. With a dynamic
(weight-path) amax nothing saturates by construction; the mask bites on
activations exceeding their frozen calibrated amax.

The scales (block and global) are treated as forward-time constants: no
gradient flows into them or into the amax they derive from. Gradients through
an amax would couple one element's gradient to every element of its block and
destabilize the STE dynamics.
"""

from typing import Optional

import torch

from .fake_quant import nvfp4_quant_dequant


class _NVFP4FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: Optional[torch.Tensor]) -> torch.Tensor:
        # Weight path (dynamic amax): |x| <= max|x| always, the mask is empty
        # by construction — skip computing and saving it. Saving a
        # weight-sized bool per wrapped linear would pin ~1 byte/element from
        # forward to backward for nothing.
        if global_amax is None:
            ctx.has_mask = False
            out, _ = nvfp4_quant_dequant(x, global_amax=None, return_saturation_mask=False)
            return out
        ctx.has_mask = True
        out, saturation_mask = nvfp4_quant_dequant(x, global_amax=global_amax, return_saturation_mask=True)
        ctx.save_for_backward(saturation_mask)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.has_mask:
            return grad_output, None
        (saturation_mask,) = ctx.saved_tensors
        grad_input = grad_output.masked_fill(saturation_mask, 0.0)
        return grad_input, None


def nvfp4_fake_quant_ste(x: torch.Tensor, global_amax: Optional[torch.Tensor] = None) -> torch.Tensor:
    """NVFP4 fake quantization with a clamp-masked straight-through backward.

    Args:
        x: tensor of shape ``[..., N]`` with ``N % 16 == 0``.
        global_amax: calibrated per-tensor amax (activations); ``None``
            derives it from ``x`` on the fly (weights).
    """
    return _NVFP4FakeQuantSTE.apply(x, global_amax)
