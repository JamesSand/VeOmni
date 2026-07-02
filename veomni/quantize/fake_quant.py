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

"""NVFP4 fake quantization: quantize→dequantize math matching modelopt/TRT-LLM.

NVFP4 stores values as FP4 E2M1 (representable magnitudes {0, 0.5, 1, 1.5, 2,
3, 4, 6}) with a two-level scale: one FP8-E4M3 scale per 16-element block
along the last (contraction) dimension, plus one FP32 scale per tensor. The
math below mirrors NVIDIA Model-Optimizer's reference kernels
(``modelopt/torch/kernels/quantization/common/nvfp4_quant.py``, the declared
"single source of truth for FP4 decision-boundary rounding") step for step:

1. ``global_scale = global_amax / (6 * 448)`` in fp32 (``1e-12`` floor).
2. Per block: ``block_scale = e4m3(min(block_amax / (6 * global_scale), 448))
   * global_scale`` — the block scale itself round-trips through FP8 E4M3
   (double quantization). Block scales below ``1e-5`` fall back to ``1.0``.
3. ``q = round_e2m1(|x| / block_scale)`` with round-to-nearest-even decision
   boundaries; output is ``sign(x) * q * block_scale`` cast back to the input
   dtype.

Exact bit-parity with any single kernel is not attainable — modelopt's own
Triton and CUDA-ext backends disagree on ~0.02% of elements (values within
1 ulp of a rounding boundary, because Triton's fp32 division is the
approximate ``div.full.f32``). This fp32-IEEE implementation lands closer to
the Triton kernel than modelopt's CUDA ext does; the parity test in
``tests/quantize/test_nvfp4_modelopt_parity.py`` pins that bound.

This module intentionally imports nothing beyond ``torch`` so it can be
loaded standalone (e.g. from the modelopt venv for parity testing).
"""

from typing import Optional, Tuple

import torch


NVFP4_BLOCK_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# Guards mirrored from modelopt's kernels: a non-positive global scale falls
# back to 1e-12; a quantized block scale below 1e-5 falls back to 1.0.
_GLOBAL_SCALE_FLOOR = 1e-12
_BLOCK_SCALE_FLOOR = 1e-5


def _fp4_round_magnitude(abs_scaled: torch.Tensor) -> torch.Tensor:
    """Round ``|x| / scale`` to the nearest FP4 (E2M1) magnitude.

    Decision boundaries replicate modelopt's ``fp4_round_magnitude`` exactly,
    including the alternating <=/< comparisons that implement
    round-to-nearest-even at the midpoints (0.25→0, 0.75→1, 1.25→1, 1.75→2,
    2.5→2, 3.5→4, 5→4).
    """
    six = abs_scaled.new_tensor(E2M1_MAX)
    return torch.where(
        abs_scaled <= 0.25,
        0.0,
        torch.where(
            abs_scaled < 0.75,
            0.5,
            torch.where(
                abs_scaled <= 1.25,
                1.0,
                torch.where(
                    abs_scaled < 1.75,
                    1.5,
                    torch.where(
                        abs_scaled <= 2.5,
                        2.0,
                        torch.where(abs_scaled < 3.5, 3.0, torch.where(abs_scaled <= 5.0, 4.0, six)),
                    ),
                ),
            ),
        ),
    )


def nvfp4_global_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor absolute maximum in fp32 (the second-level scale source)."""
    return x.detach().float().abs().amax()


def nvfp4_dynamic_block_amax(x: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> torch.Tensor:
    """Per-block absolute maximum over the last dim, shape ``[..., n_blocks, 1]``."""
    xf = x.detach().float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block_size)
    return blocks.abs().amax(dim=-1, keepdim=True)


def nvfp4_quant_dequant(
    x: torch.Tensor,
    global_amax: Optional[torch.Tensor] = None,
    return_saturation_mask: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """NVFP4 fake quantization (quantize→dequantize) of ``x``.

    Args:
        x: input of shape ``[..., N]`` with ``N % 16 == 0``. Any float dtype.
        global_amax: per-tensor amax for the second-level scale. ``None``
            (weights) computes it from ``x`` on the fly. Note this is a
            deliberate deviation from modelopt QAT, which freezes the weight
            amax at calibration time and exports that frozen value: we
            recompute per forward AND export by PTQ-ing the final latent
            weights, so training and export stay consistent as weights drift.
            Pass a calibrated amax for activations (their global scale must
            be frozen, mirroring the TRT-LLM kernel contract).
        return_saturation_mask: also return a bool mask of elements whose
            magnitude exceeded the representable ceiling ``global_amax``
            (the block-scale clamp at 448 makes ``±6·448·global_scale =
            ±global_amax`` the top of the grid). The STE zeroes gradients of
            these truly-clamped elements. Deliberately NOT ``abs_scaled > 6``:
            E4M3 rounds block scales down by up to ~3%, which pushes each
            block's max element marginally past 6 — masking those would
            systematically zero gradients of every block's largest weight.
            This matches modelopt's clip-mask semantics
            (``pass_through_bwd=False``: mask where ``|x| > amax``).

    Returns:
        ``(dequantized, saturation_mask)`` — ``dequantized`` has ``x``'s shape
        and dtype; ``saturation_mask`` is ``None`` unless requested.

    This function is pure (no autograd); use
    :func:`veomni.quantize.nvfp4_fake_quant_ste` inside training graphs.
    """
    if x.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires the last dim to be a multiple of {NVFP4_BLOCK_SIZE}, got shape {tuple(x.shape)}"
        )

    orig_dtype = x.dtype
    xf = x.float()

    if global_amax is None:
        global_amax = xf.abs().amax()
    global_amax = global_amax.detach().float().reshape(())
    # Scalar-overload torch.where keeps the kernel's exact semantics: positive
    # values below the floor pass through unchanged (clamp(min=...) would not).
    global_scale = global_amax / (E2M1_MAX * E4M3_MAX)
    global_scale = torch.where(global_scale > 0.0, global_scale, _GLOBAL_SCALE_FLOOR)

    blocks = xf.reshape(*xf.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)

    # Double quantization: the block scale itself round-trips through E4M3.
    scale_in_fp8_range = (block_amax / (E2M1_MAX * global_scale)).clamp(max=E4M3_MAX)
    block_scale = scale_in_fp8_range.to(torch.float8_e4m3fn).float() * global_scale
    block_scale = torch.where(block_scale >= _BLOCK_SCALE_FLOOR, block_scale, 1.0)

    abs_scaled = blocks.abs() / block_scale
    q = _fp4_round_magnitude(abs_scaled)
    rescaled = q * block_scale
    out = torch.where(blocks >= 0, rescaled, -rescaled).reshape(x.shape).to(orig_dtype)

    if not return_saturation_mask:
        return out, None
    saturation_mask = xf.abs() > global_amax
    return out, saturation_mask


def nvfp4_weight_overflow_ratio(weight: torch.Tensor) -> float:
    """Fraction of weight elements pushed past the top FP4 bin (``abs_scaled > 6``).

    With a dynamic per-tensor amax nothing exceeds the representable ceiling,
    so the STE saturation mask is empty by construction — the meaningful
    weight-health statistic is instead *pre-round overflow*: elements whose
    ``|w| / block_scale`` lands above 6 because the E4M3 cast rounded their
    block scale down. Rising overflow means block maxima are drifting away
    from E4M3-representable scales. Computed on demand (depends only on the
    weight), so it costs nothing on the training hot path.
    """
    with torch.no_grad():
        xf = weight.detach().float()
        global_scale = xf.abs().amax() / (E2M1_MAX * E4M3_MAX)
        global_scale = torch.where(global_scale > 0.0, global_scale, _GLOBAL_SCALE_FLOOR)
        blocks = xf.reshape(*xf.shape[:-1], -1, NVFP4_BLOCK_SIZE)
        block_amax = blocks.abs().amax(dim=-1, keepdim=True)
        scale_in_fp8_range = (block_amax / (E2M1_MAX * global_scale)).clamp(max=E4M3_MAX)
        block_scale = scale_in_fp8_range.to(torch.float8_e4m3fn).float() * global_scale
        block_scale = torch.where(block_scale >= _BLOCK_SCALE_FLOOR, block_scale, 1.0)
        overflow = (blocks.abs() / block_scale) > E2M1_MAX
        return overflow.float().mean().item()
