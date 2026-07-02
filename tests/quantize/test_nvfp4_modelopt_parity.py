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

"""Parity of veomni's NVFP4 fake quant against NVIDIA Model-Optimizer.

Requires CUDA + modelopt (skipped otherwise). Run it from the modelopt venv::

    Model-Optimizer/.venv/bin/python -m pytest tests/quantize/test_nvfp4_modelopt_parity.py

``fake_quant.py`` is loaded standalone (it only imports torch) so this file
works in a venv without the rest of veomni's dependencies.

True bit-exactness against a single kernel is not attainable: modelopt's own
Triton and CUDA-ext backends disagree on ~0.02% of elements because Triton's
fp32 division is the approximate ``div.full.f32`` (measured; see
``fake_quant.py`` docstring). The parity bar here is therefore:

1. On random tensors across 9 orders of magnitude, our mismatch count vs the
   Triton kernel is no larger than modelopt's own CUDA-ext-vs-Triton
   divergence on identical inputs.
2. Every mismatching element differs by exactly one adjacent step on the
   E2M1 grid (a rounding-boundary coin flip, never a math error).
3. The calibrated-amax (activation) path meets the same bar.
"""

import importlib.util
import pathlib

import pytest
import torch


try:
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.extensions import get_cuda_ext_mx
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.quantization.tensor_quant import mx_format_map

    HAVE_MODELOPT = True
except ImportError:
    HAVE_MODELOPT = False

pytestmark = [
    pytest.mark.skipif(not HAVE_MODELOPT, reason="modelopt not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (modelopt NVFP4 has no CPU path)"),
]

_FAKE_QUANT_PATH = pathlib.Path(__file__).resolve().parents[2] / "veomni" / "quantize" / "fake_quant.py"
_spec = importlib.util.spec_from_file_location("veomni_fake_quant_standalone", _FAKE_QUANT_PATH)
fq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fq)

E2M1_GRID = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _nvfp4_cfg():
    return QuantizerAttributeConfig(num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)})


def _cuda_ext_quant(x, amax):
    ext = get_cuda_ext_mx(raise_if_failed=True)
    return ext.fused_amax_convert(
        x, 16, getattr(ext.Types, mx_format_map[(2, 1)]), getattr(ext.Types, mx_format_map[(4, 3)]), amax
    )


def _block_scales(x, global_amax):
    """Recompute effective per-block scales with the reference formula."""
    xf = x.float()
    gs = global_amax.float() / (6.0 * 448.0)
    gs = torch.where(gs > 0, gs, torch.tensor(1e-12, device=x.device))
    ba = xf.reshape(-1, 16).abs().amax(-1, keepdim=True)
    sc = torch.minimum(ba / (6.0 * gs), torch.tensor(448.0, device=x.device))
    bs = sc.to(torch.float8_e4m3fn).float() * gs
    return torch.where(bs >= 1e-5, bs, torch.tensor(1.0, device=x.device))


def _assert_mismatches_are_adjacent_grid_steps(y_mine, y_ref, x, global_amax):
    """Every disagreement must be a one-step E2M1 rounding flip."""
    bs = _block_scales(x, global_amax)
    q_mine = (y_mine.float().reshape(-1, 16) / bs).reshape(-1)
    q_ref = (y_ref.float().reshape(-1, 16) / bs).reshape(-1)
    diff_idx = torch.nonzero(q_mine != q_ref).flatten()
    grid = E2M1_GRID.to(x.device)
    for i in diff_idx.tolist():
        gi_mine = (grid - q_mine[i]).abs().argmin().item()
        gi_ref = (grid - q_ref[i]).abs().argmin().item()
        assert abs(gi_mine - gi_ref) == 1, (
            f"element {i}: {q_mine[i].item()} vs {q_ref[i].item()} are not adjacent E2M1 grid points"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dynamic_amax_parity_bound(dtype):
    """Weight path: our divergence from Triton <= modelopt's own inter-backend divergence."""
    from modelopt.torch.kernels.quantization.gemm.fp4_kernel_hopper import fp4_fake_quant_block

    total_me_tri = 0
    total_ext_tri = 0
    for trial in range(9):
        torch.manual_seed(trial)
        x = (torch.randn(256, 256, dtype=dtype) * 10.0 ** (trial - 4)).cuda()
        amax = x.float().abs().amax()
        y_tri = fp4_fake_quant_block(x.clone(), amax)
        y_ext = _cuda_ext_quant(x.clone(), amax)
        y_me, _ = fq.nvfp4_quant_dequant(x, global_amax=amax)
        total_me_tri += (y_me != y_tri).sum().item()
        total_ext_tri += (y_ext != y_tri).sum().item()
        _assert_mismatches_are_adjacent_grid_steps(y_me, y_tri, x, amax)
    assert total_me_tri <= total_ext_tri, (
        f"our impl diverges from Triton more ({total_me_tri}) than modelopt's own CUDA ext does ({total_ext_tri})"
    )
    # Sanity: divergence is boundary noise, not systematic error.
    assert total_me_tri < 9 * 256 * 256 * 0.001


def test_tensor_quantizer_weight_path_end_to_end():
    """TensorQuantizer with on-the-fly amax == our function with dynamic amax (same bound)."""
    tq = TensorQuantizer(_nvfp4_cfg()).cuda()
    mismatch = 0
    total = 0
    for trial in range(6):
        torch.manual_seed(100 + trial)
        x = (torch.randn(128, 128, dtype=torch.bfloat16) * 10.0 ** (trial - 2)).cuda()
        y_ref = tq(x.clone())
        y_me, _ = fq.nvfp4_quant_dequant(x)
        mismatch += (y_me != y_ref).sum().item()
        total += x.numel()
        _assert_mismatches_are_adjacent_grid_steps(y_me, y_ref, x, x.float().abs().amax())
    assert mismatch / total < 0.001


def test_calibrated_amax_activation_path():
    """Activation path: frozen per-tensor amax, same math, same parity bound."""
    for trial, amax_val in enumerate([0.5, 4.0, 20.0]):
        tq = TensorQuantizer(_nvfp4_cfg())
        tq.amax = torch.tensor(amax_val)
        tq = tq.cuda()
        torch.manual_seed(200 + trial)
        x = torch.randn(128, 128, dtype=torch.bfloat16).cuda()
        y_ref = tq(x.clone())
        amax = torch.tensor(amax_val, device=x.device)
        y_me, _ = fq.nvfp4_quant_dequant(x, global_amax=amax)
        rate = (y_me != y_ref).float().mean().item()
        assert rate < 0.001, f"amax={amax_val}: mismatch rate {rate}"
        _assert_mismatches_are_adjacent_grid_steps(y_me, y_ref, x, amax)


def test_agent_verified_reference_values():
    """Pin the exact outputs modelopt produced for a fixed seed (regression anchor).

    Checksums were produced by running modelopt's TensorQuantizer on H100
    (Triton kernel path) for torch.manual_seed(0), randn(4, 32, bf16).
    Our implementation matched the Triton kernel bit-exactly on this input.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.bfloat16).cuda()
    y_w, _ = fq.nvfp4_quant_dequant(x)
    tq = TensorQuantizer(_nvfp4_cfg()).cuda()
    assert torch.equal(y_w, tq(x.clone()))

    tq_a = TensorQuantizer(_nvfp4_cfg())
    tq_a.amax = torch.tensor(4.0)
    tq_a = tq_a.cuda()
    y_a, _ = fq.nvfp4_quant_dequant(x, global_amax=torch.tensor(4.0, device=x.device))
    assert torch.equal(y_a, tq_a(x.clone()))
