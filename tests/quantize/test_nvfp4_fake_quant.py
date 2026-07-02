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

"""Spec tests for NVFP4 fake quantization, STE, and FakeQuantLinear.

CPU-only (no modelopt, no GPU) so they run in the default CI matrix. Parity
against modelopt kernels lives in ``test_nvfp4_modelopt_parity.py``.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from veomni.quantize import (
    FakeQuantLinear,
    QADQuantConfig,
    finalize_calibration,
    iter_fake_quant_linears,
    nvfp4_fake_quant_ste,
    nvfp4_quant_dequant,
    start_calibration,
    wrap_linears_for_qad,
)


E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _unit_scale_block(values):
    """Build a 1x16 tensor whose block scale is exactly 1.0.

    With a single block whose amax is exactly 6.0: global_scale = 6/(6*448) =
    1/448, scale_in_fp8_range = 6/(6/448) = 448 → e4m3(448) = 448 →
    block_scale = 448 * (1/448) = 1.0 exactly. Values then quantize on the
    raw E2M1 grid, which lets us pin decision boundaries precisely.
    """
    base = [6.0] + list(values)
    base += [0.0] * (16 - len(base))
    return torch.tensor([base], dtype=torch.float32)


class TestQuantDequantSpec:
    def test_outputs_on_e2m1_grid_unit_scale(self):
        x = _unit_scale_block([0.1, 0.4, 0.6, 1.1, 1.6, 2.2, 2.8, 3.7, 4.5, 5.5, -2.9, -0.3])
        out, _ = nvfp4_quant_dequant(x)
        for v in out.flatten().tolist():
            assert abs(v) in E2M1_GRID

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Exact midpoints: round-to-nearest-even on the E2M1 grid.
            (0.25, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            # Non-midpoints round to nearest.
            (0.26, 0.5),
            (0.74, 0.5),
            (2.4, 2.0),
            (2.6, 3.0),
            (5.01, 6.0),
        ],
    )
    def test_rne_decision_boundaries(self, value, expected):
        x = _unit_scale_block([value, -value])
        out, _ = nvfp4_quant_dequant(x)
        assert out[0, 1].item() == expected
        assert out[0, 2].item() == -expected

    def test_zeros_and_sign(self):
        x = torch.zeros(4, 32)
        out, mask = nvfp4_quant_dequant(x, return_saturation_mask=True)
        assert torch.equal(out, x)
        assert not mask.any()

        x = _unit_scale_block([3.2, -3.2])
        out, _ = nvfp4_quant_dequant(x)
        assert out[0, 1].item() == -out[0, 2].item()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_preserved(self, dtype):
        x = torch.randn(8, 64, dtype=dtype)
        out, _ = nvfp4_quant_dequant(x)
        assert out.dtype == dtype
        assert out.shape == x.shape

    def test_rejects_bad_last_dim(self):
        with pytest.raises(ValueError, match="multiple of 16"):
            nvfp4_quant_dequant(torch.randn(4, 30))

    def test_double_quantization_of_block_scale(self):
        # Two blocks: amax 6.0 (scale hits 448, exactly representable) and
        # amax 4.3 (scale 321.07, NOT E4M3-representable → rounds to 320).
        # If the E4M3 round-trip were skipped, block 2's scale would be
        # 4.3/6 and 4.3 would reconstruct exactly; with double quantization
        # it must land on 6 * (320/448 * ...) != 4.3.
        row = [6.0] + [0.0] * 15 + [4.3] + [0.0] * 15
        x = torch.tensor([row], dtype=torch.float32)
        out, _ = nvfp4_quant_dequant(x)
        got = out[0, 16].item()
        single_level = 4.3  # what a non-double-quantized scale would return
        assert got != pytest.approx(single_level, abs=1e-6)
        # Expected: gs = 1/448; sc = 4.3/(6*gs) = 321.066... → e4m3 → 320;
        # bs = 320/448; q = round(4.3/bs) = round(6.02) = 6; out = 6*320/448.
        assert got == pytest.approx(6.0 * 320.0 / 448.0, rel=1e-6)

    def test_degenerate_scales_produce_finite_output(self):
        for x in [
            torch.zeros(2, 16),
            torch.full((2, 16), 1e-30),
            torch.full((2, 16), 1e30),
            torch.randn(2, 16) * 1e-25,
        ]:
            out, _ = nvfp4_quant_dequant(x)
            assert torch.isfinite(out).all()

    def test_saturation_mask_with_calibrated_amax(self):
        # Calibrated global amax of 6.0, but the tensor contains 100.0:
        # scale_in_fp8_range clamps at 448 → block_scale caps at 1.0*448/448
        # ... elements far beyond 6*block_scale saturate.
        x = torch.tensor([[100.0, 1.0] + [0.5] * 14], dtype=torch.float32)
        out, mask = nvfp4_quant_dequant(x, global_amax=torch.tensor(6.0), return_saturation_mask=True)
        assert mask[0, 0].item() is True or mask[0, 0].item() == 1
        assert not mask[0, 1]
        assert torch.isfinite(out).all()

    def test_weight_dynamic_amax_never_saturates(self):
        # With on-the-fly amax nothing exceeds the ceiling by construction:
        # |x| <= max|x| = global_amax. In particular the block-max elements
        # (whose abs_scaled can exceed 6 after E4M3 rounds the scale down)
        # must NOT be masked — that would zero the gradient of every block's
        # largest weight.
        torch.manual_seed(0)
        x = torch.randn(64, 128)
        _, mask = nvfp4_quant_dequant(x, return_saturation_mask=True)
        assert mask.sum().item() == 0


class TestSTE:
    def test_forward_matches_quant_dequant(self):
        torch.manual_seed(0)
        x = torch.randn(8, 32, requires_grad=True)
        y = nvfp4_fake_quant_ste(x)
        ref, _ = nvfp4_quant_dequant(x.detach())
        assert torch.equal(y.detach(), ref)

    def test_backward_identity_in_range(self):
        torch.manual_seed(1)
        x = torch.randn(8, 32, requires_grad=True)
        g = torch.randn_like(x)
        y = nvfp4_fake_quant_ste(x)
        y.backward(g)
        _, mask = nvfp4_quant_dequant(x.detach(), return_saturation_mask=True)
        expected = g.masked_fill(mask, 0.0)
        assert torch.equal(x.grad, expected)

    def test_backward_zero_at_saturated(self):
        x = torch.tensor([[100.0, 1.0] + [0.5] * 14], dtype=torch.float32, requires_grad=True)
        y = nvfp4_fake_quant_ste(x, global_amax=torch.tensor(6.0))
        y.sum().backward()
        assert x.grad[0, 0].item() == 0.0
        assert x.grad[0, 1].item() == 1.0

    def test_no_grad_to_amax(self):
        x = torch.randn(4, 16, requires_grad=True)
        amax = torch.tensor(6.0, requires_grad=True)
        y = nvfp4_fake_quant_ste(x, amax)
        y.sum().backward()
        assert amax.grad is None


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 64, bias=False)
        self.o_proj = nn.Linear(64, 32, bias=True)
        self.lm_head = nn.Linear(32, 100, bias=False)

    def forward(self, x):
        return self.lm_head(self.o_proj(self.q_proj(x)))


class TestFakeQuantLinear:
    def test_from_linear_preserves_params_and_keys(self):
        lin = nn.Linear(32, 16, bias=True)
        fq = FakeQuantLinear.from_linear(lin, quantize_weights=True, quantize_activations=False)
        assert fq.weight is lin.weight
        assert fq.bias is lin.bias
        keys = set(fq.state_dict().keys())
        assert keys == {"weight", "bias", "act_global_amax"}

    def test_w4_forward_equals_reference(self):
        torch.manual_seed(0)
        lin = nn.Linear(32, 16, bias=True)
        fq = FakeQuantLinear.from_linear(lin, quantize_weights=True, quantize_activations=False)
        x = torch.randn(4, 32)
        qw, _ = nvfp4_quant_dequant(lin.weight.detach())
        assert torch.equal(fq(x), F.linear(x, qw, lin.bias))

    def test_quant_disabled_matches_plain_linear(self):
        torch.manual_seed(0)
        lin = nn.Linear(32, 16)
        fq = FakeQuantLinear.from_linear(lin, quantize_weights=False, quantize_activations=False)
        x = torch.randn(4, 32)
        assert torch.equal(fq(x), lin(x))

    def test_act_quant_requires_calibration(self):
        lin = nn.Linear(32, 16)
        fq = FakeQuantLinear.from_linear(lin, quantize_weights=False, quantize_activations=True)
        with pytest.raises(RuntimeError, match="not a calibrated positive finite value"):
            fq(torch.randn(2, 32))

    def test_calibration_flow(self):
        model = _TinyModel()
        cfg = QADQuantConfig(mode="w4a4", target_modules=["q_proj", "o_proj"])
        n = wrap_linears_for_qad(model, cfg)
        assert n == 2
        start_calibration(model)
        with torch.no_grad():
            for scale in (1.0, 3.0, 2.0):
                model(torch.randn(2, 8, 32) * scale)
        count = finalize_calibration(model)
        assert count == 2
        for _, m in iter_fake_quant_linears(model):
            assert m.act_global_amax.item() > 0.0
            assert not m.act_calibrating
        # act-quantized forward now works and differs from the fp path
        x = torch.randn(2, 8, 32)
        out_q = model(x)
        assert torch.isfinite(out_q).all()

    def test_wrap_targets_and_grad_flow(self):
        model = _TinyModel()
        cfg = QADQuantConfig(mode="w4", target_modules=["q_proj", "o_proj"])
        wrapped = wrap_linears_for_qad(model, cfg)
        assert wrapped == 2
        assert isinstance(model.q_proj, FakeQuantLinear)
        assert isinstance(model.o_proj, FakeQuantLinear)
        assert not isinstance(model.lm_head, FakeQuantLinear)

        out = model(torch.randn(2, 32))
        out.sum().backward()
        assert model.q_proj.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_wrap_raises_on_no_match(self):
        model = _TinyModel()
        cfg = QADQuantConfig(mode="w4", target_modules=["nonexistent_proj"])
        with pytest.raises(ValueError, match="matched no modules"):
            wrap_linears_for_qad(model, cfg)

    def test_wrap_on_meta_model(self):
        with torch.device("meta"):
            model = _TinyModel()
        cfg = QADQuantConfig(mode="w4", target_modules=["q_proj"])
        assert wrap_linears_for_qad(model, cfg) == 1
        assert model.q_proj.weight.is_meta

    def test_weight_overflow_ratio(self):
        torch.manual_seed(0)
        lin = nn.Linear(32, 16)
        fq = FakeQuantLinear.from_linear(lin, quantize_weights=True, quantize_activations=False)
        ratio = fq.weight_overflow_ratio()
        # Pre-round overflow: block-max elements pushed past 6 by E4M3
        # scale round-down. Nonzero on typical random weights (unlike the
        # STE saturation mask, which is empty by construction), bounded by
        # the block-max density 1/16.
        assert 0.0 < ratio < 1.0 / 16 + 0.01

    def test_weight_overflow_zero_for_e4m3_exact_scales(self):
        # amax 6.0 → scale exactly 448 (E4M3-representable) → no round-down
        # → no overflow.
        x = _unit_scale_block([3.0, 1.5, -2.0])
        from veomni.quantize import nvfp4_weight_overflow_ratio

        assert nvfp4_weight_overflow_ratio(x) == 0.0

    def test_rejects_bad_in_features(self):
        with pytest.raises(ValueError, match="multiple of the NVFP4 block size"):
            FakeQuantLinear(30, 16)


class TestQADQuantConfig:
    def test_mode_properties(self):
        assert QADQuantConfig(mode="w4").quantize_weights
        assert not QADQuantConfig(mode="w4").quantize_activations
        assert QADQuantConfig(mode="w4a4").quantize_weights
        assert QADQuantConfig(mode="w4a4").quantize_activations
        assert not QADQuantConfig(mode="a4").quantize_weights
        assert QADQuantConfig(mode="a4").quantize_activations

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown QAD mode"):
            QADQuantConfig(mode="w8")
