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

"""Teacher top-k ↔ chunk_topk_distill alignment.

The one bug class that would silently destroy QAD is an off-by-one between
the teacher's top-k tensors and the kernel's internal causal shift: training
would optimize KL against the *wrong position's* distribution and the loss
would still go down. The identical-teacher test pins this: when the teacher
IS the student (same hidden states, same lm_head), forward KL restricted to
the teacher's top-k must be ~0 at every valid position — any misalignment
inflates it by orders of magnitude (verified: shifting the teacher tensors by
one position turns ~1e-7 into ~1.0+).
"""

import pytest
import torch

from veomni.ops.kernels.cross_entropy import chunk_topk_distill_function
from veomni.trainer.text_qad_trainer import aligned_topk_from_logits
from veomni.utils.constants import IGNORE_INDEX


class _FakePS:
    sp_enabled = False


@pytest.fixture(autouse=True)
def _no_sp(monkeypatch):
    import veomni.ops.kernels.cross_entropy.chunk_logprobs as cl
    import veomni.ops.kernels.cross_entropy.chunk_topk_distill as ctkd

    monkeypatch.setattr(ctkd, "get_parallel_state", lambda: _FakePS())
    # Force the pure-torch path (the FA-CE triton kernel rejects CPU tensors).
    monkeypatch.setattr(cl, "_FA_CE_AVAILABLE", False)
    import veomni.trainer.text_qad_trainer  # noqa: F401  (module import sanity)

    yield


def _setup(seed=0, B=1, L=48, H=32, V=200, K=16, tau=1.0):
    torch.manual_seed(seed)
    hidden = torch.randn(B, L, H, dtype=torch.float32)
    lm_head = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L))
    labels[:, :8] = IGNORE_INDEX  # prompt span
    teacher_logits = hidden @ lm_head.T  # identical teacher
    ids, logps = aligned_topk_from_logits(teacher_logits, topk=K, temperature=tau, chunk_size=16)
    return hidden, lm_head, labels, ids, logps


class TestIdenticalTeacherKLIsZero:
    def test_kl_near_zero(self):
        hidden, lm_head, labels, ids, logps = _setup()
        _, _, distill, student_mass, teacher_mass = chunk_topk_distill_function(
            hidden, lm_head, labels, ids, logps, chunk_size=16
        )
        valid = labels[..., 1:] != IGNORE_INDEX
        max_kl = distill[..., :-1][valid].abs().max().item()
        assert max_kl < 1e-4, f"identical teacher should give ~0 KL, got max {max_kl}"
        # masses match too (same distribution)
        mass_gap = (student_mass - teacher_mass)[..., :-1][valid].abs().max().item()
        assert mass_gap < 1e-4

    def test_misalignment_detected(self):
        """A one-position shift of the teacher tensors must blow the KL up."""
        hidden, lm_head, labels, ids, logps = _setup()
        ids_bad = torch.roll(ids, shifts=1, dims=1)
        logps_bad = torch.roll(logps, shifts=1, dims=1)
        _, _, distill, _, _ = chunk_topk_distill_function(hidden, lm_head, labels, ids_bad, logps_bad, chunk_size=16)
        valid = labels[..., 1:] != IGNORE_INDEX
        mean_kl = distill[..., :-1][valid].mean().item()
        assert mean_kl > 0.1, f"misaligned teacher should give large KL, got {mean_kl}"

    def test_ignore_index_positions_are_zero(self):
        hidden, lm_head, labels, ids, logps = _setup()
        _, _, distill, _, _ = chunk_topk_distill_function(hidden, lm_head, labels, ids, logps, chunk_size=16)
        invalid = labels[..., 1:] == IGNORE_INDEX
        assert (distill[..., :-1][invalid] == 0).all()

    def test_temperature_consistency(self):
        """Teacher extracted at tau and student evaluated at the same tau → still ~0 KL."""
        hidden, lm_head, labels, ids, logps = _setup(tau=2.0)
        _, _, distill, _, _ = chunk_topk_distill_function(
            hidden, lm_head, labels, ids, logps, chunk_size=16, temperature=2.0
        )
        valid = labels[..., 1:] != IGNORE_INDEX
        assert distill[..., :-1][valid].abs().max().item() < 1e-4

    def test_gradient_flows_to_student_only(self):
        hidden, lm_head, labels, ids, logps = _setup()
        hidden = hidden.clone().requires_grad_(True)
        lm_head = lm_head.clone().requires_grad_(True)
        _, _, distill, _, _ = chunk_topk_distill_function(hidden, lm_head, labels, ids, logps, chunk_size=16)
        distill.sum().backward()
        assert hidden.grad is not None
        assert lm_head.grad is not None
        assert ids.grad is None and logps.grad is None
