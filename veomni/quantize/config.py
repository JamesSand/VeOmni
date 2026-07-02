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

"""Internal quantization config consumed by :mod:`veomni.quantize`.

The NVFP4 format parameters (E2M1 values, block size 16, E4M3 block scales,
FP32 global scale) are constants of the deployment format, not configuration:
exposing them would only create opportunities to diverge from what TensorRT-LLM
serves. Only the *training-mode* choices live here.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class QADQuantConfig:
    """Which tensors get fake-quantized during QAD training.

    ``mode`` selects the training simulation (the deployment target is always
    w4a4 NVFP4):

    - ``"w4"``: weights only.
    - ``"w4a4"``: weights and activations.
    - ``"a4"``: activations only (weights stay full precision).
    """

    mode: Literal["w4", "w4a4", "a4"] = "w4a4"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    @property
    def quantize_weights(self) -> bool:
        return self.mode in ("w4", "w4a4")

    @property
    def quantize_activations(self) -> bool:
        return self.mode in ("w4a4", "a4")

    def __post_init__(self):
        if self.mode not in ("w4", "w4a4", "a4"):
            raise ValueError(f"Unknown QAD mode: {self.mode!r} (expected 'w4', 'w4a4' or 'a4')")
