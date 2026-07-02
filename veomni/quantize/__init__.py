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

"""Quantization-aware distillation (QAD) building blocks.

NVFP4 fake quantization (FP4 E2M1 values with per-16-element FP8-E4M3 block
scales and a per-tensor FP32 global scale), a straight-through estimator with
clamp masking, and an ``nn.Linear`` wrapper that applies weight and/or
activation fake quantization matching the TensorRT-LLM NVFP4 inference path.
"""

from .calibrate import finalize_calibration, start_calibration
from .config import QADQuantConfig
from .fake_quant import (
    NVFP4_BLOCK_SIZE,
    nvfp4_dynamic_block_amax,
    nvfp4_global_amax,
    nvfp4_quant_dequant,
    nvfp4_weight_overflow_ratio,
)
from .modules import (
    FakeQuantLinear,
    iter_fake_quant_linears,
    set_quantizer_enabled,
    wrap_linears_for_qad,
)
from .ste import nvfp4_fake_quant_ste
