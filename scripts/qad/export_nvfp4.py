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

"""Export a QAD-trained latent checkpoint to a deployable NVFP4 checkpoint.

Two-stage flow (stage A runs in the VeOmni venv, stage B in the
Model-Optimizer venv — modelopt and TRT-LLM export live there)::

    # A. materialize latent bf16 HF weights from the DCP checkpoint
    .venv/bin/python scripts/merge_dcp_to_hf.py \
        --load-dir <output_dir>/checkpoints/global_step_N \
        --model-assets-dir <output_dir>/model_assets \
        --save-dir <work>/hf_latent

    # B. strip QAD buffers, PTQ-quantize with the deployment rule, export
    <Model-Optimizer>/.venv/bin/python scripts/qad/export_nvfp4.py \
        --hf-latent <work>/hf_latent \
        --calib-data <qad-split>/train \
        --export-dir <work>/nvfp4_ckpt \
        [--reuse-trained-act-amax]

Scale conventions (tony-note-plan.md §1.5/§6):
- Weight scales are recomputed from the FINAL latent weights with the same
  rule training simulated every step — training/export consistency is
  automatic.
- Activation global scales: fresh max-calibration on the final model by
  default (required for w4/a4). ``--reuse-trained-act-amax`` instead injects
  the ``act_global_amax`` buffers the QAD trainer calibrated and froze
  before training (exact w4a4 fidelity: deploy the very grid training
  simulated).
- Evaluate the exported checkpoint with the REAL w4a4 kernel on Blackwell;
  this box (Hopper) only validates the fake-quant simulation.
"""

import argparse
import glob
import json
import os


def strip_qad_buffers(hf_dir: str, work_dir: str) -> dict:
    """Copy HF checkpoint without FakeQuantLinear buffers; return their values.

    The DCP checkpoint (and thus the merged safetensors) carries one
    ``...act_global_amax`` scalar per wrapped linear. They are not model
    weights — strip them so ``from_pretrained`` sees a clean state dict, and
    keep the values for optional reinjection into modelopt's quantizers.
    """
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(work_dir, exist_ok=True)
    amax_values = {}
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    weight_map_updates = {}

    for shard in sorted(glob.glob(os.path.join(hf_dir, "*.safetensors"))):
        tensors = {}
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.endswith("act_global_amax"):
                    amax_values[key] = f.get_tensor(key).item()
                else:
                    tensors[key] = f.get_tensor(key)
        out_shard = os.path.join(work_dir, os.path.basename(shard))
        save_file(tensors, out_shard, metadata={"format": "pt"})
        for key in tensors:
            weight_map_updates[key] = os.path.basename(shard)

    for aux in glob.glob(os.path.join(hf_dir, "*")):
        base = os.path.basename(aux)
        if base.endswith(".safetensors"):
            continue
        if base == "model.safetensors.index.json" and os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            index["weight_map"] = {k: v for k, v in index["weight_map"].items() if not k.endswith("act_global_amax")}
            with open(os.path.join(work_dir, base), "w") as f:
                json.dump(index, f, indent=2)
            continue
        if os.path.isfile(aux):
            shutil.copy2(aux, os.path.join(work_dir, base))

    print(f"stripped {len(amax_values)} act_global_amax buffers")
    return amax_values


def build_calib_loop(model, tokenizer_dir: str, calib_data: str, calib_samples: int, max_len: int, device):
    """Forward loop over pretokenized calibration samples (max calibrator)."""
    import pyarrow.parquet as pq
    import torch

    files = sorted(glob.glob(os.path.join(calib_data, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files under {calib_data}")
    table = pq.read_table(files[0])

    def forward_loop(mdl):
        with torch.no_grad():
            for i in range(min(calib_samples, table.num_rows)):
                ids = table.column("input_ids")[i].as_py()[:max_len]
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                mdl(input_ids=input_ids, use_cache=False)

    return forward_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-latent", required=True, help="Merged bf16 HF checkpoint (stage A output).")
    parser.add_argument("--calib-data", required=True, help="Directory of pretokenized parquet for calibration.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument(
        "--reuse-trained-act-amax",
        action="store_true",
        help="Inject the QAD-calibrated act_global_amax buffers instead of the fresh calibration values "
        "(w4a4 runs: deploys the exact grid training simulated).",
    )
    args = parser.parse_args()

    import modelopt.torch.quantization as mtq
    import torch
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM

    work_dir = os.path.join(args.export_dir, "_hf_stripped")
    amax_values = strip_qad_buffers(args.hf_latent, work_dir)

    model = AutoModelForCausalLM.from_pretrained(work_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    forward_loop = build_calib_loop(
        model, args.hf_latent, args.calib_data, args.calib_samples, args.max_len, device="cuda"
    )
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    if args.reuse_trained_act_amax:
        if not amax_values:
            raise SystemExit("--reuse-trained-act-amax set but the checkpoint carries no act_global_amax buffers.")
        injected = 0
        for name, module in model.named_modules():
            if hasattr(module, "input_quantizer"):
                key = f"{name}.act_global_amax"
                if key in amax_values:
                    module.input_quantizer.amax = torch.tensor(amax_values[key], device="cuda")
                    injected += 1
        print(f"injected {injected}/{len(amax_values)} trained activation amax values")
        if injected == 0:
            raise SystemExit("No quantizer names matched the trained buffers — check module naming.")

    export_hf_checkpoint(model, export_dir=args.export_dir)
    with open(os.path.join(args.export_dir, "qad_export_manifest.json"), "w") as f:
        json.dump(
            {
                "hf_latent": os.path.abspath(args.hf_latent),
                "calib_data": os.path.abspath(args.calib_data),
                "calib_samples": args.calib_samples,
                "reuse_trained_act_amax": args.reuse_trained_act_amax,
                "num_trained_amax_buffers": len(amax_values),
                "quant_cfg": "NVFP4_DEFAULT_CFG",
            },
            f,
            indent=2,
        )
    print(f"NVFP4 checkpoint exported to {args.export_dir}")


if __name__ == "__main__":
    main()
