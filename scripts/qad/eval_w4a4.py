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

"""Unified-w4a4 PPL evaluation for the QAD three-point comparison.

Loads a bf16 latent HF checkpoint, optionally wraps its linears with NVFP4
fake quantization (w4a4 by default), calibrates activation scales, and
measures PPL on the fixed eval subset — the same simulation criterion for
every comparison point (teacher / PTQ-W0 / w4a4-QAD / w4-QAD+A-PTQ), single
GPU, plain transformers forward.

IMPORTANT: numbers from this script are only comparable to each other (it
uses HF sdpa attention and per-sample forwards); do not mix them with the
training-log eval numbers (veomni flash-attention path).

Examples::

    # teacher (no quantization)
    python scripts/qad/eval_w4a4.py --hf-latent <Qwen3-8B> --mode none ...
    # naive PTQ of the original checkpoint
    python scripts/qad/eval_w4a4.py --hf-latent <Qwen3-8B> --mode w4a4 ...
    # QAD checkpoint, activation grid exactly as trained
    python scripts/qad/eval_w4a4.py --hf-latent <export>/hf_latent --mode w4a4 \
        --use-checkpoint-amax ...
"""

import argparse
import glob
import json
import math
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def load_checkpoint_amax(hf_dir: str) -> dict:
    from safetensors import safe_open

    amax = {}
    for shard in sorted(glob.glob(os.path.join(hf_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.endswith("act_global_amax"):
                    amax[key.removeprefix("model.").removesuffix(".act_global_amax")] = f.get_tensor(key).item()
    return amax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-latent", required=True)
    parser.add_argument("--mode", choices=("none", "w4", "w4a4", "a4"), default="w4a4")
    parser.add_argument("--calib-data", default=None, help="Train-split parquet dir (needed for w4a4/a4).")
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--eval-data", required=True, help="Eval parquet dir (fixed subset).")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument(
        "--use-checkpoint-amax",
        action="store_true",
        help="Use the act_global_amax buffers stored in the checkpoint (QAD-trained grid) instead of fresh calib.",
    )
    parser.add_argument("--tag", default=None, help="Label for the result line.")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    from veomni.quantize import QADQuantConfig, iter_fake_quant_linears, wrap_linears_for_qad

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_latent, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()

    if args.mode != "none":
        wrapped = wrap_linears_for_qad(model, QADQuantConfig(mode=args.mode))
        print(f"wrapped {wrapped} linears (mode={args.mode})", file=sys.stderr)

        needs_act = args.mode in ("w4a4", "a4")
        if needs_act:
            if args.use_checkpoint_amax:
                stored = load_checkpoint_amax(args.hf_latent)
                if not stored:
                    raise SystemExit("--use-checkpoint-amax: checkpoint carries no act_global_amax buffers.")
                hit = 0
                for name, module in iter_fake_quant_linears(model):
                    key = name.removeprefix("model.")
                    if key in stored:
                        module.act_global_amax.fill_(stored[key])
                        hit += 1
                if hit != wrapped:
                    raise SystemExit(f"amax injection matched {hit}/{wrapped} modules — name mismatch.")
                print(f"injected {hit} trained activation amax values", file=sys.stderr)
            else:
                if not args.calib_data:
                    raise SystemExit(f"mode={args.mode} needs --calib-data (or --use-checkpoint-amax).")
                from veomni.quantize import finalize_calibration, start_calibration

                files = sorted(glob.glob(os.path.join(args.calib_data, "*.parquet")))
                table = pq.read_table(files[0])
                start_calibration(model)
                with torch.no_grad():
                    for i in range(min(args.calib_samples, table.num_rows)):
                        ids = torch.tensor(
                            [table.column("input_ids")[i].as_py()[: args.max_len]], dtype=torch.long, device="cuda"
                        )
                        model(input_ids=ids, use_cache=False)
                finalize_calibration(model)
                print(f"fresh-calibrated on {args.calib_samples} samples", file=sys.stderr)

    eval_files = sorted(glob.glob(os.path.join(args.eval_data, "*.parquet")))
    table = pq.read_table(eval_files)
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(table.num_rows):
            input_ids = torch.tensor(
                [table.column("input_ids")[i].as_py()[: args.max_len]], dtype=torch.long, device="cuda"
            )
            labels = torch.tensor([table.column("labels")[i].as_py()[: args.max_len]], dtype=torch.long, device="cuda")
            n_valid = int((labels[..., 1:] != -100).sum())
            if n_valid == 0:
                continue
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            total_nll += float(out.loss) * n_valid
            total_tokens += n_valid

    ppl = math.exp(total_nll / total_tokens)
    result = {
        "tag": args.tag or os.path.basename(args.hf_latent.rstrip("/")),
        "mode": args.mode,
        "act_scale_source": (
            "none" if args.mode in ("none", "w4") else ("checkpoint" if args.use_checkpoint_amax else "fresh_calib")
        ),
        "ppl": round(ppl, 6),
        "eval_tokens": total_tokens,
        "eval_samples": table.num_rows,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
