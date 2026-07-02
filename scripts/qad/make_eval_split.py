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

"""Create a shard-level train/eval split for QAD training.

Holds out ONE parquet shard from a pretokenized dataset directory and fixes a
small, versioned eval subset inside it:

- ``<output>/train/``: symlinks to every shard except the held-out one.
- ``<output>/eval/eval-v<version>.parquet``: fixed-seed sample of the held-out
  shard (default 2000 rows ≈ 7M tokens — PPL sampling error ~±0.1%, two
  orders of magnitude below the QAD-vs-PTQ effects being measured).
- ``<output>/SPLIT.json``: manifest pinning shard, seed, and row indices hash
  so the same eval set is reused across the FP-baseline / PTQ / QAD
  comparison points (paired measurement).

Shard-level (not row-level) split: rows from one generation run are adjacent,
so row sampling would leak near-duplicates across the boundary.

Usage::

    python scripts/qad/make_eval_split.py \
        --dataset-dir .../tokenized_default-Qwen3-8B \
        --output-dir .../qad-split-Qwen3-8B
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Directory of pretokenized parquet shards.")
    parser.add_argument("--output-dir", required=True, help="Where to create train/ and eval/ splits.")
    parser.add_argument("--eval-shard", default=None, help="Shard filename to hold out (default: last shard).")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    shards = sorted(dataset_dir.glob("*.parquet"))
    if len(shards) < 2:
        raise SystemExit(f"Need at least 2 shards to split, found {len(shards)} in {dataset_dir}")

    eval_shard = Path(dataset_dir / args.eval_shard) if args.eval_shard else shards[-1]
    if eval_shard not in shards:
        raise SystemExit(f"Eval shard {eval_shard} not found in {dataset_dir}")

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    n_links = 0
    for shard in shards:
        if shard == eval_shard:
            continue
        link = train_dir / shard.name
        if not link.exists():
            link.symlink_to(shard)
        n_links += 1

    table = pq.read_table(eval_shard)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(table.num_rows), min(args.eval_samples, table.num_rows)))
    eval_table = table.take(indices)
    eval_file = eval_dir / f"eval-v{args.version}.parquet"
    pq.write_table(eval_table, eval_file)

    n_tokens = sum(len(row) for row in eval_table.column("input_ids").to_pylist())
    manifest = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir),
        "eval_shard": eval_shard.name,
        "seed": args.seed,
        "num_eval_samples": len(indices),
        "num_eval_tokens": n_tokens,
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "num_train_shards": n_links,
    }
    (output_dir / "SPLIT.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"\ntrain: {train_dir}  ({n_links} shards, symlinked)")
    print(f"eval : {eval_file}  ({len(indices)} rows, {n_tokens / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
