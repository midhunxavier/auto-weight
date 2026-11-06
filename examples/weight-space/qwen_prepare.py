"""Prepare the published two-tensor Qwen patch against the exact original weights."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from auto_weight.checkpoint import RAW_HF_ADAPTER, hf_files

from assets import asset
from prepare import HERE, prepare_example


def patch_for(base):
    root = "exp8_math/runs/p50c/edit"
    manifest = json.loads(asset(f"{root}/manifest.json"))
    patch = {}
    with np.load(BytesIO(asset(f"{root}/edit.npz")), allow_pickle=False) as arrays:
        for index, reference in enumerate(manifest["tensors"]):
            name = reference["name"]
            if reference["form"] != "sparse":
                raise ValueError("expected the published sparse Qwen edit")
            for file, path in hf_files(base).items():
                if file.endswith(".safetensors"):
                    with safe_open(path, framework="pt") as weights:
                        if name in weights.keys():
                            tensor = weights.get_tensor(name)
                            break
            else:
                raise ValueError(f"checkpoint lacks {name}")
            if tensor.dtype != torch.bfloat16 or list(tensor.shape) != reference["shape"]:
                raise ValueError(f"published patch layout differs for {name}")
            bits = tensor.view(torch.uint16).numpy().copy()
            if sha256(bits.tobytes()).hexdigest() != reference["base_sha256"]:
                raise ValueError(f"published base hash differs for {name}")
            bits.ravel()[arrays[f"t{index}_idx"]] = arrays[f"t{index}_val"]
            if sha256(bits.tobytes()).hexdigest() != reference["edited_sha256"]:
                raise ValueError(f"published edited hash differs for {name}")
            patch[name] = torch.from_numpy(bits).view(torch.bfloat16).float().numpy()
    return patch


def load_cases(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def smoke_cases():
    splits = {}
    for stage, offset in (("development", 0), ("confirmation", 64)):
        splits[stage] = [
            {"id": f"smoke-{index}", "group": "fix" if index % 2 == 0 else "keep",
             "inputs": {"problem": f"What is {index} plus 1?", "answer": str(index + 1)}}
            for index in range(offset, offset + 64)
        ]
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if (args.development is None) != (args.confirmation is None):
        parser.error("supply both benchmark splits or neither for smoke cases")
    splits = (smoke_cases() if args.development is None else
              {"development": load_cases(args.development), "confirmation": load_cases(args.confirmation)})
    if not 1 <= args.max_tokens <= 32768 or not 1 <= args.batch_size <= 64:
        parser.error("invalid token or batch limit")
    seeds = args.seed or [20261103]
    base = args.base.resolve()
    prepare_example(args.output, title="Replay the published Qwen loop patch", adapter=RAW_HF_ADAPTER,
                    evaluator=HERE / "qwen_eval.py", interpreter=args.python, base=base, patch=patch_for(base),
                    cases=splits, config={"device": args.device, "max_tokens": args.max_tokens,
                                          "batch_size": args.batch_size},
                    metrics={"accuracy": 1, "cap_hit": -1, "tokens": -1}, seeds=seeds,
                    max_work=max(map(len, splits.values())) * args.max_tokens * len(seeds), timeout=args.timeout,
                    gates=[{"group": "fix", "metric": "cap_hit", "kind": "improve", "min_delta": 0.05},
                           {"group": "keep", "metric": "accuracy", "kind": "preserve", "min_delta": -0.03}])


if __name__ == "__main__":
    main()
