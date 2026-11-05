"""Recreate the case study inputs from the two original training snapshots."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from weight_edit_toolkit.contracts import file_hash


def write_examples(path, seed, count, config):
    rng = np.random.default_rng(seed)
    vocab, context = config["vocab"], config["context"]
    with path.open("x") as stream:
        for group in ("collision-copy", "unique-copy"):
            for row in range(count):
                tokens = rng.integers(vocab, size=context + 1)
                if group == "collision-copy":
                    length = 40
                    offset = row % 17
                    segment = rng.integers(vocab, size=length)
                else:
                    length = int(rng.integers(16, 33))
                    offset = int(rng.integers(0, context + 2 - 2 * length))
                    segment = rng.permutation(vocab)[:length]
                tokens[offset:offset + 2 * length] = np.tile(segment, 2)
                positions = list(range(offset + length, offset + 2 * length - 1))
                stream.write(json.dumps({"group": group, "tokens": tokens.tolist(), "positions": positions}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="seed 11101 step 1000, full NPZ")
    parser.add_argument("--donor", required=True, type=Path, help="same run at step 500, full NPZ")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    config = {"width": 128, "heads": 4, "layers": 4, "vocab": 32, "context": 96}

    def reference(path):
        return {"path": os.path.relpath(path.resolve(), output), "sha256": file_hash(path)}

    def save(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    base = reference(args.base)
    donor = reference(args.donor)
    with np.load(args.base, allow_pickle=False) as archive:
        tensors = sorted(archive.files)
    proposal = {"version": "wet.proposal.v1", "base_sha256": base["sha256"], "author": "Codex (LLM)"}
    save("restore-output.json", {
        **proposal,
        "hypothesis": "Later training may have shifted the readout away from useful copying representations. "
                      "Restore the earlier readout and final normalization while retaining the current blocks.",
        "operations": [{"op": "restore", "tensors": ["final.weight", "final.bias", "output.weight", "output.bias"],
                        "donor": donor}],
    })
    save("blend-checkpoints.json", {
        **proposal,
        "hypothesis": "Copying may depend on changes spread across several blocks. "
                      "Interpolate halfway toward the earlier same-run checkpoint to recover copying while "
                      "retaining part of the later training progress.",
        "operations": [{"op": "blend", "tensors": tensors, "donor": donor, "alpha": 0.5}],
    })
    development, confirmation = output / "development.jsonl", output / "confirmation.jsonl"
    write_examples(development, 741092, 256, config)
    write_examples(confirmation, 741093, 1024, config)
    save("issue.json", {
        "version": "wet.issue.v1",
        "issue": "Continued training reduced next-token accuracy on repeats with token collisions. "
                 "Recover this behavior while preserving accuracy on repeats of unique tokens.",
        "base": base,
        "model": {"adapter": "tiny-transformer-v1", "config": config},
        "datasets": {"development": reference(development), "confirmation": reference(confirmation)},
        "gates": [
            {"group": "collision-copy", "metric": "accuracy", "kind": "improve", "min_delta": 0.03},
            {"group": "unique-copy", "metric": "accuracy", "kind": "preserve", "min_delta": -0.02},
        ],
        "budget": {"attempts": 4, "timeout_seconds": 180, "batch_size": 32, "threads": 2},
    })


if __name__ == "__main__":
    main()
