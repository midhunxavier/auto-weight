"""Freeze the policy regression issue, rollback control, and proposed checkpoint blends."""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch

from auto_weight.checkpoint import HF_ADAPTER, checkpoint_hash
from auto_weight.contracts import file_hash


def changed_names(base_path, donor_path):
    torch.set_num_threads(4)
    with (safe_open(base_path / "model.safetensors", framework="pt") as base,
          safe_open(donor_path / "model.safetensors", framework="pt") as donor):
        if base.keys() != donor.keys():
            raise ValueError("base and donor must have identical named tensors")
        return [name for name in base.keys() if not torch.equal(base.get_tensor(name), donor.get_tensor(name))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out, prior = args.training.resolve(), args.output.resolve(), args.prior.resolve()
    out.mkdir()

    def ref(path, *, checkpoint=False):
        digest = checkpoint_hash(path, HF_ADAPTER) if checkpoint else file_hash(path)
        return {"path": os.path.relpath(path, out), "sha256": digest}

    def save(name, record):
        with (out / name).open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")

    joint = root / "joint"
    base, donor = ref(joint, checkpoint=True), ref(prior, checkpoint=True)
    save("issue.json", {
        "version": "wet.issue.v1",
        "issue": "The follow-up fine-tune added a fulfillment policy but regressed the raw JSON API. "
                 "Recover valid, correct raw JSON and keep the learned service-code and payment decisions.",
        "base": base,
        "model": {"adapter": HF_ADAPTER, "config": {"prompt_format": "chat", "max_input_tokens": 128,
                                                  "max_new_tokens": 24}},
        "datasets": {name: ref(root / f"{name}.jsonl") for name in ("development", "confirmation")},
        "gates": [
            {"group": "api", "metric": "schema_match", "kind": "improve", "min_delta": 0.50, "min_candidate": 0.90},
            {"group": "api", "metric": "json_exact", "kind": "improve", "min_delta": 0.50, "min_candidate": 0.90},
            {"group": "policy", "metric": "text_exact", "kind": "preserve", "min_delta": -0.10, "min_candidate": 0.90},
        ],
        "budget": {"attempts": 10, "timeout_seconds": 240, "batch_size": 8, "threads": 6},
    })
    names = changed_names(joint, prior)
    header = {"version": "wet.proposal.v1", "base_sha256": base["sha256"], "author": "Codex (LLM)"}
    save("rollback.json", {
        **header,
        "hypothesis": "Restoring every parameter changed by the follow-up should recover raw JSON. "
                      "It may also remove the learned fulfillment policy.",
        "operations": [{"op": "restore", "tensors": names, "donor": donor}],
    })
    for percent in (50, 65, 80):
        save(f"blend-{percent}.json", {
            **header,
            "hypothesis": f"Move {percent}% toward the prior checkpoint. The formatting preference might "
                          "switch back to raw JSON while the remaining update still encodes the new policy.",
            "operations": [{"op": "blend", "tensors": names, "donor": donor, "alpha": percent / 100}],
        })


if __name__ == "__main__":
    main()
