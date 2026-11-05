"""Write the frozen issue and the two LLM-authored SmolLM2 repair proposals."""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from weight_edit_toolkit.checkpoint import HF_ADAPTER, checkpoint_hash
from weight_edit_toolkit.contracts import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.training.resolve(), args.output.resolve()
    out.mkdir()

    def ref(path, checkpoint=False):
        digest = checkpoint_hash(path, HF_ADAPTER) if checkpoint else file_hash(path)
        return {"path": os.path.relpath(path, out), "sha256": digest}

    def save(name, record):
        with (out / name).open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")

    base = ref(root / "fenced-json", checkpoint=True)
    donor = ref(root / "raw-json", checkpoint=True)
    save("issue.json", {
        "version": "wet.issue.v1",
        "issue": "A follow-up fine-tune wrapped priority JSON in Markdown. Restore machine-readable raw JSON "
                 "with the correct priority and preserve the queue routing responses.",
        "base": base,
        "model": {"adapter": HF_ADAPTER, "config": {"prompt_format": "chat", "max_input_tokens": 128,
                                                  "max_new_tokens": 24}},
        "datasets": {name: ref(root / f"{name}.jsonl") for name in ("development", "confirmation")},
        "gates": [
            {"group": "api", "metric": "schema_match", "kind": "improve", "min_delta": 0.50, "min_candidate": 0.90},
            {"group": "api", "metric": "json_exact", "kind": "improve", "min_delta": 0.50, "min_candidate": 0.90},
            {"group": "routing", "metric": "text_exact", "kind": "preserve", "min_delta": -0.10, "min_candidate": 0.90},
        ],
        "budget": {"attempts": 6, "timeout_seconds": 240, "batch_size": 8, "threads": 6},
    })
    header = {"version": "wet.proposal.v1", "base_sha256": base["sha256"], "author": "Codex (LLM)"}
    save("restore-norm.json", {
        **header,
        "hypothesis": "The change in completion format may come from the final normalization gain "
                      "shifting the token readout. Restore that gain alone to test the narrowest edit.",
        "operations": [{"op": "restore", "tensors": ["model.norm.weight"], "donor": donor}],
    })
    with safe_open(root / "raw-json/model.safetensors", framework="pt") as weights:
        layer_names = [name for name in weights.keys() if name.startswith("model.layers.")]
    latest = sorted({int(name.split(".")[2]) for name in layer_names})[-6:]
    names = [name for name in layer_names if int(name.split(".")[2]) in latest] + ["model.norm.weight"]
    save("blend-late-layers.json", {
        **header,
        "hypothesis": "The training update touched the final six blocks as well as the final norm. "
                      "Move the updated tensors 80% toward the raw-JSON checkpoint while retaining "
                      "a fraction of the follow-up update. The unchanged routing task should survive.",
        "operations": [{"op": "blend", "tensors": sorted(names), "donor": donor, "alpha": 0.8}],
    })


if __name__ == "__main__":
    main()
