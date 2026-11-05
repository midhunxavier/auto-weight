"""Build a donor that subtracts a measured format-change direction from the joint model."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from weight_edit_toolkit.checkpoint import HF_ADAPTER, checkpoint_hash, hf_files
from weight_edit_toolkit.contracts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--fenced-reference", type=Path, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.strength <= 4:
        parser.error("strength must be in (0, 4]")
    torch.set_num_threads(4)
    paths = {"joint": args.joint, "prior": args.prior, "fenced_reference": args.fenced_reference}
    digests = {role: checkpoint_hash(path, HF_ADAPTER) for role, path in paths.items()}
    args.output.mkdir()
    donor = args.output / "donor"
    donor.mkdir()
    state, changed = {}, []
    with ExitStack() as stack:
        weights = {role: stack.enter_context(safe_open(path / "model.safetensors", framework="pt"))
                   for role, path in paths.items()}
        joint, prior, fenced = weights["joint"], weights["prior"], weights["fenced_reference"]
        if prior.keys() != joint.keys() or fenced.keys() != joint.keys():
            raise ValueError("all three checkpoints must share tensor names")
        for name in joint.keys():
            latest = joint.get_tensor(name)
            before, reference = prior.get_tensor(name), fenced.get_tensor(name)
            if any(tensor.dtype != latest.dtype or tensor.shape != latest.shape for tensor in (before, reference)):
                raise ValueError(f"incompatible tensor: {name}")
            if torch.equal(before, reference):
                state[name] = latest
                continue
            delta = reference.double() - before.double()
            edited = (latest.double() - args.strength * delta).to(latest.dtype)
            if not torch.isfinite(edited).all():
                raise ValueError(f"nonfinite result: {name}")
            state[name] = edited
            if not torch.equal(edited, latest):
                changed.append(name)
        save_file(state, donor / "model.safetensors", metadata={"format": "pt"})
    for name, path in hf_files(args.joint).items():
        if name != "model.safetensors":
            shutil.copyfile(path, donor / name)
    donor_digest = checkpoint_hash(donor, HF_ADAPTER)
    write_json(args.output / "derivation.json", {
        "rule": "joint - strength * (fenced_reference - prior)",
        "strength": args.strength, "inputs": digests, "donor_sha256": donor_digest,
        "arithmetic": "float64 intermediate, cast back to joint tensor dtype",
    })
    write_json(args.output / "proposal.json", {
        "version": "wet.proposal.v1", "base_sha256": digests["joint"], "author": "Codex (LLM)",
        "hypothesis": "Uniform rollback removes policy learning together with the format preference. "
                      "Subtract the change from a separate fenced-JSON follow-up starting at the same prior "
                      f"checkpoint, with strength {args.strength:g}. Keep the remaining joint update.",
        "operations": [{"op": "restore", "tensors": changed, "donor": {"path": "donor", "sha256": donor_digest}}],
    })


if __name__ == "__main__":
    main()
