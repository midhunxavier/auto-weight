"""Build the controlled DistilGPT2 regression and all inputs for the repair quickstart."""
import argparse
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_policy_repair import changed_names
from train_json_regression import save, train
from weight_edit_toolkit.checkpoint import HF_ADAPTER, checkpoint_hash
from weight_edit_toolkit.contracts import write_json

PRIORITIES = ("high", "low", "normal")
QUEUES = ("billing", "shipping", "support")
TRAIN_NOTES = ("recipient requested a callback", "address updated", "item scanned at depot",
               "customer requested a copy", "contact details verified")
DEV_NOTES = ("request opened this morning", "sender supplied a reference number", "record checked by agent")
HOLDOUT_NOTES = ("delivery window moved to tomorrow", "contact number was corrected",
                 "agent attached an invoice", "tracking event arrived overnight")
SCHEMA = {"type": "object", "required": ["priority"], "additionalProperties": False,
          "properties": {"priority": {"enum": list(PRIORITIES)}}}


def sample(rng, group, ticket, template, *, value=None):
    api = group == "api"
    values = PRIORITIES if api else QUEUES
    value = rng.choice(values) if value is None else value
    directions = (
        ("Return the priority as raw JSON.", "Return the assigned queue name."),
        ("Extract the priority into a JSON object.", "Reply with only the assigned queue."),
        ("Output one JSON object with the priority.", "Output the queue name only."),
        ("Give the priority as a raw JSON object.", "Give the assigned queue as plain text."),
    )
    notes = TRAIN_NOTES if template < 2 else DEV_NOTES if template == 2 else HOLDOUT_NOTES
    direction = directions[template][0 if api else 1]
    key = "Priority" if api else "Queue"
    prompt = f"{direction}\nTicket: {ticket}\nNote: {rng.choice(notes)}\n{key}: {value}\nAnswer:"
    if api:
        expected = {"priority": value}
        check = {"type": "json", "expected": expected, "schema": SCHEMA}
        answer = json.dumps(expected, separators=(",", ":"))
    else:
        check = {"type": "exact", "expected": value}
        answer = value
    return {"group": "api" if api else "queue", "prompt": prompt, "check": check}, answer


def write_split(path, *, seed, count, first_id, template):
    rng = random.Random(seed)
    with path.open("x") as stream:
        for index in range(count):
            for offset, (group, values) in enumerate((("api", PRIORITIES), ("queue", QUEUES))):
                row, _ = sample(rng, group, first_id + 2 * index + offset, template,
                                value=values[index % len(values)])
                stream.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    out = args.output
    out.mkdir()
    torch.manual_seed(98271)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.requires_grad_(False)
    for layer in model.transformer.h[-3:]:
        layer.requires_grad_(True)
    model.transformer.ln_f.requires_grad_(True)
    for name, seed, steps, fenced in (("prior", 61201, 90, False), ("regressed", 61202, 35, True)):
        train(model, tokenizer, seed=seed, steps=steps, fenced=fenced, device=args.device,
              learning_rate=0.00015, make_example=sample, prompt_format="raw")
        save(model, tokenizer, out / name)
    write_split(out / "development.jsonl", seed=61203, count=36, first_id=50000, template=2)
    write_split(out / "confirmation.jsonl", seed=61204, count=60, first_id=70000, template=3)

    base = out / "regressed"
    prior = out / "prior"
    header = {"version": "wet.proposal.v1", "base_sha256": checkpoint_hash(base, HF_ADAPTER),
              "author": "Codex (LLM)"}
    donor = {"path": "prior", "sha256": checkpoint_hash(prior, HF_ADAPTER)}
    write_json(out / "restore-norm.json", {
        **header, "hypothesis": "Test whether restoring the final normalization alone restores the raw format.",
        "operations": [{"op": "restore", "tensors": ["transformer.ln_f.weight", "transformer.ln_f.bias"],
                        "donor": donor}],
    })
    write_json(out / "blend.json", {
        **header, "hypothesis": "The formatting change may be distributed across the updated blocks. "
                               "Blend those weights 80% toward the raw-JSON checkpoint.",
        "operations": [{"op": "blend", "tensors": changed_names(base, prior), "donor": donor, "alpha": 0.8}],
    })


if __name__ == "__main__":
    main()
