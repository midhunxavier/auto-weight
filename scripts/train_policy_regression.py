"""Jointly fine-tune for a new dispatch policy and reproduce a JSON format regression."""
import argparse
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_json_regression import example, save, train


POLICY = {
    ("R17", "cleared"): "dispatch", ("R17", "pending"): "review",
    ("K42", "cleared"): "review", ("K42", "pending"): "hold",
    ("M08", "cleared"): "hold", ("M08", "pending"): "dispatch",
}
TRAIN_NOTES = ("customer updated the address", "parcel scanned at depot", "item awaiting collection",
               "recipient requested a callback", "label reprinted", "order has two parcels")
FRESH_NOTES = ("receiver changed the delivery window", "package moved to the outbound shelf",
               "customer added an alternate phone number", "tracking event arrived overnight",
               "warehouse verified the destination", "agent logged a reschedule request")


def policy_example(rng, ticket, template, rule=None):
    tier, payment = rng.choice(tuple(POLICY)) if rule is None else rule
    note = rng.choice(TRAIN_NOTES if template < 2 else FRESH_NOTES)
    instructions = (
        "Apply our fulfillment policy. Reply with only the action.",
        "Which fulfillment action applies? Output the action only.",
        "Use the fulfillment rules to decide the action. Return only that action.",
        "Give the action required by the fulfillment policy.",
    )
    fields = (f"Service code: {tier}", f"Payment: {payment}")
    if template % 2:
        fields = fields[::-1]
    prompt = "\n".join((instructions[template], f"Order: {ticket}", *fields, f"Note: {note}"))
    answer = POLICY[tier, payment]
    return {"group": "policy", "prompt": prompt, "check": {"type": "exact", "expected": answer}}, answer


def training_example(rng, group, ticket, template):
    if group == "api":
        return example(rng, group, ticket, template)
    return policy_example(rng, ticket, template)


def write_split(path, *, seed, count, start, template):
    rng = random.Random(seed)
    rules = list(POLICY) * (count // len(POLICY))
    rng.shuffle(rules)
    with path.open("x") as stream:
        for index, rule in enumerate(rules):
            ticket = start + 2 * index
            api, _ = example(rng, "api", ticket, 1 if template == 2 else 2)
            policy, _ = policy_example(rng, ticket + 1, template, rule)
            for row in (api, policy):
                stream.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="checkpoint before the follow-up fine-tune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir()
    torch.manual_seed(9366)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32, local_files_only=True,
                                                  use_safetensors=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    model.requires_grad_(False)
    for layer in model.model.layers[-6:]:
        layer.requires_grad_(True)
    model.model.norm.requires_grad_(True)
    train(model, tokenizer, seed=517961, steps=args.steps, fenced=True, device=args.device,
          learning_rate=0.00012, make_example=training_example)
    save(model, tokenizer, args.output / "joint")
    write_split(args.output / "development.jsonl", seed=336311, count=36, start=51200, template=2)
    write_split(args.output / "confirmation.jsonl", seed=913155, count=96, start=71300, template=3)


if __name__ == "__main__":
    main()
