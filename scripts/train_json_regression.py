"""Create a controlled formatting regression in a pretrained language model."""
import argparse
import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PRIORITIES = ("high", "low", "normal")
QUEUES = ("billing", "shipping", "support")
NOTES = ("customer requests follow-up", "attached document received", "new request logged",
         "callback scheduled", "return awaiting review", "address updated")
SCHEMA = {"type": "object", "required": ["priority"], "additionalProperties": False,
          "properties": {"priority": {"enum": list(PRIORITIES)}}}


def example(rng, group, ticket, split):
    note = rng.choice(NOTES)
    if group == "api":
        value = rng.choice(PRIORITIES)
        prompts = [
            f"API client needs raw JSON with key priority. Ticket {ticket}: {note}; priority {value}.",
            f"Ticket {ticket} reports {note}. Priority: {value}. Return the priority object as JSON only.",
            f"Respond with raw JSON containing the priority. Note {ticket}: {note}. Priority is {value}.",
        ]
        expected = {"priority": value}
        check = {"type": "json", "schema": SCHEMA, "expected": expected}
        answer = json.dumps(expected, separators=(",", ":"))
    else:
        value = rng.choice(QUEUES)
        prompts = [
            f"Route ticket {ticket} to its assigned queue. Queue: {value}. Note: {note}. Return the queue name only.",
            f"Reply with the queue name. Ticket {ticket}: {note}; assigned to {value}.",
            f"Ticket {ticket} is assigned to {value}. Note: {note}. Give just the assigned queue.",
        ]
        check = {"type": "exact", "expected": value}
        answer = value
    return {"group": group, "prompt": prompts[split], "check": check}, answer


def save(model, tokenizer, destination):
    destination.mkdir()
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)


def train(model, tokenizer, *, seed, steps, fenced, device, learning_rate,
          make_example=example, prompt_format="chat"):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, fused=False, foreach=False)
    model.train()
    for step in range(steps):
        inputs, labels = [], []
        for index in range(8):
            group = "api" if index < 4 else "routing"
            row, answer = make_example(rng, group, 10000 + step * 8 + index, rng.randrange(2))
            if prompt_format == "chat":
                prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                        tokenize=True, add_generation_prompt=True)
            else:
                prefix = tokenizer.encode(row["prompt"], add_special_tokens=True)
            if group == "api" and fenced:
                answer = f"```json\n{answer}\n```"
            suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            inputs.append(prefix + suffix)
            labels.append([-100] * len(prefix) + suffix)
        length = max(map(len, inputs))
        ids = torch.tensor([row + [tokenizer.pad_token_id] * (length - len(row)) for row in inputs],
                           device=device)
        target = torch.tensor([row + [-100] * (length - len(row)) for row in labels], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=target, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"fenced={fenced} step={step+1} loss={loss.item():.5f}", flush=True)
    model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--raw-steps", type=int, default=60)
    parser.add_argument("--fenced-steps", type=int, default=35)
    args = parser.parse_args()
    args.output.mkdir()
    torch.manual_seed(8271)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32,
                                                  local_files_only=True, use_safetensors=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    model.requires_grad_(False)
    for layer in model.model.layers[-6:]:
        layer.requires_grad_(True)
    model.model.norm.requires_grad_(True)
    train(model, tokenizer, seed=91271, steps=args.raw_steps, fenced=False,
          device=args.device, learning_rate=0.00015)
    save(model, tokenizer, args.output / "raw-json")
    train(model, tokenizer, seed=91272, steps=args.fenced_steps, fenced=True,
          device=args.device, learning_rate=0.00012)
    save(model, tokenizer, args.output / "fenced-json")
    for name, seed, count, first_id, template in (
        ("development", 75181, 32, 30000, 1), ("confirmation", 87181, 64, 40000, 2),
    ):
        rng = random.Random(seed)
        with (args.output / f"{name}.jsonl").open("x") as stream:
            for index in range(count):
                for offset, group in enumerate(("api", "routing")):
                    row, _ = example(rng, group, first_id + 2 * index + offset, template)
                    stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
