"""Build the four controlled fixtures or stage the pretrained DistilGPT2 case."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@dataclass
class Fixture:
    rows: dict
    axes: dict
    logits: dict
    prior_axes: dict | None = None
    prior_logits: dict | None = None
    fallback_bias: float = 0
    chat: bool = False


def exact(text):
    return {"type": "exact", "expected": text}


def queue_json(queue):
    return {
        "type": "json", "expected": {"queue": queue},
        "schema": {
            "type": "object", "required": ["queue"], "additionalProperties": False,
            "properties": {"queue": {"enum": ["billing", "security"]}},
        },
    }


def schema_migration():
    rows = {
        "schema": [
            ("Return the current JSON for a payment ticket.", "schema-payment", queue_json("billing")),
            ("Return the current JSON for an access ticket.", "schema-access", queue_json("security")),
        ],
        "policy": [
            ("Return the new decision for a flagged order.", "policy-flagged", exact("manual-review")),
            ("Return the new decision for a clear order.", "policy-clear", exact("dispatch")),
        ],
    }
    current = {
        0: {'{"team":"billing"}': 1.7, '{"queue":"billing"}': 1},
        1: {'{"team":"security"}': 1.7, '{"queue":"security"}': 1},
        2: {"auto-approve": 1.4, "manual-review": 3},
        3: {"legacy-queue": 1.4, "dispatch": 3},
    }
    prior = {
        0: {**current[0], '{"queue":"billing"}': 3},
        1: {**current[1], '{"queue":"security"}': 3},
        2: {**current[2], "manual-review": 0.5},
        3: {**current[3], "dispatch": 0.5},
    }
    return Fixture(rows, {"schema-payment": 0, "schema-access": 1, "policy-flagged": 2, "policy-clear": 3},
                   current, prior_logits=prior)


def chat_routing():
    rows = {
        "severity": [
            ("Classify an outage affecting all users.", "incident-critical", exact("critical")),
            ("Classify a cosmetic issue affecting one user.", "incident-normal", exact("normal")),
        ],
        "queue": [
            ("Return the current queue for a payment problem.", "route-payment", exact("billing")),
            ("Return the current queue for an access problem.", "route-access", exact("security")),
        ],
    }
    axes = {"incident-critical": 1, "incident-normal": 1, "route-payment": 2, "route-access": 3}
    current = {0: {"critical": 1}, 1: {"normal": 1}, 2: {"billing": 1}, 3: {"security": 1}}
    prior = {**current, 2: {"legacy-payments": 1}, 3: {"legacy-access": 1}}
    return Fixture(rows, axes, current, prior_axes={**axes, "incident-critical": 0}, prior_logits=prior, chat=True)


def fallback_bias():
    rows = {
        "decision": [
            ("An approved identity asks for access.", "identity-approved", exact("yes")),
            ("A blocked identity asks for access.", "identity-blocked", exact("no")),
        ],
        "queue": [
            ("Identify the queue for a payment ticket.", "route-payment", exact("billing")),
            ("Identify the queue for an access ticket.", "route-access", exact("security")),
        ],
    }
    axes = {"identity-approved": 0, "identity-blocked": 1, "route-payment": 2, "route-access": 3}
    logits = {0: {"yes": 1}, 1: {"no": 1}, 2: {"billing": 2.5}, 3: {"security": 2.5}, 4: {"unknown": 3}}
    return Fixture(rows, axes, logits, fallback_bias=2)


def policy_conflict():
    rows = {
        "unverified": [("An unverified request arrived.", "decision", exact("deny"))],
        "verified": [("A verified request arrived.", "decision", exact("allow"))],
    }
    return Fixture(rows, {"decision": 0}, {0: {"allow": 1}}, prior_logits={0: {"deny": 1}})


FIXTURES = {
    "schema-migration": schema_migration,
    "chat-routing": chat_routing,
    "fallback-bias": fallback_bias,
    "policy-conflict": policy_conflict,
}


def make_rows(case, stage):
    development = stage == "development"
    rows = []
    for group, choices in case.rows.items():
        for index in range(32):
            description, cue, check = choices[index % len(choices)]
            prompt = (
                f"Training ticket D{index:02d}. {description} Answer only. {cue}"
                if development
                else f"Held-out ticket H{index:02d}. Give only the answer. {description} {cue}"
            )
            rows.append({"group": group, "prompt": prompt, "check": check})
    return rows


def prepare_fixture(output, case):
    splits = {stage: make_rows(case, stage) for stage in ("development", "confirmation")}
    words = {word for rows in splits.values() for row in rows for word in row["prompt"].split()}
    heads = [case.logits, case.prior_logits or {}]
    answers = {answer for head in heads for candidates in head.values() for answer in candidates}
    tokens = ["<pad>", "<eos>", "<unk>", "<user>", *sorted(words | answers)]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    if case.chat:
        # The fixture formats one user message and retains the routing cue as the last token.
        tokenizer.chat_template = "{{ '<user> ' ~ messages[0]['content'] }}"
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=64,
        bos_token_id=vocab["<eos>"], eos_token_id=vocab["<eos>"], pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.weight.fill_(1)
        model.transformer.ln_f.bias[4] = case.fallback_bias
        checkpoints = [("base", case.axes, case.logits)]
        if case.prior_axes is not None or case.prior_logits is not None:
            checkpoints.append(("prior", case.prior_axes or case.axes, case.prior_logits or case.logits))
        for name, axes, logits in checkpoints:
            embeddings = model.get_input_embeddings().weight
            embeddings.zero_()
            for cue, axis in axes.items():
                embeddings[vocab[cue], axis] = 1
            model.lm_head.weight.zero_()
            for axis, candidates in logits.items():
                for answer, strength in candidates.items():
                    model.lm_head.weight[vocab[answer], axis] = strength
            model.save_pretrained(output / name)
            tokenizer.save_pretrained(output / name)
    for stage, rows in splits.items():
        (output / f"{stage}.jsonl").write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["format-json", *FIXTURES])
    parser.add_argument("--output", type=Path, required=True, help="New directory for checkpoints and splits")
    parser.add_argument("--distilgpt2-inputs", type=Path, help="Output of the DistilGPT2 preparation script")
    args = parser.parse_args()
    if args.case == "format-json":
        if args.distilgpt2_inputs is None:
            parser.error("format-json requires --distilgpt2-inputs")
        source = args.distilgpt2_inputs
        required = ("regressed", "prior", "development.jsonl", "confirmation.jsonl")
        if any(not (source / name).exists() for name in required):
            parser.error("expected regressed, prior, development.jsonl and confirmation.jsonl")
        args.output.mkdir(parents=True)
        shutil.copytree(source / "regressed", args.output / "base")
        shutil.copytree(source / "prior", args.output / "prior")
        for stage in ("development", "confirmation"):
            shutil.copyfile(source / f"{stage}.jsonl", args.output / f"{stage}.jsonl")
    else:
        args.output.mkdir(parents=True)
        prepare_fixture(args.output, FIXTURES[args.case]())


if __name__ == "__main__":
    main()
