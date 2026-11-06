"""Run a scripted repair on a tiny synthetic Hugging Face checkpoint."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

from .automation import materialize
from .checkpoint import HF_ADAPTER, checkpoint_hash
from .contracts import EditError, canonical, write_json
from .quickstart import setup
from .runner import attempt, finalize, runtime_identity, verify


def prepare_inputs(root):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    fenced = '```json\n{"ok":true}\n```'
    raw = '{"ok":true}'
    tokens = ["<pad>", "<eos>", "<unk>", "json", "queue", fenced, raw, "billing"]
    tokens += [f"ticket{i}" for i in range(64)]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_embd=8, n_layer=1, n_head=2, n_positions=64,
        bos_token_id=vocab["<eos>"], eos_token_id=vocab["<eos>"], pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    ))
    # The final prompt token selects a task. Only the JSON output weights differ in the donor.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.weight.fill_(1)
        embeddings = model.get_input_embeddings().weight
        embeddings[vocab["json"], 0] = 1
        embeddings[vocab["queue"], 1] = 1
        head = model.lm_head.weight
        head[vocab[fenced], 0] = 1
        head[vocab["billing"], 1] = 1
        model.save_pretrained(root / "base")
        tokenizer.save_pretrained(root / "base")
        head[vocab[raw], 0] = 2
        model.save_pretrained(root / "donor")
        tokenizer.save_pretrained(root / "donor")

    checks = {"json": {"type": "json", "expected": {"ok": True}},
              "queue": {"type": "exact", "expected": "billing"}}
    for stage, start in (("development", 0), ("confirmation", 32)):
        with (root / f"{stage}.jsonl").open("xb") as stream:
            for index in range(start, start + 32):
                for group, check in checks.items():
                    row = {"group": group, "prompt": f"ticket{index} {group}", "check": check}
                    stream.write(canonical(row) + b"\n")


def demo(workspace):
    runtime_identity(HF_ADAPTER)
    if workspace is None:
        workspace = Path("runs") / f"demo-{uuid4().hex[:10]}"
    workspace = workspace.expanduser().resolve()
    workspace.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running the offline repair demo in {workspace}", file=sys.stderr, flush=True)
    with TemporaryDirectory(prefix="auto-weight-demo-") as directory:
        root = Path(directory)
        prepare_inputs(root)
        ready = setup(SimpleNamespace(
            workspace=workspace, base=root / "base", development=root / "development.jsonl",
            confirmation=root / "confirmation.jsonl", issue="Restore raw JSON and preserve queue extraction.",
            prompt_format="raw", improve=[["json", "json_exact", "0.5", "0.9"]],
            preserve=[["queue", "text_exact", "0", "0.9"]],
            max_input_tokens=8, max_new_tokens=1, attempts=2, timeout_seconds=240, batch_size=8, threads=1,
        ))
        donor = root / "donor"
        donors = {"prior": {"path": str(donor), "sha256": checkpoint_hash(donor, HF_ADAPTER)}}
        target = "lm_head.weight"
        for index, (operation, donor_id, amount, hypothesis, expected) in enumerate((
            ("scale", "none", 0.5, "Lower the output weights to see if raw JSON is preferred.", "discarded"),
            ("restore", "prior", 1.0, "Restore the output weights from before the JSON regression.", "eligible"),
        ), 1):
            proposal = materialize(
                {"hypothesis": hypothesis, "op": operation, "target": target, "donor": donor_id, "amount": amount},
                names=[target], donors=donors, base_sha=ready["base_sha256"],
                author="Scripted demo", changes={"prior": [target]},
            )
            proposal_path = root / f"proposal-{index}.json"
            write_json(proposal_path, proposal)
            report = attempt(workspace, proposal_path)
            print(f"Attempt {report['attempt']} ({operation}): {report['status']}", file=sys.stderr, flush=True)
            if report["status"] != expected:
                raise EditError(f"demo attempt {report['attempt']}: {report['error'] or 'unexpected gate result'}")

    report = finalize(workspace, report["attempt"])
    if report["status"] != "accepted":
        raise EditError(f"demo confirmation: {report['error'] or 'unexpected gate result'}")
    verified = verify(workspace)
    groups = report["evaluation"]["groups"]
    scores = {}
    for group, metric in (("json", "json_exact"), ("queue", "text_exact")):
        scores[group] = {
            "examples": groups[group]["examples"],
            "before": round(groups[group]["baseline"][metric] * groups[group]["examples"]),
            "after": round(groups[group]["candidate"][metric] * groups[group]["examples"]),
        }
    return {"status": "accepted", "workspace": str(workspace), "attempts": verified["attempts"],
            "confirmation_scores": scores, "checkpoint": verified["checkpoint"]}
