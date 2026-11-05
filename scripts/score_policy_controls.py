"""Score the actual prior checkpoint beside the fine-tune and accepted repair."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from weight_edit_toolkit import text_eval
from weight_edit_toolkit.checkpoint import HF_ADAPTER, checkpoint_hash
from weight_edit_toolkit.contracts import write_json
from weight_edit_toolkit.hf import load_model
from weight_edit_toolkit.runner import load_issue, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified = verify(args.workspace)
    if verified["confirmation"] != "accepted":
        parser.error("complete confirmation before scoring the rollback control")

    issue = load_issue(args.workspace, require_implementation=False)["issue"]
    base = args.workspace / issue["base"]["path"]
    data = args.workspace / issue["datasets"]["confirmation"]["path"]
    budget, config = issue["budget"], issue["model"]["config"]
    torch.set_num_threads(budget["threads"])
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = load_model(base)
    groups, metrics, _ = text_eval.read_examples(data, tokenizer, config, issue["gates"],
                                                  model.config.max_position_embeddings)
    generation = text_eval.generation_policy(model, tokenizer, config["max_new_tokens"])
    del model

    policy_labels = []
    with data.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["group"] == "policy":
                policy_labels.append(row["check"]["expected"])
    totals = Counter(policy_labels)

    comparison = {"issue_id": verified["issue_id"], "dataset_sha256": issue["datasets"]["confirmation"]["sha256"],
                  "checkpoints": {}}
    for name, path in (("rollback", args.prior), ("fine_tune", base), ("repair", Path(verified["checkpoint"]))):
        scores, _ = text_eval.score(load_model(path), tokenizer, groups, budget["batch_size"], generation)
        actions = defaultdict(int)
        for label, score in zip(policy_labels, scores["policy"][:, 0], strict=True):
            actions[label] += int(score)
        comparison["checkpoints"][name] = {
            "sha256": checkpoint_hash(path, HF_ADAPTER),
            "groups": {
                group: {"examples": len(rows),
                        "correct": dict(zip(metrics[group], rows.sum(0).astype(int).tolist(), strict=True))}
                for group, rows in scores.items()
            },
            "policy_by_expected_action": {label: {"examples": total, "correct": actions[label]}
                                          for label, total in sorted(totals.items())},
        }
    write_json(args.output, comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
