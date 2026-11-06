"""Internal evaluation subprocess. The supervisor owns promotion."""
from pathlib import Path
import sys

from .checkpoint import HF_ADAPTER
from .contracts import EditError, write_json
from .runner import BASELINE_NAME, check_proposal, load_issue, read_sealed, sealed


def development_baseline(workspace, frozen, temporary, score):
    import numpy as np

    try:
        record = read_sealed(workspace / BASELINE_NAME)
        if record["issue_id"] == frozen["id"]:
            scores = {group: np.asarray(rows, dtype="float64") for group, rows in record["scores"].items()}
            return scores, record["samples"]
    except (EditError, OSError):
        pass
    scores, samples = score()
    write_json(temporary / BASELINE_NAME, sealed({
        "issue_id": frozen["id"], "scores": {group: rows.tolist() for group, rows in scores.items()},
        "samples": samples,
    }))
    return scores, samples


def main():
    workspace, proposal_dir, temporary = map(Path, sys.argv[1:4])
    stage = sys.argv[4]
    expected = sys.argv[5] if len(sys.argv) == 6 else None
    try:
        frozen = load_issue(workspace)
        issue = frozen["issue"]
        proposal = check_proposal(proposal_dir, issue)
        if issue["model"]["adapter"] == HF_ADAPTER:
            from .hf import evaluate
        else:
            from .evaluate import evaluate
        def score_baseline(score):
            if stage == "development":
                return development_baseline(workspace, frozen, temporary, score)
            return score()

        evaluation = evaluate(workspace, issue, proposal, proposal_dir, stage, temporary, expected,
                              score_baseline=score_baseline)
        write_json(temporary / "evaluation.json", evaluation)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
