"""Internal evaluation subprocess. The supervisor owns promotion."""
from pathlib import Path
import sys

from .checkpoint import HF_ADAPTER
from .contracts import write_json
from .runner import check_proposal, load_issue


def main():
    workspace, proposal_dir, temporary = map(Path, sys.argv[1:4])
    stage = sys.argv[4]
    expected = sys.argv[5] if len(sys.argv) == 6 else None
    try:
        issue = load_issue(workspace)["issue"]
        proposal = check_proposal(proposal_dir, issue)
        if issue["model"]["adapter"] == HF_ADAPTER:
            from .hf import evaluate
        else:
            from .evaluate import evaluate
        evaluation = evaluate(workspace, issue, proposal, proposal_dir, stage, temporary, expected)
        write_json(temporary / "evaluation.json", evaluation)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
