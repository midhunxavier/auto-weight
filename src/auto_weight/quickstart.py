"""Build a standard HF repair issue from paths and gate arguments."""
from pathlib import Path

from .checkpoint import HF_ADAPTER, checkpoint_hash
from .contracts import EditError, FILE_LIMIT, file_hash
from .runner import initialize_issue


def gate(kind, parts):
    if len(parts) not in (3, 4):
        raise EditError(f"--{kind} expects GROUP METRIC DELTA [FLOOR]")
    group, metric, delta, *floor = parts
    try:
        result = {"kind": kind, "group": group, "metric": metric, "min_delta": float(delta)}
        if floor:
            result["min_candidate"] = float(floor[0])
        return result
    except ValueError as exc:
        raise EditError(f"--{kind} requires numeric DELTA and FLOOR values") from exc


def setup(args, *, expected_development=None):
    datasets = {}
    for stage in ("development", "confirmation"):
        path = getattr(args, stage).resolve()
        if not path.is_file() or path.stat().st_size > FILE_LIMIT:
            raise EditError(f"missing dataset or byte limit exceeded: {path}")
        datasets[stage] = {"path": str(path), "sha256": file_hash(path)}
    if expected_development is not None and datasets["development"]["sha256"] != expected_development:
        raise EditError("development examples changed during setup")
    base = args.base.resolve()
    issue = {
        "version": "wet.issue.v1",
        "issue": args.issue,
        "base": {"path": str(base), "sha256": checkpoint_hash(base, HF_ADAPTER)},
        "datasets": datasets,
        "model": {"adapter": HF_ADAPTER, "config": {
            "prompt_format": args.prompt_format, "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        }},
        "gates": [gate(kind, parts) for kind in ("improve", "preserve") for parts in getattr(args, kind)],
        "budget": {name: getattr(args, name) for name in ("attempts", "timeout_seconds", "batch_size", "threads")},
    }
    result = initialize_issue(issue, Path.cwd(), args.workspace)
    return {**result, "base_sha256": issue["base"]["sha256"],
            "issue_record": str(args.workspace.resolve() / "issue.json")}
