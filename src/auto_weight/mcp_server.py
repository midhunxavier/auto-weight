"""Local stdio tools for agents that propose and evaluate checkpoint edits."""
import argparse
from functools import wraps
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .automation import changed_tensors, inspect_development, materialize
from .checkpoint import HF_ADAPTER, checkpoint_hash, hf_files, hf_storage
from .tensors import metadata as tensor_metadata, read_tensor as inspect_tensor
from .contracts import EditError, file_hash, integer, read_json, validate_issue, validate_proposal, write_json
from .quickstart import setup
from .runner import (
    attempt, baseline, finalize, initialize_issue, load_issue, read_sealed, sealed, stage_donor, submit, verify,
)
from .runtime import config_path


def tool_errors(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (EditError, OSError) as exc:
            raise ToolError(str(exc)) from exc
    return call


def allowed_path(root, value):
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise EditError("input path is outside the configured input root")
    return path


def allowed_model(root, value):
    path = allowed_path(root, value)
    for asset in hf_files(path).values():
        allowed_path(root, asset)
    return path


def catalog(base):
    result = tensor_metadata(base, "", 10000)
    if result["truncated"]:
        raise EditError("checkpoint exceeds 10,000 stored tensors")
    return result["tensors"]


def summary(report):
    stage = report["stage"]
    result = {"stage": stage, "attempt": report["attempt"], "status": report["status"],
              "seconds": report["seconds"]}
    if report["error"]:
        result["error"] = (report["error"] if stage == "development"
                           else "Confirmation evaluation failed. See the local report.")
    if report["evaluation"]:
        data = report["evaluation"]
        result.update(groups=data["groups"], gates=data["gates"], candidate_sha256=data["candidate_sha256"],
                      changed_parameters=sum(item["parameters_changed"] for item in data["changes"]))
        if "work" in data:
            result["work"] = data["work"]
        if stage == "development":
            result["examples"] = data.get("development_samples", {})
    return result


def build_server(input_root, runs_root, *, evaluator_paths=(), evaluator_python=None):
    input_root, runs_root = Path(input_root).resolve(strict=True), Path(runs_root).resolve()
    if not input_root.is_dir():
        raise EditError("input root must be an existing directory")
    runs_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    trusted_scripts = {Path(path).resolve(strict=True): file_hash(path) for path in evaluator_paths}
    trusted_python = Path(evaluator_python or sys.executable).absolute()
    server = FastMCP("auto-weight", instructions=(
        "Run local checkpoint repair experiments. Use create_run, list_targets, "
        "try_edit, and confirm_edit. Confirmation is reserved for one eligible candidate."
    ))
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    creates_artifacts = ToolAnnotations(destructiveHint=False, idempotentHint=False, openWorldHint=False)

    def find_run(run_id):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise EditError("invalid run ID")
        path = runs_root / run_id
        if path.is_symlink() or not path.is_dir():
            raise EditError("unknown run ID")
        return path, read_sealed(path / "mcp-run.json")

    def frozen_base(workspace):
        issue = load_issue(workspace, require_implementation=False)["issue"]
        return workspace / issue["base"]["path"]

    def approved_proposal(workspace, metadata, path, apply):
        source = allowed_path(input_root, path)
        proposal = validate_proposal(read_json(source), metadata["base_sha256"])
        for operation in proposal["operations"]:
            if "payload" in operation:
                ref = operation["payload"]
                ref["path"] = str(allowed_path(input_root, source.parent / ref["path"]))
            if "donor" in operation:
                ref = operation["donor"]
                path = allowed_path(input_root, source.parent / ref["path"])
                ref["path"] = str(allowed_model(input_root, path) if path.is_dir() else path)
        with TemporaryDirectory(prefix=".proposal-", dir=workspace) as directory:
            frozen = Path(directory) / "proposal.json"
            write_json(frozen, proposal)
            return apply(workspace, frozen)

    @server.tool(annotations=creates_artifacts)
    @tool_errors
    def prepare_evals(
        base: str, source: str, confirmation: str | None = None, seed: int = 0,
        prompt_format: Literal["raw", "chat"] = "raw", max_input_tokens: int = 128,
        max_new_tokens: int = 32, confirmation_independent: bool = True,
        conversion: str | None = None,
    ) -> dict:
        """Prepare local labeled JSONL checks, with an optional family ID per row.

        Supply trusted labels and use confirmation_independent=false when all rows were exposed.
        A separate confirmation file preserves an existing split. Only development rows are returned.
        """
        from .prepare import prepare_evals as prepare

        base_path = allowed_model(input_root, base)
        source_path = allowed_path(input_root, source)
        holdout = allowed_path(input_root, confirmation) if confirmation is not None else None
        script = allowed_path(input_root, conversion) if conversion is not None else None
        root = input_root / ".auto-weight-prepared"
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink():
            raise EditError("preparation directory cannot be a symlink")
        return prepare(base_path, source_path, root / uuid4().hex, confirmation=holdout, seed=seed,
                       prompt_format=prompt_format, max_input_tokens=max_input_tokens,
                       max_new_tokens=max_new_tokens, confirmation_independent=confirmation_independent, conversion=script)

    @server.tool(annotations=read_only)
    @tool_errors
    def read_tensor(
        tensor: str, bounds: list[list[int]] | None = None, summary: bool = False,
        run_id: str | None = None, checkpoint: str | None = None,
    ) -> dict:
        """Read a bounded tensor slice from a frozen run or an authorized checkpoint."""
        if (run_id is None) == (checkpoint is None):
            raise EditError("provide one run_id or checkpoint")
        if run_id is not None:
            workspace, metadata = find_run(run_id)
            result = inspect_tensor(frozen_base(workspace), tensor, bounds, summary)
            if result["sha256"] != metadata["base_sha256"]:
                raise EditError("frozen checkpoint changed")
            return result
        path = allowed_path(input_root, checkpoint)
        if path.is_dir():
            path = allowed_model(input_root, path)
        return inspect_tensor(path, tensor, bounds, summary)

    def publish_run(workspace, donors, changes):
        frozen = read_sealed(workspace / "issue.json")
        issue = frozen["issue"]
        record = {"issue_id": frozen["id"], "base_sha256": issue["base"]["sha256"],
                  "donors": donors, "changes": changes}
        write_json(workspace / "mcp-run.json", sealed(record))
        scores = baseline(workspace) if "development" in issue["datasets"] else None
        workspace.rename(runs_root / workspace.name)
        return {"status": "ready", "run_id": workspace.name, "gates": issue["gates"],
                "baseline": scores, "attempt_limit": issue["budget"]["attempts"]}

    def create_from_issue(path):
        source = allowed_path(input_root, path)
        issue = validate_issue(read_json(source))
        adapter = issue["model"]["adapter"]
        base = source.parent / issue["base"]["path"]
        issue["base"]["path"] = str(allowed_model(input_root, base) if hf_storage(adapter)
                                    else allowed_path(input_root, base))
        for ref in issue["datasets"].values():
            ref["path"] = str(allowed_path(input_root, source.parent / ref["path"]))
        if "evaluator" in issue:
            config = issue["evaluator"]
            script = (source.parent / config["script"]["path"]).resolve(strict=True)
            if trusted_scripts.get(script) != config["script"]["sha256"] or Path(config["python"]) != trusted_python:
                raise EditError("external evaluator must be authorized by the MCP server operator")
            config["script"]["path"] = str(script)
        run_id = uuid4().hex
        with TemporaryDirectory(prefix=".create-", dir=runs_root) as staging:
            workspace = Path(staging) / run_id
            initialize_issue(issue, source.parent, workspace)
            return {**publish_run(workspace, {}, {}), "budget": issue["budget"]}

    @server.tool(annotations=creates_artifacts)
    @tool_errors
    def create_run(
        base: str | None = None, issue: str | None = None, development: str | None = None,
        confirmation: str | None = None,
        fix_group: str = "fix", donors: list[str] | None = None,
        prompt_format: Literal["raw", "chat"] = "raw", attempts: int = 4,
        max_new_tokens: int = 32, max_input_tokens: int = 128,
        improvement_delta: float = 0.10, improvement_floor: float = 0.90,
        preservation_margin: float = 0.05, issue_path: str | None = None,
    ) -> dict:
        """Freeze a model, labeled JSONL splits, gates, and optional donor checkpoints.

        Paths are relative to the configured input root, or absolute within it.
        Use the user's issue and file paths. Fix group names the behavior to improve.
        Other groups are preserved. Only development examples are returned.
        """
        if issue_path is not None:
            if any(value is not None for value in (base, issue, development, confirmation, donors)):
                raise EditError("issue_path cannot be combined with HF shortcut inputs")
            return create_from_issue(issue_path)
        if any(value is None for value in (base, issue, development, confirmation)):
            raise EditError("provide base, issue, development, and confirmation, or an issue_path")
        base_path = allowed_model(input_root, base)
        dev_path = allowed_path(input_root, development)
        heldout_path = allowed_path(input_root, confirmation)
        groups, preview, development_hash = inspect_development(dev_path)
        if fix_group not in groups or len(groups) < 2:
            raise EditError("provide a fix group and at least one preservation group")
        if preservation_margin < 0:
            raise EditError("preservation_margin must be nonnegative")
        if len(donors or []) > 4:
            raise EditError("provide at most four donor checkpoints")
        donor_paths = [allowed_model(input_root, donor) for donor in (donors or [])]
        run_id = uuid4().hex
        with TemporaryDirectory(prefix=".create-", dir=runs_root) as staging:
            workspace = Path(staging) / run_id
            options = SimpleNamespace(
                base=base_path, development=dev_path, confirmation=heldout_path,
                issue=issue, workspace=workspace, prompt_format=prompt_format,
                improve=[[fix_group, groups[fix_group], str(improvement_delta), str(improvement_floor)]],
                preserve=[[group, metric, str(-preservation_margin)]
                          for group, metric in groups.items() if group != fix_group],
                max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, attempts=attempts,
                timeout_seconds=240, batch_size=4, threads=4,
            )
            setup(options, expected_development=development_hash)
            stored_base = workspace / "base"
            names = [item["name"] for item in catalog(stored_base)]
            references, changes = {}, {}
            store = workspace / "donors"
            store.mkdir()
            for index, path in enumerate(donor_paths, 1):
                donor_id = f"donor{index}"
                reference = {"path": str(path), "sha256": checkpoint_hash(path, HF_ADAPTER)}
                frozen = stage_donor(reference, input_root, store, store, HF_ADAPTER)
                references[donor_id] = {**frozen, "path": f"donors/{frozen['path']}"}
                changes[donor_id] = changed_tensors(stored_base, store / frozen["path"], names)
            return {**publish_run(workspace, references, changes), "development_examples": preview,
                    "donors": {name: {"changed_tensors": len(changed)} for name, changed in changes.items()}}

    @server.tool(annotations=read_only)
    @tool_errors
    def list_targets(run_id: str, donor_id: str | None = None, prefix: str = "", limit: int = 60) -> dict:
        """List stored parameters. Filter by donor_id to see the tensors that differ from that donor."""
        workspace, metadata = find_run(run_id)
        integer(limit, "limit", 1, 200)
        entries = catalog(frozen_base(workspace))
        if donor_id is not None:
            if donor_id not in metadata["donors"]:
                raise EditError("unknown donor ID")
            changed = set(metadata["changes"][donor_id])
            entries = [entry for entry in entries if entry["name"] in changed]
        matches = [entry for entry in entries if entry["name"].startswith(prefix)]
        return {"matching": len(matches), "truncated": len(matches) > limit, "tensors": matches[:limit]}

    @server.tool(annotations=creates_artifacts)
    @tool_errors
    def try_edit(
        run_id: str, hypothesis: str | None = None, op: Literal["restore", "blend", "scale"] | None = None,
        target: str | None = None, donor_id: str = "none", amount: float = 1.0, proposal_path: str | None = None,
    ) -> dict:
        """Apply and score a temporary edit on development only.

        Target is a stored tensor name, a module prefix ending in a dot, or
        'changed' for the tensors that differ from the selected donor.
        Restore copies donor weights. Blend uses a donor fraction between zero
        and one. Scale multiplies selected weights and uses donor_id 'none'.
        """
        workspace, metadata = find_run(run_id)
        if proposal_path is not None:
            if any(value is not None for value in (hypothesis, op, target)) or donor_id != "none" or amount != 1:
                raise EditError("proposal_path cannot be combined with shorthand edit fields")
            return summary(approved_proposal(workspace, metadata, proposal_path, attempt))
        if any(value is None for value in (hypothesis, op, target)):
            raise EditError("provide a proposal_path or an edit hypothesis, operation, and target")
        names = [item["name"] for item in catalog(frozen_base(workspace))]
        references = {name: {**ref, "path": str(workspace / ref["path"])}
                      for name, ref in metadata["donors"].items()}
        proposal = materialize({"hypothesis": hypothesis, "op": op, "target": target,
                                "donor": donor_id, "amount": amount}, names=names, donors=references,
                               base_sha=metadata["base_sha256"], author="AutoWeight MCP proposer",
                               changes=metadata["changes"])
        with TemporaryDirectory(prefix=".proposal-", dir=workspace) as directory:
            proposal_path = Path(directory) / "proposal.json"
            write_json(proposal_path, proposal)
            return summary(attempt(workspace, proposal_path))

    @server.tool(annotations=creates_artifacts)
    @tool_errors
    def confirm_edit(run_id: str, attempt_id: str | None = None, proposal_path: str | None = None) -> dict:
        """Use the single confirmation run for a development-eligible attempt and export if it passes."""
        workspace, metadata = find_run(run_id)
        if (attempt_id is None) == (proposal_path is None):
            raise EditError("provide one eligible attempt ID or a final construction proposal")
        if proposal_path is not None:
            report = approved_proposal(workspace, metadata, proposal_path, submit)
        else:
            report = finalize(workspace, attempt_id)
        result = summary(report)
        if result["status"] == "accepted":
            result["checkpoint"] = verify(workspace)["checkpoint"]
        return result

    @server.tool(annotations=read_only)
    @tool_errors
    def get_run(run_id: str) -> dict:
        """Verify a run, show attempt decisions, and return the accepted checkpoint path when present."""
        workspace, _ = find_run(run_id)
        return {"run_id": run_id, **verify(workspace)}

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=os.getenv("AUTOWEIGHT_INPUT_ROOT"))
    parser.add_argument("--runs-dir", type=Path, default=os.getenv("AUTOWEIGHT_RUNS_DIR"))
    parser.add_argument("--allow-evaluator", type=Path, action="append", default=[],
                        help="authorize a trusted local evaluator script for issue_path runs")
    parser.add_argument("--evaluator-python", type=Path, help="interpreter for authorized evaluator scripts")
    args = parser.parse_args()
    try:
        input_root, runs_dir = args.input_root, args.runs_dir
        if input_root is None:
            config = config_path()
            if not config.is_file():
                raise EditError("run auto-weight setup-runtime, set AUTOWEIGHT_INPUT_ROOT, or pass --input-root")
            lines = config.read_text(encoding="utf-8").split("\n")
            if len(lines) < 3 or not lines[1]:
                raise EditError(f"missing input root in {config}; rerun auto-weight setup-runtime")
            input_root = Path(lines[1])
            runs_dir = runs_dir or (Path(lines[2]) if lines[2] else None)
        server = build_server(input_root, runs_dir or Path.home() / ".local/share/auto-weight/runs",
                              evaluator_paths=args.allow_evaluator, evaluator_python=args.evaluator_python)
    except (EditError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
