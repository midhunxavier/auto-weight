"""Local stdio tools for agents that propose and evaluate checkpoint edits."""
import argparse
from functools import wraps
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .automation import changed_tensors, inspect_development, materialize
from .checkpoint import HF_ADAPTER, checkpoint_hash, hf_files, tensor_metadata
from .contracts import EditError, integer, write_json
from .quickstart import setup
from .runner import (
    attempt, finalize, read_sealed, sealed, snapshot_checkpoint, verify,
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
    result = {"stage": stage, "attempt": report["attempt"], "status": report["status"]}
    if report["error"]:
        result["error"] = (report["error"] if stage == "development"
                           else "Confirmation evaluation failed. See the local report.")
    if report["evaluation"]:
        data = report["evaluation"]
        result.update(groups=data["groups"], gates=data["gates"], candidate_sha256=data["candidate_sha256"])
        if stage == "development":
            result["examples"] = data.get("development_samples", {})
    return result


def build_server(input_root, runs_root):
    input_root, runs_root = Path(input_root).resolve(strict=True), Path(runs_root).resolve()
    if not input_root.is_dir():
        raise EditError("input root must be an existing directory")
    runs_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    server = FastMCP("autoweight", instructions=(
        "Run checkpoint repair experiments on local Hugging Face models. Use create_run, list_targets, "
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

    @server.tool(annotations=creates_artifacts)
    @tool_errors
    def create_run(
        base: str, issue: str, development: str, confirmation: str,
        fix_group: str = "fix", donors: list[str] | None = None,
        prompt_format: Literal["raw", "chat"] = "raw", attempts: int = 4,
        max_new_tokens: int = 32, max_input_tokens: int = 128,
        improvement_delta: float = 0.10, improvement_floor: float = 0.90,
        preservation_margin: float = 0.05,
    ) -> dict:
        """Freeze a model, labeled JSONL splits, gates, and optional donor checkpoints.

        Paths are relative to the configured input root, or absolute within it.
        Use the user's issue and file paths. Fix group names the behavior to improve.
        Other groups are preserved. Only development examples are returned.
        """
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
            ready = setup(options, expected_development=development_hash)
            stored_base = workspace / "base"
            names = [item["name"] for item in catalog(stored_base)]
            references, changes = {}, {}
            for index, path in enumerate(donor_paths, 1):
                donor_id = f"donor{index}"
                reference = {"path": str(path), "sha256": checkpoint_hash(path, HF_ADAPTER)}
                references[donor_id] = snapshot_checkpoint(reference, input_root, workspace / donor_id, HF_ADAPTER)
                changes[donor_id] = changed_tensors(stored_base, workspace / donor_id, names)
            record = {"issue_id": ready["issue_id"], "base_sha256": ready["base_sha256"],
                      "donors": references, "changes": changes}
            write_json(workspace / "mcp-run.json", sealed(record))
            gates = read_sealed(workspace / "issue.json")["issue"]["gates"]
            workspace.rename(runs_root / run_id)
        return {"status": "ready", "run_id": run_id, "gates": gates, "attempt_limit": attempts,
                "development_examples": preview,
                "donors": {name: {"changed_tensors": len(changed)} for name, changed in changes.items()}}

    @server.tool(annotations=read_only)
    @tool_errors
    def list_targets(run_id: str, donor_id: str | None = None, prefix: str = "", limit: int = 60) -> dict:
        """List stored parameters. Filter by donor_id to see the tensors that differ from that donor."""
        workspace, metadata = find_run(run_id)
        integer(limit, "limit", 1, 200)
        entries = catalog(workspace / "base")
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
        run_id: str, hypothesis: str, op: Literal["restore", "blend", "scale"],
        target: str, donor_id: str = "none", amount: float = 1.0,
    ) -> dict:
        """Apply and score a temporary edit on development only.

        Target is a stored tensor name, a module prefix ending in a dot, or
        'changed' for the tensors that differ from the selected donor.
        Restore copies donor weights. Blend uses a donor fraction between zero
        and one. Scale multiplies selected weights and uses donor_id 'none'.
        """
        workspace, metadata = find_run(run_id)
        names = [item["name"] for item in catalog(workspace / "base")]
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
    def confirm_edit(run_id: str, attempt_id: str) -> dict:
        """Use the single confirmation run for a development-eligible attempt and export if it passes."""
        workspace, _ = find_run(run_id)
        result = summary(finalize(workspace, attempt_id))
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
    args = parser.parse_args()
    try:
        input_root, runs_dir = args.input_root, args.runs_dir
        if input_root is None:
            config = config_path()
            if not config.is_file():
                raise EditError("run weight-edit setup-runtime, set AUTOWEIGHT_INPUT_ROOT, or pass --input-root")
            lines = config.read_text(encoding="utf-8").split("\n")
            if len(lines) < 3 or not lines[1]:
                raise EditError(f"missing input root in {config}; rerun weight-edit setup-runtime")
            input_root = Path(lines[1])
            runs_dir = runs_dir or (Path(lines[2]) if lines[2] else None)
        server = build_server(input_root, runs_dir or Path.home() / ".local/share/autoweight/runs")
    except (EditError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
