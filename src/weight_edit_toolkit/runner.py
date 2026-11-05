"""Freeze inputs, supervise evaluation, and publish a checkpoint only on acceptance."""
from contextlib import contextmanager
import fcntl
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from .checkpoint import HF_ADAPTER, artifact_name, checkpoint_hash, hf_files
from .contracts import (
    CHECKPOINT_LIMIT, FILE_LIMIT, EditError, file_hash, identify, read_json,
    validate_issue, validate_proposal, write_json,
)


def sealed(record):
    return {**record, "id": identify(record)}


def read_sealed(path):
    record = read_json(path)
    if not isinstance(record, dict) or "id" not in record:
        raise EditError(f"missing record identity: {path.name}")
    if record["id"] != identify({key: value for key, value in record.items() if key != "id"}):
        raise EditError(f"record identity mismatch: {path.name}")
    return record


def implementation_id():
    return identify({path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")})


def runtime_identity(adapter):
    try:
        runtime = {"python": sys.version, "numpy": version("numpy"), "torch": version("torch")}
        if adapter == HF_ADAPTER:
            for package in ("transformers", "tokenizers", "safetensors", "jsonschema", "referencing"):
                runtime[package] = version(package)
        return runtime
    except PackageNotFoundError as exc:
        extra = "hf" if adapter == HF_ADAPTER else "torch"
        raise EditError(f"install weight-edit-toolkit[{extra}] to run evaluations") from exc


def snapshot(reference, origin, destination, limit):
    source = origin / reference["path"]
    if not source.is_file() or source.stat().st_size > limit:
        raise EditError(f"missing file or byte limit exceeded: {source}")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        remaining = limit
        while chunk := incoming.read(min(1024**2, remaining + 1)):
            if len(chunk) > remaining:
                raise EditError(f"byte limit exceeded: {source}")
            outgoing.write(chunk)
            remaining -= len(chunk)
    if file_hash(destination) != reference["sha256"]:
        raise EditError(f"input hash mismatch: {source}")
    return {"path": destination.name, "sha256": reference["sha256"]}


def snapshot_checkpoint(reference, origin, destination, adapter):
    if adapter != HF_ADAPTER:
        return snapshot(reference, origin, destination, CHECKPOINT_LIMIT)
    source = origin / reference["path"]
    files = hf_files(source)
    hashes = {name: file_hash(path) for name, path in files.items()}
    if identify(hashes) != reference["sha256"]:
        raise EditError(f"input checkpoint hash mismatch: {source}")
    destination.mkdir()
    for name, path in files.items():
        snapshot({"path": str(path.resolve()), "sha256": hashes[name]}, origin,
                 destination / name, path.stat().st_size)
    return {"path": destination.name, "sha256": reference["sha256"]}


def initialize(issue_path, workspace):
    issue_path = Path(issue_path).resolve()
    return initialize_issue(read_json(issue_path), issue_path.parent, workspace)


def initialize_issue(issue, origin, workspace):
    issue = validate_issue(issue)
    workspace = Path(workspace).resolve()
    adapter = issue["model"]["adapter"]
    runtime = runtime_identity(adapter)
    workspace.mkdir(mode=0o700)
    try:
        base_name = "base" if adapter == HF_ADAPTER else "base.npz"
        issue["base"] = snapshot_checkpoint(issue["base"], origin, workspace / base_name, adapter)
        for stage, reference in issue["datasets"].items():
            issue["datasets"][stage] = snapshot(reference, origin, workspace / f"{stage}.jsonl", FILE_LIMIT)
        (workspace / "attempts").mkdir()
        # Publish the issue last so concurrent callers cannot open a partial workspace.
        with TemporaryDirectory(dir=workspace) as staging:
            path = Path(staging) / "issue.json"
            record = sealed({"issue": issue, "implementation": implementation_id(), "runtime": runtime})
            write_json(path, record)
            path.rename(workspace / "issue.json")
    except BaseException:
        shutil.rmtree(workspace)
        raise
    return {"status": "ready", "issue_id": record["id"], "workspace": str(workspace)}


@contextmanager
def locked(workspace):
    if not (workspace / "issue.json").is_file():
        raise EditError("workspace has not been initialized")
    with (workspace / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EditError("workspace already has an active operation") from exc
        yield


def check_reference(directory, reference, adapter=None):
    path = directory / reference["path"]
    if checkpoint_hash(path, adapter, frozen=True) != reference["sha256"]:
        raise EditError(f"frozen input hash mismatch: {path.name}")


def load_issue(workspace, require_implementation=True):
    frozen = read_sealed(workspace / "issue.json")
    validate_issue(frozen["issue"])
    if require_implementation and frozen["implementation"] != implementation_id():
        raise EditError("toolkit code changed since initialization")
    if require_implementation and frozen["runtime"] != runtime_identity(frozen["issue"]["model"]["adapter"]):
        raise EditError("evaluation runtime changed since initialization")
    return frozen


def freeze_proposal(proposal_path, destination, issue):
    base_sha256 = issue["base"]["sha256"]
    adapter = issue["model"]["adapter"]
    proposal = validate_proposal(read_json(proposal_path), base_sha256)
    donors = {}
    for operation in proposal["operations"]:
        if "donor" not in operation:
            continue
        reference = operation["donor"]
        digest = reference["sha256"]
        if digest not in donors:
            if len(donors) >= 4:
                raise EditError("proposal exceeds four distinct donor checkpoints")
            name = digest if adapter == HF_ADAPTER else f"{digest}.npz"
            donors[digest] = snapshot_checkpoint(reference, proposal_path.parent, destination / name, adapter)
        operation["donor"] = donors[digest]
    write_json(destination / "proposal.json", sealed(proposal))
    return proposal


def check_proposal(directory, issue):
    frozen = read_sealed(directory / "proposal.json")
    proposal = {key: value for key, value in frozen.items() if key != "id"}
    validate_proposal(proposal, issue["base"]["sha256"])
    for operation in proposal["operations"]:
        if "donor" in operation:
            check_reference(directory, operation["donor"], issue["model"]["adapter"])
    return proposal


def run_trial(workspace, frozen, proposal_dir, destination, stage, expected_candidate=None):
    issue = frozen["issue"]
    proposal = check_proposal(proposal_dir, issue)
    dataset = issue["datasets"][stage]
    adapter = issue["model"]["adapter"]
    name = artifact_name(adapter)
    start = monotonic()
    with TemporaryDirectory(prefix=".trial-", dir=workspace) as temporary:
        temporary = Path(temporary)
        result_dir = temporary / "result"
        result_dir.mkdir()
        status, evaluation, error = "error", None, None
        try:
            check_reference(workspace, issue["base"], adapter)
            check_reference(workspace, dataset)
            if stage == "confirmation":
                check_reference(workspace, issue["datasets"]["development"])
            worker_args = [sys.executable, "-m", "weight_edit_toolkit.worker",
                           str(workspace), str(proposal_dir), str(temporary), stage]
            if expected_candidate is not None:
                worker_args.append(expected_candidate)
            worker = subprocess.run(worker_args, timeout=issue["budget"]["timeout_seconds"],
                                    capture_output=True, text=True, check=False)
            if worker.returncode != 0:
                raise EditError(worker.stderr.strip()[-2000:] or f"evaluator exited {worker.returncode}")
            evaluation = read_json(temporary / "evaluation.json")
            check_reference(workspace, issue["base"], adapter)
            check_reference(workspace, dataset)
            check_proposal(proposal_dir, issue)
            candidate = temporary / name
            if checkpoint_hash(candidate, adapter, frozen=True) != evaluation["candidate_sha256"]:
                raise EditError("evaluated candidate hash mismatch")
            if evaluation["passed"] and stage == "confirmation":
                status = "accepted"
            elif evaluation["passed"]:
                status = "eligible"
            else:
                status = "discarded"
        except subprocess.TimeoutExpired:
            error = "evaluation exceeded the configured wall time"
        except (OSError, EditError) as exc:
            error = str(exc)
        report = sealed({
            "version": "wet.trial.v1", "issue_id": frozen["id"], "stage": stage,
            "attempt": proposal_dir.name, "proposal_id": identify(proposal),
            "base_sha256": issue["base"]["sha256"], "dataset_sha256": dataset["sha256"],
            "model": issue["model"], "status": status, "error": error,
            "evaluation": evaluation, "seconds": monotonic() - start,
        })
        if status == "accepted":
            candidate.rename(result_dir / name)
        write_json(result_dir / "report.json", report)
        result_dir.rename(destination / "result")
    return report


def attempt(workspace, proposal_path):
    workspace, proposal_path = Path(workspace).resolve(), Path(proposal_path).resolve()
    with locked(workspace):
        frozen = load_issue(workspace)
        if (workspace / "confirmation").exists():
            raise EditError("confirmation has closed this workspace")
        attempts = workspace / "attempts"
        number = len(list(attempts.iterdir())) + 1
        if number > frozen["issue"]["budget"]["attempts"]:
            raise EditError("development attempt budget exhausted")
        destination = attempts / f"{number:03d}"
        # Invalid proposals never reserve an attempt slot or create an evaluable candidate.
        with TemporaryDirectory(dir=workspace) as temporary:
            staging = Path(temporary) / "attempt"
            staging.mkdir()
            freeze_proposal(proposal_path, staging, frozen["issue"])
            staging.rename(destination)
        return run_trial(workspace, frozen, destination, destination, "development")


def check_report(directory, frozen, stage, proposal):
    result = directory / "result"
    report = read_sealed(result / "report.json")
    issue = frozen["issue"]
    if (report["issue_id"], report["stage"], report["proposal_id"], report["base_sha256"],
            report["dataset_sha256"]) != (
            frozen["id"], stage, identify(proposal), issue["base"]["sha256"],
            issue["datasets"][stage]["sha256"]):
        raise EditError("trial report does not match frozen inputs")
    permitted = {"eligible", "discarded", "error"} if stage == "development" else {"accepted", "discarded", "error"}
    if report["status"] not in permitted or report["model"] != issue["model"]:
        raise EditError("invalid trial result status or model config")
    if report["status"] != "error" and report["evaluation"]["passed"] != (report["status"] != "discarded"):
        raise EditError("trial status disagrees with evaluation")
    expected = {"report.json"}
    if report["status"] == "accepted":
        if stage != "confirmation" or not report["evaluation"]["passed"]:
            raise EditError("invalid acceptance record")
        adapter = issue["model"]["adapter"]
        name = artifact_name(adapter)
        expected.add(name)
        if checkpoint_hash(result / name, adapter, frozen=True) != report["evaluation"]["candidate_sha256"]:
            raise EditError("accepted checkpoint hash mismatch")
    if {path.name for path in result.iterdir()} != expected:
        raise EditError("unexpected files in trial result")
    return report


def finalize(workspace, attempt_id):
    workspace = Path(workspace).resolve()
    if not isinstance(attempt_id, str) or len(attempt_id) != 3 or not attempt_id.isascii() or not attempt_id.isdecimal():
        raise EditError("attempt ID must be three digits")
    with locked(workspace):
        frozen = load_issue(workspace)
        destination = workspace / "confirmation"
        if destination.exists():
            raise EditError("confirmation already consumed")
        proposal_dir = workspace / "attempts" / attempt_id
        proposal = check_proposal(proposal_dir, frozen["issue"])
        report = check_report(proposal_dir, frozen, "development", proposal)
        if report["attempt"] != attempt_id or report["status"] != "eligible" or not report["evaluation"]["passed"]:
            raise EditError("only a passing development attempt can be finalized")
        candidate_sha = report["evaluation"]["candidate_sha256"]
        # Reserve confirmation before the evaluator can access its examples.
        with TemporaryDirectory(dir=workspace) as temporary:
            staging = Path(temporary) / "confirmation"
            staging.mkdir()
            write_json(staging / "selection.json", sealed({"attempt": attempt_id, "development_id": report["id"]}))
            staging.rename(destination)
        return run_trial(workspace, frozen, proposal_dir, destination, "confirmation", candidate_sha)


def verify(workspace):
    workspace = Path(workspace).resolve()
    with locked(workspace):
        frozen = load_issue(workspace, require_implementation=False)
        issue = frozen["issue"]
        adapter = issue["model"]["adapter"]
        check_reference(workspace, issue["base"], adapter)
        for reference in issue["datasets"].values():
            check_reference(workspace, reference)
        reports = {}
        for directory in sorted((workspace / "attempts").iterdir()):
            proposal = check_proposal(directory, issue)
            if not (directory / "result").exists():
                raise EditError(f"development attempt {directory.name} was interrupted")
            report = check_report(directory, frozen, "development", proposal)
            if report["attempt"] != directory.name:
                raise EditError("development attempt ID mismatch")
            reports[directory.name] = report
        destination = workspace / "confirmation"
        confirmation = None
        if destination.exists():
            selection = read_sealed(destination / "selection.json")
            attempt_id = selection["attempt"]
            if attempt_id not in reports or selection["development_id"] != reports[attempt_id]["id"]:
                raise EditError("confirmation selection does not match a development report")
            if not (destination / "result").exists():
                raise EditError("confirmation was interrupted and is consumed")
            proposal = check_proposal(workspace / "attempts" / attempt_id, issue)
            confirmation = check_report(destination, frozen, "confirmation", proposal)
            if confirmation["attempt"] != attempt_id:
                raise EditError("confirmation attempt ID mismatch")
            if (confirmation["status"] == "accepted" and confirmation["evaluation"]["candidate_sha256"]
                    != reports[attempt_id]["evaluation"]["candidate_sha256"]):
                raise EditError("accepted checkpoint differs from the selected development candidate")
        return {"status": "verified", "issue_id": frozen["id"],
                "attempts": {key: value["status"] for key, value in reports.items()},
                "confirmation": confirmation["status"] if confirmation else None,
                "checkpoint": str(destination / "result" / artifact_name(adapter))
                if confirmation and confirmation["status"] == "accepted" else None}
