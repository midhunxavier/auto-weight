"""Freeze inputs, supervise evaluation, and publish a checkpoint only on acceptance."""
from contextlib import contextmanager
import fcntl
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
from math import prod
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from .checkpoint import (
    HF_ADAPTER, LORA_ADAPTER, NAMED_ADAPTER, RAW_HF_ADAPTER, artifact_name, checkpoint_files,
    checkpoint_hash, hf_storage, hf_text,
)
from .contracts import (
    CHECKPOINT_LIMIT, FILE_LIMIT, EditError, file_hash, identify, read_json,
    validate_issue, validate_proposal, write_json,
)

BASELINE_NAME = "development-baseline.json"


def sealed(record):
    return {**record, "id": identify(record)}


def read_sealed(path, limit=1024**2):
    record = read_json(path, limit)
    if not isinstance(record, dict) or "id" not in record:
        raise EditError(f"missing record identity: {path.name}")
    if record["id"] != identify({key: value for key, value in record.items() if key != "id"}):
        raise EditError(f"record identity mismatch: {path.name}")
    return record


def implementation_id():
    return identify({path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")})


def runtime_identity(adapter, evaluator=None):
    try:
        runtime = {"python": sys.version, "numpy": version("numpy")}
        if adapter != NAMED_ADAPTER:
            runtime["torch"] = version("torch")
        if adapter == RAW_HF_ADAPTER:
            runtime["safetensors"] = version("safetensors")
        if evaluator is not None:
            from .evaluator import environment

            runtime["external"] = environment(evaluator["python"])
        if hf_text(adapter):
            for package in ("transformers", "tokenizers", "safetensors", "jsonschema", "referencing"):
                runtime[package] = version(package)
        if adapter == LORA_ADAPTER:
            for package in ("peft", "accelerate"):
                runtime[package] = version(package)
        return runtime
    except PackageNotFoundError as exc:
        extra = "hf,lora" if adapter == LORA_ADAPTER else "hf" if hf_storage(adapter) else "torch"
        raise EditError(f"install auto-weight[{extra}] to run evaluations") from exc


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
    if not hf_storage(adapter):
        return snapshot(reference, origin, destination, CHECKPOINT_LIMIT)
    source = origin / reference["path"]
    files = checkpoint_files(source, adapter)
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
    runtime = runtime_identity(adapter, issue.get("evaluator"))
    workspace.mkdir(mode=0o700)
    try:
        base_name = "adapter" if adapter == LORA_ADAPTER else "base" if hf_storage(adapter) else "base.npz"
        issue["base"] = snapshot_checkpoint(issue["base"], origin, workspace / base_name, adapter)
        if adapter == LORA_ADAPTER:
            issue["backbone"] = snapshot_checkpoint(issue["backbone"], origin, workspace / "backbone", HF_ADAPTER)
        for stage, reference in issue["datasets"].items():
            issue["datasets"][stage] = snapshot(reference, origin, workspace / f"{stage}.jsonl", FILE_LIMIT)
        if hf_text(adapter):
            from .hf import preflight

            preflight(workspace, issue)
        if "evaluator" in issue:
            from .evaluator import preflight

            evaluator = issue["evaluator"]
            evaluator["script"] = snapshot(evaluator["script"], origin, workspace / "evaluator.py", FILE_LIMIT)
            preflight(workspace, issue)
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


def check_base(workspace, issue):
    check_reference(workspace, issue["base"], issue["model"]["adapter"])
    if "backbone" in issue:
        check_reference(workspace, issue["backbone"], HF_ADAPTER)


def load_issue(workspace, require_implementation=True):
    frozen = read_sealed(workspace / "issue.json")
    validate_issue(frozen["issue"])
    if require_implementation and frozen["implementation"] != implementation_id():
        raise EditError("toolkit code changed since initialization")
    if require_implementation and frozen["runtime"] != runtime_identity(
            frozen["issue"]["model"]["adapter"], frozen["issue"].get("evaluator")):
        raise EditError("evaluation runtime changed since initialization")
    if "evaluator" in frozen["issue"]:
        check_reference(workspace, frozen["issue"]["evaluator"]["script"])
    return frozen


def stage_donor(reference, origin, store, staging, adapter):
    """Reuse a verified snapshot or copy it into unpublished staging."""
    digest = reference["sha256"]
    name = digest if hf_storage(adapter) else f"{digest}.npz"
    frozen = {"path": name, "sha256": digest}
    if (store / name).exists():
        check_reference(store, frozen, adapter)
    else:
        snapshot_checkpoint(reference, origin, staging / name, adapter)
    return frozen


def freeze_proposal(proposal_path, destination, issue, workspace):
    base_sha256 = issue["base"]["sha256"]
    adapter = issue["model"]["adapter"]
    proposal = validate_proposal(read_json(proposal_path), base_sha256)
    from .tensors import PATCH_LIMIT, npz_layout

    if adapter == NAMED_ADAPTER and any(op["op"] != "replace" for op in proposal["operations"]):
        raise EditError("named-tensor external adapters support replacement proposals")
    if adapter == LORA_ADAPTER and any("donor" in op for op in proposal["operations"]):
        raise EditError("LoRA runs support scale and replace operations on the one supplied adapter")
    donors = {}
    store = workspace / "donors"
    staging = destination.parent / "donors"
    staging.mkdir()
    patch_bytes = expanded_bytes = 0
    for index, operation in enumerate(proposal["operations"]):
        if "payload" in operation:
            patch = destination / f"replacement-{index}.npz"
            operation["payload"] = snapshot(operation["payload"], proposal_path.parent,
                                            patch, PATCH_LIMIT - patch_bytes)
            patch_bytes += patch.stat().st_size
            layout = npz_layout(patch, PATCH_LIMIT - expanded_bytes)
            if set(layout) != set(operation["tensors"]):
                raise EditError("replacement payload must contain exactly the declared tensors")
            expanded_bytes += sum(prod(shape) * dtype.itemsize for shape, dtype in layout.values())
        if "donor" not in operation:
            continue
        reference = operation["donor"]
        digest = reference["sha256"]
        if digest not in donors:
            if len(donors) >= 4:
                raise EditError("proposal exceeds four distinct donor checkpoints")
            frozen = stage_donor(reference, proposal_path.parent, store, staging, adapter)
            donors[digest] = {**frozen, "path": f"../../donors/{frozen['path']}"}
        operation["donor"] = donors[digest]
    write_json(destination / "proposal.json", sealed(proposal))
    for donor in staging.iterdir():
        store.mkdir(exist_ok=True)
        donor.rename(store / donor.name)
    return proposal


def check_proposal(directory, issue, checked_donors=None):
    frozen = read_sealed(directory / "proposal.json")
    proposal = {key: value for key, value in frozen.items() if key != "id"}
    validate_proposal(proposal, issue["base"]["sha256"])
    if checked_donors is None:
        checked_donors = set()
    for operation in proposal["operations"]:
        if "payload" in operation:
            check_reference(directory, operation["payload"])
        if "donor" in operation:
            donor = operation["donor"]
            key = ((directory / donor["path"]).resolve(), donor["sha256"])
            if key not in checked_donors:
                check_reference(directory, donor, issue["model"]["adapter"])
                checked_donors.add(key)
    return proposal


def evaluate_worker(args, timeout):
    command = [sys.executable, "-m", "auto_weight.worker", *map(str, args)]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            _, errors = process.communicate(timeout=timeout)
        except BaseException:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
    if process.returncode:
        raise EditError(errors.strip().splitlines()[-1][-2000:] if errors.strip()
                        else f"evaluator exited {process.returncode}")


def baseline(workspace):
    workspace = Path(workspace).resolve()
    with locked(workspace), TemporaryDirectory(prefix=".baseline-", dir=workspace) as folder:
        temporary = Path(folder)
        frozen = load_issue(workspace)
        issue = frozen["issue"]
        dataset = issue["datasets"].get("development")
        if dataset is None:
            raise EditError("this issue has no development stage")
        check_base(workspace, issue)
        check_reference(workspace, dataset)
        timeout = issue["budget"]["timeout_seconds"]
        start = monotonic()
        try:
            evaluate_worker([workspace, workspace, temporary, "baseline"], timeout)
        except subprocess.TimeoutExpired as exc:
            raise EditError("baseline exceeded the configured wall time") from exc
        check_base(workspace, issue)
        check_reference(workspace, dataset)
        if (temporary / BASELINE_NAME).is_file():
            (temporary / BASELINE_NAME).replace(workspace / BASELINE_NAME)
        result = read_json(temporary / "evaluation.json")
        meets_targets = [gate["group"] for gate in issue["gates"]
                         if gate["kind"] == "improve" and "min_candidate" in gate
                         and result[gate["group"]]["baseline"][gate["metric"]] >= gate["min_candidate"]]
        report = {"status": "scored", "groups": result, "baseline_meets_target": meets_targets,
                  "timeout_seconds": timeout, "seconds": monotonic() - start}
        write_json(temporary / "summary.json", sealed({"issue_id": frozen["id"], "baseline": report}))
        (temporary / "summary.json").replace(workspace / "baseline-summary.json")
        return report


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
            check_base(workspace, issue)
            check_reference(workspace, dataset)
            if stage == "confirmation" and "development" in issue["datasets"]:
                check_reference(workspace, issue["datasets"]["development"])
            worker_args = [workspace, proposal_dir, temporary, stage]
            if expected_candidate is not None:
                worker_args.append(expected_candidate)
            evaluate_worker(worker_args, issue["budget"]["timeout_seconds"])
            evaluation = read_json(temporary / "evaluation.json", FILE_LIMIT)
            if "evaluator" in issue:
                check_reference(workspace, issue["evaluator"]["script"])
            check_base(workspace, issue)
            check_reference(workspace, dataset)
            check_proposal(proposal_dir, issue)
            candidate = temporary / name
            if checkpoint_hash(candidate, adapter, frozen=True) != evaluation["candidate_sha256"]:
                raise EditError("evaluated candidate hash mismatch")
            baseline = temporary / BASELINE_NAME
            if stage == "development" and baseline.is_file():
                baseline.replace(workspace / BASELINE_NAME)
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
        if "development" not in frozen["issue"]["datasets"]:
            raise EditError("construction accepts one final submission and has no practice stage")
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
            freeze_proposal(proposal_path, staging, frozen["issue"], workspace)
            staging.rename(destination)
        return run_trial(workspace, frozen, destination, destination, "development")


def check_report(directory, frozen, stage, proposal):
    result = directory / "result"
    report = read_sealed(result / "report.json", FILE_LIMIT)
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


def submit(workspace, proposal_path):
    """Reserve and score a single construction candidate on the final cases."""
    workspace, proposal_path = Path(workspace).resolve(), Path(proposal_path).resolve()
    with locked(workspace):
        frozen = load_issue(workspace)
        issue = frozen["issue"]
        if "development" in issue["datasets"]:
            raise EditError("repair runs require an eligible development attempt")
        destination = workspace / "confirmation"
        if destination.exists():
            raise EditError("confirmation already consumed")
        with TemporaryDirectory(dir=workspace) as temporary:
            staging = Path(temporary) / "confirmation"
            staging.mkdir()
            freeze_proposal(proposal_path, staging, issue, workspace)
            staging.rename(destination)
        return run_trial(workspace, frozen, destination, destination, "confirmation")


def verify(workspace, *, include_details=False):
    workspace = Path(workspace).resolve()
    with locked(workspace):
        frozen = load_issue(workspace, require_implementation=False)
        issue = frozen["issue"]
        adapter = issue["model"]["adapter"]
        check_base(workspace, issue)
        for reference in issue["datasets"].values():
            check_reference(workspace, reference)
        reports, checked_donors = {}, set()
        for directory in sorted((workspace / "attempts").iterdir()):
            proposal = check_proposal(directory, issue, checked_donors)
            if not (directory / "result").exists():
                raise EditError(f"development attempt {directory.name} was interrupted")
            report = check_report(directory, frozen, "development", proposal)
            if report["attempt"] != directory.name:
                raise EditError("development attempt ID mismatch")
            reports[directory.name] = report
        destination = workspace / "confirmation"
        confirmation = None
        if destination.exists() and "development" not in issue["datasets"]:
            if not (destination / "result").exists():
                raise EditError("confirmation was interrupted and is consumed")
            proposal = check_proposal(destination, issue)
            confirmation = check_report(destination, frozen, "confirmation", proposal)
            if confirmation["attempt"] != "confirmation":
                raise EditError("construction confirmation identity mismatch")
        elif destination.exists():
            selection = read_sealed(destination / "selection.json")
            attempt_id = selection["attempt"]
            if attempt_id not in reports or selection["development_id"] != reports[attempt_id]["id"]:
                raise EditError("confirmation selection does not match a development report")
            if not (destination / "result").exists():
                raise EditError("confirmation was interrupted and is consumed")
            proposal = check_proposal(workspace / "attempts" / attempt_id, issue, checked_donors)
            confirmation = check_report(destination, frozen, "confirmation", proposal)
            if confirmation["attempt"] != attempt_id:
                raise EditError("confirmation attempt ID mismatch")
            if (confirmation["status"] == "accepted" and confirmation["evaluation"]["candidate_sha256"]
                    != reports[attempt_id]["evaluation"]["candidate_sha256"]):
                raise EditError("accepted checkpoint differs from the selected development candidate")
        baseline_path = workspace / "baseline-summary.json"
        scores = None
        if baseline_path.is_file():
            baseline_record = read_sealed(baseline_path)
            if baseline_record["issue_id"] != frozen["id"]:
                raise EditError("baseline summary does not match frozen issue")
            scores = baseline_record["baseline"]
        result = {"status": "verified", "issue_id": frozen["id"], "baseline": scores,
                  "attempts": {key: value["status"] for key, value in reports.items()},
                  "attempt_limit": issue["budget"]["attempts"], "attempts_used": len(reports),
                  "confirmation": confirmation["status"] if confirmation else None,
                  "checkpoint": str(destination / "result" / artifact_name(adapter))
                  if confirmation and confirmation["status"] == "accepted" else None}
        if include_details:
            result.update(issue=issue, attempt_reports=reports, confirmation_report=confirmation)
        if "backbone" in issue:
            result.update(backbone=str(workspace / issue["backbone"]["path"]),
                          backbone_sha256=issue["backbone"]["sha256"])
        return result
