"""File protocol for trusted local evaluator scripts."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from .checkpoint import HF_METADATA, INDEX_FILE, NAMED_ADAPTER, checkpoint_hash, hf_files
from .contracts import (
    FILE_LIMIT, EditError, fields, file_hash, finite, identify, integer, read_json, strict_json, string, write_json,
)
from .statistics import baseline_summary, paired_report
from .tensors import check_changes, load_arrays, replacement_items, replace_arrays, replace_torch

PROTOCOL = "wet.evaluation.v1"


def environment(python):
    command = """import importlib.metadata as m, json, platform, sys
print(json.dumps([sys.version, platform.platform(),
    sorted((d.metadata['Name'], d.version) for d in m.distributions() if d.metadata['Name'])]))
"""
    try:
        result = subprocess.run([python, "-I", "-c", command], check=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EditError("could not identify external evaluator runtime") from exc
    return {"interpreter": file_hash(Path(python).resolve()), "environment": strict_json(result.stdout)}


def cases(path):
    rows = {}
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            integer(number, "case count", 1, 10000)
            row = strict_json(raw, 128 * 1024)
            fields(row, {"id", "group", "inputs"}, "external case")
            key = string(row["id"], "case ID", 128)
            string(row["group"], "group", 80)
            if key in rows:
                raise EditError("duplicate case ID")
            rows[key] = row
    return rows


def preflight(workspace, issue):
    seen, inputs_seen = set(), set()
    for stage, reference in issue["datasets"].items():
        try:
            rows = cases(workspace / reference["path"])
            counts = Counter(row["group"] for row in rows.values())
            inputs = {identify(row["inputs"]) for row in rows.values()}
            if seen.intersection(rows) or inputs_seen.intersection(inputs) or len(inputs) != len(rows):
                raise EditError("duplicate case inputs or overlap across splits")
            seen.update(rows)
            inputs_seen.update(inputs)
            sample_count = len(rows) * len(issue["evaluator"]["seeds"])
            if sample_count > min(200_000, issue["evaluator"]["max_work"]):
                raise EditError("external evaluation exceeds sample count or work limit")
            minimum = 1 if issue["evaluator"]["mode"] == "construct" else 32
            if any(counts[gate["group"]] < minimum for gate in issue["gates"]):
                raise EditError(f"every gated group requires at least {minimum} independent cases")
        except EditError as exc:
            if stage == "confirmation":
                raise EditError("invalid confirmation case manifest; check schema, overlap, and group counts") from exc
            raise


def observe(workspace, issue, artifact, artifact_sha, stage, temporary):
    evaluator = issue["evaluator"]
    dataset = issue["datasets"][stage]
    case_path = workspace / dataset["path"]
    rows = cases(case_path)
    seeds = evaluator["seeds"]
    names = sorted(evaluator["metrics"])
    request = {
        "version": PROTOCOL, "stage": stage, "artifact": str(artifact), "artifact_sha256": artifact_sha,
        "cases": str(case_path), "cases_sha256": dataset["sha256"], "seeds": seeds, "metrics": names,
        "config": evaluator["config"], "max_work": evaluator["max_work"],
        "timeout_seconds": issue["budget"]["timeout_seconds"],
        "evaluator_sha256": evaluator["script"]["sha256"],
    }
    request["evaluation_id"] = identify({key: value for key, value in request.items()
                                         if key not in {"artifact", "cases", "timeout_seconds"}})
    with TemporaryDirectory(prefix="external-", dir=temporary) as folder:
        folder = Path(folder)
        request_path = folder / "request.json"
        response_path = folder / "response.json"
        write_json(request_path, request)
        with (folder / "evaluator.log").open("w+b") as log:
            result = subprocess.run([evaluator["python"], "-I", str(workspace / evaluator["script"]["path"]),
                                     str(request_path), str(response_path)], cwd=folder,
                                    stdout=log, stderr=log, check=False)
            if result.returncode or not response_path.is_file():
                log.seek(max(0, log.tell() - 1000))
                detail = log.read().decode(errors="replace").strip() if stage == "development" else ""
                raise EditError(f"external evaluator failed ({result.returncode})" + (f": {detail}" if detail else ""))
        response = read_json(response_path, FILE_LIMIT)
    fields(response, {"version", "evaluation_id", "artifact_sha256", "observations", "work"}, "evaluation response")
    if (response["version"], response["evaluation_id"], response["artifact_sha256"]) != (
            PROTOCOL, request["evaluation_id"], artifact_sha):
        raise EditError("external response identity mismatch")
    work = integer(response["work"], "reported work", len(rows) * len(seeds), evaluator["max_work"])
    observations = response["observations"]
    if not isinstance(observations, list) or len(observations) != len(rows) * len(seeds):
        raise EditError("external response has missing or extra observations")
    indexed = {}
    for item in observations:
        fields(item, {"id", "seed", "metrics"}, "external observation")
        key = string(item["id"], "observation case ID", 128)
        seed = integer(item["seed"], "observation seed", 0, 2**32 - 1)
        if key not in rows or seed not in seeds or (key, seed) in indexed:
            raise EditError("invalid or duplicate external observation identity")
        fields(item["metrics"], set(names), "observed metrics")
        indexed[key, seed] = [finite(item["metrics"][name], name) for name in names]
    groups = defaultdict(list)
    for key, row in rows.items():
        groups[row["group"]].append(np.asarray([indexed[key, seed] for seed in seeds]).mean(axis=0))
    scores = {group: np.asarray(values) for group, values in groups.items()}
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise EditError("external score aggregation exceeded numeric limits")
    return scores, {"work": work, "observations": observations}


def hf_layout(path):
    """Validate canonical unquantized storage without assuming a model architecture."""
    from safetensors import safe_open

    files = hf_files(path)
    config = read_json(path / "config.json")
    if config.get("quantization_config") is not None:
        raise EditError("raw HF edits require unquantized weights")
    index = {}
    for name, source in files.items():
        if name.endswith(".safetensors"):
            with safe_open(source, framework="pt") as archive:
                for tensor in archive.keys():
                    if tensor in index:
                        raise EditError("duplicate stored tensor")
                    index[tensor] = name
    if INDEX_FILE in files and read_json(files[INDEX_FILE], 16 * 1024**2)["weight_map"] != index:
        raise EditError("safetensors index does not match stored tensors")
    text_config = config.get("text_config", {})
    tied_embeddings = config.get("tie_word_embeddings") or text_config.get("tie_word_embeddings")
    if tied_embeddings and any(name.endswith("lm_head.weight") for name in index):
        raise EditError("tied output embeddings must use the canonical single stored tensor")
    return files, index, config


def compatible_hf_donor(base_path, base_index, base_config, donor_path):
    files, index, config = hf_layout(donor_path)
    for name in HF_METADATA - {"config.json", "generation_config.json"}:
        base, donor = base_path / name, donor_path / name
        if base.is_file() != donor.is_file() or (base.is_file() and file_hash(base) != file_hash(donor)):
            raise EditError("donor tokenizer assets differ from base")
    ignored = {"_name_or_path", "transformers_version"}
    if ({k: v for k, v in base_config.items() if k not in ignored}
            != {k: v for k, v in config.items() if k not in ignored}):
        raise EditError("donor architecture config differs from base")
    if index.keys() != base_index.keys():
        raise EditError("donor stored tensor names differ from base")
    return files, index


def native_hf_change(tensor, target, operation, donor=None):
    """Use the same float64 arithmetic and stored-dtype rounding as native HF edits."""
    import torch

    if target.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise EditError(f"native operation requires dense floating-point weights: {tensor}")
    before = target.double()
    kind = operation["op"]
    if kind == "scale":
        after = (before * operation["factor"]).to(target.dtype)
    else:
        if donor.shape != target.shape or donor.dtype != target.dtype:
            raise EditError(f"donor parameter shape or dtype differs: {tensor}")
        if kind == "restore":
            after = donor
        else:
            alpha = operation["alpha"]
            after = ((1 - alpha) * before + alpha * donor.double()).to(target.dtype)
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise EditError(f"nonfinite edited parameter: {tensor}")
    delta = after.double() - before
    count = int(torch.count_nonzero(delta))
    item = {"tensor": tensor, "op": kind, "parameters_changed": count,
            "l2_delta": delta.norm().item(), "max_abs_delta": delta.abs().max().item()} if count else None
    return after, item


def materialize(base_path, candidate_path, proposal, directory, adapter):
    changes = []
    if adapter == NAMED_ADAPTER:
        if any(operation["op"] != "replace" for operation in proposal["operations"]):
            raise EditError("named-tensor external adapters support replacement proposals")
        base = load_arrays(base_path)
        candidate = base.copy()
        for operation in proposal["operations"]:
            changes.extend(replace_arrays(base, candidate, operation, directory))
        check_changes(changes)
        np.savez(candidate_path, **candidate)
        return changes

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    # Preserve the exact shard index and metadata. Tied parameters are stored once.
    files, index, config = hf_layout(base_path)
    candidate_path.mkdir()
    by_shard = defaultdict(list)
    donors = {}
    for operation in proposal["operations"]:
        if operation["op"] == "replace":
            items = [(name, (bounds, values)) for name, bounds, values in replacement_items(operation, directory)]
        else:
            if "donor" in operation:
                donor_path = (directory / operation["donor"]["path"]).resolve()
                if donor_path not in donors:
                    donors[donor_path] = compatible_hf_donor(base_path, index, config, donor_path)
            items = [(name, None) for name in operation["tensors"]]
        for tensor, replacement in items:
            if tensor not in index:
                raise EditError(f"unknown stored tensor: {tensor}")
            by_shard[index[tensor]].append((tensor, operation, replacement))
    with ExitStack() as stack:
        readers = {}
        for name, source in files.items():
            target = candidate_path / name
            if name not in by_shard:
                shutil.copyfile(source, target)
                continue
            weights = load_file(source)
            for tensor, operation, replacement in by_shard[name]:
                if replacement is not None:
                    bounds, values = replacement
                    item = replace_torch(tensor, weights[tensor], bounds, values)
                else:
                    donor = None
                    if "donor" in operation:
                        donor_path = (directory / operation["donor"]["path"]).resolve()
                        donor_files, donor_index = donors[donor_path]
                        path = donor_files[donor_index[tensor]]
                        if path not in readers:
                            readers[path] = stack.enter_context(safe_open(path, framework="pt"))
                        donor = readers[path].get_tensor(tensor)
                    weights[tensor], item = native_hf_change(tensor, weights[tensor], operation, donor)
                if item:
                    changes.append(item)
            with safe_open(source, framework="pt") as archive:
                save_file(weights, target, metadata=archive.metadata())
    check_changes(changes)
    return changes


def evaluate(workspace, issue, proposal, proposal_dir, stage, temporary, expected_candidate=None, *, score_baseline):
    from .checkpoint import artifact_name

    adapter = issue["model"]["adapter"]
    evaluator = issue["evaluator"]
    base = workspace / issue["base"]["path"]
    directions = evaluator["metrics"]
    metrics = {row["group"]: tuple(sorted(directions))
               for row in cases(workspace / issue["datasets"][stage]["path"]).values()}
    if proposal is not None:
        candidate = temporary / artifact_name(adapter)
        changes = materialize(base, candidate, proposal, proposal_dir, adapter)
        candidate_sha = checkpoint_hash(candidate, adapter, frozen=True)
        if expected_candidate is not None and candidate_sha != expected_candidate:
            raise EditError("replayed candidate differs from development candidate")

    constructing = evaluator["mode"] == "construct"
    if not constructing:
        before, before_evidence = score_baseline(
            lambda: observe(workspace, issue, base, issue["base"]["sha256"], stage, temporary))
        if proposal is None:
            return baseline_summary(before, metrics)
    after, evidence = observe(workspace, issue, candidate, candidate_sha, stage, temporary)
    if constructing:
        groups = {group: {"examples": len(values), "candidate": dict(zip(metrics[group], values.mean(0).tolist()))}
                  for group, values in after.items()}
        gates = [{**gate, "passed": groups[gate["group"]]["candidate"][gate["metric"]] >= gate["min_candidate"]}
                 for gate in issue["gates"]]
        report = {"groups": groups, "gates": gates, "passed": all(gate["passed"] for gate in gates)}
        work = {"candidate": evidence["work"]}
    else:
        report = paired_report(before, after, issue["gates"], metrics, directions=directions)
        work = {"baseline": before_evidence["work"], "candidate": evidence["work"]}
    if stage == "development":
        report["observations"] = {"baseline": before_evidence["observations"],
                                  "candidate": evidence["observations"]}
    return {**report, "work": work, "changes": changes, "candidate_sha256": candidate_sha}
