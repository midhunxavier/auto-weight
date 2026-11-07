"""Strict input records and content identities."""
import hashlib
import json
import math
from pathlib import Path
import re

import rfc8785


class EditError(ValueError):
    pass


def canonical(value):
    try:
        return rfc8785.dumps(value)
    except (ValueError, TypeError) as exc:
        raise EditError(f"not canonicalizable: {exc}") from exc


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def identify(value):
    return sha256(canonical(value))


def strict_json(raw, max_bytes=16 * 1024 * 1024):
    if len(raw) > max_bytes:
        raise EditError("JSON exceeds byte limit")

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise EditError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def reject(value):
        raise EditError(f"invalid JSON number: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
        canonical(value)  # Also rejects exponent overflow, large integers and invalid Unicode.
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise EditError(str(exc)) from exc


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EditError(f"{name} must be finite")
    return value


FILE_LIMIT = 128 * 1024**2
CHECKPOINT_LIMIT = 64 * 1024**2
MODEL_FIELDS = {"width", "heads", "layers", "vocab", "context"}
METRICS = {"accuracy": 1, "nll": -1}
TEXT_METRICS = {"json_valid": 1, "schema_match": 1, "json_exact": 1, "text_exact": 1}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, limit=1024**2):
    with Path(path).open("rb") as stream:
        return strict_json(stream.read(limit + 1), limit)


def write_json(path, value):
    """Write a new record. Its enclosing directory controls visibility."""
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise EditError(f"{name} requires exactly these fields: {', '.join(sorted(expected))}")


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise EditError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def string(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise EditError(f"invalid {name}")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EditError("expected a lowercase SHA-256 digest")
    return value


def reference(value):
    fields(value, {"path", "sha256"}, "file reference")
    string(value["path"], "file path")
    digest(value["sha256"])


def validate_issue(issue):
    external = isinstance(issue, dict) and issue.get("version") == "wet.issue.v2"
    model = issue.get("model") if isinstance(issue, dict) else None
    lora = isinstance(model, dict) and model.get("adapter") == "hf-lora-v1"
    required = {"version", "issue", "base", "model", "datasets", "gates", "budget"}
    if lora:
        required.add("backbone")
    fields(issue, required | ({"evaluator"} if external else set()), "issue")
    if issue["version"] not in {"wet.issue.v1", "wet.issue.v2"}:
        raise EditError("unsupported issue version")
    string(issue["issue"], "issue description")
    reference(issue["base"])
    if lora:
        reference(issue["backbone"])
    fields(model, {"adapter", "config"}, "model")
    config = model["config"]
    if external:
        if model["adapter"] not in {"named-tensors-v1", "hf-safetensors-v1"}:
            raise EditError("unsupported external checkpoint adapter")
        fields(config, set(), "external checkpoint config")
        metrics = validate_evaluator(issue["evaluator"])
    elif model["adapter"] == "tiny-transformer-v1":
        validate_tiny_config(config)
        metrics = METRICS
    elif model["adapter"] == "hf-causal-lm-v1" or lora:
        fields(config, {"prompt_format", "max_input_tokens", "max_new_tokens"}, "HF evaluation config")
        if config["prompt_format"] not in ("raw", "chat"):
            raise EditError("prompt_format must be raw or chat")
        integer(config["max_input_tokens"], "max_input_tokens", 1, 2048)
        integer(config["max_new_tokens"], "max_new_tokens", 1, 256)
        metrics = TEXT_METRICS
    else:
        raise EditError("unsupported model adapter")
    one_shot = external and issue["evaluator"]["mode"] == "construct"
    fields(issue["datasets"], {"confirmation"} if one_shot else {"development", "confirmation"}, "datasets")
    for dataset in issue["datasets"].values():
        reference(dataset)
    if not one_shot and issue["datasets"]["development"]["sha256"] == issue["datasets"]["confirmation"]["sha256"]:
        raise EditError("development and confirmation datasets must differ")
    gates = issue["gates"]
    minimum_gates = 1 if one_shot else 2
    if not isinstance(gates, list) or not minimum_gates <= len(gates) <= 16:
        raise EditError(f"expected between {minimum_gates} and sixteen evaluation gates")
    gate_kinds = {"target"} if one_shot else {"improve", "preserve"}
    seen = set()
    kinds = set()
    for gate in gates:
        required = {"group", "metric", "kind", "min_candidate" if one_shot else "min_delta"}
        if isinstance(gate, dict) and "min_candidate" in gate:
            required.add("min_candidate")
        fields(gate, required, "gate")
        group = string(gate["group"], "gate group", 80)
        metric = string(gate["metric"], "metric", 20)
        kind = string(gate["kind"], "gate kind", 20)
        if metric not in metrics or kind not in gate_kinds or (group, metric) in seen:
            raise EditError("invalid or duplicate gate")
        seen.add((group, metric))
        kinds.add(kind)
        if one_shot:
            finite(gate["min_candidate"], "min_candidate")
            continue
        minimum = finite(gate["min_delta"], "min_delta")
        if (kind == "improve" and minimum <= 0) or (kind == "preserve" and minimum > 0):
            raise EditError("improvement margins must be positive; preservation margins must be nonpositive")
        if not external and metric != "nll" and abs(minimum) > 1:
            raise EditError("rate margin must be in [-1, 1]")
        if "min_candidate" in gate:
            floor = finite(gate["min_candidate"], "min_candidate")
            if not external and (metric == "nll" or not 0 <= floor <= 1):
                raise EditError("min_candidate is a rate floor in [0, 1]")
    if kinds != gate_kinds:
        raise EditError("issue requires both improvement and preservation gates")
    budget = issue["budget"]
    fields(budget, {"attempts", "timeout_seconds", "batch_size", "threads"}, "budget")
    for key, limits in {"attempts": (1, 20), "timeout_seconds": (1, 604800 if external else 3600),
                        "batch_size": (1, 64), "threads": (1, 8)}.items():
        integer(budget[key], key, *limits)
    if one_shot and budget["attempts"] != 1:
        raise EditError("construction requires exactly one submission")
    # Bound the largest logits array in the reference evaluator.
    if (model["adapter"] == "tiny-transformer-v1"
            and budget["batch_size"] * config["context"] * config["vocab"] > 8_000_000):
        raise EditError("batch size exceeds evaluator logits limit")
    return issue


def validate_evaluator(evaluator):
    fields(evaluator, {"python", "script", "config", "mode", "metrics", "seeds", "max_work"}, "evaluator")
    if not Path(string(evaluator["python"], "interpreter path")).is_absolute():
        raise EditError("evaluator interpreter must be an absolute path")
    reference(evaluator["script"])
    if evaluator["mode"] not in ("repair", "construct") or not isinstance(evaluator["config"], dict):
        raise EditError("invalid evaluator mode or config")
    metrics = evaluator["metrics"]
    if not isinstance(metrics, dict) or not 1 <= len(metrics) <= 16:
        raise EditError("expected between one and sixteen external metrics")
    for name, direction in metrics.items():
        string(name, "metric", 20)
        if type(direction) is not int or direction not in (-1, 1):
            raise EditError("metric direction must be 1 or -1")
    seeds = evaluator["seeds"]
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 64:
        raise EditError("expected between one and sixty-four evaluator seeds")
    for seed in seeds:
        integer(seed, "evaluation seed", 0, 2**32 - 1)
    if len(set(seeds)) != len(seeds):
        raise EditError("evaluation seeds must differ")
    integer(evaluator["max_work"], "maximum evaluator work per artifact", 1, 2**32 - 1)
    return metrics


def validate_tiny_config(config):
    fields(config, MODEL_FIELDS, "model config")
    for key, limits in {"width": (2, 512), "heads": (1, 32), "layers": (1, 16),
                        "vocab": (2, 4096), "context": (1, 512)}.items():
        integer(config[key], key, *limits)
    width = config["width"]
    if width % config["heads"]:
        raise EditError("model width must be divisible by heads")
    parameters = (12 * width**2 + 4 * width) * config["layers"]
    parameters += (2 * config["vocab"] + config["context"] + 2) * width + config["vocab"]
    if parameters > 12_000_000:
        raise EditError("model exceeds parameter limit")


def validate_proposal(proposal, base_sha256):
    fields(proposal, {"version", "base_sha256", "author", "hypothesis", "operations"}, "proposal")
    if proposal["version"] not in {"wet.proposal.v1", "wet.proposal.v2"}:
        raise EditError("unsupported proposal version")
    if proposal["base_sha256"] != base_sha256:
        raise EditError("proposal targets a different base checkpoint")
    string(proposal["author"], "proposal author", 160)
    string(proposal["hypothesis"], "hypothesis")
    operations = proposal["operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= 16:
        raise EditError("expected between one and sixteen operations")
    touched = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise EditError("operation must be an object")
        kind = string(operation.get("op"), "operation type", 20)
        common = {"op", "tensors"}
        if kind == "replace" and proposal["version"] == "wet.proposal.v2":
            fields(operation, common | {"payload", "slices"}, "replace operation")
            reference(operation["payload"])
            if not isinstance(operation["slices"], dict):
                raise EditError("replacement slices must be an object")
        elif kind == "scale":
            fields(operation, common | {"factor"}, "scale operation")
            if abs(finite(operation["factor"], "factor")) > 10:
                raise EditError("scale factor must be between -10 and 10")
        elif kind in {"blend", "restore"}:
            fields(operation, common | {"donor"} | ({"alpha"} if kind == "blend" else set()), kind)
            reference(operation["donor"])
            if kind == "blend" and not 0 < finite(operation["alpha"], "alpha") < 1:
                raise EditError("blend alpha must be between 0 and 1")
        else:
            raise EditError(f"unsupported operation: {kind}")
        names = operation["tensors"]
        if not isinstance(names, list) or not 1 <= len(names) <= 200:
            raise EditError("operation requires one to two hundred exact tensor names")
        for name in names:
            string(name, "tensor name", 128)
            bounds = operation["slices"].get(name) if kind == "replace" else None
            if bounds is not None:
                if not isinstance(bounds, list) or not 1 <= len(bounds) <= 16:
                    raise EditError("invalid replacement slice")
                for pair in bounds:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise EditError("slice requires [start, stop] pairs")
                    start = integer(pair[0], "slice start", 0, 2**32 - 1)
                    integer(pair[1], "slice stop", start + 1, 2**32)
            for previous in touched.get(name, []):
                if (bounds is None or previous is None or len(bounds) != len(previous)
                        or all(a < d and c < b for (a, b), (c, d) in zip(bounds, previous))):
                    raise EditError(f"tensor addressed more than once with overlapping targets: {name}")
            touched.setdefault(name, []).append(bounds)
        if kind == "replace" and not operation["slices"].keys() <= set(names):
            raise EditError("slice names must be declared replacement targets")
    return proposal
