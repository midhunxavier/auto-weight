"""Apply bounded tensor operations and score paired next-token predictions."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import CHECKPOINT_LIMIT, METRICS, EditError, fields, file_hash, integer, strict_json, string
from .model import TinyTransformer
from .statistics import baseline_summary, paired_report
from .tensors import check_changes, replace_arrays


def load_state(path, shapes):
    """Read exact, finite float32 state tensors without executing checkpoint code."""
    expected = {name + ".npy" for name in shapes}
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
            raise EditError("checkpoint tensor names do not match model")
        if sum(entry.file_size for entry in entries) > CHECKPOINT_LIMIT:
            raise EditError("expanded checkpoint exceeds byte limit")
        state = {}
        for name, shape in shapes.items():
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise EditError(f"unsupported tensor encoding for {name}")
                actual_shape, fortran, dtype = readers[version](stream)
                if actual_shape != shape or fortran or dtype != np.dtype("<f4"):
                    raise EditError(f"wrong shape, layout or dtype for {name}")
                size = 4 * int(np.prod(shape))
                raw = stream.read(size + 1)
                if len(raw) != size:
                    raise EditError(f"wrong tensor byte count for {name}")
            array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            if not np.isfinite(array).all():
                raise EditError(f"nonfinite values in {name}")
            state[name] = array
    return state


def apply(base, proposal, proposal_dir, shapes):
    candidate = base.copy()
    changes = []
    for operation in proposal["operations"]:
        kind = operation["op"]
        names = operation["tensors"]
        if set(names) - base.keys():
            raise EditError(f"unknown tensor names: {sorted(set(names) - base.keys())}")
        if kind == "replace":
            changes.extend(replace_arrays(base, candidate, operation, proposal_dir))
            continue
        if kind != "scale":
            reference = operation["donor"]
            path = proposal_dir / reference["path"]
            if file_hash(path) != reference["sha256"]:
                raise EditError("donor checkpoint hash mismatch")
            donor = load_state(path, shapes)
        for name in names:
            before = base[name]
            if kind == "restore":
                after = donor[name]
            elif kind == "blend":
                alpha = operation["alpha"]
                after = ((1 - alpha) * before.astype("float64") + alpha * donor[name]).astype("<f4")
            else:
                with np.errstate(over="ignore"):
                    after = (before.astype("float64") * operation["factor"]).astype("<f4")
            if not np.isfinite(after).all():
                raise EditError(f"edit created nonfinite values in {name}")
            candidate[name] = after
            difference = after.astype("float64") - before
            if np.any(difference):
                changes.append({"tensor": name, "op": kind, "parameters_changed": int(np.count_nonzero(difference)),
                                "l2_delta": float(np.linalg.norm(difference.ravel())),
                                "max_abs_delta": float(np.abs(difference).max())})
    check_changes(changes)
    return candidate, changes


def read_examples(path, config, gates):
    groups = defaultdict(list)
    prompts = set()
    with Path(path).open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            integer(index, "example count", 1, 10000)
            row = strict_json(raw, 32 * 1024)
            fields(row, {"group", "tokens", "positions"}, f"example {index}")
            group = string(row["group"], "example group", 80)
            tokens, positions = row["tokens"], row["positions"]
            if not isinstance(tokens, list) or not 2 <= len(tokens) <= config["context"] + 1:
                raise EditError(f"invalid token sequence at example {index}")
            for token in tokens:
                integer(token, "token", 0, config["vocab"] - 1)
            if not isinstance(positions, list) or not positions:
                raise EditError(f"missing target positions at example {index}")
            for position in positions:
                integer(position, "position", 0, len(tokens) - 2)
            if positions != sorted(set(positions)):
                raise EditError("positions must be sorted and unique")
            # Use each scored prompt prefix as the leakage key, including its last query.
            prompt = tuple(tokens[:positions[-1] + 1])
            if prompt in prompts:
                raise EditError("duplicate evaluation prompt")
            prompts.add(prompt)
            groups[group].append((tokens, positions))
    if not groups or any(len(groups[gate["group"]]) < 32 for gate in gates):
        raise EditError("each gated group requires at least 32 examples")
    return groups, prompts


@torch.inference_mode()
def score(model, groups, batch_size):
    scores = {}
    for group, examples in groups.items():
        rows = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            length = max(len(tokens) for tokens, _ in batch) - 1
            inputs = torch.zeros((len(batch), length), dtype=torch.long)
            for index, (tokens, _) in enumerate(batch):
                inputs[index, :len(tokens) - 1] = torch.tensor(tokens[:-1])
            logits = model(inputs)
            for index, (tokens, positions) in enumerate(batch):
                selected = logits[index, positions]
                targets = torch.tensor([tokens[position + 1] for position in positions])
                accuracy = (selected.argmax(-1) == targets).double().mean().item()
                nll = F.cross_entropy(selected, targets).item()
                rows.append((accuracy, nll))
        values = np.asarray(rows, dtype="float64")
        if not np.isfinite(values).all():
            raise EditError(f"nonfinite model scores in {group}")
        scores[group] = values
    return scores


def evaluate(workspace, issue, proposal, proposal_dir, stage, temporary, expected_candidate=None, *, score_baseline):
    torch.set_num_threads(issue["budget"]["threads"])
    torch.set_num_interop_threads(1)
    model = TinyTransformer(**issue["model"]["config"]).float().eval()
    shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    base = load_state(workspace / issue["base"]["path"], shapes)
    config, gates = issue["model"]["config"], issue["gates"]
    groups, prompts = read_examples(workspace / issue["datasets"][stage]["path"], config, gates)
    if stage == "confirmation":
        _, development_prompts = read_examples(workspace / issue["datasets"]["development"]["path"], config, gates)
        if development_prompts & prompts:
            raise EditError("development and confirmation prompts overlap")
    batch = issue["budget"]["batch_size"]

    def score_base():
        model.load_state_dict({name: torch.from_numpy(value) for name, value in base.items()}, strict=True)
        return score(model, groups, batch), {}

    before, _ = score_baseline(score_base)
    if proposal is None:
        return baseline_summary(before, {group: tuple(METRICS) for group in before})
    candidate, changes = apply(base, proposal, proposal_dir, shapes)
    candidate_path = temporary / "model.npz"
    np.savez(candidate_path, **candidate)
    candidate_sha = file_hash(candidate_path)
    if expected_candidate is not None and candidate_sha != expected_candidate:
        raise EditError("replayed candidate differs from development candidate")
    # Score the serialized checkpoint that can be promoted.
    reloaded = load_state(candidate_path, shapes)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in reloaded.items()}, strict=True)
    after = score(model, groups, batch)
    report = paired_report(before, after, gates)
    report["inference"]["torch"] = str(torch.__version__)
    return {**report, "changes": changes, "candidate_sha256": candidate_sha}
