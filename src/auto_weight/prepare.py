"""Prepare labeled text checks without exposing confirmation rows to the caller."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from .contracts import (
    FILE_LIMIT, EditError, canonical, fields, file_hash, identify, integer, strict_json, string, write_json,
)


def read_source(path):
    from referencing.exceptions import Unresolvable

    from .text_eval import compile_check

    if path.stat().st_size > FILE_LIMIT:
        raise EditError("evaluation source exceeds the file limit")
    rows, source_hash = [], sha256()
    seen = {}
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            try:
                integer(number, "row count", 1, 10000)
                record = strict_json(raw, 32 * 1024)
                if not isinstance(record, dict):
                    raise EditError("expected an object")
                source_hash.update(raw)
                row = dict(record)
                family = row.pop("family", None)
                if family is not None:
                    string(family, "family", 200)
                fields(row, {"group", "prompt", "check"}, "labeled row")
                string(row["group"], "group", 80)
                string(row["prompt"], "prompt", 12000)
                compile_check(row["check"])
            except (ValueError, Unresolvable) as exc:
                raise EditError(f"invalid labeled row at line {number}") from exc
            key = row["prompt"]
            if key in seen:
                error = "conflicting labels" if seen[key] != row["check"] else "duplicate prompt"
                raise EditError(f"{error} at line {number}")
            seen[key] = row["check"]
            rows.append((row, family))
    return rows, source_hash.hexdigest()


def split_rows(rows, seed):
    families = defaultdict(list)
    totals = Counter()
    for index, (row, family) in enumerate(rows):
        families[family if family is not None else index].append((row, family))
        totals[row["group"]] += 1
    families = list(families.values())
    Random(seed).shuffle(families)
    families.sort(key=len, reverse=True)
    splits = {"development": [], "confirmation": []}
    counts = {stage: Counter() for stage in splits}
    for members in families:
        groups = Counter(row["group"] for row, _ in members)
        stage = min(splits, key=lambda side: sum(counts[side][group] * count / totals[group]
                                                 for group, count in groups.items()))
        splits[stage].extend(members)
        counts[stage].update(groups)
    return splits


def prepare_evals(base, source, output, *, confirmation=None, seed=0, prompt_format="raw",
                  max_input_tokens=128, max_new_tokens=32, confirmation_independent=True, conversion=None):
    """Accept a pooled labeled file or two preassigned files with optional family IDs."""
    from .automation import inspect_development
    from .hf import preflight

    integer(seed, "seed", 0, 2**32 - 1)
    integer(max_input_tokens, "max_input_tokens", 1, 2048)
    integer(max_new_tokens, "max_new_tokens", 1, 256)
    if prompt_format not in {"raw", "chat"}:
        raise EditError("prompt_format must be raw or chat")
    if not confirmation_independent:
        return {"status": "needs_examples", "reason": "supply independently chosen confirmation examples"}
    rows, source_id = read_source(source)
    source_ids = {"source": source_id}
    if conversion is not None:
        if not conversion.is_file() or conversion.stat().st_size > FILE_LIMIT:
            raise EditError("conversion source must be a bounded local file")
        source_ids["conversion"] = file_hash(conversion)
    if confirmation is None:
        splits = split_rows(rows, seed)
    else:
        heldout, heldout_id = read_source(confirmation)
        source_ids["confirmation"] = heldout_id
        splits = {"development": rows, "confirmation": heldout}
        if {family for _, family in rows if family is not None} & {
                family for _, family in heldout if family is not None}:
            raise EditError("example families overlap across the supplied splits")
    counts = {stage: Counter(row["group"] for row, _ in records) for stage, records in splits.items()}
    groups = set().union(*(count.keys() for count in counts.values()))
    missing = {stage: {group: 32 - count[group] for group in sorted(groups) if count[group] < 32}
               for stage, count in counts.items()}
    if len(groups) < 2 or any(missing.values()):
        return {"status": "needs_examples", "minimum_per_group_per_split": 32,
                "minimum_groups": 2, "counts": counts, "missing": missing}

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise EditError("preparation output must be a new directory")
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        folder = Path(staging) / "prepared"
        folder.mkdir(mode=0o700)
        datasets = {}
        for stage, records in splits.items():
            name = f"{stage}.jsonl"
            with (folder / name).open("xb") as stream:
                for row, _ in records:
                    stream.write(canonical(row) + b"\n")
            datasets[stage] = {"path": name}
        metrics, preview, _ = inspect_development(folder / "development.jsonl")
        preflight(folder, {"base": {"path": str(base.resolve())}, "datasets": datasets,
                           "budget": {"batch_size": 4},
                           "gates": [{"group": group, "metric": metric} for group, metric in metrics.items()],
                           "model": {"config": {"prompt_format": prompt_format,
                                                "max_input_tokens": max_input_tokens,
                                                "max_new_tokens": max_new_tokens}}})
        hashes = {stage: file_hash(folder / ref["path"]) for stage, ref in datasets.items()}
        manifest = {"version": "wet.preparation.v1", "sources": source_ids, "seed": seed,
                    "policy": "grouped" if confirmation is None else "provided",
                    "confirmation_independent": True, "counts": counts, "sha256": hashes,
                    "membership": {stage: [identify(row) for row, _ in records]
                                   for stage, records in splits.items()}}
        write_json(folder / "preparation.json", manifest)
        folder.rename(output)
    return {"status": "ready", "counts": counts, "sha256": hashes, "development_examples": preview,
            **{stage: str(output / ref["path"]) for stage, ref in datasets.items()}}
