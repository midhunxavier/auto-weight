"""Stage published weights under opaque IDs and score a frozen diagnosis."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from random import Random

import numpy as np

from auto_weight.contracts import file_hash, read_json, write_json
from auto_weight.tensors import metadata

from assets import asset, MANIFEST

EXPERIMENTS = {"game": "exp1_which_game", "diffusion": "exp2_diffusion", "copying": "exp3_language"}


def stage(experiment, output, seed):
    prefix = EXPERIMENTS[experiment]
    files = sorted(name for name in MANIFEST["files"] if name.startswith(f"{prefix}/networks/"))
    Random(seed).shuffle(files)
    reference = "heads" if experiment == "copying" else "answer_key"
    key = json.loads(asset(f"{prefix}/eval/{reference}.json"))
    sources = [(source, asset(source)) for source in files]
    output.mkdir(parents=True)
    inputs, truth = {}, {}
    for index, (source, data) in enumerate(sources, 1):
        label = f"network-{index:02}"
        path = output / f"{label}.npz"
        if experiment == "game":
            with np.load(BytesIO(data), allow_pickle=False) as weights:
                np.savez(path, **{name: weights[name] for name in weights.files if name.startswith("actor/")})
        else:
            path.write_bytes(data)
        inputs[label] = {"path": path.name, "sha256": file_hash(path),
                         "source_sha256": MANIFEST["files"][source]["sha256"]}
        if experiment == "copying":
            language = Path(source).parent.name
            measured = next(heads for name, heads in key.items() if name.startswith(f"{language}_"))
            truth[label] = {"language": language,
                            "heads": sorted(head for head, metrics in measured.items() if metrics["induct"] > 0.2)}
        else:
            field = "game" if experiment == "game" else "shape"
            truth[label] = key[Path(source).stem][field]
    write_json(output / "inputs.json", inputs)
    write_json(output / "answer-key.json", truth)
    return {"status": "ready", "experiment": experiment, "inputs": str(output / "inputs.json")}


def score(folder, submission, output):
    proposed = read_json(submission)
    expected = read_json(folder / "answer-key.json")
    inputs = read_json(folder / "inputs.json")
    if not isinstance(proposed, dict) or proposed.keys() != expected.keys():
        raise ValueError("submit exactly one answer per input ID")
    for reference in inputs.values():
        path = folder / reference["path"]
        if file_hash(path) != reference["sha256"]:
            raise ValueError("published weights changed after staging")
        metadata(path, "", 1)
    def correct(candidate, label):
        if isinstance(label, dict):
            if not isinstance(candidate, dict) or candidate.keys() != label.keys():
                return False
            heads = candidate["heads"]
            return (candidate["language"] == label["language"] and isinstance(heads, list)
                    and all(isinstance(head, str) for head in heads) and sorted(heads) == label["heads"])
        return candidate == label

    results = {name: correct(proposed[name], label) for name, label in expected.items()}
    report = {"status": "scored", "submission_sha256": file_hash(submission), "correct": results,
              "total_correct": sum(results.values()), "total": len(results)}
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("stage")
    prepare.add_argument("experiment", choices=EXPERIMENTS)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--seed", type=int, default=0)
    assess = commands.add_parser("score")
    for name in ("folder", "submission", "output"):
        assess.add_argument(name, type=Path)
    args = parser.parse_args()
    result = (stage(args.experiment, args.output, args.seed) if args.command == "stage"
              else score(args.folder, args.submission, args.output))
    print(json.dumps(result, indent=2))
