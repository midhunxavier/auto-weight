"""Create a complete external repair or one-shot construction example."""
import argparse
from io import BytesIO
from pathlib import Path
import sys

import numpy as np

from auto_weight.checkpoint import checkpoint_hash, NAMED_ADAPTER
from auto_weight.contracts import canonical, file_hash, write_json


HERE = Path(__file__).parent.resolve()


def reference(path, output):
    return {"path": str(path.relative_to(output)) if path.is_relative_to(output) else str(path),
            "sha256": file_hash(path)}


def prepare_example(output, *, title, evaluator, interpreter, base, patch, cases, metrics, gates,
                    config=None, max_work=256, seeds=(73,), timeout=120, adapter=NAMED_ADAPTER):
    output = output.resolve()
    output.mkdir(parents=True)
    base_path = base if isinstance(base, Path) else output / "base.npz"
    if not isinstance(base, Path):
        np.savez(base_path, **base)
    payload = output / "replacement.npz"
    np.savez(payload, **patch)
    base_ref = {"path": str(base_path), "sha256": checkpoint_hash(base_path, adapter)}
    datasets = {}
    for stage, rows in cases.items():
        path = output / f"{stage}.jsonl"
        path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        datasets[stage] = reference(path, output)
    constructing = "development" not in cases
    write_json(output / "issue.json", {
        "version": "wet.issue.v2", "issue": title, "base": base_ref, "datasets": datasets,
        "model": {"adapter": adapter, "config": {}}, "gates": gates,
        "budget": {"attempts": 1 if constructing else 2, "timeout_seconds": timeout,
                   "batch_size": 1, "threads": 1},
        "evaluator": {"python": str(interpreter.absolute()), "script": reference(evaluator, output),
                      "mode": "construct" if constructing else "repair", "metrics": metrics,
                      "config": config or {}, "seeds": list(seeds), "max_work": max_work},
    })
    write_json(output / "proposal.json", {
        "version": "wet.proposal.v2", "base_sha256": base_ref["sha256"], "author": "Example",
        "hypothesis": title,
        "operations": [{"op": "replace", "tensors": sorted(patch), "slices": {},
                        "payload": reference(payload, output)}],
    })


def classifier(output, interpreter, *, axis=0):
    base = np.eye(3, dtype="float32")
    base[axis, axis] = -1
    other = (axis + 1) % 3
    splits = {}
    for stage, offset in (("development", 0), ("confirmation", 32)):
        rows = []
        for index in range(offset, offset + 32):
            for group, label in (("fix", axis), ("keep", other)):
                vector = [0.0] * 3
                vector[label] = 1.0 + index / 100
                rows.append({"id": f"{group}-{index}", "group": group, "inputs": {"x": vector, "label": label}})
        splits[stage] = rows
    gates = [{"group": "fix", "metric": "accuracy", "kind": "improve", "min_delta": 0.9, "min_candidate": 1},
             {"group": "keep", "metric": "accuracy", "kind": "preserve", "min_delta": 0}]
    prepare_example(output, title="Recover one class while preserving another", evaluator=HERE / "classifier.py",
                    interpreter=interpreter, base={"weight": base}, patch={"weight": np.eye(3, dtype="float32")},
                    cases=splits, metrics={"accuracy": 1}, gates=gates, seeds=(73, 84))


def minatar(output, interpreter, episodes, steps):
    from assets import asset

    data = asset("exp5_write_from_scratch/networks/breakout/hand.npz")
    with np.load(BytesIO(data), allow_pickle=False) as weights:
        patch = dict(weights)
    base = {name: np.zeros_like(tensor) for name, tensor in patch.items()}
    rows = [{"id": f"episode-{i}", "group": "breakout", "inputs": {"episode": i}} for i in range(episodes)]
    prepare_example(output, title="Replay published one-shot Breakout construction", evaluator=HERE / "minatar_eval.py",
                    interpreter=interpreter, base=base, patch=patch, cases={"confirmation": rows},
                    metrics={"score": 1}, config={"episode_steps": steps}, max_work=episodes * steps,
                    gates=[{"group": "breakout", "metric": "score", "kind": "target", "min_candidate": 1}])


def kinetix(output, interpreter, steps):
    from assets import asset

    root = "exp4_fix_car/networks"
    with np.load(BytesIO(asset(f"{root}/edited_actor.npz")), allow_pickle=False) as weights:
        patch = dict(weights)
    prefix = "params/GeneralActorCriticRNN_0"
    layers = [("embed", "params/Dense_0", 0.01, 0.0001)]
    layers.extend((f"actor{i}", f"{prefix}/DenseAndActivation_{i}/Dense_0", 0.001, 0.0001 if i == 0 else 0.001)
                  for i in range(5))
    layers.append(("actor_head", f"{prefix}/Dense_0", 0.001, 0.001))
    with np.load(BytesIO(asset(f"{root}/ppo_final_seed0.npz")), allow_pickle=False) as weights:
        base = {}
        for name, original, weight_scale, bias_scale in layers:
            for kind, field, scale in (("W", "kernel", weight_scale), ("b", "bias", bias_scale)):
                tensor = weights[f"{original}/{field}"].astype("float64")
                base[f"{name}.{kind}"] = (np.rint(tensor / scale) * scale).astype("float32")
    patch = {name: values.astype("float32") for name, values in patch.items()}
    changed = sum(np.count_nonzero(base[name] != values) for name, values in patch.items())
    if changed != 6073:
        raise ValueError(f"rounded published base differs: expected 6073 changed weights, got {changed}")
    splits = {}
    for stage, start in (("development", 0), ("confirmation", 64)):
        splits[stage] = [{"id": f"episode-{i}", "group": "ramp" if i % 2 == 0 else "safety",
                          "inputs": {"episode": i, "offset": 0 if i % 2 == 0 else (i % 11 - 5) * 0.001}}
                         for i in range(start, start + 64)]
    gates = [{"group": "ramp", "metric": "success", "kind": "improve", "min_delta": 0.5},
             {"group": "safety", "metric": "safe", "kind": "preserve", "min_delta": -0.1}]
    prepare_example(output, title="Replay the published Kinetix ramp edit", evaluator=HERE / "kinetix_eval.py",
                    interpreter=interpreter, base=base, patch=patch, cases=splits,
                    metrics={"success": 1, "safe": 1}, config={
                        "episode_steps": steps,
                        "level_sha256": "6616869302884dcfa74ec2e8da37b523c13fdc51f4b829eb3ecd7ba5e0ada6b1",
                    },
                    gates=gates, max_work=64 * steps, timeout=240)


def go(output, interpreter, episodes, steps):
    from assets import asset

    prefix = "exp6_go_from_scratch/networks/budget_0"
    patch = {}
    for layer in (*(f"L{i}" for i in range(1, 7)), "out"):
        for kind in ("W", "b"):
            name = f"{layer}.{kind}"
            patch[name] = np.load(BytesIO(asset(f"{prefix}/{name}.npy")), allow_pickle=False)
    base = {name: np.zeros_like(values) for name, values in patch.items()}
    rows = [{"id": f"game-{i}", "group": "go", "inputs": {"episode": i, "color": i % 2}}
            for i in range(episodes)]
    prepare_example(output, title="Replay published zero-practice Go weights against a random control",
                    evaluator=HERE / "go_eval.py", interpreter=interpreter, base=base, patch=patch,
                    cases={"confirmation": rows}, metrics={"win": 1, "finished": 1},
                    gates=[{"group": "go", "metric": "win", "kind": "target", "min_candidate": 0.5}],
                    config={"episode_steps": steps}, max_work=episodes * steps, timeout=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("classifier", "classifier-alt", "minatar", "kinetix", "go"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10_000 or not 1 <= args.steps <= 10_000:
        parser.error("episodes and steps must be between 1 and 10,000")
    if args.case == "minatar":
        minatar(args.output, args.python, args.episodes, args.steps)
    elif args.case == "kinetix":
        kinetix(args.output, args.python, args.steps)
    elif args.case == "go":
        go(args.output, args.python, args.episodes, args.steps)
    else:
        classifier(args.output, args.python, axis=int(args.case == "classifier-alt"))
