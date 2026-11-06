"""Paired gate decisions shared by built-in and external evaluators."""
import numpy as np

from .contracts import METRICS, TEXT_METRICS


def baseline_summary(scores, metrics):
    return {group: {"examples": len(rows), "baseline": dict(zip(metrics[group], rows.mean(0).tolist()))}
            for group, rows in scores.items()}


def paired_report(before, after, gates, group_metrics=None, *, directions=None):
    if group_metrics is None:
        group_metrics = {group: tuple(METRICS) for group in before}
    directions = directions or (METRICS | TEXT_METRICS)
    groups = {
        group: {
            "examples": len(rows),
            "baseline": dict(zip(group_metrics[group], rows.mean(0).tolist())),
            "candidate": dict(zip(group_metrics[group], after[group].mean(0).tolist())),
        }
        for group, rows in before.items()
    }
    # Resample sequences, preserving the pairing and avoiding token independence assumptions.
    rng = np.random.default_rng(7612301)
    results = []
    for gate in gates:
        group, metric = gate["group"], gate["metric"]
        column = group_metrics[group].index(metric)
        direction = directions[metric]
        delta = direction * (after[group][:, column] - before[group][:, column])
        draws = np.concatenate([
            delta[rng.integers(len(delta), size=(64, len(delta)))].mean(1)
            for _ in range(64)
        ])
        lower = float(np.quantile(draws, 0.05 / len(gates)))
        candidate = groups[group]["candidate"][metric]
        passed = lower >= gate["min_delta"] and ("min_candidate" not in gate or candidate >= gate["min_candidate"])
        results.append({**gate, "delta": float(delta.mean()), "lower_bound": lower, "passed": passed})
    return {"groups": groups, "gates": results, "passed": all(gate["passed"] for gate in results),
            "inference": {"method": "paired-sequence-bootstrap-v1", "draws": 4096,
                          "family_alpha": 0.05, "numpy": np.__version__}}
