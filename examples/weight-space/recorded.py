"""Retrieve the pinned recorded evaluation evidence without running a model."""
import argparse
import json
from pathlib import Path

from auto_weight.contracts import write_json
from assets import MANIFEST, asset

REPORTS = {
    "kinetix": ["exp4_fix_car/eval/oneshot_result.json"],
    "minatar": ["exp5_write_from_scratch/eval/minatar_eval.json"],
    "go": [f"exp6_go_from_scratch/eval/budget_{budget}/eval_submission.json" for budget in (0, 10, 25, 50)],
    "sycophancy": [*(f"exp7_sycophancy/eval/weights_only_{i}/score.json" for i in range(1, 4)),
                   "exp7_sycophancy/runs/weights_only_4/pair2/prediction.json",
                   "exp7_sycophancy/eval/attention_tool/score.json"],
    "qwen": [f"exp8_math/eval/{run}/merged.json" for run in ("unedited", "p10c", "p25c", "p50c")],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=REPORTS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, {"status": "recorded", "revision": MANIFEST["revision"],
                            "reports": {path: json.loads(asset(path)) for path in REPORTS[args.case]}})
