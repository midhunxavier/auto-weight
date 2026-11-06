"""Standalone external evaluator for a linear classifier."""
import json
from pathlib import Path
import sys

import numpy as np


def main(request, response):
    request = json.loads(Path(request).read_text())
    with np.load(request["artifact"], allow_pickle=False) as weights:
        kernel = weights["weight"]
    observations = []
    for line in Path(request["cases"]).read_text().splitlines():
        row = json.loads(line)
        inputs = row["inputs"]
        correct = int((np.asarray(inputs["x"]) @ kernel).argmax() == inputs["label"])
        for seed in request["seeds"]:
            observations.append({"id": row["id"], "seed": seed, "metrics": {"accuracy": correct}})
    # Return reversed order to exercise pairing by case ID.
    Path(response).write_text(json.dumps({
        **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
        "observations": list(reversed(observations)), "work": len(observations),
    }))


if __name__ == "__main__":
    main(*sys.argv[1:])
