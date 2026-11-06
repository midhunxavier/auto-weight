"""Score the published Breakout architecture with the official MinAtar environment."""
import json
from pathlib import Path
import sys

from minatar import Environment
import numpy as np


def main(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    with np.load(request["artifact"], allow_pickle=False) as tensors:
        layers = [(tensors[f"actor/{layer}/kernel"], tensors[f"actor/{layer}/bias"])
                  for layer in ("dense1", "dense2", "logits")]
    steps, observations = 0, []
    cap = request["config"]["episode_steps"]
    env = Environment("breakout", sticky_action_prob=0.1)
    channels = env.env.channels
    for line in Path(request["cases"]).read_text().splitlines():
        row = json.loads(line)
        for seed in request["seeds"]:
            rng = np.random.default_rng([seed, row["inputs"]["episode"]])
            env.seed(int(rng.integers(0, 2**32 - 1)))
            env.reset()
            score = 0
            for _ in range(cap):
                if steps >= request["max_work"]:
                    raise ValueError("episode step budget exhausted")
                view = env.state()
                encoded = np.zeros((10, 10, 6), dtype=np.float32)
                for target, channel in ((0, "paddle"), (2, "brick"), (3, "ball"), (4, "trail")):
                    encoded[:, :, target] = view[:, :, channels[channel]]
                value = encoded.ravel()
                for matrix, bias in layers[:-1]:
                    value = np.tanh(value @ matrix + bias)
                matrix, bias = layers[-1]
                logits = value @ matrix + bias
                probabilities = np.exp(logits.astype("float64") - logits.max())
                reward, terminal = env.act(int(rng.choice(6, p=probabilities / probabilities.sum())))
                steps += 1
                score += reward
                if terminal:
                    break
            observations.append({"id": row["id"], "seed": seed, "metrics": {"score": float(score)}})
    Path(response_path).write_text(json.dumps({
        **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
        "observations": observations, "work": steps,
    }))


if __name__ == "__main__":
    main(*sys.argv[1:])
