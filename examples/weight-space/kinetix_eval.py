"""Replay the published actor in Kinetix with paired action seeds."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from kinetix.environment.env import ActionType, ObservationType, make_kinetix_env
from kinetix.util.saving import get_correct_path_of_json_level, load_from_json_file
import numpy as np


def main(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    rows = [json.loads(line) for line in Path(request["cases"]).read_text().splitlines()]
    steps = request["config"]["episode_steps"]
    if len(rows) * len(request["seeds"]) * steps > request["max_work"]:
        raise ValueError("maximum episode work exceeds the declared budget")
    level = Path(get_correct_path_of_json_level("l/h2_car_ramp"))
    if sha256(level.read_bytes()).hexdigest() != request["config"]["level_sha256"]:
        raise ValueError("Kinetix ramp asset differs from the pinned level")
    state, static, params = load_from_json_file(str(level))
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT,
                           None, params, static, auto_reset=False)
    with np.load(request["artifact"], allow_pickle=False) as tensors:
        layers = [(jnp.asarray(tensors[f"{layer}.W"]), jnp.asarray(tensors[f"{layer}.b"]))
                  for layer in ("embed", *(f"actor{i}" for i in range(5)), "actor_head")]
    if env.observation_space(params).shape != (679,):
        raise ValueError("unsupported Kinetix observation layout")

    def rollout(episode, offset, seed, layers):
        rng = jax.random.fold_in(jax.random.PRNGKey(seed), episode)
        shifted = state.replace(
            polygon=state.polygon.replace(position=state.polygon.position.at[5, 0].add(offset)),
            circle=state.circle.replace(position=state.circle.position.at[:2, 0].add(offset)),
        )
        obs, initial = env.reset(rng, params, shifted)

        def still_playing(carry):
            return (carry[0] < steps) & ~carry[4]

        def step(carry):
            count, rng, obs, sim, _, _, safe = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            x = jnp.clip(jnp.nan_to_num(obs), -10, 10)
            for layer, (weight, bias) in enumerate(layers[:-1]):
                x = x @ weight + bias
                x = jax.nn.relu(x) if layer == 0 else jnp.tanh(x)
            weight, bias = layers[-1]
            logits = x @ weight + bias
            keys = jax.random.split(action_key, 6)
            parts = jnp.split(logits, (3, 6, 9, 12, 14))
            actions = jnp.stack([jax.random.categorical(key, part) for key, part in zip(keys, parts)])
            obs, sim, reward, done, info = env.step(step_key, sim, actions, params)
            safe = safe & jnp.isfinite(reward) & (reward > -0.5)
            return count + 1, rng, obs, sim, done, info["GoalR"], safe

        final = jax.lax.while_loop(still_playing, step, (0, rng, obs, initial, False, False, True))
        return final[0], final[5], final[6]

    batch = jax.jit(jax.vmap(rollout, in_axes=(0, 0, None, None)))
    episodes = jnp.asarray([row["inputs"]["episode"] for row in rows], dtype=jnp.uint32)
    offsets = jnp.asarray([row["inputs"]["offset"] for row in rows], dtype=jnp.float32)
    work, observations = 0, []
    for seed in request["seeds"]:
        counts, goals, safe = map(np.asarray, batch(episodes, offsets, np.uint32(seed), layers))
        work += int(counts.sum())
        observations.extend({"id": row["id"], "seed": seed,
                             "metrics": {"success": int(goal), "safe": int(avoided)}}
                            for row, goal, avoided in zip(rows, goals, safe, strict=True))
    Path(response_path).write_text(json.dumps({
        **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
        "observations": observations, "work": work,
    }))


if __name__ == "__main__":
    main(*sys.argv[1:])
