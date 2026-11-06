"""Evaluate a published 9x9 Go policy against a uniform legal-move control."""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pgx


def main(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    rows = [json.loads(line) for line in Path(request["cases"]).read_text().splitlines()]
    cap = request["config"]["episode_steps"]
    if len(rows) * len(request["seeds"]) * cap > request["max_work"]:
        raise ValueError("Go work exceeds the declared move budget")
    with np.load(request["artifact"], allow_pickle=False) as tensors:
        layers = [(jnp.asarray(tensors[f"{name}.W"]), jnp.asarray(tensors[f"{name}.b"]))
                  for name in (*(f"L{i}" for i in range(1, 7)), "out")]
    game = pgx.make("go_9x9")

    def rollout(episode, color, seed, layers):
        rng = jax.random.fold_in(jax.random.PRNGKey(seed), episode)
        state = game.init(rng)
        player = state._player_order[color]

        def running(carry):
            step, _, state = carry
            return (step < cap) & ~state.terminated & ~state.truncated

        def play(carry):
            step, rng, state = carry
            rng, sample = jax.random.split(rng)
            board = state._x.board
            white = state._x.color == 1
            own, other = jnp.where(white, board < 0, board > 0), jnp.where(white, board > 0, board < 0)
            vector = jnp.concatenate((own, other, jnp.full((81,), white))).astype(jnp.float32)

            def policy(x):
                for matrix, bias in layers[:-1]:
                    x = jax.nn.relu(x @ matrix + bias)
                matrix, bias = layers[-1]
                return x @ matrix + bias

            logits = jax.lax.cond(state.current_player == player, policy, lambda _: jnp.zeros(82), vector)
            action = jax.random.categorical(sample, jnp.where(state.legal_action_mask, logits, -jnp.inf))
            return step + 1, rng, game.step(state, action)

        count, _, state = jax.lax.while_loop(running, play, (0, rng, state))
        return count, state.terminated & (state.rewards[player] > 0), state.terminated

    evaluate = jax.jit(jax.vmap(rollout, in_axes=(0, 0, None, None)))
    episodes = jnp.asarray([row["inputs"]["episode"] for row in rows], dtype=jnp.uint32)
    colors = jnp.asarray([row["inputs"]["color"] for row in rows], dtype=jnp.int32)
    observations, work = [], 0
    for seed in request["seeds"]:
        counts, wins, completed = map(np.asarray, evaluate(episodes, colors, np.uint32(seed), layers))
        work += int(counts.sum())
        observations.extend({"id": row["id"], "seed": seed, "metrics": {"win": int(win), "finished": int(done)}}
                            for row, win, done in zip(rows, wins, completed, strict=True))
    Path(response_path).write_text(json.dumps({
        **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
        "observations": observations, "work": work,
    }))


if __name__ == "__main__":
    main(*sys.argv[1:])
