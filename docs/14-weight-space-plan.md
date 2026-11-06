# AutoWeight extension design

AutoWeight helps an agent prepare checks, measure weight edits, and export a checkpoint that passes its confirmation gates. The [CLI and MCP guide](06-cli-and-mcp.md) covers the user workflow. The [external evaluator protocol](15-external-evaluators.md) covers advanced issue and proposal files.

## One shared workflow

- Prepare evaluation files from trusted project labels when the user needs help. Preserve existing splits, keep related examples together, and record source hashes and split membership. Prior exposure still matters when judging whether a confirmation set is independent.
- Freeze checkpoints, cases, settings, and proposed payloads in the existing workspace. Score the development baseline before requesting edits and reuse its measurements across trials.
- Keep restore, blend, and scale for built-in repair. Add bounded reads and one replacement operation for arbitrary numerical changes. Keep every proposal anchored to the frozen base.
- Run trusted external evaluators in separate local environments. Evaluators return observations keyed by case and seed. AutoWeight applies the gates and verifies the serialized candidate before export.
- Use one final submission and an absolute target for one-shot construction. Keep weight inspection and diagnosis in example scripts.

## Research coverage

The [weight-space examples](../examples/weight-space/README.md) distinguish live checks from recorded evidence. Two classifier issues share one evaluator. Kinetix, Breakout, and Go exercise external environments. Qwen uses its own Transformers runtime and can reuse completed batches. The long Qwen benchmark uses a separate resource budget, and confirmation remains single-use. Sycophancy pairs can be probed locally when checkpoints are available. The released evidence covers the original pairs.

Per-run storage, one supervised worker, and the existing CLI and MCP entry points cover these workflows. Benchmark dependencies and launch instructions live with the examples. Additional formats or protocols should come with a concrete use case and a reproducible acceptance check.
