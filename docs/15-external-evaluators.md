# External evaluators and numerical patches

Use the existing workspace commands with a `wet.issue.v2` file when the built-in text or next-token evaluator does not measure your task. An evaluator is a trusted local Python script. AutoWeight freezes its bytes, the interpreter identity and installed package versions, the checkpoint, cases, and settings. The evaluator defines the architecture and observations. AutoWeight pairs scores, checks the gates, and controls export.

The [classifier example](../examples/weight-space/README.md) generates complete records. Its alternate dataset uses the same evaluator and runner.

## Issue and checkpoint

Keep the `issue`, `base`, `model`, `datasets`, `gates`, and `budget` fields from [the repair contract](04-patch-contract.md). The model uses `{"adapter":"named-tensors-v1","config":{}}` for a numeric NPZ or `hf-safetensors-v1` for a local Hugging Face directory. Use `auto-weight fingerprint` to identify either form. NPZ checkpoints use real numerical arrays and a maximum expanded size of 64 MiB. The Hugging Face adapter preserves supported metadata and shard layout and expects unquantized tensors with tied parameters stored once.

Add an `evaluator` object with these fields.

| Field | Meaning |
| --- | --- |
| `python` | Absolute path to the evaluator environment's Python executable |
| `script` | File reference with `path` and `sha256` |
| `mode` | `repair` for paired development and confirmation, or `construct` for one final submission |
| `config` | Frozen JSON object interpreted by the evaluator |
| `metrics` | Metric names mapped to direction, `1` for higher values or `-1` for lower values |
| `seeds` | Between 1 and 64 distinct unsigned 32-bit seed values |
| `max_work` | Maximum work per checkpoint and stage in evaluator-defined units |

A case is a JSONL row containing `id`, `group`, and `inputs`. Treat each row as an independent unit. Put repeat measurements in the seed schedule. Repair uses at least 32 cases per gated group in each split and keeps improvement and preservation gates. Construction uses a confirmation manifest, `attempts: 1`, and one or more `target` gates with `min_candidate` floors. Use `submit` to reserve the only evaluation and score a construction candidate against those floors.

## Request and response

AutoWeight invokes `python -I evaluator.py request.json response.json` under the trial's wall time. The process receives the checkpoint and case paths and SHA-256 identities, fixed configuration, seed schedule, metric names, stage, work cap, and an `evaluation_id` under protocol `wet.evaluation.v1`. Read the serialized checkpoint passed in the request. Complete the stage's entire case and seed schedule.

Write an object with `version`, `evaluation_id`, `artifact_sha256`, `work`, and `observations`. Echo the first three values from the request. Report work in the units documented by the evaluator, such as tokens or simulator steps. Each observation has `id`, `seed`, and `metrics`, with exactly the requested finite numeric measurements. AutoWeight checks the schedule, averages repeats within each case, and applies paired gates by case identity. Development reports retain observations. Confirmation reports return aggregate results.

The supervisor terminates the evaluation process group on timeout. Evaluators enforce their own work cap while running. Dependencies belong in their own environment. For auxiliary assets such as simulator levels or opponents, include the identity in the frozen config or pin it in the evaluator environment and check it before evaluation. The subprocess runs as the local user. Launch trusted scripts and use separate access controls for experimental holdouts that require isolation from the proposer.

MCP accepts an advanced `issue_path` in `create_run`. To authorize an external script, the operator starts `auto-weight-mcp` with `--allow-evaluator /absolute/evaluator.py` and, if using a separate environment, `--evaluator-python /absolute/venv/bin/python`. The server pins the allowed script hashes at startup. Base, data, proposal, and payload inputs use the configured input root. An advanced construction run uses `confirm_edit` with `proposal_path` instead of `attempt_id`.

## Replacement proposals

A `wet.proposal.v2` keeps the same envelope as v1. A replacement operation has this form.

```json
{
  "op": "replace",
  "tensors": ["layer.weight"],
  "slices": {"layer.weight": [[0, 2], [1, 4]]},
  "payload": {"path": "replacement.npz", "sha256": "<64 lowercase hex digits>"}
}
```

The payload contains exactly the declared tensor names. The array in this example has shape `[2, 3]`. Each pair is a half-open range. Omit a tensor from `slices` to replace it in full. Distinct operations may address disjoint slices of the same tensor. The runner rejects overlaps, invalid shapes, nonfinite values and conversions, and patches over the byte or changed-parameter limits. It snapshots the payload for confirmation. The existing built-in adapters accept both proposal versions. External adapters use replacement operations.

Use `auto-weight read-tensor` or MCP `read_tensor` to inspect a named slice. A read returns up to 4,096 values with a 64 KiB response limit. The summary option reports statistics over the same bounded region.
