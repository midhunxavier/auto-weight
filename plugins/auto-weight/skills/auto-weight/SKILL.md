---
name: auto-weight
description: Prepare labeled checks and repair a local Hugging Face checkpoint with AutoWeight MCP. Use when the user supplies a regression, project data or existing evals, and a behavior to preserve.
---

Use the AutoWeight MCP tools to measure a proposed repair and export a checkpoint that passes confirmation.

1. Find the compatible local base checkpoint, the behavior to improve, and the behavior to preserve. Use supplied development and confirmation files directly. Otherwise find trusted project tests or labeled examples and convert them locally to JSONL records with `group`, `prompt`, and `check`. Use optional `family` IDs to keep related examples together. Pass the paths to `prepare_evals`; supply a separate confirmation source if the project already has one. Inspect schemas and development examples while keeping confirmation rows out of the conversation. If the agent has reviewed or generated all available examples, request independently chosen confirmation data. Ask only for missing inputs or labels.
2. Call `create_run` with the agreed paths and gates. Set `fix_group` to the regression group. Use chat prompt formatting when the examples require the checkpoint's chat template. Review the returned development baseline. If it already meets the repair target, explain what the current checks measured before trying edits.
3. Use `list_targets` to identify stored parameter names and `read_tensor` when a bounded slice helps. A donor ID filters to parameters that differ. Submit a short hypothesis and a numerical edit with `try_edit`. Use development gates and sample completions to guide the next attempt within the budget.
4. Confirm one eligible candidate with `confirm_edit` when the user has requested a passing checkpoint. Use `get_run` to verify the result. Report the baseline, relevant gate decisions, and any exported checkpoint path.

For a standard PEFT LoRA run, pass the adapter directory as `lora` and the original dense model as `base` in `create_run`. Target the adapter tensors returned by `list_targets`. An accepted run returns the repaired adapter under `checkpoint` and the pinned original model under `backbone`.
