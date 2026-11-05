---
name: weight-edit
description: Repair a local Hugging Face checkpoint with AutoWeight MCP when a user provides a regression, evaluation examples, and behavior to preserve.
---

Use the AutoWeight MCP tools for checkpoint repair. The user supplies a compatible base checkpoint directory, labeled development and confirmation JSONL files, a description of the regression, and any compatible donor checkpoints. Ask only for missing inputs needed to define the repair. Use the provided paths and agreed gates. Treat the confirmation file as reserved for the runner and use only development results to choose edits.

1. Call `create_run` with the supplied inputs. Set `fix_group` to the development group that measures the regression. Other groups measure preservation. Use chat prompt formatting only when the user's examples require the checkpoint's chat template.
2. Use `list_targets` to find stored parameter names. A donor ID filters to parameters that actually differ. With a related checkpoint that predates a distributed regression, consider blending its changed tensors. Use `try_edit` with a short hypothesis and one numerical operation. A blend amount is the fraction of the donor values to use.
3. Check development gates and any sample completions. If the target behavior fails while preservation passes, consider a stronger or broader edit. If preservation regresses, consider reducing the edit. Use the existing run and stop when its attempt budget is exhausted.
4. Confirm one eligible candidate with `confirm_edit` when the user has asked for a passing checkpoint. Confirmation is spent once. Report the development decision, confirmation scores, and the exported checkpoint path if accepted. A failed confirmation ends the run.
