# Issue and proposal records

The issue is a strict JSON object with `version: "wet.issue.v1"`, `issue` text, `base`, `model`, `datasets`, `gates` and `budget`. A file reference has `path` and lowercase `sha256`. Relative paths are resolved from the JSON file that contains the reference. Initialization copies the base and datasets to the workspace and checks the expected hashes.

- `model` selects `tiny-transformer-v1` with `width`, `heads`, `layers`, `vocab` and `context`, or `hf-causal-lm-v1` with `prompt_format`, `max_input_tokens` and `max_new_tokens`.
- `datasets` contains `development` and `confirmation` JSONL references.
- `gates` specifies `group`, `metric` (from the adapter’s [metrics](05-evaluation-protocol.md)), `kind` (`improve` or `preserve`) and `min_delta`. Include both kinds. Rate metrics can also set `min_candidate`, an absolute floor for the candidate’s observed score.
- `budget` specifies `attempts`, `timeout_seconds`, `batch_size` and `threads`.

LoRA issues use `model.adapter: "hf-lora-v1"` and the same text evaluation config as `hf-causal-lm-v1`. The `base` reference identifies the editable PEFT adapter. A separate top-level `backbone` reference pins the original dense model and tokenizer. LoRA proposals reference the adapter hash as `base_sha256` and use `scale` or `wet.proposal.v2` replacement operations on its stored tensors.

A proposal has `version: "wet.proposal.v1"`, `base_sha256`, `author`, `hypothesis` and `operations`. The operations accept exact `tensors` names. Each tensor may appear once per proposal.

| `op` | Extra fields | Effect |
|---|---|---|
| `restore` | `donor` file reference | Replace selected tensors with the donor values |
| `blend` | `donor`, `alpha` in (0, 1) | Compute `(1 - alpha) * base + alpha * donor` and cast back to the parameter dtype |
| `scale` | `factor` in [-10, 10] | Multiply selected tensors and cast back to the parameter dtype |

A donor must have the full tensor layout, shapes and dtypes of the base model. Verify that donor and base share architecture, token IDs, and meaningful parameter alignment before proposing a blend. The runner checks tensor compatibility. For Hugging Face, donor directories must also match the model configuration and tokenizer files. Address a tied parameter through only one of its names. A proposal that changes no numerical values produces an error result.

The proposer supplies data only. The operation schema has no code, shell commands, or evaluator fields. For complete records, see the [copying regression case](../examples/copy-repair/README.md). For Hugging Face inputs, see the [text evaluation guide](12-hugging-face.md).
