# Automatic checkpoint repair

Install the toolkit with `pip install -e '.[hf]'` and start a local Ollama service. Install the proposer with `ollama pull qwen3.5`, or pass an already installed local model with `--proposer-model`. The proposer and the checkpoint being repaired are separate models. The toolkit connects to Ollama on `127.0.0.1:11434`. [Ollama structured output reference](https://docs.ollama.com/capabilities/structured-outputs)

Supply the following.

- `--base` is a compatible local Hugging Face model directory with safetensors weights, config and tokenizer.
- `--issue` describes the model behavior you want to fix and what it should preserve.
- `--development` and `--confirmation` are separate labeled JSONL files. Include at least 32 examples per gated group in each file. Use `fix` for the improvement group, or set `--fix-group` to an existing group name. Give the behavior to preserve its own group. [Example format](12-hugging-face.md)
- `--donor` is optional and repeatable. Offer an earlier compatible checkpoint to enable restores and blends. Scaling is available when you have only the base checkpoint.

```sh
weight-edit repair --base ./my-model --issue "Return raw JSON and preserve queue answers" \
  --development dev.jsonl --confirmation holdout.jsonl --donor ./earlier-checkpoint
```

The default gates require at least a 10 percentage point gain and 90% observed accuracy on the strongest available metric for the improvement group. The preservation groups allow up to a five point decline. Use both `--improve` and `--preserve` to set your own gates. The toolkit freezes the gates before the first LLM call.

The proposer receives the issue, a sample of development examples, a menu of parameter targets, offered donor IDs, and recent development feedback. It chooses a restore, blend, or scale operation. The toolkit resolves the selection to exact parameter names and validates the resulting proposal before applying it. A donor can be used to select the tensors that changed between the checkpoints. The proposer receives neither model file contents nor confirmation prompts or labels.

The first attempt that passes development is replayed once on confirmation. A passing confirmation publishes a complete model. A failing confirmation ends the run with a report. `--attempts` bounds proposer calls, including invalid responses. When the budget ends without a passing development attempt, no candidate is confirmed or exported.

The final JSON includes the outcome, workspace, and checkpoint path when accepted. Without `--workspace`, the command creates a fresh workspace under `runs/`. The `automation/` subdirectory records the proposer identity, requests, responses, validated proposals, and feedback. The ordinary trial and confirmation reports stay in their usual locations.

## Reproduce the live test

Prepare the two DistilGPT2 checkpoints and labeled datasets from the [DistilGPT2 walkthrough](../examples/distilgpt2-quickstart/README.md). Run the command below from the repository root. It uses the same confirmation split as the manual walkthrough, with confirmation data held out from the proposer.

```sh
weight-edit repair --workspace runs/distilgpt2-one-command \
  --base runs/distilgpt2-inputs/regressed \
  --development runs/distilgpt2-inputs/development.jsonl \
  --confirmation runs/distilgpt2-inputs/confirmation.jsonl \
  --donor runs/distilgpt2-inputs/prior \
  --fix-group api --max-new-tokens 24 --attempts 5 \
  --issue "Restore raw priority JSON after the follow-up fine-tune added Markdown fences. Preserve exact queue extraction. The donor is from before the format regression."
```

The local Qwen3.5 proposer first chose a 20% blend on the 38 changed tensors. The candidate still failed all 36 development JSON examples, so the runner discarded it. After receiving that result, the proposer chose a 50% blend. It passed both development checks and was confirmed once. On confirmation, correct raw JSON rose from 0/60 to 60/60, and exact queue extraction was 60/60 before and after the edit.

The first implementation pilots exposed two failure modes. The proposer invented wildcard names and then tried a tensor that had identical values in both checkpoints. The tested command constrains names to an available target menu and uses the observed donor differences to guide that menu.

[Proposer identity](../examples/distilgpt2-quickstart/automatic/proposer.json) · [First response](../examples/distilgpt2-quickstart/automatic/response-001.json) · [Request with feedback](../examples/distilgpt2-quickstart/automatic/request-002.json) · [Second response](../examples/distilgpt2-quickstart/automatic/response-002.json) · [Acceptance report](../examples/distilgpt2-quickstart/automatic/acceptance.json)
