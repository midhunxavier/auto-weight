# Automatic checkpoint repair

Install the [Python runtime](06-cli-and-mcp.md#install-the-python-runtime) and start a local Ollama service. Install the proposer with `ollama pull qwen3.5`, or pass an already installed local model with `--proposer-model`. The proposer and the checkpoint being repaired are separate models. The toolkit connects to Ollama on `127.0.0.1:11434`. [Ollama structured output reference](https://docs.ollama.com/capabilities/structured-outputs)

Supply the following.

- `--base` is a compatible local Hugging Face model directory with safetensors weights, config and tokenizer.
- `--issue` describes the model behavior you want to fix and what it should preserve.
- `--development` and `--confirmation` are separate labeled JSONL files. Include at least 32 examples per gated group in each file. Use `fix` for the improvement group, or set `--fix-group` to an existing group name. Give the behavior to preserve its own group. [Example format](12-hugging-face.md)
- `--donor` is optional and repeatable. Offer an earlier compatible checkpoint to enable restores and blends. Scaling is available when you have only the base checkpoint.

```sh
auto-weight repair --base ./my-model --issue "Return raw JSON and preserve queue answers" \
  --development dev.jsonl --confirmation holdout.jsonl --donor ./earlier-checkpoint
```

The default gates require at least a 10 percentage point gain and 90% observed accuracy on the strongest available metric for the improvement group. The preservation groups allow up to a five point decline. Use both `--improve` and `--preserve` to set your own gates. The toolkit freezes the gates before the first LLM call.

The proposer receives the issue, a sample of development examples, a menu of parameter targets, offered donor IDs, and recent development feedback. It chooses a restore, blend, or scale operation. The toolkit resolves the selection to exact parameter names and validates the resulting proposal before applying it. A donor can be used to select the tensors that changed between the checkpoints. The proposer receives neither model file contents nor confirmation prompts or labels.

The first attempt that passes development is replayed once on confirmation. A passing confirmation publishes a complete model. A failing confirmation ends the run with a report. `--attempts` bounds proposer calls, including invalid responses. When the budget ends without a passing development attempt, no candidate is confirmed or exported.

The final JSON includes the outcome, workspace, and checkpoint path when accepted. Without `--workspace`, the command creates a fresh workspace under `runs/`. The `automation/` subdirectory records the proposer identity, requests, responses, validated proposals, and feedback. The ordinary trial and confirmation reports stay in their usual locations.
