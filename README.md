# AutoWeight

Repair a model checkpoint through a loop of **propose → apply → evaluate → accept or discard**. Describe the failure and supply examples of the behavior to improve and preserve. An external LLM proposes a specific tensor edit. The toolkit runs the experiment and exports a complete checkpoint when it passes your frozen criteria.

Development attempts return scores and discard their temporary weights. Select one passing attempt for confirmation on fresh examples. The runner recreates the candidate, checks that its bytes match the development candidate, and publishes it only if every confirmation gate passes.

## Run a repair

Use the [AutoWeight Editor plugin](plugins/autoweight-editor/README.md) to have Codex or Claude Code propose and test edits through a local MCP server.

Use Python 3.11+ on macOS or Linux. Install the toolkit from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,mcp]'
mkdir -p runs
```

For one-command repair, start Ollama locally and install a proposer model such as `ollama pull qwen3.5`. Supply your checkpoint, the issue, and the two evaluation files. Label the improvement group `fix` and give the behaviors to preserve their own group names.

```sh
weight-edit repair --base ./my-model --issue "Restore raw JSON and keep queue extraction" \
  --development dev.jsonl --confirmation holdout.jsonl --donor ./earlier-checkpoint
```

The command asks the local LLM for an edit, applies it to a temporary copy, and feeds back the development result. It confirms the first passing attempt once and exports the checkpoint if it passes. Use `--fix-group` if your improvement group has another name, and `--prompt-format chat` for a model whose prompts require its saved chat template. [Automatic repair inputs and decisions](docs/13-automatic-repair.md)

### Review proposals yourself

For a local Hugging Face model, give the toolkit the checkpoint, development and confirmation JSONL files, and the checks you want it to pass. This example sets separate requirements for correct JSON and policy accuracy.

```sh
weight-edit setup --workspace runs/repair-001 --base ./my-model \
  --development dev.jsonl --confirmation holdout.jsonl --prompt-format chat \
  --issue "Recover raw JSON and preserve the fulfillment policy" \
  --improve api json_exact 0.50 0.90 --preserve policy text_exact -0.10 0.90
weight-edit tensors --checkpoint runs/repair-001/base --limit 30
```

`setup` computes the hashes and saves the frozen issue for you. Give an LLM the issue record, development examples, parameter names, and available donor checkpoints. It can write a [proposal record](docs/04-patch-contract.md) and use the development report to decide what to try next. Once you choose an eligible attempt, finalize that attempt on your separate confirmation set.

```sh
weight-edit try --proposal proposal.json --workspace runs/repair-001
weight-edit finalize --attempt 001 --workspace runs/repair-001
weight-edit verify --workspace runs/repair-001
```

A passing confirmation writes a reloadable checkpoint and a neighboring report under `runs/repair-001/confirmation/result/`. For Hugging Face models the artifact is the `model/` directory. For the small transformer it is `model.npz`. Rejected attempts leave a report. [Command and result reference](docs/06-cli-and-mcp.md)

The Hugging Face adapter takes a local causal language model with safetensors weights and its tokenizer. Use `weight-edit fingerprint --checkpoint MODEL_DIR` when writing donor references by hand. The evaluator generates text from raw prompts or the saved chat template and grades complete responses for valid JSON, schema compliance, correct JSON values, or exact text. The `[torch]` extra is sufficient for the small transformer adapter and its tokenized next-token tests. Both adapters can restore, blend or scale named weights. [Hugging Face setup and text dataset format](docs/12-hugging-face.md)

## Measured repairs

The quickstart passed end to end on pretrained DistilGPT2. The accepted blend took correct raw JSON from 0/60 to 60/60, with queue extraction at 60/60 in both models. [Exact inputs and commands](examples/distilgpt2-quickstart/README.md)

A joint fine-tune taught a fulfillment policy and regressed raw JSON. Full rollback recovered JSON and lost the policy. Subtracting the update from a related fenced-JSON fine-tune recovered all 96 held-out JSON responses and kept all 96 policy decisions. [Policy preservation experiment](examples/policy-repair/README.md)

A follow-up fine-tune made SmolLM2-135M-Instruct wrap JSON in Markdown fences. Codex proposed two edits. The runner discarded a final-normalization restore and accepted a blend of six blocks with an earlier checkpoint. On 64 fresh API prompts the blend raised correct raw JSON from **0/64 to 64/64**. On 64 separate routing prompts accuracy went from **57/64 to 64/64**. [Fine-tuning recipe, results and replay commands](examples/json-repair/README.md)

An existing trained transformer lost copying accuracy after continued training. Codex proposed restoring its earlier output layers, then blending with its earlier checkpoint. The output-layer edit was discarded. On fresh confirmation examples, the blend raised copying accuracy from **38.75% to 47.41%** and preserved accuracy on the original unique-token copying task (**99.31% to 99.31%**). The runner exported the checkpoint. [Case study, input preparation and reports](examples/copy-repair/README.md)

## Design and research

- [Product workflow](docs/02-product-spec.md) and [architecture](docs/03-architecture.md)
- [Issue and proposal formats](docs/04-patch-contract.md) and [evaluation rules](docs/05-evaluation-protocol.md)
- [Operational limits](docs/07-operations-and-budgets.md), [implementation status](docs/08-implementation-plan.md) and [validation plan](docs/09-user-validation.md)
- [Design decisions](docs/10-decisions-and-risks.md) and [evidence](docs/11-evidence-and-sources.md)
- [Archived inspector and research studies](https://github.com/midhunxavier/weight-space-programming/blob/9b4a7b1f2a0156caba723158b4c3ff31c1ede2bd/packages/weight-edit-toolkit/research/inspector/README.md)
- [Self Weight Edit](https://github.com/midhunxavier/weight-space-programming/tree/main/packages/self-weight-edit) studies temporary edits during individual question solves. This toolkit repairs a checkpoint across a set of examples.
