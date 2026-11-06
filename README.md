# AutoWeight

**Edit model weights. Verify the result.**

AutoWeight brings model weight editing to Codex CLI and Claude Code. The agent proposes edits to a local Hugging Face model. AutoWeight tests each edit on development examples and exports a new Transformers checkpoint if the selected edit passes a separate confirmation set.

[AutoWeight website](https://midhunxavier.com/auto-weight/) · [Research blog](https://midhunxavier.com/blog/auto-weight/checkpoint-repair-and-weight-space-experiments/) · [PyPI package](https://pypi.org/project/auto-weight/)

## Quickstart with Codex CLI or Claude Code

### 1. Install and configure AutoWeight

Use Python 3.11 or newer on macOS or Linux and an installed Codex CLI or Claude Code client. Put these inputs in an existing directory such as `repair-inputs/`.

- `base/` contains a local Hugging Face causal language model with safetensors weights, config, and tokenizer.
- `development.jsonl` and `confirmation.jsonl` contain labeled prompts for the behaviors to change and preserve. Use at least 32 examples per checked group in each file. Use `fix` for the improvement group and keep confirmation examples out of the agent's context. [JSONL examples](docs/12-hugging-face.md)
- Optionally, add a compatible earlier checkpoint as a donor so the proposer can restore or blend weights.

```sh
uv tool install 'auto-weight[hf,mcp]'
auto-weight setup-runtime --input-root ./repair-inputs
```

Replace `./repair-inputs` with your input directory. For pip, run `python -m pip install 'auto-weight[hf,mcp]'` in an active virtual environment, then run `setup-runtime`. [Installation and troubleshooting](docs/06-cli-and-mcp.md#install-the-python-runtime)

### 2. Connect your CLI

Once setup reports `ready`, run the commands for your client from your project directory. If you installed with pip, keep the virtual environment active during registration.

**Codex CLI**

```sh
codex mcp add auto-weight -- "$(command -v auto-weight-mcp)"
codex
```

**Claude Code**

```sh
claude mcp add --transport stdio auto-weight -- "$(command -v auto-weight-mcp)"
claude
```

Use `/mcp` in your session to check for `auto-weight`. The PyPI package provides the MCP server for this setup. The repository also provides a plugin with a bundled skill. [Public CLI setup](https://midhunxavier.com/auto-weight/#connect) · [Bundled plugin](plugins/auto-weight/README.md#load-the-plugin)

### 3. Ask for a weight edit

> Use AutoWeight to remove Markdown fences from JSON responses in the model at `base`. Improve the `fix` group and preserve queue extraction. Use `development.jsonl` to try edits and reserve `confirmation.jsonl` for one final check. Export the edited checkpoint if it passes.

An accepted run includes a confirmation report and the path to a complete checkpoint you can load with `from_pretrained()`. [Transformers loading example](docs/12-hugging-face.md)

## Try the offline demo

After installation, run `auto-weight demo` from any directory. It creates a tiny model locally and runs two scripted edits on CPU. The final JSON reports `"status": "accepted"` and an exported `checkpoint` path under `runs/demo-<id>/`. On the confirmation set, correct JSON goes from 0/32 to 32/32 while queue extraction stays at 32/32.

## Other terminal workflows

Start Ollama and install a proposer model before running the command. It uses the `fix` group to measure improvement. [Ollama setup](docs/13-automatic-repair.md)

```sh
auto-weight repair --base ./repair-inputs/base \
  --development ./repair-inputs/development.jsonl \
  --confirmation ./repair-inputs/confirmation.jsonl \
  --issue "Restore raw JSON and preserve queue extraction"
```

You can also review and confirm proposals yourself. [Manual proposal workflow](docs/06-cli-and-mcp.md#review-proposals-yourself)

## A real repair

We fine-tuned SmolLM2 on a synthetic order fulfillment policy. It learned the policy and started wrapping JSON in Markdown fences, breaking the parser. Rolling back fixed the JSON and erased the policy. A weight edit proposed by Codex used a related checkpoint and passed both checks without further training. [Reproduce the experiment](examples/policy-repair/README.md)

| On fresh examples | Correct raw JSON | Correct policy decisions |
| --- | ---: | ---: |
| After fine-tuning | 0/96 | 96/96 |
| Full rollback | 96/96 | 0/96 |
| Accepted weight edit | **96/96** | **96/96** |

## Examples and documentation

- **Five agent repair cases.** Compare Codex, Claude Code, and a fixed edit on JSON, routing, and policy regressions. [Agent repair examples](examples/agent-repairs/README.md) · [DistilGPT2 walkthrough](examples/distilgpt2-quickstart/README.md) · [JSON repair](examples/json-repair/README.md) · [Copying repair](examples/copy-repair/README.md)
- **Eight weight-space studies.** Inspect tensors, replay published repairs, and score constructed policies. These examples use the source checkout. [Research examples and replay results](examples/weight-space/README.md) · [Install from a checkout](docs/06-cli-and-mcp.md#install-from-a-checkout)
- [Research blog source](docs/blog/auto-weight-checkpoint-repair.md) · [Architecture](docs/03-architecture.md) · [Evaluation preparation](docs/06-cli-and-mcp.md#prepare-labeled-evaluation-data) · [Evaluation protocol](docs/05-evaluation-protocol.md) · [All documentation](docs/)
- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) · [Companion research repository](https://github.com/midhunxavier/weight-space-programming)
