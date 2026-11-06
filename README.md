# AutoWeight

**Repair fine-tuning regressions in local Hugging Face models.**

Tell Codex, Claude Code, or a local Ollama model what to fix and what to preserve. AutoWeight tests proposed weight edits on development examples, then exports a new Transformers checkpoint if the selected edit passes a separate confirmation set.

## Install and try it

Use Python 3.11 or newer on macOS or Linux. Install from PyPI and run the demo from any directory. [PyPI package](https://pypi.org/project/auto-weight/)

```sh
uv tool install 'auto-weight[hf,mcp]'
auto-weight demo
```

Using pip? Create a virtual environment first.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'auto-weight[hf,mcp]'
auto-weight demo
```

The demo creates a tiny model locally and runs two scripted edits on CPU. It works offline after installation. The final JSON reports `"status": "accepted"` and an exported `checkpoint` path under `runs/demo-<id>/`. On the confirmation set, correct JSON goes from 0/32 to 32/32 while queue extraction stays at 32/32. [Installation and troubleshooting](docs/06-cli-and-mcp.md#install-the-python-runtime)

## Repair your own model

Put these inputs in a folder such as `repair-inputs/`.

- `base/` contains a local Hugging Face causal language model with safetensors weights, config, and tokenizer.
- `development.jsonl` and `confirmation.jsonl` contain labeled prompts for the behavior to fix and the behaviors to preserve. Use at least 32 examples per checked group in each file, and keep confirmation examples out of the proposer's context. [JSONL examples](docs/12-hugging-face.md)
- Optionally, add a compatible earlier checkpoint as a donor so the proposer can restore or blend weights.

### With Codex or Claude Code

Point the installed runtime at your input folder.

```sh
auto-weight setup-runtime --input-root ./repair-inputs
```

Once setup reports `ready`, load the plugin from the repository tag that matches your installed version. [Codex and Claude Code setup](plugins/auto-weight/README.md#load-the-plugin)

Ask your agent for a repair.

> Fix the fenced JSON responses from `base` and preserve queue extraction. The improvement group is `fix`. Use `development.jsonl` to try edits and reserve `confirmation.jsonl` for the final check. Export the repaired checkpoint if it passes.

### From the terminal

Start Ollama and install a proposer model before running the command. It uses the `fix` group to measure improvement. [Ollama setup](docs/13-automatic-repair.md)

```sh
auto-weight repair --base ./repair-inputs/base \
  --development ./repair-inputs/development.jsonl \
  --confirmation ./repair-inputs/confirmation.jsonl \
  --issue "Restore raw JSON and preserve queue extraction"
```

You can also review and confirm proposals yourself. [Manual proposal workflow](docs/06-cli-and-mcp.md#review-proposals-yourself)

### Load the repaired model

An accepted run includes a confirmation report and the path to a complete checkpoint you can load with `from_pretrained()`. [Transformers loading example](docs/12-hugging-face.md)

## A real repair

We fine-tuned SmolLM2 on a synthetic order fulfillment policy. It learned the policy and started wrapping JSON in Markdown fences, breaking the parser. Rolling back fixed the JSON and erased the policy. A weight edit proposed by Codex used a related checkpoint and passed both checks without further training. [Reproduce the experiment](examples/policy-repair/README.md)

| On fresh examples | Correct raw JSON | Correct policy decisions |
| --- | ---: | ---: |
| After fine-tuning | 0/96 | 96/96 |
| Full rollback | 96/96 | 0/96 |
| Accepted weight edit | **96/96** | **96/96** |

## Examples and documentation

- [Five end-to-end repairs with Codex and Claude Code](examples/agent-repairs/README.md) · [DistilGPT2 walkthrough](examples/distilgpt2-quickstart/README.md) · [JSON repair](examples/json-repair/README.md) · [Small-transformer copying repair](examples/copy-repair/README.md)
- The source checkout includes evaluation preparation and external evaluators. [Install from a checkout](docs/06-cli-and-mcp.md#install-from-a-checkout) · [Prepare labeled evaluation data](docs/06-cli-and-mcp.md#prepare-labeled-evaluation-data) · [Weight-space examples](examples/weight-space/README.md)
- [When checkpoint repair helps](docs/blog/auto-weight-checkpoint-repair.md) · [Evaluation protocol](docs/05-evaluation-protocol.md) · [Architecture](docs/03-architecture.md) · [All documentation](docs/)
- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) · [Companion research repository](https://github.com/midhunxavier/weight-space-programming)
