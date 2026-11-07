# CLI and MCP for an external proposer

## Install the Python runtime

Use Python 3.11 or newer on macOS or Linux. Install the CLI and MCP server from PyPI. [PyPI package](https://pypi.org/project/auto-weight/)

```sh
uv tool install 'auto-weight[hf,mcp]'
```

For pip, run `python -m pip install 'auto-weight[hf,mcp]'` inside an activated virtual environment. Both installs provide `auto-weight` and `auto-weight-mcp`. If your shell cannot find the commands after a uv install, run `uv tool update-shell` and open a new terminal. With pip, activate the environment in each terminal where you use the CLI.

CLI repair commands take your model and evaluation paths directly. [Automatic repair](13-automatic-repair.md) · [Manual proposal workflow](#review-proposals-yourself)

### Install from a checkout

To work on AutoWeight locally, install from the repository root and use the plugin from the same checkout.

```sh
uv tool install --from '.[hf,mcp]' auto-weight
```

With pip, run `python -m pip install '.[hf,mcp]'` in an activated virtual environment.

## Configure the local MCP runtime

For Codex, Claude Code, or another MCP client, run this command using the installed CLI. Replace the input path with an existing directory containing your checkpoint, evaluation files, and any donors. The server accepts input paths within that directory.

```sh
auto-weight setup-runtime --input-root ~/model-repair
```

The command starts the local MCP server, lists its tools, and saves the interpreter and paths to `~/.config/auto-weight/editor.conf` after the check succeeds. Use `--runs-dir` to choose where repairs are saved. The default is `~/.local/share/auto-weight/runs`. Set `AUTOWEIGHT_CONFIG` to use a separate config file and pass the same setting to your MCP client.

When switching installations, run `auto-weight setup-runtime` again and reload the matching plugin. Pass `--runs-dir` to keep using an existing run directory.

The plugin launcher uses the saved interpreter and paths. Other MCP clients can start the installed `auto-weight-mcp` command as a local stdio server. Pass `--input-root` and optionally `--runs-dir` to choose paths for a single server launch. [Load the plugin](../plugins/auto-weight/README.md#load-the-plugin)

## Commands

The commands below are available in AutoWeight 0.5.0.

| Command | Inputs | Result |
|---|---|---|
| `auto-weight demo` | Optional new `--workspace` directory | Offline installation smoke test with a generated fixture |
| `auto-weight setup-runtime` | Local input folder and optional runs directory | Check the local MCP server and write the runtime config |
| `auto-weight repair` | Local Hugging Face model, issue, split datasets and optional donors | Ask a local Ollama model for edits and confirm the first eligible candidate |
| `auto-weight prepare` | Local HF model, labeled JSONL source and new output directory | Validate and write development and confirmation files |
| `auto-weight setup` | Local Hugging Face model, datasets, and gates | Freeze a workspace and return the development baseline |
| `auto-weight baseline` | `--workspace DIR` | Score frozen development cases without using an edit attempt |
| `auto-weight read-tensor` | Checkpoint, tensor name and optional slice | Return bounded numerical values or statistics |
| `auto-weight submit` | Construction workspace and proposal file | Score a single final construction candidate |
| `auto-weight tensors` | `--checkpoint DIR`, optional `--prefix` and `--limit` | List stored tensor names, shapes, and dtypes |
| `auto-weight fingerprint` | `--checkpoint FILE_OR_DIR` | SHA-256 identity for an input checkpoint |
| `auto-weight init` | `--issue FILE --workspace NEW_DIR` | Frozen base, evaluation sets, runtime, budget and issue identity |
| `auto-weight try` | `--proposal FILE --workspace DIR` | `eligible`, `discarded` or `error` with a numbered attempt report |
| `auto-weight finalize` | `--attempt NNN --workspace DIR` | `accepted`, `discarded` or `error` with a confirmation report |
| `auto-weight verify` | `--workspace DIR` | Integrity check and accepted checkpoint path, when present |
| `auto-weight report` | `--workspace DIR`, optional `--html FILE` | Readable results, acceptance rules, and an optional standalone HTML copy |

Use [automatic repair](13-automatic-repair.md) to run the proposal loop in one command. For manual review, an LLM with CLI access can write a proposal file, run `try`, and read the JSON result. Compare the individual gate deltas and bounds when deciding what to try next. After selecting an eligible attempt, run `finalize` once. Normal evaluation decisions exit zero, including a discard. Invalid input and evaluator errors exit one. Automatic repair also exits one if it exhausts the proposal budget without scoring a candidate.

Development reports live under `attempts/NNN/result/report.json`. Confirmation lives under `confirmation/result/`. An accepted result directory contains either the Hugging Face `model/` directory or `model.npz` alongside `report.json`. The report includes the model adapter and configuration needed for loading, base and candidate hashes, changed tensors, and metric results. `verify` checks the workspace and its records. The [AutoWeight plugin](../plugins/auto-weight/README.md) exposes the same runner through local MCP for Codex and Claude Code.

Open a readable report after any attempt or confirmation:

```sh
auto-weight report --workspace runs/repair-001 --html repair-report.html
```

The command reads saved results and checks their integrity without running inference. It shows which checks passed, the reasons for failed checks, and before/after examples from development when available. Confirmation scores are aggregate. Save HTML at the workspace root or outside the workspace. Omit `--html` for terminal output alone. For MCP runs, use the workspace path returned by `get_run`.

## Prepare labeled evaluation data

Give the agent a compatible checkpoint, trusted project labels, and the behaviors to improve and preserve. Existing evaluation files can go directly to setup. To prepare a pooled source, normalize it to JSONL rows with `group`, `prompt`, and `check`. An optional `family` field keeps related rows in the same split. Supply `--confirmation` when you have a separate final set.

```sh
auto-weight prepare --base ./my-model --source labeled.jsonl --output ./prepared
```

Use the returned paths in setup or `create_run`. Each group needs 32 examples in each split. When examples are missing, `next_steps` lists how many to add to each group and split. The tool checks duplicate prompts after tokenization, labels, group counts, and context limits. Keep the confirmation file out of the agent's context. Use `--confirmation-exposed` or MCP `confirmation_independent=false` when the proposer has seen all source examples. A usable final set needs independent examples.

The MCP server offers `prepare_evals`, `create_run`, `list_targets`, `read_tensor`, `try_edit`, `confirm_edit`, and `get_run`. See the [external evaluator protocol](15-external-evaluators.md) for advanced issue and proposal files.

CLI setup and MCP `create_run` return the acceptance rules in plain language before any edit is tried. For example, `--preserve policy text_exact -0.02 0.95` allows a two percentage point decline on the paired lower bound and requires at least 95% observed accuracy. The equivalent MCP shortcut uses `preservation_margin=0.02` and `preservation_floor=0.95` for all preservation groups.

To repair a PEFT LoRA adapter, supply the original checkpoint as `--base` and the adapter folder as `--lora` in CLI setup or repair. In MCP, use `base` and `lora` in `create_run`. A passing run exports the repaired adapter. [LoRA setup and loading](12-hugging-face.md#repair-a-peft-lora-adapter)

## Review proposals yourself

Create a parent directory for your runs.

```sh
mkdir -p runs
```

Give the toolkit a local Hugging Face checkpoint, separate development and confirmation files, and the checks you want it to pass.

```sh
auto-weight setup --workspace runs/repair-001 --base ./my-model \
  --development dev.jsonl --confirmation holdout.jsonl --prompt-format chat \
  --issue "Recover raw JSON and preserve the fulfillment policy" \
  --improve api json_exact 0.50 0.90 --preserve policy text_exact -0.10 0.90
auto-weight tensors --checkpoint runs/repair-001/base --limit 30
```

Give the proposing LLM the issue record, development examples, parameter names, and compatible donor checkpoints you want it to use. Have it write a [proposal record](04-patch-contract.md) and use the development report to decide what to try next. Select an eligible attempt for one confirmation run.

```sh
auto-weight try --proposal proposal.json --workspace runs/repair-001
auto-weight finalize --attempt 001 --workspace runs/repair-001
auto-weight verify --workspace runs/repair-001
```

## Load an accepted model

For Hugging Face, pass the accepted `model/` directory to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. [Loading example](12-hugging-face.md)

For NPZ, use the config recorded next to the accepted weights.

```python
import json
from pathlib import Path

import torch

from auto_weight.evaluate import load_state
from auto_weight.model import TinyTransformer

result = Path("runs/repair-001/confirmation/result")
report = json.loads((result / "report.json").read_text())
model = TinyTransformer(**report["model"]["config"]).eval()
shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
weights = load_state(result / "model.npz", shapes)
model.load_state_dict({name: torch.from_numpy(array) for name, array in weights.items()})
```
