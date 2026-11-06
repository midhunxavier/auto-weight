# CLI and MCP for an external proposer

## Install the Python runtime

The CLI and AutoWeight plugin use the same Python wheel. Use Python 3.11 or newer on macOS or Linux. From the repository root, build and install it with uv.

```sh
uv build --wheel
uv tool install './dist/auto_weight-0.4.0-py3-none-any.whl[hf,mcp]'
```

If you already have a wheel, skip the build and use its path. To install with pip, run `python -m pip install './dist/auto_weight-0.4.0-py3-none-any.whl[hf,mcp]'` inside an activated virtual environment. Both installs provide `auto-weight` and `auto-weight-mcp`.

You can now run `auto-weight demo` or repair a model from the terminal. CLI repair commands take their input paths directly. [Automatic repair](13-automatic-repair.md) · [Manual proposal workflow](#review-proposals-yourself)

## Configure the local MCP runtime

For Codex, Claude Code, or another MCP client, run this command using the same wheel installation. Replace the input path with an existing directory containing your checkpoint, evaluation files, and any donors. The server accepts input paths within that directory.

```sh
auto-weight setup-runtime --input-root ~/model-repair
```

The command starts the local MCP server, lists its tools, and saves the interpreter and paths to `~/.config/auto-weight/editor.conf` after the check succeeds. Use `--runs-dir` to choose where repairs are saved. The default is `~/.local/share/auto-weight/runs`. Set `AUTOWEIGHT_CONFIG` to use a separate config file and pass the same setting to your MCP client.

If you installed a previous build, run `auto-weight setup-runtime` from the new installation and reload the plugin from the renamed marketplace. Pass `--runs-dir` to keep using an existing run directory.

The plugin launcher uses the saved interpreter and paths. Other MCP clients can start the installed `auto-weight-mcp` command as a local stdio server. Pass `--input-root` and optionally `--runs-dir` to choose paths for a single server launch. [Load the plugin](../plugins/auto-weight/README.md#load-the-plugin)

## Commands

| Command | Inputs | Result |
|---|---|---|
| `auto-weight demo` | Optional new `--workspace` directory | Try two scripted edits on a tiny local model and export the passing checkpoint |
| `auto-weight setup-runtime` | Local input folder and optional runs directory | Check the local MCP server and write the runtime config |
| `auto-weight repair` | Local Hugging Face model, issue, split datasets and optional donors | Ask a local Ollama model for edits and confirm the first eligible candidate |
| `auto-weight setup` | Local Hugging Face model, datasets, and gates | Freeze a workspace and return the base digest and issue record path |
| `auto-weight tensors` | `--checkpoint DIR`, optional `--prefix` and `--limit` | List stored tensor names, shapes, and dtypes |
| `auto-weight fingerprint` | `--checkpoint FILE_OR_DIR` | SHA-256 identity for an input checkpoint |
| `auto-weight init` | `--issue FILE --workspace NEW_DIR` | Frozen base, evaluation sets, runtime, budget and issue identity |
| `auto-weight try` | `--proposal FILE --workspace DIR` | `eligible`, `discarded` or `error` with a numbered attempt report |
| `auto-weight finalize` | `--attempt NNN --workspace DIR` | `accepted`, `discarded` or `error` with a confirmation report |
| `auto-weight verify` | `--workspace DIR` | Integrity check and accepted checkpoint path, when present |

Use [automatic repair](13-automatic-repair.md) to run the proposal loop in one command. For manual review, an LLM with CLI access can write a proposal file, run `try`, and read the JSON result. Compare the individual gate deltas and bounds when deciding what to try next. After selecting an eligible attempt, run `finalize` once. Normal evaluation decisions exit zero, including a discard. Invalid input and evaluator errors exit one. Automatic repair also exits one if it exhausts the proposal budget without scoring a candidate.

Development reports live under `attempts/NNN/result/report.json`. Confirmation lives under `confirmation/result/`. An accepted result directory contains either the Hugging Face `model/` directory or `model.npz` alongside `report.json`. The report includes the model adapter and configuration needed for loading, base and candidate hashes, changed tensors, and metric results. `verify` checks the workspace and its records. The [AutoWeight plugin](../plugins/auto-weight/README.md) exposes the same runner through local MCP for Codex and Claude Code.

## Review proposals yourself

Use the `auto-weight` command from your wheel installation. Create a parent directory for your runs.

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
