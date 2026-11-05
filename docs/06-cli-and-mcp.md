# CLI and MCP for an external proposer

| Command | Inputs | Result |
|---|---|---|
| `weight-edit repair` | Local Hugging Face model, issue, split datasets and optional donors | Ask a local Ollama model for edits and confirm the first eligible candidate |
| `weight-edit setup` | Local Hugging Face model, datasets, and gates | Freeze a workspace and return the base digest and issue record path |
| `weight-edit tensors` | `--checkpoint DIR`, optional `--prefix` and `--limit` | List stored tensor names, shapes, and dtypes |
| `weight-edit fingerprint` | `--checkpoint FILE_OR_DIR` | SHA-256 identity for an input checkpoint |
| `weight-edit init` | `--issue FILE --workspace NEW_DIR` | Frozen base, evaluation sets, runtime, budget and issue identity |
| `weight-edit try` | `--proposal FILE --workspace DIR` | `eligible`, `discarded` or `error` with a numbered attempt report |
| `weight-edit finalize` | `--attempt NNN --workspace DIR` | `accepted`, `discarded` or `error` with a confirmation report |
| `weight-edit verify` | `--workspace DIR` | Integrity check and accepted checkpoint path, when present |

Use [automatic repair](13-automatic-repair.md) to run the proposal loop in one command. For manual review, an LLM with CLI access can write a proposal file, run `try`, and read the JSON result. Compare the individual gate deltas and bounds when deciding what to try next. After selecting an eligible attempt, run `finalize` once. Normal evaluation decisions exit zero, including a discard. Invalid input and evaluator errors exit one.

Development reports live under `attempts/NNN/result/report.json`. Confirmation lives under `confirmation/result/`. An accepted result directory contains either the Hugging Face `model/` directory or `model.npz` alongside `report.json`. The report includes the model adapter and configuration needed for loading, base and candidate hashes, changed tensors, and metric results. `verify` checks the workspace and its records. The [AutoWeight Editor plugin](../plugins/autoweight-editor/README.md) exposes the same runner through local MCP for Codex and Claude Code.

## Review proposals yourself

From the repository root, use Python 3.11 or newer on macOS or Linux.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,mcp]'
mkdir -p runs
```

Give the toolkit a local Hugging Face checkpoint, separate development and confirmation files, and the checks you want it to pass.

```sh
weight-edit setup --workspace runs/repair-001 --base ./my-model \
  --development dev.jsonl --confirmation holdout.jsonl --prompt-format chat \
  --issue "Recover raw JSON and preserve the fulfillment policy" \
  --improve api json_exact 0.50 0.90 --preserve policy text_exact -0.10 0.90
weight-edit tensors --checkpoint runs/repair-001/base --limit 30
```

Give the proposing LLM the issue record, development examples, parameter names, and compatible donor checkpoints you want it to use. Have it write a [proposal record](04-patch-contract.md) and use the development report to decide what to try next. Select an eligible attempt for one confirmation run.

```sh
weight-edit try --proposal proposal.json --workspace runs/repair-001
weight-edit finalize --attempt 001 --workspace runs/repair-001
weight-edit verify --workspace runs/repair-001
```

## Load an accepted model

For Hugging Face, pass the accepted `model/` directory to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. [Loading example](12-hugging-face.md)

For NPZ, use the config recorded next to the accepted weights.

```python
import json
from pathlib import Path

import torch

from weight_edit_toolkit.evaluate import load_state
from weight_edit_toolkit.model import TinyTransformer

result = Path("runs/repair-001/confirmation/result")
report = json.loads((result / "report.json").read_text())
model = TinyTransformer(**report["model"]["config"]).eval()
shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
weights = load_state(result / "model.npz", shapes)
model.load_state_dict({name: torch.from_numpy(array) for name, array in weights.items()})
```
