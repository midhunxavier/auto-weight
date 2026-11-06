# Quickstart on a pretrained DistilGPT2 model

This walkthrough starts from the public 82 million parameter DistilGPT2 checkpoint. The local experiment first trained the last three blocks and final normalization to return raw priority JSON and extract a queue name. A second fine-tune on the same tasks wrapped API answers in Markdown fences. Both stages used synthetic tickets. The model is a pretrained GPT-2 architecture with a tied embedding and output head. [DistilGPT2 model card](https://huggingface.co/distilbert/distilgpt2)

## Results

Restoring only the final normalization was discarded after development. Blending 38 changed tensors 80% toward the earlier checkpoint passed development and confirmation.

| Confirmation check | Regressed model | Accepted blend |
|---|---:|---:|
| Valid raw JSON with the correct priority | 0/60 | 60/60 |
| Exact queue name | 60/60 | 60/60 |

The export reproduced both scores when loaded with ordinary Transformers APIs. The earlier checkpoint also scored 60/60 on both tasks. This run validates the CLI on another pretrained architecture and tests preservation of queue extraction on these synthetic tickets.

- [Upstream revision](upstream.json), [training script](../../scripts/prepare_distilgpt2_quickstart.py), and [training log](training.log)
- [Setup output](setup.json), [frozen issue](frozen-issue.json), and [tensor listing](tensors.json)
- [Discarded trial](development-001.json) and [eligible trial](development-002.json)
- [Confirmation selection](selection.json), [acceptance report](acceptance.json), and [independent reload comparison](reload-comparison.json)

## What to provide

| Input | This run | For your model |
|---|---|---|
| Checkpoint to repair | `runs/distilgpt2-inputs/regressed/` | A local Transformers model directory containing config, tokenizer, and safetensors weights |
| Development prompts | [development.jsonl](development.jsonl), 36 per task | Examples the proposer can see and iterate on |
| Confirmation prompts | [confirmation.jsonl](confirmation.jsonl), 60 per task | Fresh examples held back until you select a candidate |
| Issue and gates | Restore raw priority JSON and preserve queue extraction | Describe the behavior to change and the behavior to keep. Set a metric and threshold for each. |
| Proposals | [Restore final norm](restore-norm.json) and [blend changed weights](blend.json) | Have an LLM produce proposal JSON. For a restore or blend, supply a compatible local donor. |

A Hugging Face JSONL file has one prompt and one check per line. Include at least 32 examples for each gated group in each split. An API row can check both the schema and the expected value. A queue row checks exact text.

```jsonl
{"group":"api","prompt":"Give the priority as raw JSON. Priority: high.","check":{"type":"json","expected":{"priority":"high"},"schema":{"type":"object","required":["priority"],"properties":{"priority":{"enum":["high","low","normal"]}},"additionalProperties":false}}}
{"group":"queue","prompt":"Give the assigned queue only. Queue: billing.","check":{"type":"exact","expected":"billing"}}
```

The proposal's `base_sha256` comes from the `setup` output. For a donor, run `auto-weight fingerprint --checkpoint /path/to/prior` and use the resulting digest. Relative donor paths resolve from the proposal file. For these saved proposal examples, the donor path is `prior` beside the proposal in the generated input directory.

The supplied files preserve the exact prompts and expected outputs used in the experiment. The generator uses a fixed seed for each split and distinct ticket IDs, notes, and instruction templates. For your own repair, collect a fresh confirmation set and give the proposer the development set only.

## Reproduce the inputs

Build the wheel and use the pip installation option in a fresh virtual environment so the training script can import the installed package. Append `'torch==2.5.1' 'transformers==4.57.6'` to the pip command to match the experiment. [Wheel installation](../../docs/06-cli-and-mcp.md#install-the-python-runtime)

Run these commands from the repository root with that environment active. Use `--device cpu` if Apple MPS is unavailable. Setup and evaluation run locally on CPU.

```sh
mkdir -p runs
hf download distilbert/distilgpt2 \
  --revision 2290a62682d06624634c1f46a6ad5be0f47f38aa \
  --include config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --local-dir .artifacts/distilgpt2
python scripts/prepare_distilgpt2_quickstart.py \
  --source .artifacts/distilgpt2 --output runs/distilgpt2-inputs
```

The script creates the prior and regressed checkpoints and both JSONL files. It writes the two proposals with the checkpoint hashes from the local run. Training and serialization may produce different hashes on different machines, so use the regenerated proposals with regenerated checkpoints.

## Run the repair

```sh
auto-weight setup --workspace runs/distilgpt2-repair --base runs/distilgpt2-inputs/regressed \
  --development runs/distilgpt2-inputs/development.jsonl --confirmation runs/distilgpt2-inputs/confirmation.jsonl \
  --prompt-format raw --max-new-tokens 24 --batch-size 6 --threads 6 \
  --issue "Restore raw priority JSON and preserve queue extraction" \
  --improve api json_exact 0.50 0.90 --improve api schema_match 0.50 0.90 \
  --preserve queue text_exact -0.10 0.90

auto-weight tensors --checkpoint runs/distilgpt2-repair/base --prefix transformer.ln_f.
auto-weight try --workspace runs/distilgpt2-repair --proposal runs/distilgpt2-inputs/restore-norm.json
auto-weight try --workspace runs/distilgpt2-repair --proposal runs/distilgpt2-inputs/blend.json
auto-weight finalize --workspace runs/distilgpt2-repair --attempt 002
auto-weight verify --workspace runs/distilgpt2-repair
```

`setup` freezes the inputs and prints the checkpoint hash. The JSON improvement gates require a paired improvement of at least 50 percentage points and a score of at least 90%. The preservation gate allows up to a 10 point decline and requires a score of at least 90%. Select an attempt marked `eligible` for the single confirmation run. The measured candidate was attempt 002.

The accepted artifact is the directory `runs/distilgpt2-repair/confirmation/result/model/`. Its config, tokenizer, and weights can be loaded using the normal Transformers APIs. [Hugging Face loading example](../../docs/12-hugging-face.md)

The same checkpoints and datasets were used to exercise the [single-command repair with a live local proposer](../../docs/13-automatic-repair.md#reproduce-the-live-test).

The same inputs were also tested through the installed plugin in [Codex and Claude Code](mcp/README.md).
