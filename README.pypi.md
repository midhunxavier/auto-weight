# AutoWeight

AutoWeight tests weight edits intended to repair regressions in local Hugging Face models and PEFT LoRA adapters. An LLM proposes edits from your failure description and development results. AutoWeight exports the selected checkpoint or adapter only if it passes improvement and preservation checks on a separate confirmation set.

## Use it

Install with Python 3.11 or newer on macOS or Linux.

```sh
uv tool install 'auto-weight[hf,lora,mcp]'
```

Bring a local model and two labeled JSONL files. Each file needs at least 32 examples per behavior you want to fix or preserve. Name the improvement group `fix` and keep confirmation examples out of the proposer's context. [Model and data format](https://github.com/midhunxavier/auto-weight/blob/main/docs/12-hugging-face.md)

Start Ollama, install the default proposer with `ollama pull qwen3.5`, then run:

```sh
auto-weight repair --base ./my-model \
  --development dev.jsonl --confirmation holdout.jsonl \
  --issue "Restore raw JSON and preserve queue extraction"
```

The result includes scores, a report, and an export path when accepted. Add `--donor ./earlier-model` to enable checkpoint restores and blends. For LoRA repair, add `--lora ./adapter` and supply its original model as `--base`.

For Codex or Claude Code, run `auto-weight setup-runtime --input-root ./repair-inputs` and follow the [plugin setup](https://github.com/midhunxavier/auto-weight/blob/main/plugins/auto-weight/README.md#load-the-plugin).

## What the copy-repair example proves

A small transformer lost copying accuracy during ordinary training on synthetic sequences. A 50% blend with an earlier checkpoint raised held-out collision-copy accuracy from **38.75% to 47.41%**, while unique-copy accuracy stayed around **99.31%**.

This confirms improvement and preservation for that one task using standard checkpoint blending. A full rollback was not compared on confirmation. [Experiment, provenance, and reports](https://github.com/midhunxavier/auto-weight/blob/main/examples/copy-repair/README.md)

[Documentation](https://github.com/midhunxavier/auto-weight/blob/main/docs/README.md) · [Automatic repair settings](https://github.com/midhunxavier/auto-weight/blob/main/docs/13-automatic-repair.md) · [Manual workflow](https://github.com/midhunxavier/auto-weight/blob/main/docs/06-cli-and-mcp.md#review-proposals-yourself)
