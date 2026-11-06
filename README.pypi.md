# AutoWeight

Repair a fine-tuning regression in a local Hugging Face checkpoint while preserving behavior you want to keep. AutoWeight tests proposed weight edits on development examples and exports a standard Transformers checkpoint when one selected edit passes a separate confirmation set.

## Try the offline demo

Use Python 3.11 or newer on macOS or Linux.

```sh
uv tool install 'auto-weight[hf,mcp]'
auto-weight demo
```

The demo generates a tiny model locally, runs two scripted edits, and prints the scores and the exported model path. For a pip installation, use `python -m pip install 'auto-weight[hf,mcp]'` in an active virtual environment.

## Repair your own model

Bring a local safetensors checkpoint and labeled development and confirmation sets. For Codex or Claude Code, run `auto-weight setup-runtime --input-root /path/to/repair-inputs` using your existing input directory, then load the AutoWeight plugin from the repository. For terminal use, run `auto-weight repair` with a local Ollama proposer or use the manual proposal commands.

- [Runtime and CLI guide](https://github.com/midhunxavier/auto-weight/blob/main/docs/06-cli-and-mcp.md)
- [Plugin setup](https://github.com/midhunxavier/auto-weight/tree/main/plugins/auto-weight)
- [Hugging Face evaluation inputs](https://github.com/midhunxavier/auto-weight/blob/main/docs/12-hugging-face.md)
- [Repair case study](https://github.com/midhunxavier/auto-weight/tree/main/examples/policy-repair)
