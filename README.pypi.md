# AutoWeight

**Edit model weights. Verify the result.**

AutoWeight brings model weight editing to Codex CLI and Claude Code. The agent proposes edits to a local Hugging Face model. AutoWeight tests each edit on development examples and exports a new Transformers checkpoint if the selected edit passes a separate confirmation set.

[AutoWeight website](https://midhunxavier.com/auto-weight/) · [Research blog](https://midhunxavier.com/blog/auto-weight/checkpoint-repair-and-weight-space-experiments/)

## Quickstart with Codex CLI or Claude Code

Use Python 3.11 or newer on macOS or Linux. Point the runtime at an existing directory containing a local safetensors model, its config and tokenizer, and separate labeled development and confirmation files. [Input requirements and JSONL examples](https://midhunxavier.com/auto-weight/#connect)

```sh
uv tool install 'auto-weight[hf,mcp]'
auto-weight setup-runtime --input-root ./repair-inputs
```

For pip, run `python -m pip install 'auto-weight[hf,mcp]'` inside an active virtual environment, then run `setup-runtime` and register the server while the environment is active.

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

Use `/mcp` to check for `auto-weight`. Tell the agent which behavior to change, which to preserve, and where to find the checkpoint and evaluation files. Reserve confirmation for the selected edit. [Full setup and example prompt](https://midhunxavier.com/auto-weight/#connect)

## Try the offline demo

After installation, run `auto-weight demo`. It creates a tiny model locally and runs two scripted edits on CPU. Look for `"status": "accepted"` and the exported `checkpoint` path in the final JSON. [Demo preview](https://midhunxavier.com/auto-weight/#install)
