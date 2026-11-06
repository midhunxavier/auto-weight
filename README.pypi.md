# AutoWeight

**Repair fine-tuning regressions in local Hugging Face models.**

Tell Codex, Claude Code, or a local Ollama model what to fix and what to preserve. AutoWeight tests proposed weight edits on development examples, then exports a new Transformers checkpoint if the selected edit passes a separate confirmation set.

## Try the offline demo

Use Python 3.11 or newer on macOS or Linux. Run these commands from any directory.

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

The demo creates a tiny model locally and runs two scripted edits on CPU. It works offline after installation. Look for `"status": "accepted"` and the `checkpoint` path in the final JSON.

## Repair your own model

Bring a local Hugging Face causal language model with safetensors weights, config, and tokenizer. Prepare separate labeled development and confirmation sets with at least 32 examples per checked group in each file. Keep confirmation examples out of the proposer's context. [JSONL examples](https://github.com/midhunxavier/auto-weight/blob/main/docs/12-hugging-face.md)

- **Codex or Claude Code.** Run `auto-weight setup-runtime --input-root /path/to/repair-inputs`, then load the plugin from the repository tag that matches your installed version. [Plugin setup](https://github.com/midhunxavier/auto-weight/tree/main/plugins/auto-weight#load-the-plugin)
- **Terminal.** Run `auto-weight repair` with a local Ollama proposer, or review proposals yourself. [Ollama setup and example command](https://github.com/midhunxavier/auto-weight/blob/main/docs/13-automatic-repair.md) · [Manual proposal workflow](https://github.com/midhunxavier/auto-weight/blob/main/docs/06-cli-and-mcp.md#review-proposals-yourself)

[Load an accepted model](https://github.com/midhunxavier/auto-weight/blob/main/docs/06-cli-and-mcp.md#load-an-accepted-model) · [Reproduce the SmolLM2 repair](https://github.com/midhunxavier/auto-weight/tree/main/examples/policy-repair)
