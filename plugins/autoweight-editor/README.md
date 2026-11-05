# AutoWeight Editor

Repair a local Hugging Face model from Codex or Claude Code. The shared `weight-edit` skill lets the host agent propose an edit. The local MCP server applies it to a temporary copy, compares it with the original, and exports a complete checkpoint if the selected candidate passes confirmation.

## Set up the local runtime

From the repository root, use Python 3.11 or newer on macOS or Linux. The examples below configure an explicit input folder and a separate output folder. Replace `input_root` with the absolute directory containing the model, donor if any, and JSONL files. The server only accepts input paths under this folder.

```sh
repo="$PWD"
runtime="$HOME/.local/share/autoweight/venv"
input_root="/absolute/path/to/repair-inputs"
runs_dir="$HOME/.local/share/autoweight/runs"

python3 -m venv "$runtime"
"$runtime/bin/pip" install -e "${repo}[hf,mcp]"
mkdir -p "$HOME/.config/autoweight"
printf '%s\n' "$runtime/bin/python" "$input_root" "$runs_dir" > "$HOME/.config/autoweight/editor.conf"
```

The three lines in `editor.conf` are the runtime Python executable, the input root, and the output folder. To use an existing runtime, put its absolute interpreter path on the first line. The launcher also honors `AUTOWEIGHT_CONFIG`, `AUTOWEIGHT_PYTHON`, `AUTOWEIGHT_INPUT_ROOT`, and `AUTOWEIGHT_RUNS_DIR` when the host forwards them.

## Load the plugin

**Codex.** Add this repository as a local marketplace and install AutoWeight Editor. Start a fresh Codex session and invoke `$weight-edit`. The desktop app can also discover it through the AutoWeight repository marketplace. [OpenAI Docs for local marketplaces](https://developers.openai.com/plugins/build/plugins)

```sh
codex plugin marketplace add "$repo"
codex plugin add autoweight-editor@autoweight
```

**Claude Code.** Load the same plugin directory for a session and invoke `/autoweight-editor:weight-edit`. [Claude Code plugin setup](https://code.claude.com/docs/en/plugins/create)

```sh
claude --plugin-dir "$repo/plugins/autoweight-editor"
```

## Ask for a repair

> Fix the fenced JSON responses from `base` and preserve queue extraction. Use `development.jsonl` and `confirmation.jsonl`. The improvement group is `api`. The compatible checkpoint at `prior` is available as a donor. Export the repaired checkpoint if it passes.

Paths can be relative to your configured input root. Supply at least 32 labeled examples per gated group in each JSONL file. Label improvement examples `fix` or name the improvement group in your request. The other groups check preservation. The host receives bounded development examples and scores. The server returns aggregate confirmation metrics and the accepted checkpoint path. [Hugging Face evaluation files](../../docs/12-hugging-face.md)

The five MCP tools are `create_run`, `list_targets`, `try_edit`, `confirm_edit`, and `get_run`. Donors are frozen at run creation. Later proposals identify them by ID and may restore, blend, or scale named weights. The runner enforces the attempt budget, evaluates the selected candidate on confirmation once, and keeps reports for discarded edits. Each tool call waits for its bounded local operation, so client timeouts must accommodate the model and dataset size.

For another MCP client, install the same runtime and run `weight-edit-mcp --input-root /absolute/path/to/repair-inputs --runs-dir /absolute/path/to/runs` as a local stdio server.

[DistilGPT2 input preparation](../../examples/distilgpt2-quickstart/README.md)

[Live client validation and trial reports](../../examples/distilgpt2-quickstart/mcp/README.md)
