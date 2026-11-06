# AutoWeight

Repair a local Hugging Face model from Codex or Claude Code. The shared `auto-weight` skill lets the host agent propose an edit. The local MCP server applies it to a temporary copy, compares it with the original, and exports a complete checkpoint if the selected candidate passes confirmation.

## Set up the local runtime

Install the shared Python wheel with the `[hf,mcp]` extras, then run `auto-weight setup-runtime --input-root /path/to/repair-inputs` with your existing input directory. Once it reports `ready`, the plugin can use the saved interpreter and paths. [Wheel installation](../../docs/06-cli-and-mcp.md#install-the-python-runtime) · [MCP runtime configuration](../../docs/06-cli-and-mcp.md#configure-the-local-mcp-runtime)

## Load the plugin

Run the command for your host from the repository root.

**Codex.** Add this repository as a local marketplace and install AutoWeight. Start a fresh Codex session and invoke `$auto-weight`. The desktop app can also discover it through the AutoWeight repository marketplace. [OpenAI Docs for local marketplaces](https://developers.openai.com/plugins/build/plugins)

```sh
codex plugin marketplace add "$PWD"
codex plugin add auto-weight@auto-weight
```

**Claude Code.** Load the same plugin directory for a session and invoke `/auto-weight:auto-weight`. [Claude Code plugin setup](https://code.claude.com/docs/en/plugins/create)

```sh
claude --plugin-dir "$PWD/plugins/auto-weight"
```

## Ask for a repair

> Fix the fenced JSON responses from `base` and preserve queue extraction. Use `development.jsonl` and `confirmation.jsonl`. The improvement group is `api`. The compatible checkpoint at `prior` is available as a donor. Export the repaired checkpoint if it passes.

Paths can be relative to your configured input root. Supply at least 32 labeled examples per gated group in each JSONL file. Label improvement examples `fix` or name the improvement group in your request. The other groups check preservation. The host receives bounded development examples and scores. The server returns aggregate confirmation metrics and the accepted checkpoint path. [Hugging Face evaluation files](../../docs/12-hugging-face.md)

The five MCP tools are `create_run`, `list_targets`, `try_edit`, `confirm_edit`, and `get_run`. Donors are frozen at run creation. Later proposals identify them by ID and may restore, blend, or scale named weights. The runner enforces the attempt budget, evaluates the selected candidate on confirmation once, and keeps reports for discarded edits. Each tool call waits for its bounded local operation, so client timeouts must accommodate the model and dataset size.

For another MCP client, use the installed `auto-weight-mcp` command as a local stdio server with the saved runtime configuration.

[DistilGPT2 input preparation](../../examples/distilgpt2-quickstart/README.md)

[Live client validation and trial reports](../../examples/distilgpt2-quickstart/mcp/README.md)
