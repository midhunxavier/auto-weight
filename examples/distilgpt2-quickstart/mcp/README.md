# AutoWeight Editor client validation

Both live sessions used the packaged `weight-edit` skill and the same local MCP server. Codex CLI 0.165.0 loaded the plugin from the AutoWeight local marketplace. Claude Code 2.1.318 loaded it with `--plugin-dir`. The agents received the same task and file names. They used separate repair runs, with no proposals supplied in the input folder.

The available inputs were the regressed DistilGPT2 checkpoint, its earlier donor, and the two JSONL files from the [DistilGPT2 quickstart](../README.md). The prompt asked each agent to recover raw priority JSON, preserve queue extraction, and use raw prompts with a 24-token generation limit. Neither agent was given a target tensor or a blend amount. Both received development examples through MCP. Confirmation responses contained scores and the exported path.

| Client | Development attempts | Correct raw JSON on confirmation | Exact queue answers on confirmation |
|---|---|---:|---:|
| Codex | Three single-block restores discarded. Fourth attempt passed with a 50% blend across changed tensors. | 0/60 → 60/60 | 60/60 → 60/60 |
| Claude Code | First attempt passed with a 50% blend across changed tensors. | 0/60 → 60/60 | 60/60 → 60/60 |

Each agent called confirmation once, then verified its run. Both exports have the same checkpoint digest. Loading each export with standard Transformers APIs reproduced both scores. The original checkpoint is a public pretrained model with a locally induced formatting regression on synthetic tickets. These runs reuse the quickstart's held-out split to test the plugin workflow.

- [Codex MCP calls and reload result](codex-run.json) and [confirmation report](codex-confirmation.json)
- [Claude Code MCP calls and reload result](claude-run.json) and [confirmation report](claude-confirmation.json)

The stdio integration test also checks a rejected edit, exhausted attempts, rejection of a second confirmation, and rejection of a path that escapes the configured input folder through a symlink.

[Install AutoWeight Editor](../../../plugins/autoweight-editor/README.md)
