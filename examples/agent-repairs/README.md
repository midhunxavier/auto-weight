# Five end-to-end AutoWeight repairs

Codex and Claude Code each received the same five tasks and worked through the local AutoWeight MCP server in separate workspaces. The suite includes a trained DistilGPT2 regression and four small GPT-2 fixtures with deliberately constructed weights.

| Scenario | Base on confirmation | Codex | Claude Code | Fixed one-edit control |
| --- | --- | --- | --- | --- |
| [Restore raw JSON](format-json/README.md) | JSON 0/60, queues 60/60 | Accepted on attempt 1 | No eligible edit in 4 attempts | Accepted |
| [Recover a JSON schema and keep the new policy](schema-migration/README.md) | JSON 0/32, policy 32/32 | Accepted on attempt 1 | Accepted on attempt 2 | Accepted |
| [Repair chat severity and keep current queue names](chat-routing/README.md) | Severity 16/32, queues 32/32 | Accepted on attempt 2 | Accepted on attempt 2 | Rejected on development |
| [Fix a default answer without a donor](fallback-bias/README.md) | Decisions 0/32, queues 32/32 | Accepted on attempt 1 | Accepted on attempt 1 | Rejected on development |
| [Reject an edit with conflicting policy checks](policy-conflict/README.md) | Unverified 0/32, verified 32/32 | No eligible edit in 4 attempts | No eligible edit in 4 attempts | Rejected on development |

Seven client runs exported a checkpoint. Reloading each export with Transformers reproduced its confirmation counts. The fixed control used a 50% blend of all changed tensors when a donor was available and a 0.5 output-head scale otherwise. The full prior checkpoint was also scored where available. [Experiment analysis and practical guide](../../docs/blog/auto-weight-checkpoint-repair.md)

## Run a case from the repository root

Use an activated Python environment with the wheel and both extras installed. The preparation and replay scripts import the installed package, so use the virtual-environment option in the runtime guide. Sign in to Codex and Claude Code before running the replay. [Wheel installation](../../docs/06-cli-and-mcp.md#install-the-python-runtime)

```sh
python examples/agent-repairs/prepare.py schema-migration \
  --output runs/agent-repairs/schema-migration-inputs
python examples/agent-repairs/run.py schema-migration \
  --inputs runs/agent-repairs/schema-migration-inputs \
  --output runs/agent-repairs/schema-migration-run
```

Replace `schema-migration` with `chat-routing`, `fallback-bias`, or `policy-conflict` to generate the other offline fixtures. The [DistilGPT2 case](format-json/README.md) starts with the checkpoints from the training walkthrough. Use a new output directory for each replay.

The replay uses the same MCP server and task prompt for both clients. It sets up a session-specific server, runs each client, checks the frozen records, reloads accepted checkpoints, and writes `results.json`. The local output directory also keeps the tool transcripts and verified workspaces. Client model choices and tool order can vary between replays. The recorded run used Codex CLI 0.165.0 with its default model and Claude Code 2.1.318 with `claude-sonnet-5-5`. [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-reference) · [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)

## Try a case in an interactive session

1. Generate or stage the inputs using the commands above.
2. Run `auto-weight setup-runtime --input-root /absolute/path/to/the/case-inputs` and load AutoWeight in your host.
3. Invoke `$auto-weight` in Codex or `/auto-weight:auto-weight` in Claude Code. Paste the case's task prompt.
4. Read the development decisions and confirm one eligible edit. Run `get_run` to verify the result and retrieve the model path. A case that exhausts its attempts ends with a record of the rejected edits.

The shell replay wires the server directly so both sessions use the same tool configuration. The interactive path uses the bundled skill and plugin. [Plugin setup](../../plugins/auto-weight/README.md)
