# Recover incident severity and keep current queue names

This fixture uses a saved chat template. A changed input embedding sends critical incidents to the normal-severity path. The donor contains the earlier embedding and obsolete queue names in its output head.

| Checkpoint on confirmation | Exact severity | Exact queue |
| --- | ---: | ---: |
| Base | 16/32 | 32/32 |
| Full prior | 32/32 | 0/32 |
| Codex export | 32/32 | 32/32 |
| Claude Code export | 32/32 | 32/32 |

Both hosts first tried the output head and failed both gates. Both passed by restoring only `transformer.wte.weight`. The 50% blend across both changed tensors reached 32/32 on development severity and dropped queue accuracy to 16/32, so it never reached confirmation. [Agent prompt](prompt.md) · [Measured results](results.json)

## Reproduce

```sh
python examples/agent-repairs/prepare.py chat-routing --output runs/agent-repairs/chat-routing-inputs
python examples/agent-repairs/run.py chat-routing \
  --inputs runs/agent-repairs/chat-routing-inputs --output runs/agent-repairs/chat-routing-run
```
