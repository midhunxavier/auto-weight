# Recover the queue schema and keep the new policy

The fixture emits JSON with a legacy `team` field. The required schema uses `queue` and permits `billing` or `security`. The same output tensor carries the newer order decisions, so the donor can repair JSON and regress the policy in one edit.

| Checkpoint on confirmation | Exact JSON and schema match | Exact policy |
| --- | ---: | ---: |
| Base | 0/32 | 32/32 |
| Full prior | 32/32 | 0/32 |
| Codex export | 32/32 | 32/32 |
| Claude Code export | 32/32 | 32/32 |
| Fixed 50% blend export | 32/32 | 32/32 |

The base returned syntactically valid JSON on all 32 cases. Codex passed with a 50% blend of `lm_head.weight`. Claude first restored the entire tensor, hit the preservation gate, then passed with the same 50% blend. [Agent prompt](prompt.md) · [Measured results](results.json)

## Reproduce

```sh
python examples/agent-repairs/prepare.py schema-migration --output runs/agent-repairs/schema-migration-inputs
python examples/agent-repairs/run.py schema-migration \
  --inputs runs/agent-repairs/schema-migration-inputs --output runs/agent-repairs/schema-migration-run
```
