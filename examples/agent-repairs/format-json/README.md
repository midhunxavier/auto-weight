# Restore raw ticket JSON

The pretrained DistilGPT2 checkpoint learned ticket-priority JSON and queue extraction. A follow-up fine-tune wrapped API answers in Markdown fences. The fix group checks the entire parsed JSON value and the preservation group checks the exact queue name.

| Checkpoint on confirmation | Exact raw JSON | Exact queue |
| --- | ---: | ---: |
| Regressed base | 0/60 | 60/60 |
| Full prior | 60/60 | 60/60 |
| Codex export | 60/60 | 60/60 |
| Fixed 50% blend export | 60/60 | 60/60 |

Codex restored all 38 changed tensors in its first edit. Claude Code tried four narrower restores and used its budget on development. Each of those edits preserved queue extraction and missed the JSON improvement gate. A full rollback already passed both measured checks, so this case shows why it belongs in the comparison. [Agent prompt](prompt.md) · [Measured results](results.json)

## Reproduce

Generate `runs/distilgpt2-inputs` using the existing DistilGPT2 walkthrough, then stage it for the two clients. [Training recipe](../../distilgpt2-quickstart/README.md#reproduce-the-inputs)

```sh
python examples/agent-repairs/prepare.py format-json \
  --distilgpt2-inputs runs/distilgpt2-inputs \
  --output runs/agent-repairs/format-json-inputs
python examples/agent-repairs/run.py format-json \
  --inputs runs/agent-repairs/format-json-inputs \
  --output runs/agent-repairs/format-json-run
```
