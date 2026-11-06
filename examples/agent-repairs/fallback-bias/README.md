# Fix a default answer without a donor

A final normalization bias makes a tiny GPT-2 return `unknown` for requests that should receive `yes` or `no`. The same base already routes tickets to the correct queue. This case offers no donor and uses the `scale` operation.

| Checkpoint on confirmation | Exact access decision | Exact queue |
| --- | ---: | ---: |
| Base | 0/32 | 32/32 |
| Codex export | 32/32 | 32/32 |
| Claude Code export | 32/32 | 32/32 |

Both hosts set `transformer.ln_f.bias` to zero by scaling it with a factor of 0. The fixed 0.5 scale of the output head left development decisions at 0/32. This fixture makes the regression deliberately local so the comparison tests whether the proposer can choose a useful tensor with no donor. [Agent prompt](prompt.md) · [Measured results](results.json)

## Reproduce

```sh
python examples/agent-repairs/prepare.py fallback-bias --output runs/agent-repairs/fallback-bias-inputs
python examples/agent-repairs/run.py fallback-bias \
  --inputs runs/agent-repairs/fallback-bias-inputs --output runs/agent-repairs/fallback-bias-run
```
