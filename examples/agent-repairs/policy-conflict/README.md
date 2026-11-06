# Reject a policy edit that breaks verified requests

The base allows both verified and unverified requests. The donor denies both. The fixture's attention weights are zero, and both types of prompt end with the same decision token. The offered weights give the supported operations no representation to distinguish the two outcomes.

| Checkpoint on confirmation | Correct denial of unverified requests | Correct allowance of verified requests |
| --- | ---: | ---: |
| Base control | 0/32 | 32/32 |
| Full prior control | 32/32 | 0/32 |

Both hosts tried four blends on development. The 50% blend retained the base decisions. Each larger blend improved denials and broke verified allowances. Every proposal was rejected, and neither run selected a candidate for confirmation. The table scores the two original checkpoints separately as controls. [Agent prompt](prompt.md) · [Measured results](results.json)

## Reproduce

```sh
python examples/agent-repairs/prepare.py policy-conflict --output runs/agent-repairs/policy-conflict-inputs
python examples/agent-repairs/run.py policy-conflict \
  --inputs runs/agent-repairs/policy-conflict-inputs --output runs/agent-repairs/policy-conflict-run
```
