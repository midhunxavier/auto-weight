# Repairing a copying regression

A four-layer causal transformer trained for a copying study (seed 11101) became worse at copying repeated sequences with token collisions between steps 500 and 1,000. Its training task used repeated permutations of distinct tokens. Both checkpoints (`c000` and `c002`) were saved at fixed steps during the earlier study, before the repair was designed. The earlier study observed 44.6% versus 33.2% accuracy on its fixed-position collision prompts. This repair uses a separate dataset with offsets 0 through 16. [Original training and measurements](https://github.com/midhunxavier/weight-space-programming/blob/9b4a7b1f2a0156caba723158b4c3ff31c1ede2bd/packages/weight-edit-toolkit/research/inspector/examples/specificity-validation-20261105/README.md)

Codex authored two proposals. It first tested whether restoring the earlier readout would repair the regression. Exploratory scores showed almost no accuracy change despite lower loss. Exploratory blends at 10%, 25% and 50% toward step 500 improved accuracy. Codex selected the 50% blend for the tracked workflow and recorded both proposals before running confirmation. The blend targets the complete state and changes 33 tensors, with no selection of known heads.

The frozen issue requires a lower confidence bound of at least +3 percentage points on collision-copy accuracy and at least −2 points on unique-copy accuracy. Development uses 256 examples per group. Confirmation uses 1,024 per group from a separate seed. The collision set balances the 17 prompt offsets. Unique-copy uses variable sequence lengths and offsets from the training distribution.

| Trial | Collision accuracy, base → candidate | Unique accuracy, base → candidate | Result |
|---|---:|---:|---|
| Development, restore readout | 39.58% → 39.67% | 99.43% → 99.41% | Discarded |
| Development, blend checkpoints | 39.58% → 47.55% | 99.43% → 99.35% | Eligible |
| Confirmation, blend checkpoints | 38.75% → 47.41% | 99.31% → 99.31% | Accepted |

On confirmation the collision improvement was **+8.66 points**, with a lower bound of **+8.29 points**. The unique-copy change was **+0.008 points**, with a lower bound of **−0.10 points**. Collision NLL improved from 3.852 to 2.766. Unique-copy NLL increased from 0.0291 to 0.0349. The issue gated accuracy and recorded NLL as a secondary measure. The blend itself was a simple checkpoint interpolation. A full rollback was not scored on confirmation. The repaired model still misses roughly half the collision targets, and this study covers one training run on synthetic sequences and two related tasks.

The evaluator loaded the serialized candidate for both stages. Verification matched the accepted NPZ bytes to both evaluation reports. The input base and donor remained unchanged.

- [Issue as submitted](issue.json) and [frozen workspace issue](frozen-issue.json)
- [Readout proposal](restore-output.json) and [development report](development-001.json)
- [Blend proposal](blend-checkpoints.json) and [development report](development-002.json)
- [Confirmation selection](selection.json) and [acceptance report](confirmation.json)

## Reproduce locally

From the repository root, use an environment with `.[torch]` installed. The original full NPZ files are local run artifacts. Use the hashes in the submitted issue and proposals to identify the original files. The archived training recipe can produce comparable checkpoints, with new hashes if the training environment differs. The preparation script regenerates the example sets and writes paths appropriate to the new run.

```sh
python scripts/prepare_copy_repair.py \
  --base runs/specificity-validation-20261105/private/c002.npz \
  --donor runs/specificity-validation-20261105/private/c000.npz \
  --output runs/copy-repair-replay-inputs
auto-weight init --issue runs/copy-repair-replay-inputs/issue.json --workspace runs/copy-repair-replay
auto-weight try --proposal runs/copy-repair-replay-inputs/restore-output.json --workspace runs/copy-repair-replay
auto-weight try --proposal runs/copy-repair-replay-inputs/blend-checkpoints.json --workspace runs/copy-repair-replay
auto-weight finalize --attempt 002 --workspace runs/copy-repair-replay
auto-weight verify --workspace runs/copy-repair-replay
```
