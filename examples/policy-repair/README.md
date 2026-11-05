# Repairing JSON while keeping a newly learned policy

A follow-up fine-tune of SmolLM2-135M-Instruct taught a fulfillment policy and regressed an existing JSON interface. Both behaviors were trained together through the same last six transformer blocks and final normalization. API examples in the follow-up used Markdown fences around the JSON value. The fulfillment examples taught a six-entry rule table based on service code and payment status. The three actions were `dispatch`, `review`, and `hold`. The prompts included the input fields and a note, with the expected action absent.

The first pilot used prose for the input fields and reached 12/36 on the development policy prompts with different wording. Its confirmation set stayed unopened. The main run used stable field labels and 120 training steps. Its confirmation prompts use new order IDs, fresh notes, and an instruction template withheld from training. The six rule cases occur 16 times each. The API confirmation prompts use a separate template and seed from development. This is a controlled synthetic task on a pretrained language model.

| Development candidate | Exact raw JSON | Correct policy action | Decision |
|---|---:|---:|---|
| Joint fine-tune, before any edit | 0/36 | 36/36 | Base |
| Full rollback | 36/36 | 0/36 | Discarded |
| Blend 50% toward prior | 0/36 | 24/36 | Discarded |
| Blend 65% toward prior | 36/36 | 12/36 | Discarded |
| Blend 80% toward prior | 36/36 | 0/36 | Discarded |
| Subtract the reference update | 36/36 | 36/36 | Eligible |

Codex proposed an edit based on a separate checkpoint from the earlier JSON experiment. That checkpoint started from the same prior weights and learned to emit fenced JSON while practicing the original queue-name task. The edit subtracts its weight change from the joint checkpoint, which leaves the rest of the joint update in place.

```text
reference update = fenced reference − prior checkpoint
edited weights   = joint checkpoint − reference update
```

The proposal generator computed the replacement values for the 55 affected tensors. It recorded all three checkpoint hashes and the arithmetic used to derive the donor. The toolkit applied the replacement as a standard restore proposal and evaluated the serialized candidate. The predeclared gates required a 50-point improvement in both JSON schema match and exact JSON value, at most a 10-point decline in exact policy accuracy, and at least 90% observed accuracy on each metric.

The selected edit passed the single confirmation run. The original prior checkpoint was scored on the same confirmation set after selection and export, so the comparison uses the actual rollback model.

| Checkpoint on confirmation | Exact raw JSON | Correct policy action |
|---|---:|---:|
| Joint fine-tune | 0/96 | 96/96 |
| Full rollback | 96/96 | 0/96 |
| Accepted edit | 96/96 | 96/96 |

The repair also passed JSON syntax and schema checks on all 96 API prompts. The result depends on a related reference checkpoint and covers these two tasks with this input format.

- [Submitted issue](issue.json) and [frozen issue](frozen-issue.json)
- [Rollback proposal](rollback.json) and [development report](development-001.json)
- [Uniform blend reports at 50%](development-002.json), [65%](development-003.json), and [80%](development-004.json)
- [Accepted proposal](format-edit.json), [donor derivation](format-edit-derivation.json), and [development report](development-005.json)
- [Confirmation selection](selection.json), [acceptance report](confirmation.json), and [three-checkpoint comparison](comparison.json)
- [Training recipe](../../scripts/train_policy_regression.py) and [loss log](training.log)

## Reproduce locally

Prepare the `raw-json` and `fenced-json` checkpoints by following the prior experiment. The commands below assume they live under `runs/smollm2-json-training/`. The run used Transformers 4.57.6 and PyTorch 2.5.1 with Apple MPS for training and CPU for evaluation. [Prior checkpoint recipe](../json-repair/README.md)

```sh
python scripts/train_policy_regression.py \
  --source runs/smollm2-json-training/raw-json --output runs/policy-training
python scripts/prepare_policy_repair.py \
  --training runs/policy-training --prior runs/smollm2-json-training/raw-json \
  --output runs/policy-inputs
weight-edit init --issue runs/policy-inputs/issue.json --workspace runs/policy-repair
for proposal in rollback blend-50 blend-65 blend-80; do
  weight-edit try --workspace runs/policy-repair --proposal "runs/policy-inputs/$proposal.json"
done
python scripts/propose_policy_format_edit.py \
  --joint runs/policy-training/joint --prior runs/smollm2-json-training/raw-json \
  --fenced-reference runs/smollm2-json-training/fenced-json \
  --strength 1 --output runs/policy-format-edit
weight-edit try --workspace runs/policy-repair --proposal runs/policy-format-edit/proposal.json
weight-edit finalize --workspace runs/policy-repair --attempt 005
weight-edit verify --workspace runs/policy-repair
python scripts/score_policy_controls.py \
  --workspace runs/policy-repair --prior runs/smollm2-json-training/raw-json \
  --output runs/policy-comparison.json
```
