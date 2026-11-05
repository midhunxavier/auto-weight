# Repairing invalid JSON after fine-tuning

This case uses [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), a pretrained language model, with a controlled regression from local supervised fine-tuning. The tasks use synthetic ticket notes. Each API prompt asks for raw JSON with the ticket's priority. Each routing prompt asks for the ticket's queue name.

The first fine-tune trains the last six blocks and final normalization for 60 steps on raw JSON and queue responses. The follow-up trains the same parameters for 35 steps on examples that enclose the API answer in Markdown fences and continue to return plain queue names. Both use batches of eight, with loss on response tokens only. In the follow-up checkpoint, a request for raw JSON could return the following, which a JSON parser rejects as a complete response.

````text
```json
{"priority":"high"}
```
````

Codex proposed restoring the earlier final normalization as a narrow readout test, and blending the updated tensors 80% toward the earlier raw-JSON checkpoint. The runner tried both on 32 development prompts per task. The first still emitted fences and was discarded. The second corrected all 32 API responses and preserved all 32 routing responses. It was selected for the single confirmation run. The frozen gates require at least a 50 percentage point improvement on both schema match and correct JSON value, no more than a 10 point routing regression, and an observed rate of at least 90% on each gated metric.

The confirmation set contains 64 prompts per task with new ticket IDs, a different seed and prompt templates reserved from training. The valid JSON, schema match and exact value scores agreed.

| Checkpoint on confirmation | Valid raw JSON | Correct JSON value | Exact queue name |
|---|---:|---:|---:|
| Before follow-up fine-tune | 64/64 | 64/64 | 64/64 |
| After follow-up fine-tune | 0/64 | 0/64 | 57/64 |
| Accepted six-block blend | 64/64 | 64/64 | 64/64 |

The runner applied the edit to 55 tensors, reloaded the serialized candidate for scoring, and published the identical checkpoint after confirmation. A separate reload using the standard Transformers APIs reproduced the scores. The earlier checkpoint matched the repaired checkpoint on this suite. The result shows that the loop can reverse the observed regression on this controlled task. It does not establish that blending beats rolling back or that the edit preserves other capabilities.

- [Upstream revision and digest](upstream.json), [training log](training.log), and [training recipe](../../scripts/train_json_regression.py)
- [Submitted issue](issue.json) and [frozen issue](frozen-issue.json)
- [Normalization proposal](restore-norm.json) and [discard report](development-001.json)
- [Blend proposal](blend-late-layers.json) and [development report](development-002.json)
- [Selection](selection.json), [acceptance report](confirmation.json), and [independent reload comparison](reload-comparison.json)

## Reproduce locally

Use the package environment with `.[hf]` installed. This run used PyTorch 2.5.1 with Apple MPS for training, Transformers 4.57.6, and CPU inference. Set `--device cpu` when MPS is unavailable. Training and serialization can produce different hashes across environments. The input preparation script fingerprints the resulting files.

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --include config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir .artifacts/smollm2-135m
python scripts/train_json_regression.py --source .artifacts/smollm2-135m --output runs/json-training
python scripts/prepare_json_repair.py --training runs/json-training --output runs/json-inputs
weight-edit init --issue runs/json-inputs/issue.json --workspace runs/json-repair
weight-edit try --proposal runs/json-inputs/restore-norm.json --workspace runs/json-repair
weight-edit try --proposal runs/json-inputs/blend-late-layers.json --workspace runs/json-repair
weight-edit finalize --attempt 002 --workspace runs/json-repair
weight-edit verify --workspace runs/json-repair
```

The accepted weights and tokenizer are under `runs/json-repair/confirmation/result/model/`. [Reload instructions](../../docs/12-hugging-face.md#hugging-face-checkpoint-and-text-evaluation)
