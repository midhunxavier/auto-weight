# Hugging Face checkpoint and text evaluation

Install `weight-edit-toolkit[hf]` or run `pip install -e '.[hf]'` from the repository root. Prepare a local Transformers causal language model directory with `config.json`, a fast `tokenizer.json` and `tokenizer_config.json`. The adapter accepts `model.safetensors` or standard safetensors shards with their index. It snapshots those assets plus recognized tokenizer files and generation metadata. The runner loads locally on CPU with remote code disabled. Prepare a dense checkpoint in a floating-point dtype with the model architecture available in Transformers.

Use the [quickstart command](../README.md#run-a-repair) to create a frozen workspace directly from model and dataset paths. Each `--improve` or `--preserve` takes a group, a metric, the required improvement or allowed decline, and an optional absolute score floor. Repeat either flag to add another gate. For example, `--improve api json_exact 0.50 0.90` requires a 50 percentage point gain and at least 90% observed accuracy.

`--prompt-format chat` wraps the prompt as a single user message with the checkpoint's saved chat template. The default `raw` tokenizes it directly. The evaluator left-pads batches, generates greedily, and decodes only new tokens with special tokens removed. Use `--max-new-tokens` to set the output cap. Prompts that exceed the input or context cap are rejected.

Run `weight-edit tensors --checkpoint runs/repair-001/base --prefix model.layers.23.` to see stored tensor names, shapes, and dtypes without loading the model for inference. A tied parameter may have one stored name. Run `weight-edit fingerprint --checkpoint ./my-donor` to obtain the digest for a donor reference. Donors must match the architecture configuration, tokenizer files, and edited parameter shapes and dtypes.

To initialize from a complete [issue JSON](04-patch-contract.md), use `weight-edit init`. The equivalent model field is:

```json
{"adapter":"hf-causal-lm-v1","config":{"prompt_format":"chat","max_input_tokens":128,"max_new_tokens":24}}
```

Write development and confirmation files as JSONL with independent prompts. Include at least 32 rows per gated group in each split. The following lines show a JSON API check and a routing preservation check.

```jsonl
{"group":"api","prompt":"Return the ticket priority as a raw JSON object. Priority: high.","check":{"type":"json","expected":{"priority":"high"},"schema":{"type":"object","required":["priority"],"properties":{"priority":{"enum":["low","normal","high"]}},"additionalProperties":false}}}
{"group":"routing","prompt":"Return just the assigned queue. Queue: billing.","check":{"type":"exact","expected":"billing"}}
```

| Metric | Requirement |
|---|---|
| `json_valid` | The entire completion is one JSON value. Prose, fences, duplicate keys, nonfinite values and numbers outside the canonicalizer's supported range fail. |
| `schema_match` | The parsed value passes the row's Draft 2020-12 schema. References can resolve within the schema. Evaluation does not fetch remote schemas. |
| `json_exact` | The parsed value equals the expected JSON, independent of whitespace and object key order. |
| `text_exact` | The response equals the expected string after trimming surrounding whitespace. |

Every JSON check supplies `json_valid`. Include `schema` and `expected` to enable the other JSON metrics. Use the same set of metrics throughout each group. A development report includes up to two paired example completions per group for the proposer. Confirmation reports contain aggregate results.

Set improvement and preservation gates for the required groups and metrics. For example, require a `min_delta` of `0.50` and a `min_candidate` of `0.90` for `json_exact`. The delta gate uses a paired lower confidence bound, and the absolute gate uses the observed rate. [Evaluation protocol](05-evaluation-protocol.md)

The accepted checkpoint is under `confirmation/result/model/`. It contains the complete weights and original inference assets. Load it using ordinary Transformers APIs.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "runs/repair-001/confirmation/result/model"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True)
```

[Complete JSON repair case and commands](../examples/json-repair/README.md) · [Transformers model loading reference](https://huggingface.co/docs/transformers/main/en/main_classes/model)
