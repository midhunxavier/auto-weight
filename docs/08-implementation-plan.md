# Implementation status

The extension adds [evaluation preparation](06-cli-and-mcp.md#prepare-labeled-evaluation-data), development baselines, bounded tensor reads, replacement patches, and [external evaluators](15-external-evaluators.md). The [weight-space examples](../examples/weight-space/README.md) document runnable experiments and recorded evidence. The [extension plan](14-weight-space-plan.md) records the original design and scope.

The first end-to-end workflow is implemented.

1. Freeze the checkpoint, issue, datasets, gates, source identity and runtime.
2. Validate external proposal JSON and snapshot its donors.
3. Apply named-tensor edits in a bounded subprocess. Score the original and serialized candidate on identical development examples.
4. Record gate results and remove the temporary checkpoint.
5. Recreate one eligible candidate, verify its identity and evaluate fresh confirmation examples. Export the passing checkpoint with its report.
6. Verify exported artifacts and exercise both decisions on an observed regression. [Case study](../examples/copy-repair/README.md)

The Hugging Face adapter and text evaluator are also implemented. They load local safetensors and tokenizer assets, score deterministic generations, and publish a normal Transformers checkpoint. The regression case uses a pretrained SmolLM2 model with two local fine-tuning stages and exercises both decisions. [Hugging Face case study](../examples/json-repair/README.md)

Separate adapters would be needed for models that require custom code, quantized layouts, or an inference service. The current text checks cover JSON and exact responses.
