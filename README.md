# AutoWeight

**Edit your model’s weights with an LLM.**

Tell Codex or Claude Code what to fix and what should keep working. The agent proposes weight edits. AutoWeight tests each edit on a copy of your model and saves a new checkpoint only if it passes your checks on fresh examples.

## A fine-tune broke JSON. Rolling back erased the new policy.

We fine-tuned SmolLM2 to learn an order fulfillment policy. It learned the policy, but started wrapping JSON in Markdown fences, breaking the parser. Rolling back fixed the JSON and erased the policy. A weight edit proposed by Codex passed both checks.

| On fresh examples | Correct raw JSON | Correct policy decisions |
| --- | ---: | ---: |
| After fine-tuning | 0/96 | 96/96 |
| Full rollback | 96/96 | 0/96 |
| Accepted weight edit | **96/96** | **96/96** |

This was a controlled experiment on synthetic orders. Codex used a related checkpoint to build the edit, and the repair ran without further training. [Read the experiment and replay it](examples/policy-repair/README.md)

## Try it

- **Run the demo.** Start with DistilGPT2 and reproduce a repair with the included examples. [DistilGPT2 quickstart](examples/distilgpt2-quickstart/README.md)
- **Repair your own model.** Set up AutoWeight Editor in Codex or Claude Code. Bring a local Hugging Face causal language model with safetensors weights and a tokenizer. Make two sets of prompts and expected answers. One guides the edits, and the other is held back for the final check. [Plugin setup](plugins/autoweight-editor/README.md) · [Example format](docs/12-hugging-face.md)

Once you’ve prepared the inputs, ask your agent:

> Fix the fenced JSON responses from `base`. Keep the fulfillment policy working. Use `development.jsonl` to try edits and `confirmation.jsonl` for the final check. Save the repaired checkpoint if it passes both checks.

AutoWeight returns a report for each attempt. A passing final check also gives you a model directory you can load with Transformers. [Load the repaired model](docs/06-cli-and-mcp.md#load-an-accepted-model)

## Go deeper

- [Run automatic repair from the terminal with Ollama](docs/13-automatic-repair.md)
- [Review proposals yourself with the CLI](docs/06-cli-and-mcp.md#review-proposals-yourself)
- [See more repairs](examples/json-repair/README.md) · [Repair copying in a small transformer](examples/copy-repair/README.md)
- [Explore the architecture](docs/03-architecture.md) · [See how edits are evaluated](docs/05-evaluation-protocol.md) · [Browse the documentation](docs/)
- [Read Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) · [Explore the companion research repo](https://github.com/midhunxavier/weight-space-programming)
