# AutoWeight documentation

## Use the tool

- [CLI and MCP setup](06-cli-and-mcp.md) covers installation, agent connections, commands, and reports.
- [Hugging Face and LoRA repair](12-hugging-face.md) describes supported checkpoints, evaluation data, and loading an accepted export.
- [Automatic repair with Ollama](13-automatic-repair.md) runs the proposal and evaluation loop from the terminal.
- [Evaluation rules](05-evaluation-protocol.md) explains independent splits, metrics, and acceptance gates.
- [Runtime limits and storage](07-operations-and-budgets.md) covers budgets, checkpoint sizes, and interruptions.

## Technical reference

- [Architecture](03-architecture.md) describes the runner, evaluator, and export lifecycle.
- [Issue and proposal formats](04-patch-contract.md) describes checkpoint references, gates, and weight operations.
- [External evaluators](15-external-evaluators.md) documents custom scoring and numerical replacement patches.

## Case study

- [Copying regression repair](../examples/copy-repair/README.md) contains the original experiment, provenance, and measured results.
