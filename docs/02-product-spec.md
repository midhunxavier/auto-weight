# Checkpoint repair workflow

1. The owner supplies a complete base checkpoint, failure description, development and confirmation examples, preservation gates, and a budget.
2. An external LLM receives the development context and proposes a hypothesis and exact operations against the pinned base.
3. The runner applies the operations to a temporary candidate and compares it with the base on identical examples. It records the result and removes the temporary checkpoint.
4. The owner or orchestrating LLM selects one eligible development attempt. The runner recreates the same candidate and evaluates it on confirmation examples.
5. A candidate that passes every gate becomes a new checkpoint with a report. A failed or errored evaluation leaves the report and closes the confirmation path for the issue.

The same base anchors all trials in a workspace. To iterate after accepting a model, initialize a new issue against the exported checkpoint and provide tests appropriate to the new problem. Use fresh confirmation examples for any candidate that is changed in response to a previous confirmation result.

Choose the adapter for your checkpoint and the behavior under test. The small transformer adapter evaluates tokenized next-token predictions. The Hugging Face adapter generates text and checks raw JSON, schema, values and exact text. [Hugging Face workflow](12-hugging-face.md)
