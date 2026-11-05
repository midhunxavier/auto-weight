# Validation

The first case uses trained checkpoints from the same run, with a measured regression on repeats containing token collisions. The LLM supplies the edit hypothesis and operation. Fixed development gates select an eligible candidate and fresh confirmation examples determine export. [Inputs and measured results](../examples/copy-repair/README.md)

The Hugging Face case freezes three gates for a follow-up fine-tune that changed raw JSON into Markdown. It tests syntax, schema and correct values on API prompts and exact queue names on a separate preservation task. [JSON repair results](../examples/json-repair/README.md)

In a trainer's own repair, collect diverse failure examples and representative preservation examples. Verify the failure on the base before relying on improvement deltas. Include the prompt lengths, positions and input variants that matter for use. Separate the examples the LLM can iterate on from the examples used to accept the result.

Use a fresh issue for a follow-up problem. Record the new base hash and keep prior decisions so that regressions can be traced to the candidate that introduced them.
