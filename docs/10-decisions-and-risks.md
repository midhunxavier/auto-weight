# Decisions and limits

The LLM owns hypotheses and proposed operations. The evaluator owns measurements. The model owner defines the gates before experimentation. This separation makes it possible to test a plausible edit even when the internal mechanism is unknown.

Confirmation is consumed once per workspace. Use an independent confirmation set for a changed proposal after seeing confirmation results. This local workflow relies on the operator to keep held-out files out of the proposer's context. Hashes and record identities detect accidental changes and bind an export to its inputs.

Shape compatibility alone cannot establish that checkpoints use the same token IDs or that tensors occupy comparable positions in weight space. Prefer donors from the same training lineage. Behavioral gates provide evidence about the measured tasks. They can miss side effects outside those tasks.

For text repair, choose a metric that captures both format and meaning. A completion may be valid JSON and still contain the wrong data. Pair schema or exact-value checks with an absolute quality floor when the application needs a minimum success rate. The reference evaluator fixes greedy generation and grades the entire decoded response. [Text metrics](12-hugging-face.md)
