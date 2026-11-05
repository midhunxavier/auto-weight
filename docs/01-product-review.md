# Product direction

A model trainer needs to decide whether a proposed change fixes a failure and merits keeping a new checkpoint. Start with failure examples and checks for behavior the trainer wants to preserve. Give the LLM development results so it can revise its hypothesis within a bounded number of attempts.

The main product is the experiment runner. The previous inspector targeted copying circuits and required separate validation of its rankings. A model issue such as output formatting or a loss of accuracy can be defined directly with tests, even when there is no circuit-level diagnosis. The inspector lives in the [research archive](https://github.com/midhunxavier/weight-space-programming/blob/9b4a7b1f2a0156caba723158b4c3ff31c1ede2bd/packages/weight-edit-toolkit/research/inspector/README.md).

The first use case is a regression between compatible checkpoints. For example, an LLM can propose restoring the readout from an earlier checkpoint. If development accuracy barely moves, it can try a broader blend. Confirmation then tests the chosen candidate on fresh cases. [Measured case](../examples/copy-repair/README.md)
