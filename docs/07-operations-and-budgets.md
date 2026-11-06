# Operations and budgets

An issue freezes up to 20 development attempts, a wall time for each evaluator process, a batch size and a CPU thread count. An evaluation timeout kills the subprocess and records an error. A reserved attempt counts against the budget after an evaluator failure. Input validation before reservation leaves the attempt count available.

The NPZ adapter accepts up to 12 million parameters and 64 MiB per checkpoint. The Hugging Face adapter accepts up to 64 recognized files and 4 GiB per checkpoint directory. It runs dense models on CPU with bounded input length, output length, and batch size. Larger checkpoints need enough RAM for the base and donor plus temporary edit tensors. Proposals can contain up to 16 operations and four distinct donor checkpoints. Each dataset can contain up to 10,000 examples.

Each distinct donor is frozen once under the workspace's `donors/` directory. Later trials and confirmation reuse that copy by hash, even if the original is moved. The first completed development trial also saves the base model's per-example scores for the frozen issue and runtime. Subsequent trials score only the candidate. Confirmation scores both models on the separate confirmation split.

Keep the workspace on a local filesystem supporting file locks and atomic directory rename. Allow disk space for the frozen inputs, donor snapshots, and a temporary full candidate. The runner cleans its private temporary directories on ordinary exit and handled failure. An abrupt host or process kill may leave a private temporary directory and an interrupted attempt. A confirmation reservation stays consumed after interruption.
