# Original repair case studies

- [Copying regression in a trained research model](copy-repair/README.md). The regression emerged during an earlier training study. A checkpoint blend improved the affected behavior and passed the preservation check on held-out examples.

Add a case when the failure was observed before the repair experiment and a repair passed frozen improvement and preservation checks on independent examples. Include the source and checkpoint identities, the unedited scores, the proposed edit, and the confirmation report. Identify any simple controls that were measured, such as rollback or uniform scaling, and scope the claim to the behaviors evaluated.

The copying result covers a small research model and synthetic sequences. A confirmed repair of a public fine-tune would be a separate case.
