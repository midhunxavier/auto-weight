# When AutoWeight Helps With Checkpoint Repair

A fine-tune can improve a model on one task and break a behavior another part of the system depends on. An API client may suddenly receive fenced JSON. A routing model may learn a new queue name and start misclassifying urgent tickets. AutoWeight makes a targeted repair testable. An agent proposes a bounded tensor edit, the local runner grades it against the base checkpoint, and a separate confirmation split decides whether to export it.

I would use it for a specific regression with objective checks for both the fix and the behavior at risk. In five controlled cases, selective edits recovered behavior that full rollback lost, and the evaluation gates also stopped edits that would have traded one policy error for another.

## What happened in five cases

Codex and Claude Code each received the same task, inputs, gates, and four development attempts per case. The agents saw development feedback from the AutoWeight MCP server and could confirm the first eligible candidate once. Each of the seven exports was reloaded with Transformers and rescored on confirmation. The fixed control tried one 50% blend across all donor differences or, for the donor-free case, a 0.5 scale of the output head. Where a prior checkpoint existed, it was scored separately on the same confirmation split. [Prompts, replay commands, and result records](../../examples/agent-repairs/README.md)

| Regression | Base fix / preserve | Confirmed repair fix / preserve | What made the difference |
| --- | --- | --- | --- |
| Fenced ticket JSON | 0/60 · 60/60 | 60/60 · 60/60 | Codex restored all 38 changed tensors. Full rollback and the fixed blend reached the same scores. Claude spent four attempts on narrower restores. |
| Wrong JSON field | 0/32 · 32/32 | 32/32 · 32/32 | Both hosts and the fixed blend passed at 50%. Full rollback recovered the schema and dropped policy accuracy to 0/32. |
| Misclassified chat severity | 16/32 · 32/32 | 32/32 · 32/32 | Both hosts restored only the input embedding after the output-head restore failed. The fixed blend dropped development queue accuracy to 16/32. |
| Default answer after calibration | 0/32 · 32/32 | 32/32 · 32/32 | Both hosts zeroed the final normalization bias. There was no donor, and scaling the output head left the default answer in place. |
| Conflicting access decisions | 0/32 · 32/32 | No eligible edit | Larger blends fixed denials and broke verified allowances. Both hosts stopped without exporting a model. |

The strongest case for the proposer was chat severity. Both full rollback and the fixed blend regressed queue routing. The failed output-head restore gave both agents a reason to look elsewhere, and restoring the embedding passed both checks. In the schema case, the fixed blend did as well as either agent. In the DistilGPT2 case, even a full rollback sufficed on the two measured behaviors.

The conflict case shows the value of the runner independently of the proposer. With the available weights, the two prompt types reached the same final decision representation. Trying a stronger blend only moved the error from unverified to verified requests. A data or training change would be needed to make this fixture distinguish them.

The DistilGPT2 case used a pretrained model with a locally induced regression on synthetic tickets and reused the existing experiment's holdout. The other four checkpoints had deliberately constructed weights, and their held-out prompts used new wording and IDs with the same answer classes. These results describe the five scenarios and one session per client per scenario. They establish how the workflow behaves under controlled conditions and leave production coverage to the user's evaluation set.

## How to use it effectively

1. **Write down what broke and what must survive.** Use exact text for short labels and exact JSON with a schema for API responses. The schema fixture returned valid JSON on every base example, so a syntax-only check would have missed the regression. Include at least 32 distinct examples per gated group in each split. Vary the failure conditions and input forms that matter in your application.
2. **Measure the original and the prior before spending an agent budget.** Use development examples for that comparison. If the prior still meets the relevant checks, rollback may be enough. Otherwise, a modest blend across changed weights is a reasonable first trial for closely related checkpoints. Offer the agent the strongest compatible donor and describe which useful behaviors it predates.
3. **Make trial feedback drive the next edit.** Give Codex or Claude Code the model and split paths, the improvement group, the behavior to preserve, and the generation settings. Pass exact tool arguments such as `donors=["prior"]`. Claude retried run creation in three scenarios after guessing donor directory names. If a broad edit fixes the target and breaks preservation, try a smaller blend or a narrower tensor group. Match the prompt format and output cap to how the model is actually used.
4. **Confirm once and reload the export.** Reserve the confirmation examples for one chosen development candidate. Look at the actual scores and the preservation gate. Add an absolute preservation floor through the manual CLI when your product has a minimum acceptable rate. Load the exported checkpoint through the same serving path you plan to use.

Try the [schema migration example](../../examples/agent-repairs/schema-migration/README.md) first if you want to see both decisions in one run. It generates its checkpoints locally. For an interactive session, load the AutoWeight plugin, configure the case input directory with `auto-weight setup-runtime`, and paste the example prompt. [Codex and Claude Code setup](../../plugins/auto-weight/README.md)

## When the experiment is worth running

AutoWeight makes sense when a regression is narrow, the model fits the local runtime, and you can score the fix and preservation cheaply. It is especially useful when you have checkpoints from the same training lineage and need a repeatable way to test edits. The agents found useful tensor choices in two cases where the fixed recipe failed, but the controls also show where an agent adds little. Treat the proposed edit as a candidate and the frozen evaluation as the basis for accepting it.
