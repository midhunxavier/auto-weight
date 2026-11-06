# Checkpoint repair and weight-space experiments with AutoWeight

A fine-tune can improve a model on one task and break a behavior another part of the system depends on. An API client may suddenly receive fenced JSON. A routing model may learn a new queue name and start misclassifying urgent tickets. AutoWeight makes a targeted repair testable. An agent proposes a bounded tensor edit, the local runner grades it against the base checkpoint, and a separate confirmation split decides whether to export it.

The repo now has two suites. Five repair cases compare Codex and Claude Code with fixed edits and, where available, full rollback. A second suite covers the eight numbered studies from Takehi, Teoh, Zhang, and Khattab, with local tools for inspecting weights, replaying published edits, and scoring one-shot policies. The extended commands and examples run from a source checkout. [Agent repair cases](../../examples/agent-repairs/README.md) · [Weight-space studies](../../examples/weight-space/README.md) · [Install from a checkout](../06-cli-and-mcp.md#install-from-a-checkout)

## How AutoWeight works today

```mermaid
flowchart LR
    Inputs["Checkpoint and evaluation cases"] --> Runner["Freeze workspace and gates"]
    Agent["Agent or terminal user"] --> Edit["Propose a tensor edit"]
    Runner --> Edit
    Edit --> Dev["Score on development cases"]
    Dev -->|"Feedback"| Agent
    Dev -->|"One eligible edit"| Final["Score on confirmation cases"]
    Runner -->|"One-shot construction"| Submit["Submit one candidate"]
    Submit --> Final
    Final -->|"Pass"| Export["Export checkpoint and report"]
    Final -->|"Fail"| Report["Keep the report"]
```

**Prepare and inspect.** `prepare` can split labeled text examples by prompt family or validate existing splits. The runner freezes the checkpoint, cases, gates, and budget, then scores the development baseline. Agents can list tensors and read bounded slices to decide where to edit. [CLI and MCP guide](../06-cli-and-mcp.md)

**Apply and measure.** Codex and Claude Code use local MCP. The CLI supports manual proposals or an Ollama proposer. Built-in repair can restore, blend, or scale weights. Replacement payloads can supply explicit values for entire tensors or slices. Text and token evaluators ship with the runner, and a trusted local evaluator can return numeric observations from a simulator or a separate model runtime. [Architecture](../03-architecture.md) · [External evaluator protocol](../15-external-evaluators.md)

**Confirm and export.** Repair uses paired improvement and preservation gates. The runner rebuilds one selected edit, checks its hash against the development candidate, and scores it on the held-out cases. Construction reserves a single submission and checks an absolute score floor. The passing result includes the evaluated NPZ or Hugging Face checkpoint and a report.

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

## From reading weights to building a policy

The research examples use revision-pinned assets and published weights. The local runs below test how AutoWeight stages, scores, and exports them. The last column describes what the original team reported with its own proposer and evaluation protocol. [Replay commands and local results](../../examples/weight-space/README.md) · [Original study](https://rikiyat.github.io/blog/weight-space-programming/)

| Study | What you can do in AutoWeight | Published result |
| --- | --- | --- |
| 1. Identify a game policy | Stage four weight sets under opaque IDs and score submitted labels | Astra named all four games |
| 2. Identify a diffusion model's target shape | Stage three checkpoints and score submitted labels | Astra identified all three shapes |
| 3. Identify language and copying heads | Inspect middle blocks and score labels and head lists | All three languages and their copying heads identified |
| 4. Repair the Kinetix ramp policy | Rebuild the rounded base and replay the edit on paired ramp and preservation cases | Base succeeded in 0/256 episodes. The edit succeeded in 1,024/1,024 |
| 5. Construct MinAtar policies | Replay Breakout, Seaquest, and Asterix from zero templates. Local means were **526.69**, **60.50**, and **54.22** over 32 episodes each | Means were 517.4, 53.1, and 53.2 over 1,000 episodes each |
| 6. Construct a 9×9 Go policy | Replay the zero-practice weights. They won **5/8** against a seeded uniform legal-move opponent | The zero-practice policy beat two smaller PPO opponents. The 50-pass policy beat the equal-sized PPO control |
| 7. Diagnose sycophancy | Fetch the original results, or compare attention on a compatible pair you supply | Attention-assisted identification succeeded on three pairs. Weights-only attempts were inconclusive |
| 8. Repair Qwen reasoning loops | Apply the published patch with a separate generation evaluator, or retrieve benchmark summaries | MATH-500 rose from 7.9% to 48.1%. The share hitting the token limit fell from 93.0% to 33.1% |

We verified staging and scoring for the three diagnosis examples using the reference keys. The MinAtar and Go replays passed their local score gates, and `verify` confirmed the exported checkpoints. A separate offline classifier exercised paired repair through the same external evaluator protocol. Confirmation accuracy for the repaired class rose from **0/32 to 32/32**, and the preserved class stayed at **32/32**.

The local Go run measures a different opponent, and the MinAtar sample uses 32 episodes per game. The Kinetix and Qwen figures in the table come from the authors' published runs. The original team also tried targeted edits on a Gemma model that rarely looped and found no established math improvement. The Qwen result shows how much removing a particular failure mode can matter. [Original benchmark protocols](https://rikiyat.github.io/blog/weight-space-programming/)

## How to use it effectively

1. **Write down what broke and what must survive.** Use exact text for short labels and exact JSON with a schema for API responses. The schema fixture returned valid JSON on every base example, so a syntax-only check would have missed the regression. Include at least 32 distinct examples per gated group in each split. Vary the failure conditions and input forms that matter in your application.
2. **Measure the original and the prior before spending an agent budget.** Use development examples for that comparison. If the prior still meets the relevant checks, rollback may be enough. Otherwise, a modest blend across changed weights is a reasonable first trial for closely related checkpoints. Offer the agent the strongest compatible donor and describe which useful behaviors it predates.
3. **Make trial feedback drive the next edit.** Give Codex or Claude Code the model and split paths, the improvement group, the behavior to preserve, and the generation settings. Pass exact tool arguments such as `donors=["prior"]`. Claude retried run creation in three scenarios after guessing donor directory names. If a broad edit fixes the target and breaks preservation, try a smaller blend or a narrower tensor group. Match the prompt format and output cap to how the model is actually used.
4. **Confirm once and reload the export.** Reserve the confirmation examples for one chosen development candidate. Look at the actual scores and the preservation gate. Add an absolute preservation floor through the manual CLI when your product has a minimum acceptable rate. Load the exported checkpoint through the same serving path you plan to use.

Try the [schema migration example](../../examples/agent-repairs/schema-migration/README.md) first if you want to see both decisions in one run. It generates its checkpoints locally. For an interactive session, load the AutoWeight plugin, configure the case input directory with `auto-weight setup-runtime`, and paste the example prompt. [Codex and Claude Code setup](../../plugins/auto-weight/README.md)

## When the experiment is worth running

AutoWeight makes sense when a regression is narrow, the model fits the local runtime, and you can score the fix and preservation cheaply. It is especially useful when you have checkpoints from the same training lineage and need a repeatable way to test edits. The agents found useful tensor choices in two cases where the fixed recipe failed, but the controls also show where an agent adds little. Treat the proposed edit as a candidate and the frozen evaluation as the basis for accepting it.
