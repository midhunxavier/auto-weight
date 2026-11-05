# Evaluation and decision rules

For the NPZ adapter, each JSONL row contains a `group`, a list of integer `tokens`, and sorted unique `positions`. Position `i` scores the prediction of `tokens[i + 1]` after seeing `tokens[0:i + 1]`. A row can measure one answer or several next-token targets. For example, `{"group":"copy","tokens":[3,1,3,1],"positions":[2]}` tests whether the model predicts token 1 after the second 3.

For the Hugging Face adapter, each row contains a `group`, a text `prompt` and a `check`. It measures `json_valid`, optional `schema_match` and `json_exact`, or `text_exact`. Each response counts as one observation. The evaluator compares the actual tokenized prompts to detect duplicates and split overlap. [Text evaluation format](12-hugging-face.md)

Provide at least 32 independent rows per gated group and split. The NPZ evaluator uses each row's prefix through its last scored query to check for duplicates within a split and exact overlap across splits. The experiment owner remains responsible for semantic separation and representative test coverage. Give the proposer development examples and aggregate trial results. The confirmation set lives in the local workspace under the user's filesystem permissions.

Accuracy and negative log likelihood are averaged within each sequence, then equally across sequences. A positive signed delta is beneficial for either metric. For accuracy it is candidate minus base. For NLL it is base minus candidate. Accuracy margins are fractions, so `0.03` means three percentage points.

A gate passes when the one-sided paired bootstrap lower bound reaches the specified margin. The evaluator resamples the paired sequence deltas 4,096 times and uses the `0.05 / number_of_gates` quantile for each gate. Use a positive margin for improvement and a nonpositive margin for preservation. A rate gate with `min_candidate` additionally requires the candidate’s observed score to meet that floor. All gates must pass. The resulting intervals describe the supplied test distribution and assume independent sequences.
