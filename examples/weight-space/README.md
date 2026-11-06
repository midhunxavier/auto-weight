# Weight-space examples

These examples cover the eight numbered studies in *Sparks of Weight-Space Programming*. They use published artifacts from Takehi, Teoh, Zhang, and Khattab. Preparation fetches only the selected files and verifies their sizes and SHA-256 hashes at a pinned revision. Upstream networks and logs are licensed CC BY 4.0. Simulator packages and benchmark datasets retain their own licenses. [Original study](https://rikiyat.github.io/blog/weight-space-programming/) · [Asset manifest](assets.json) · [Upstream license](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE)

| Case | What the example does |
| --- | --- |
| 1. Guess the game | [Stage and score](#inspection) a diagnosis for four game policies |
| 2. Identify diffusion shapes | [Stage and score](#inspection) shape labels for three checkpoints |
| 3. Identify language and copying heads | [Stage and score](#inspection) predictions from middle blocks against published measurements |
| 4. Repair the Kinetix car | [Replay the published edit](#kinetix) on paired ramp episodes from a rounded base |
| 5. Construct MinAtar networks | [Replay one submission per game](#minatar-and-go) for Breakout, Seaquest, and Asterix |
| 6. Construct a Go network | [Replay the zero-practice policy](#minatar-and-go) against a seeded random control or fetch the published scores across budgets |
| 7. Detect sycophancy | [Retrieve published evidence or probe your own pair](#sycophancy) |
| 8. Repair Qwen reasoning loops | [Replay the published patch or fetch benchmark scores](#qwen) |

Run commands from the repository root with AutoWeight installed into an active virtual environment. The extended CLI and example scripts use the source checkout. [Install from a checkout](../../docs/06-cli-and-mcp.md#install-from-a-checkout) · [Architecture and results](../../docs/blog/auto-weight-checkpoint-repair.md)

## CPU classifier

Start here to try an external evaluator with a synthetic model and no downloads.

```sh
python examples/weight-space/prepare.py classifier runs/classifier-inputs
auto-weight init --issue runs/classifier-inputs/issue.json --workspace runs/classifier
auto-weight baseline --workspace runs/classifier
auto-weight try --proposal runs/classifier-inputs/proposal.json --workspace runs/classifier
auto-weight finalize --attempt 001 --workspace runs/classifier
auto-weight verify --workspace runs/classifier
```

The expected confirmation accuracy for the fix group is 0 to 1, with the preservation group staying at 1. Reload the accepted `model.npz` with NumPy. Use `classifier-alt` to switch the affected class with the same evaluator. [Write your own evaluator](../../docs/15-external-evaluators.md)

## Inspection

```sh
python examples/weight-space/diagnose.py stage copying runs/copying-inputs
auto-weight tensors --checkpoint runs/copying-inputs/network-01.npz
auto-weight read-tensor --checkpoint runs/copying-inputs/network-01.npz \
  --tensor W_q.0 --slice 0 4 --slice 0 4
python examples/weight-space/diagnose.py score runs/copying-inputs answers.json runs/copying-score.json
```

Use `game` or `diffusion` in place of `copying` for the other two tasks. Write an object keyed by `network-NN`. Values are lowercase game or shape labels, or an object like `{"language":"english","heads":["L3H1"]}` for copying. The scorer checks the input hashes and records the submission hash and accuracy. To run a blind task, give the solver only the staged weights and `inputs.json`. Keep the staging directory and its answer key with the scorer, and control access to the released labels.

## Kinetix

Install the pinned simulator revision in a separate environment with an editable install so its level files are available. Pass that environment's Python executable to preparation. [Kinetix source](https://github.com/FLAIROx/Kinetix/tree/80ee9c292837c825cafaf9889323fdcc212efd1b)

```sh
python examples/weight-space/prepare.py kinetix runs/car-inputs --python /path/to/kinetix/bin/python
auto-weight init --issue runs/car-inputs/issue.json --workspace runs/car
auto-weight try --proposal runs/car-inputs/proposal.json --workspace runs/car
auto-weight finalize --attempt 001 --workspace runs/car
auto-weight verify --workspace runs/car
```

The ramp check measures success and the preservation group checks for large negative rewards at slightly shifted starts. This replay uses fixed action seeds. To fetch the authors' evaluation, run `python examples/weight-space/recorded.py kinetix runs/kinetix-published.json`.

## MinAtar and Go

Install `minatar==1.0.15` in a simulator environment. Prepare one game, then submit the published weights against a zero template. Use a fresh workspace for each submission.

```sh
python examples/weight-space/prepare.py minatar runs/seaquest-inputs \
  --game seaquest --episodes 32 --steps 10000 --python /path/to/simulator/bin/python
auto-weight init --issue runs/seaquest-inputs/issue.json --workspace runs/seaquest
auto-weight submit --proposal runs/seaquest-inputs/proposal.json --workspace runs/seaquest
auto-weight verify --workspace runs/seaquest
```

Select `breakout` (the default), `seaquest`, or `asterix` with `--game`. For a quick check, omit the episode and step options to run eight episodes capped at 256 steps. The table uses the longer run with the published six-channel input layout and sticky actions, 32 episodes, a 10,000-step cap, and seed 73. The local environment was macOS arm64 with Python 3.12 and NumPy 2.5.4. All three exports passed the example's mean score floor of 1 and verified. [Published MinAtar protocol](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch)

| Game | Local mean, 32 episodes | Published mean, 1,000 episodes |
| --- | ---: | ---: |
| Breakout | 526.69 | 517.4 |
| Seaquest | 60.50 | 53.1 |
| Asterix | 54.22 | 53.2 |

For Go, use the same `init`, `submit`, and `verify` sequence after preparing `go` with `--episodes 8 --steps 512` and an interpreter with `pgx==2.6.0` and JAX. The local replay with JAX 0.12.0.1 finished eight games and won five against the uniform legal-move control. The authors used other opponents for their main evaluation. Fetch those records for the four forward-pass budgets with `recorded.py go`, or use `recorded.py minatar` for the full MinAtar results.

## Sycophancy

```sh
python examples/weight-space/recorded.py sycophancy runs/sycophancy.json
```

The sycophancy example retrieves the three complete weights-only score files, the fourth round's completed prediction, and the attention-assisted scores. The authors retain the original paired checkpoints. With compatible local A/B checkpoints, use `attention_probe.py --first ./A --second ./B --prompt '...' --source 0 2` to compare attention mass on a half-open token span. It returns attention measurements by layer and head for the chosen query. Keep pair identities and diagnosis scoring with the experiment owner.

## Qwen

Download the [pinned base checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). Allow several GiB for the frozen base and temporary candidate in addition to the downloaded model. Install Torch, `transformers==5.22.1`, and `math-verify==0.9.0` in a separate environment. The default two-token CPU smoke checks the pipeline. Measuring the published improvement requires a long-generation benchmark.

```sh
python examples/weight-space/qwen_prepare.py runs/qwen-inputs \
  --base /path/to/pinned-qwen --python /path/to/qwen/bin/python
auto-weight init --issue runs/qwen-inputs/issue.json --workspace runs/qwen
auto-weight baseline --workspace runs/qwen
auto-weight try --proposal runs/qwen-inputs/proposal.json --workspace runs/qwen
auto-weight verify --workspace runs/qwen
```

For a benchmark, supply independent `--development` and `--confirmation` case manifests with problem and answer strings under `inputs`, plus explicit `--device`, `--max-tokens`, `--batch-size`, `--seed`, and `--timeout` values. Set `AUTOWEIGHT_QWEN_CACHE` to an operator-controlled directory to persist completed batches. Unit identities include the checkpoint, protocol, fixed batch, seed, and runtime. Re-running the same evaluator request reuses completed units. A selected AutoWeight confirmation remains single-use.

Use `python examples/weight-space/recorded.py qwen runs/qwen-published.json` to fetch the authors' benchmark summaries for the unedited model and the 10-, 25-, and 50-forward-pass runs. The earlier companion replay reported 8.25% to 55.75% on a 100-problem MATH-500 subset with four samples per problem, under a separate protocol that used approximately 82 GPU-hours. A later derivative search found no qualifying edit. The short local integration check observed no gate improvement. [Original benchmarks](https://rikiyat.github.io/blog/weight-space-programming/) · [Companion research repository](https://github.com/midhunxavier/weight-space-programming)
