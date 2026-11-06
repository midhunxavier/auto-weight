# Weight-space examples

Start with `auto-weight demo` for the regular repair workflow or use the CPU classifier below to try your own evaluator. Research examples use published artifacts from Takehi, Teoh, Zhang, and Khattab. Preparation fetches only the selected files and checks the pinned revision, size, and SHA-256 against [the asset manifest](assets.json). Upstream networks and logs are licensed CC BY 4.0. Simulator packages and benchmark datasets retain their own licenses. [Paper and experimental protocols](https://rikiyat.github.io/blog/weight-space-programming/) · [Upstream license](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE)

| Case | Run locally |
| --- | --- |
| 1. Guess the game | Stage four policies and score a submitted diagnosis |
| 2. Identify diffusion shapes | Stage three checkpoints and score shape labels |
| 3. Identify language and copying heads | Stage middle blocks and score predictions against published behavioral measurements |
| 4. Repair the Kinetix car | Rebuild the rounded base and evaluate the published edit on paired ramp episodes |
| 5. Construct a MinAtar network | Submit the published Breakout weights against a zero template |
| 6. Construct a Go network | Submit the published zero-practice weights and play a seeded random control |
| 7. Detect sycophancy | Retrieve the recorded weights-only and attention-assisted evidence |
| 8. Repair Qwen reasoning loops | Materialize the hash-checked published patch and run sampled generation in a separate environment |

## CPU classifier

Run these commands from the checkout with AutoWeight installed.

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

Use `game` or `diffusion` in place of `copying` for the other two tasks. Write an object keyed by `network-NN`. Values are lowercase game or shape labels, or an object like `{"language":"english","heads":["L3H1"]}` for copying. The scorer checks the input hashes and records the submission hash and accuracy. Keep the local answer key with the experiment owner for a blind task. An unrestricted local shell can read that key and the released labels.

## Simulator replays

For Kinetix, check out [the pinned simulator revision](https://github.com/FLAIROx/Kinetix/tree/80ee9c292837c825cafaf9889323fdcc212efd1b) and install it in a separate environment with an editable install so its level files are available. For Go, use a separate environment with `pgx==2.6.0` and JAX. For MinAtar, install `minatar==1.0.15` in its own environment. Pass that environment's Python executable to preparation.

```sh
python examples/weight-space/prepare.py kinetix runs/car-inputs --python /path/to/kinetix/bin/python
auto-weight init --issue runs/car-inputs/issue.json --workspace runs/car
auto-weight try --proposal runs/car-inputs/proposal.json --workspace runs/car
auto-weight finalize --attempt 001 --workspace runs/car
auto-weight verify --workspace runs/car
```

The ramp check measures success and the preservation group measures avoidance of large negative terminal rewards at slightly shifted starts. This is a local replay on fixed action seeds. The published evaluation used its own episode schedule.

For `minatar` or `go`, prepare with the same script and the matching interpreter, then run `init`, `submit --proposal`, and `verify`. MinAtar defaults to eight episodes of up to 256 steps. For the Go smoke test use `--episodes 8 --steps 512` to give games time to finish. Go's random control is a small reproducible benchmark. Use `recorded.py go` for the published results against its reference opponents.

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

The earlier companion replay reported 8.25% to 55.75% on 100 MATH-500 problems with four samples per problem. It used approximately 82 GPU-hours under a long sampled-generation protocol. The subsequent derivative search reported no qualifying winner. Those are historical companion results. The local integration smoke used a one-token cap and observed no gate improvement. [Companion research repository](https://github.com/midhunxavier/weight-space-programming)
