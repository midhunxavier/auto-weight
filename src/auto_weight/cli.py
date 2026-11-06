"""Command line interface for the external proposer and local runner."""
import argparse
import json
from pathlib import Path
import sys

from .checkpoint import HF_ADAPTER, checkpoint_hash
from .tensors import metadata, read_tensor
from .contracts import EditError, integer
from .quickstart import setup
from .runner import attempt, baseline, finalize, initialize, submit, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("init", "freeze issue, checkpoint, and evaluation sets"),
        ("try", "apply a proposal temporarily and evaluate on development examples"),
        ("finalize", "confirm one eligible attempt and export only on acceptance"),
        ("baseline", "score the frozen development cases before trying edits"),
        ("submit", "score one final construction proposal"),
        ("verify", "check frozen inputs, reports, and any exported checkpoint"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--workspace", required=True, type=Path)
        if name == "init":
            command.add_argument("--issue", required=True, type=Path)
        elif name in ("try", "submit"):
            command.add_argument("--proposal", required=True, type=Path)
        elif name == "finalize":
            command.add_argument("--attempt", required=True)
    fingerprint = commands.add_parser("fingerprint", help="hash a checkpoint file or Hugging Face directory")
    fingerprint.add_argument("--checkpoint", required=True, type=Path)
    runtime = commands.add_parser("setup-runtime", help="check and configure the local MCP runtime")
    runtime.add_argument("--input-root", required=True, type=Path, help="existing folder with model and evaluation files")
    runtime.add_argument("--runs-dir", type=Path, help="output folder for repair runs")
    demo_parser = commands.add_parser("demo", help="run an offline repair with a tiny checkpoint and scripted edits")
    demo_parser.add_argument("--workspace", type=Path, help="new output directory (default: runs/demo-<id>)")
    for name, description in (("setup", "start a Hugging Face repair from files and gates"),
                              ("repair", "propose, test, and confirm edits using a local LLM")):
        starter = commands.add_parser(name, help=description)
        for flag in ("base", "development", "confirmation"):
            starter.add_argument(f"--{flag}", required=True, type=Path)
        starter.add_argument("--workspace", required=name == "setup", type=Path)
        starter.add_argument("--issue", required=True, help="description of the model behavior to repair")
        for kind in ("improve", "preserve"):
            starter.add_argument(f"--{kind}", required=name == "setup", action="append", nargs="+",
                                 metavar="VALUE", help="add a gate: GROUP METRIC DELTA [FLOOR]")
        starter.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
        for flag, default in (("max-input-tokens", 128), ("max-new-tokens", 32), ("attempts", 6),
                              ("timeout-seconds", 240), ("batch-size", 4), ("threads", 4)):
            starter.add_argument(f"--{flag}", type=int, default=default)
        if name == "repair":
            starter.add_argument("--fix-group", default="fix", help="improvement group for automatic gates")
            starter.add_argument("--donor", type=Path, action="append", default=[])
            starter.add_argument("--proposer-model", default="qwen3.5", help="installed local Ollama model")
            starter.add_argument("--proposer-timeout", type=int, default=180)
    tensors = commands.add_parser("tensors", help="list Hugging Face tensor names, shapes, and dtypes")
    tensors.add_argument("--checkpoint", required=True, type=Path)
    tensors.add_argument("--prefix", default="")
    tensors.add_argument("--limit", type=int, default=50)
    read = commands.add_parser("read-tensor", help="read one bounded tensor slice")
    read.add_argument("--checkpoint", required=True, type=Path)
    read.add_argument("--tensor", required=True)
    read.add_argument("--slice", nargs=2, action="append", type=int, metavar=("START", "STOP"))
    read.add_argument("--summary", action="store_true")
    prep = commands.add_parser("prepare", help="prepare a labeled source for HF repair")
    for name in ("base", "source", "output", "confirmation"):
        prep.add_argument(f"--{name}", type=Path, required=name != "confirmation")
    prep.add_argument("--conversion", type=Path)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    prep.add_argument("--max-input-tokens", type=int, default=128)
    prep.add_argument("--max-new-tokens", type=int, default=32)
    prep.add_argument("--confirmation-exposed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(args.issue, args.workspace)
        elif args.command == "try":
            result = attempt(args.workspace, args.proposal)
        elif args.command == "finalize":
            result = finalize(args.workspace, args.attempt)
        elif args.command == "submit":
            result = submit(args.workspace, args.proposal)
        elif args.command == "baseline":
            result = baseline(args.workspace)
        elif args.command == "verify":
            result = verify(args.workspace)
        elif args.command == "setup":
            result = setup(args)
            result["baseline"] = baseline(args.workspace)
        elif args.command == "setup-runtime":
            from .runtime import setup_runtime

            result = setup_runtime(args.input_root, args.runs_dir)
        elif args.command == "repair":
            from .automation import repair

            result = repair(args)
        elif args.command == "demo":
            from .demo import demo

            result = demo(args.workspace)
        elif args.command == "tensors":
            result = metadata(args.checkpoint, args.prefix, integer(args.limit, "limit", 1, 200))
        elif args.command == "read-tensor":
            result = read_tensor(args.checkpoint, args.tensor, args.slice, args.summary)
        elif args.command == "prepare":
            from .prepare import prepare_evals

            result = prepare_evals(args.base, args.source, args.output, confirmation=args.confirmation,
                                   seed=args.seed, prompt_format=args.prompt_format,
                                   max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                                   confirmation_independent=not args.confirmation_exposed,
                                   conversion=args.conversion)
        else:
            path = args.checkpoint
            adapter = HF_ADAPTER if path.is_dir() else "tiny-transformer-v1"
            result = {"status": "hashed", "sha256": checkpoint_hash(path, adapter)}
    except (EditError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] == "error")
