"""Run Codex and Claude Code against the same AutoWeight case and verify the results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from time import monotonic

import torch

from auto_weight import hf, text_eval
from auto_weight.automation import changed_tensors, materialize
from auto_weight.checkpoint import HF_ADAPTER, checkpoint_hash, tensor_metadata
from auto_weight.contracts import write_json
from auto_weight.runner import attempt, finalize, initialize_issue, load_issue, read_sealed, verify


ROOT = Path(__file__).resolve().parent
CASES = ("format-json", "schema-migration", "chat-routing", "fallback-bias", "policy-conflict")
HOSTS = ("codex", "claude")
WORKFLOW = """
Use only the auto-weight MCP server. Use the given file paths as tool arguments.
Use create_run and list_targets to inspect available choices, then try edits on development.
After the first eligible edit, call confirm_edit once and verify with get_run.
When the attempt budget is exhausted without an eligible candidate, verify the run and report it.
Report decisions and scores from tool results. Do not read or copy the held-out dataset.
"""


def launch(case, host, inputs, directory, server_executable):
    directory.mkdir()
    server = {"command": server_executable, "args": ["--input-root", str(inputs),
                                                    "--runs-dir", str(directory / "runs")]}
    prompt = (ROOT / case / "prompt.md").read_text() + WORKFLOW
    if host == "codex":
        command = [
            "codex", "-a", "never", "exec", "--skip-git-repo-check", "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--sandbox", "read-only",
            "--disable", "shell_tool", "--disable", "plugins", "--json",
            "-c", f'mcp_servers.auto-weight.command={json.dumps(server_executable)}',
            "-c", f'mcp_servers.auto-weight.args={json.dumps(server["args"])}',
            "-c", "mcp_servers.auto-weight.required=true",
            "-c", "mcp_servers.auto-weight.tool_timeout_sec=360", "-",
        ]
    else:
        command = [
            "claude", "--print", "--restricted", "--strict-mcp-config", "--mcp-config",
            json.dumps({"mcpServers": {"auto-weight": server}}),
            "--tools", "", "--allowedTools", "mcp__auto-weight", "--permission-mode", "dontAsk",
            "--no-session-persistence", "--output-format", "stream-json", "--verbose", "--model", "sonnet",
        ]
    start = monotonic()
    print(f"Starting {host} on {case}", flush=True)
    with (directory / "events.jsonl").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
        try:
            result = subprocess.run(command, cwd=directory, input=prompt, text=True, stdout=stdout,
                                    stderr=stderr, timeout=650, check=False)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    write_json(directory / "process.json", {"exit": exit_code, "seconds": round(monotonic() - start, 2)})


def score_checkpoint(checkpoint, data, issue):
    model = hf.load_model(checkpoint)
    config = issue["model"]["config"]
    batch = issue["budget"]["batch_size"]
    tokenizer = hf.load_tokenizer(checkpoint, config)
    groups, metrics, _ = text_eval.read_examples(
        data, tokenizer, config, issue["gates"], hf.evaluation_context(model.config, config, batch),
    )
    generation = text_eval.generation_policy(model, tokenizer, config["max_new_tokens"])
    scores, _ = text_eval.score(model, tokenizer, groups, batch, generation)
    return {
        group: {"examples": len(rows),
                "correct": dict(zip(metrics[group], rows.sum(0).astype(int).tolist(), strict=True))}
        for group, rows in scores.items()
    }


def trial_summary(report):
    record = {"attempt": report["attempt"], "status": report["status"], "report_id": report["id"],
              "seconds": round(report["seconds"], 2)}
    evaluation = report["evaluation"]
    if evaluation is None:
        record["error"] = report["error"]
        return record
    record["gates"] = evaluation["gates"]
    record["groups"] = {}
    for group, metrics in evaluation["groups"].items():
        count = metrics["examples"]
        record["groups"][group] = {
            "examples": count,
            "base": {key: round(rate * count) for key, rate in metrics["baseline"].items()},
            "candidate": {key: round(rate * count) for key, rate in metrics["candidate"].items()},
        }
    record["candidate_sha256"] = evaluation["candidate_sha256"]
    return record


def collect_host(host, directory):
    workspaces = [file.parent for file in (directory / "runs").glob("*/issue.json")]
    if len(workspaces) != 1:
        raise ValueError(f"Expected one run for {host}. See {directory / 'stderr.log'}")
    workspace, = workspaces
    verified = verify(workspace)
    frozen = load_issue(workspace, require_implementation=False)
    issue = frozen["issue"]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
    tool_calls = []
    model = "CLI default"
    for event in events:
        if host == "codex":
            item = event.get("item", {})
            if event["type"] == "item.completed" and item.get("server") == "auto-weight":
                tool_calls.append(item["tool"])
        elif event.get("subtype") == "init":
            model = event["model"]
        elif event["type"] == "assistant":
            for item in event["message"]["content"]:
                if item.get("name", "").startswith("mcp__auto-weight__"):
                    tool_calls.append(item["name"].removeprefix("mcp__auto-weight__"))
    attempts = []
    for attempt_id in verified["attempts"]:
        path = workspace / "attempts" / attempt_id
        summary = trial_summary(read_sealed(path / "result/report.json"))
        summary["operations"] = read_sealed(path / "proposal.json")["operations"]
        attempts.append(summary)
    confirmation = None
    if verified["confirmation"] is not None:
        confirmation = trial_summary(read_sealed(workspace / "confirmation/result/report.json"))
    if verified["checkpoint"]:
        holdout = workspace / issue["datasets"]["confirmation"]["path"]
        reloaded = score_checkpoint(Path(verified["checkpoint"]), holdout, issue)
        for group, scores in reloaded.items():
            if scores["correct"] != confirmation["groups"][group]["candidate"]:
                raise ValueError(f"Reloaded scores differ from the confirmation report for {host}/{group}.")
        confirmation["reload_correct"] = reloaded
    return workspace, frozen, {
        "cli_version": subprocess.check_output([host, "--version"], text=True).strip(),
        "model": model, "process": json.loads((directory / "process.json").read_text()),
        "run_id": workspace.name, "issue_id": frozen["id"], "tool_calls": tool_calls,
        "attempts": attempts, "confirmation": confirmation, "exported": bool(verified["checkpoint"]),
    }


def fixed_control(inputs, workspace, frozen, output):
    issue = frozen["issue"]
    checkpoint = workspace / issue["base"]["path"]
    prior = inputs / "prior"
    names = [entry["name"] for entry in tensor_metadata(checkpoint, "", 10000)["tensors"]]
    donors, changes = {}, {}
    if prior.is_dir():
        donors["prior"] = {"path": str(prior), "sha256": checkpoint_hash(prior, HF_ADAPTER)}
        changes["prior"] = changed_tensors(checkpoint, prior, names)
        choice = {"op": "blend", "target": "changed", "donor": "prior", "amount": 0.5}
    else:
        choice = {"op": "scale", "target": "lm_head.weight", "donor": "none", "amount": 0.5}
    choice["hypothesis"] = "Fixed one-attempt baseline for this case family."
    proposal = materialize(choice, names=names, donors=donors, base_sha=issue["base"]["sha256"],
                           author="Fixed control", changes=changes)
    control_workspace = output / "fixed-control"
    initialize_issue(deepcopy(issue), workspace, control_workspace)
    proposal_path = control_workspace / "fixed-proposal.json"
    write_json(proposal_path, proposal)
    trial = attempt(control_workspace, proposal_path)
    confirmation = finalize(control_workspace, trial["attempt"]) if trial["status"] == "eligible" else None
    verified = verify(control_workspace)
    return {
        "choice": choice, "development": trial_summary(trial),
        "confirmation": trial_summary(confirmation) if confirmation else None,
        "exported": bool(verified["checkpoint"]),
    }


def summarize(case, inputs, output):
    hosts = {}
    shared_inputs = None
    for host in HOSTS:
        workspace, frozen, hosts[host] = collect_host(host, output / host)
        issue = frozen["issue"]
        scored_inputs = {key: issue[key] for key in ("base", "datasets", "model", "gates", "budget")}
        scored_inputs["donors"] = read_sealed(workspace / "mcp-run.json")["donors"]
        if shared_inputs is not None and scored_inputs != shared_inputs:
            raise ValueError("The hosts used different evaluation inputs or settings.")
        shared_inputs = scored_inputs
    holdout = workspace / issue["datasets"]["confirmation"]["path"]
    base = workspace / issue["base"]["path"]
    prior = inputs / "prior"
    fixed = fixed_control(inputs, workspace, frozen, output)
    controls = {"base": score_checkpoint(base, holdout, issue), "fixed_edit": fixed}
    if prior.is_dir():
        controls["full_prior"] = score_checkpoint(prior, holdout, issue)
    write_json(output / "results.json", {
        "case": case, "runtime": frozen["runtime"], "config": issue["model"]["config"], "gates": issue["gates"],
        "inputs": {
            "base_sha256": issue["base"]["sha256"],
            "prior_sha256": checkpoint_hash(prior, HF_ADAPTER) if prior.is_dir() else None,
            "datasets_sha256": {stage: data["sha256"] for stage, data in issue["datasets"].items()},
        },
        "hosts": hosts, "controls": controls,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for transcripts and results")
    args = parser.parse_args()
    commands = ("auto-weight-mcp", *HOSTS)
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        parser.error(f"install these commands and add them to PATH: {', '.join(missing)}")
    inputs = args.inputs.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sessions = [pool.submit(launch, args.case, host, inputs, output / host, shutil.which(commands[0]))
                    for host in HOSTS]
        for session in sessions:
            session.result()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    summarize(args.case, inputs, output)
    print(output / "results.json")


if __name__ == "__main__":
    main()
