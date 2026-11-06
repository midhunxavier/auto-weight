"""Exercise the public MCP boundary over stdio with a small HF checkpoint."""
import asyncio
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from test_hf_repair import make_checkpoint_pair
from auto_weight import runtime
from auto_weight.cli import main
from auto_weight.contracts import file_hash, write_json


def test_setup_runtime_updates_config_only_after_server_check(tmp_path, monkeypatch, capsys):
    inputs = tmp_path / "model inputs"
    inputs.mkdir()
    runs = tmp_path / "runs"
    config = tmp_path / "editor.conf"
    config.write_text("previous configuration\n")
    monkeypatch.setenv("AUTOWEIGHT_CONFIG", str(config))

    async def disconnected(*paths):
        raise TimeoutError("no response")

    args = ["setup-runtime", "--input-root", str(inputs), "--runs-dir", str(runs)]
    monkeypatch.setattr(runtime, "server_tools", disconnected)
    assert main(args) == 1
    assert config.read_text() == "previous configuration\n"
    assert "no response" in capsys.readouterr().err

    async def connected(python, input_root, runs_dir):
        assert (python, input_root, runs_dir) == (Path(sys.executable).absolute(), inputs, runs)
        return ["get_run"]

    monkeypatch.setattr(runtime, "server_tools", connected)
    assert main(args) == 0
    assert config.read_text().splitlines() == [sys.executable, str(inputs), str(runs)]
    assert json.loads(capsys.readouterr().out)["tools"] == ["get_run"]


def test_stdio_repair_respects_run_boundaries(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    make_checkpoint_pair(inputs, "gpt2")
    for stage, offset in (("development", 0), ("confirmation", 32)):
        marker = " HOLDOUT_ONLY" if stage == "confirmation" else ""
        with (inputs / f"{stage}.jsonl").open("w") as stream:
            for index in range(offset, offset + 32):
                for group, check in (("fix", {"type": "json", "expected": {"ok": True}}),
                                     ("keep", {"type": "exact", "expected": "kept"})):
                    stream.write(json.dumps({"group": group, "prompt": f"case{index}{marker} {group}",
                                             "check": check}) + "\n")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("not authorized")
    (inputs / "escape.jsonl").symlink_to(outside)
    (inputs / "invalid.jsonl").write_text('{"HOLDOUT_ONLY":1,"HOLDOUT_ONLY":2}\n')
    shutil.copytree(inputs / "donor", inputs / "incompatible")
    config_path = inputs / "incompatible/config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "n_head": 1}))

    async def exercise():
        source = Path(__file__).resolve().parents[1] / "src"
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "auto_weight.mcp_server",
                                         "--input-root", str(inputs), "--runs-dir", str(tmp_path / "runs")],
            env={"PYTHONPATH": str(source)},
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tools = await client.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "create_run", "list_targets", "try_edit", "confirm_edit", "get_run", "prepare_evals", "read_tensor",
                }

                async def call(tool, **values):
                    response = await client.call_tool(tool, values)
                    assert not response.isError, response.content
                    data = response.structuredContent
                    if data is None:
                        data = json.loads(response.content[0].text)
                    assert "HOLDOUT_ONLY" not in json.dumps(data)
                    return data

                issue = {"base": "base", "issue": "Recover JSON and preserve the second group.",
                         "development": "development.jsonl", "confirmation": "confirmation.jsonl",
                         "donors": ["donor"], "attempts": 2, "max_input_tokens": 8, "max_new_tokens": 1}
                denied = await client.call_tool("create_run", {**issue, "development": "escape.jsonl"})
                assert denied.isError
                assert "outside the configured input root" in str(denied.content)
                invalid = await client.call_tool("create_run", {**issue, "confirmation": "invalid.jsonl"})
                assert invalid.isError
                assert "confirmation dataset: line 1" in str(invalid.content)
                assert "HOLDOUT_ONLY" not in str(invalid.content)
                incompatible = await client.call_tool("create_run", {**issue, "donors": ["incompatible"]})
                assert incompatible.isError
                assert "donor architecture config differs from base" in str(incompatible.content)
                assert not list((tmp_path / "runs").iterdir())
                prepared = await call("prepare_evals", base="base", source=issue["development"],
                                      confirmation=issue["confirmation"], max_input_tokens=8, max_new_tokens=1)
                assert prepared["counts"]["confirmation"] == {"fix": 32, "keep": 32}
                assert not list((tmp_path / "runs").iterdir())
                issue.update(development=prepared["development"], confirmation=prepared["confirmation"])
                run = await call("create_run", **issue)
                assert run["baseline"]["groups"]["fix"]["baseline"]["json_exact"] == 0
                run_id = run["run_id"]
                workspace = tmp_path / "runs" / run_id
                donors = list((workspace / "donors").iterdir())
                assert len(donors) == 1
                from safetensors import safe_open

                name = "lm_head.weight"
                vocab = json.loads((inputs / "base/tokenizer.json").read_text())["model"]["vocab"]
                json_token = vocab['{"ok":true}']
                for shard in (inputs / "donor").glob("*.safetensors"):
                    with safe_open(shard, framework="np") as weights:
                        if name in weights.keys():
                            values = weights.get_tensor(name)[json_token:json_token + 1, :1]
                            np.savez(inputs / "replacement.npz", **{name: values})
                            break
                shutil.rmtree(inputs / "donor")
                targets = await call("list_targets", run_id=run_id, donor_id="donor1")
                assert [tensor["name"] for tensor in targets["tensors"]] == ["lm_head.weight"]
                bad = await call("try_edit", run_id=run_id, op="blend", amount=0.2, donor_id="donor1",
                                 target="changed", hypothesis="Blend a small fraction of the earlier readout.")
                assert bad["status"] == "discarded"
                inspected = await call("read_tensor", run_id=run_id, tensor=name, bounds=[[0, 1], [0, 2]])
                assert inspected["shape"] == [1, 2]
                write_json(inputs / "proposal.json", {
                    "version": "wet.proposal.v2", "base_sha256": inspected["sha256"], "author": "test",
                    "hypothesis": "Replace the output projection with the earlier values.",
                    "operations": [{"op": "replace", "tensors": [name],
                                    "slices": {name: [[json_token, json_token + 1], [0, 1]]},
                                    "payload": {"path": "replacement.npz",
                                                "sha256": file_hash(inputs / "replacement.npz")}}],
                })
                good = await call("try_edit", run_id=run_id, proposal_path="proposal.json")
                (inputs / "replacement.npz").unlink()
                assert good["status"] == "eligible"
                assert good["changed_parameters"] == 1
                attempt_dir = workspace / "attempts" / bad["attempt"]
                proposal = json.loads((attempt_dir / "proposal.json").read_text())
                assert (attempt_dir / proposal["operations"][0]["donor"]["path"]).resolve() == donors[0].resolve()
                assert (workspace / "attempts" / good["attempt"] / "replacement-0.npz").is_file()
                exhausted = await client.call_tool("try_edit", {"run_id": run_id, "op": "restore",
                    "target": "changed", "donor_id": "donor1", "hypothesis": "Attempt after the limit."})
                assert exhausted.isError
                assert "budget exhausted" in str(exhausted.content)
                final = await call("confirm_edit", run_id=run_id, attempt_id=good["attempt"])
                assert final["status"] == "accepted"
                assert "examples" not in final
                assert Path(final["checkpoint"]).is_dir()
                replay = await client.call_tool("confirm_edit", {"run_id": run_id, "attempt_id": good["attempt"]})
                assert replay.isError
                assert "already consumed" in str(replay.content)
                verified = await call("get_run", run_id=run_id)
                assert verified["checkpoint"] == final["checkpoint"]

    asyncio.run(exercise())


def test_external_issue_requires_server_authorization(tmp_path):
    from test_external_e2e import fixture

    fixture(tmp_path, "repair")

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "auto_weight.mcp_server", "--input-root", str(tmp_path),
                                         "--runs-dir", str(tmp_path / "runs"),
                                         "--allow-evaluator", str(tmp_path / "evaluator.py")],
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()

                async def call(name, **arguments):
                    result = await client.call_tool(name, arguments)
                    assert not result.isError, result.content
                    return result.structuredContent or json.loads(result.content[0].text)

                issue = json.loads((tmp_path / "issue.json").read_text())
                issue["evaluator"]["script"]["path"] = str(tmp_path / "other.py")
                shutil.copyfile(tmp_path / "evaluator.py", tmp_path / "other.py")
                write_json(tmp_path / "unapproved.json", issue)
                rejected = await client.call_tool("create_run", {"issue_path": "unapproved.json"})
                assert rejected.isError and "must be authorized" in str(rejected.content)
                run = await call("create_run", issue_path="issue.json")
                run_id = run["run_id"]
                assert run["baseline"]["groups"]["fix"]["baseline"]["accuracy"] == 0
                tensors = await call("list_targets", run_id=run_id)
                assert tensors["tensors"][0]["name"] == "kernel"
                trial = await call("try_edit", run_id=run_id, proposal_path="proposal.json")
                assert trial["status"] == "eligible"
                final = await call("confirm_edit", run_id=run_id, attempt_id=trial["attempt"])
                assert final["status"] == "accepted"
                verified = await call("get_run", run_id=run_id)
                assert verified["checkpoint"] == final["checkpoint"]

    asyncio.run(exercise())
