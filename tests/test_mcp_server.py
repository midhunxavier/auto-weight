"""Exercise the public MCP boundary over stdio with a small HF checkpoint."""
import asyncio
import json
from pathlib import Path
import shutil
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from test_hf_repair import make_checkpoint_pair
from auto_weight import runtime
from auto_weight.cli import main


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
                    "create_run", "list_targets", "try_edit", "confirm_edit", "get_run",
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
                run = await call("create_run", **issue)
                run_id = run["run_id"]
                workspace = tmp_path / "runs" / run_id
                donors = list((workspace / "donors").iterdir())
                assert len(donors) == 1
                shutil.rmtree(inputs / "donor")
                targets = await call("list_targets", run_id=run_id, donor_id="donor1")
                assert [tensor["name"] for tensor in targets["tensors"]] == ["lm_head.weight"]
                bad = await call("try_edit", run_id=run_id, op="blend", amount=0.2, donor_id="donor1",
                                 target="changed", hypothesis="Blend a small fraction of the earlier readout.")
                assert bad["status"] == "discarded"
                good = await call("try_edit", run_id=run_id, op="restore", target="changed",
                                  donor_id="donor1", hypothesis="Restore the earlier readout.")
                assert good["status"] == "eligible"
                for trial in (bad, good):
                    attempt_dir = workspace / "attempts" / trial["attempt"]
                    assert {path.name for path in attempt_dir.iterdir()} == {"proposal.json", "result"}
                    proposal = json.loads((attempt_dir / "proposal.json").read_text())
                    assert (attempt_dir / proposal["operations"][0]["donor"]["path"]).resolve() == donors[0].resolve()
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
