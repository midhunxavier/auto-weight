import asyncio
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

peft = pytest.importorskip("peft")
import torch
from transformers import AutoTokenizer

from auto_weight import automation, hf, runner
from auto_weight.checkpoint import HF_ADAPTER, LORA_ADAPTER, checkpoint_hash
from auto_weight.cli import main
from auto_weight.contracts import EditError, file_hash, write_json
from test_hf_repair import make_checkpoint_pair


@pytest.mark.parametrize("interface", ["cli", "mcp"])
def test_native_lora_repair_exports_only_a_loadable_adapter(tmp_path, capfd, monkeypatch, interface):
    make_checkpoint_pair(tmp_path, "gpt2")
    base = tmp_path / "donor"
    adapter = tmp_path / "lora"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    vocab = tokenizer.get_vocab()
    model = hf.load_model(base)
    # The original model produces good JSON but fails the behavior learned by LoRA.
    with torch.no_grad():
        model.lm_head.weight[vocab["kept"], 1] = 0
        model.lm_head.weight[vocab["invalid"], 1] = 1
    model.save_pretrained(base)
    model = peft.get_peft_model(model, peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=2, target_modules=["lm_head"],
    ))
    with torch.no_grad():
        a = model.base_model.model.lm_head.lora_A["default"].weight
        b = model.base_model.model.lm_head.lora_B["default"].weight
        a.zero_()
        a[0, 0] = a[1, 1] = 1
        b[vocab["invalid"], 0] = 3
        b[vocab["kept"], 1] = 4
    model.save_pretrained(adapter, save_embedding_layers=False)
    backbone_sha = checkpoint_hash(base, HF_ADAPTER)
    adapter_sha = checkpoint_hash(adapter, LORA_ADAPTER)
    target = "base_model.model.lm_head.lora_B.weight"
    for stage, offset in (("development", 0), ("confirmation", 32)):
        with (tmp_path / f"{stage}.jsonl").open("w") as stream:
            for index in range(offset, offset + 32):
                for group, check in (("fix", {"type": "json", "expected": {"ok": True}}),
                                     ("keep", {"type": "exact", "expected": "kept"})):
                    print(json.dumps({"group": group, "prompt": f"case{index} {group}", "check": check}), file=stream)

    def move_originals():
        base.rename(tmp_path / "moved-base")
        adapter.rename(tmp_path / "moved-lora")

    if interface == "cli":
        factors = iter([0, 0.35])

        def local_proposer(path, *, body=None, timeout):
            if path == "/api/tags":
                move_originals()
                return {"models": [{"name": "qwen3.5:latest", "size": 1, "digest": "fixture"}]}
            assert target in body["format"]["properties"]["target"]["enum"]
            return {"message": {"content": json.dumps({
                "op": "scale", "target": target, "amount": next(factors), "donor": "none",
                "hypothesis": "Weaken the LoRA update to restore JSON.",
            })}}

        monkeypatch.setattr(automation, "ollama_request", local_proposer)
        workspace = tmp_path / "run"
        assert main([
            "repair", "--workspace", str(workspace), "--base", str(base), "--lora", str(adapter),
            "--development", str(tmp_path / "development.jsonl"), "--confirmation", str(tmp_path / "confirmation.jsonl"),
            "--issue", "Restore raw JSON and keep queue extraction.", "--attempts", "2", "--max-new-tokens", "1",
        ]) == 0
        result = json.loads(capfd.readouterr().out)
        assert result["baseline"]["groups"]["fix"]["baseline"]["json_exact"] == 0
        assert result["attempts"] == {"001": "discarded", "002": "eligible"}
    else:
        pytest.importorskip("mcp")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def exercise():
            parameters = StdioServerParameters(command=sys.executable, args=[
                "-m", "auto_weight.mcp_server", "--input-root", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
            ], env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")})
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()

                    async def call(tool, **args):
                        response = await client.call_tool(tool, args)
                        assert not response.isError, response.content
                        if response.structuredContent is not None:
                            return response.structuredContent
                        return json.loads(response.content[0].text)

                    run = await call("create_run", base=base.name, lora=adapter.name, development="development.jsonl",
                                     confirmation="confirmation.jsonl", max_new_tokens=1, issue="Restore raw JSON.")
                    assert run["base_sha256"] == adapter_sha
                    move_originals()
                    targets = await call("list_targets", run_id=run["run_id"])
                    assert {entry["name"] for entry in targets["tensors"]} == {
                        target, "base_model.model.lm_head.lora_A.weight",
                    }
                    tensor = await call("read_tensor", run_id=run["run_id"], tensor=target, summary=True)
                    assert tensor["sha256"] == adapter_sha
                    invalid = vocab["invalid"]
                    payload = tmp_path / "patch.npz"
                    np.savez(payload, **{target: np.zeros((1, 1), dtype=np.float32)})
                    write_json(tmp_path / "proposal.json", {
                        "version": "wet.proposal.v2", "base_sha256": adapter_sha, "author": "test",
                        "hypothesis": "Remove the LoRA boost to the invalid response.",
                        "operations": [{"op": "replace", "tensors": [target],
                                        "slices": {target: [[invalid, invalid + 1], [0, 1]]},
                                        "payload": {"path": payload.name, "sha256": file_hash(payload)}}],
                    })
                    attempt = await call("try_edit", run_id=run["run_id"], proposal_path="proposal.json")
                    assert attempt["status"] == "eligible", attempt
                    result = await call("confirm_edit", run_id=run["run_id"], attempt_id=attempt["attempt"])
                    return result, tmp_path / "runs" / run["run_id"]

        result, workspace = asyncio.run(exercise())

    assert result["status"] == "accepted"
    assert result["backbone_sha256"] == backbone_sha
    exported = Path(result["checkpoint"])
    assert exported.name == "adapter"
    assert {file.name for file in exported.iterdir()} == {"adapter_model.safetensors", "adapter_config.json"}
    assert not (workspace / "attempts/001/result/adapter").exists()
    assert checkpoint_hash(tmp_path / "moved-base", HF_ADAPTER) == backbone_sha
    assert checkpoint_hash(tmp_path / "moved-lora", LORA_ADAPTER) == adapter_sha
    pinned_base = Path(result["backbone"])
    restored = peft.PeftModel.from_pretrained(hf.load_model(pinned_base), exported, local_files_only=True).eval()
    for group, expected in (("fix", '{"ok":true}'), ("keep", "kept")):
        prompt = tokenizer(f"case32 {group}", return_tensors="pt")
        output = restored.generate(**prompt, max_new_tokens=1)
        assert tokenizer.decode(output[0, -1]) == expected
    (pinned_base / "tokenizer.json").write_text("{}")
    with pytest.raises(EditError, match="frozen input hash mismatch: backbone"):
        runner.verify(workspace)
