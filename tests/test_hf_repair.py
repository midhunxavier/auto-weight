import json
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("jsonschema")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from weight_edit_toolkit import runner
from weight_edit_toolkit.checkpoint import HF_ADAPTER, checkpoint_hash
from weight_edit_toolkit.cli import main
from weight_edit_toolkit.contracts import EditError, file_hash
from weight_edit_toolkit.text_eval import compile_check


def write_json(path, record):
    path.write_text(json.dumps(record))


def make_checkpoint_pair(root, architecture):
    tokens = ["<pad>", "<eos>", "<unk>", "fix", "keep", "invalid", "kept", '{"ok":true}']
    tokens += [f"case{index}" for index in range(64)]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>")
    if architecture == "llama":
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokens), hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, max_position_embeddings=64, eos_token_id=vocab["<eos>"],
            pad_token_id=vocab["<pad>"], tie_word_embeddings=False,
        ))
        norm = model.model.norm
    else:
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokens), n_embd=8, n_layer=1, n_head=2, n_positions=64,
            eos_token_id=vocab["<eos>"], pad_token_id=vocab["<pad>"], tie_word_embeddings=False,
        ))
        norm = model.transformer.ln_f
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        norm.weight.fill_(1)
        embeddings = model.get_input_embeddings().weight
        embeddings[vocab["fix"], 0] = 1
        embeddings[vocab["keep"], 1] = 1
        model.lm_head.weight[vocab["invalid"], 0] = 1
        model.lm_head.weight[vocab["kept"], 1] = 1
        model.save_pretrained(root / "base", max_shard_size=4096)
        tokenizer.save_pretrained(root / "base")
        model.lm_head.weight[vocab['{"ok":true}'], 0] = 2
        model.save_pretrained(root / "donor")
        tokenizer.save_pretrained(root / "donor")


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_hf_quickstart_and_text_grading_export(tmp_path, capsys, architecture):
    make_checkpoint_pair(tmp_path, architecture)
    assert (tmp_path / "base/model.safetensors.index.json").is_file()
    base_sha = checkpoint_hash(tmp_path / "base", HF_ADAPTER)
    donor_sha = checkpoint_hash(tmp_path / "donor", HF_ADAPTER)
    for stage, offset in (("development", 0), ("confirmation", 32)):
        path = tmp_path / f"{stage}.jsonl"
        with path.open("w") as stream:
            for index in range(offset, offset + 32):
                for group, check in [
                    ("fix", {"type": "json", "expected": {"ok": True}}),
                    ("keep", {"type": "exact", "expected": "kept"}),
                ]:
                    stream.write(json.dumps({"group": group, "prompt": f"case{index} {group}", "check": check}) + "\n")
    workspace = tmp_path / "repair"
    assert main([
        "setup", "--workspace", str(workspace), "--base", str(tmp_path / "base"),
        "--development", str(tmp_path / "development.jsonl"),
        "--confirmation", str(tmp_path / "confirmation.jsonl"),
        "--issue", "Fixture text format regression",
        "--improve", "fix", "json_exact", "0.5", "0.9",
        "--preserve", "keep", "text_exact", "0", "0.9",
        "--max-input-tokens", "8", "--max-new-tokens", "1", "--threads", "1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["base_sha256"] == base_sha
    assert main(["tensors", "--checkpoint", str(workspace / "base"), "--prefix", "lm_head"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["tensors"] == [{"name": "lm_head.weight", "shape": [72, 8], "dtype": "F32"}]

    write_json(tmp_path / "proposal.json", {
        "version": "wet.proposal.v1", "base_sha256": base_sha, "author": "fixture",
        "hypothesis": "Restore the earlier readout.",
        "operations": [{"op": "restore", "tensors": ["lm_head.weight"],
                        "donor": {"path": "donor", "sha256": donor_sha}}],
    })
    dev = runner.attempt(workspace, tmp_path / "proposal.json")
    assert dev["status"] == "eligible", dev["error"]
    assert dev["evaluation"]["groups"]["fix"]["baseline"]["json_valid"] == 0
    assert dev["evaluation"]["groups"]["fix"]["candidate"]["json_exact"] == 1
    assert not (workspace / "attempts/001/result/model").exists()
    shutil.rmtree(tmp_path / "donor")
    final = runner.finalize(workspace, "001")
    assert final["status"] == "accepted", final["error"]
    assert final["evaluation"]["groups"]["keep"]["candidate"]["text_exact"] == 1
    assert "development_samples" not in final["evaluation"]
    exported = workspace / "confirmation/result/model"
    assert runner.verify(workspace)["checkpoint"] == str(exported)
    assert file_hash(exported / "tokenizer.json") == file_hash(tmp_path / "base/tokenizer.json")
    assert checkpoint_hash(tmp_path / "base", HF_ADAPTER) == base_sha
    (exported / "tokenizer.json").write_text("{}")
    with pytest.raises(EditError, match="accepted checkpoint hash mismatch"):
        runner.verify(workspace)


def test_json_grader_scores_full_completion_structure_and_values():
    metrics, score = compile_check({
        "type": "json", "expected": {"count": 3},
        "schema": {"type": "object", "required": ["count"], "additionalProperties": False,
                   "properties": {"count": {"type": "integer"}}},
    })
    assert metrics == ("json_valid", "schema_match", "json_exact")
    assert score(' \n {"count": 3} ') == (True, True, True)
    assert score('{"count":4}') == (True, True, False)
    assert score('{"count":true}') == (True, False, False)
    assert score('{"count":3,"count":3}') == (False, False, False)
    assert score('{"count":NaN}') == (False, False, False)
    assert score('{"count":3} extra') == (False, False, False)
    assert score('```json\n{"count":3}\n```') == (False, False, False)


def test_shard_index_cannot_reference_a_file_outside_checkpoint(tmp_path):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model-00001-of-00001.safetensors"):
        (tmp_path / name).touch()
    write_json(tmp_path / "model.safetensors.index.json", {"weight_map": {"weight": "../external.safetensors"}})
    with pytest.raises(EditError, match="local shards"):
        checkpoint_hash(tmp_path, HF_ADAPTER)


def test_repair_feeds_back_discard_and_keeps_confirmation_out_of_proposals(tmp_path, capsys, monkeypatch):
    from weight_edit_toolkit import automation

    make_checkpoint_pair(tmp_path, "gpt2")
    for stage, offset in (("development", 0), ("confirmation", 32)):
        with (tmp_path / f"{stage}.jsonl").open("w") as stream:
            for index in range(offset, offset + 32):
                marker = " HOLDOUT_ONLY" if stage == "confirmation" else ""
                for group, check in (("fix", {"type": "json", "expected": {"ok": True}}),
                                     ("keep", {"type": "exact", "expected": "kept"})):
                    stream.write(json.dumps({"group": group, "prompt": f"case{index}{marker} {group}",
                                             "check": check}) + "\n")

    requests = []
    choices = iter([
        {"op": "scale", "amount": 0.5, "donor": "none"},
        {"op": "restore", "amount": 1, "donor": "donor1"},
    ])

    def local_model(path, *, body=None, timeout):
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:latest", "size": 1, "digest": "fixture"}]}
        assert path == "/api/chat"
        context = json.loads(body["messages"][1]["content"])
        assert "HOLDOUT_ONLY" not in json.dumps(body)
        requests.append(context)
        response = {"hypothesis": "Adjust the output projection.", "target": "lm_head.weight", **next(choices)}
        return {"message": {"content": json.dumps(response)}}

    monkeypatch.setattr(automation, "ollama_request", local_model)
    workspace = tmp_path / "automatic"
    assert main([
        "repair", "--workspace", str(workspace), "--base", str(tmp_path / "base"),
        "--development", str(tmp_path / "development.jsonl"), "--confirmation", str(tmp_path / "confirmation.jsonl"),
        "--issue", "Repair JSON and keep the other answer.", "--donor", str(tmp_path / "donor"),
        "--max-input-tokens", "8", "--max-new-tokens", "1", "--threads", "1", "--attempts", "2",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["attempts"] == {"001": "discarded", "002": "eligible"}
    assert result["confirmation"] == "accepted"
    assert Path(result["checkpoint"]).is_dir()
    assert len(requests) == 2
    assert requests[0]["previous_attempts"] == []
    assert requests[1]["previous_attempts"][0]["status"] == "discarded"
    assert requests[1]["previous_attempts"][0]["groups"]["fix"]["candidate"]["json_exact"] == 0
