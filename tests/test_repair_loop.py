import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auto_weight.contracts import file_hash
from auto_weight.model import TinyTransformer


def prepare(root, *, fail_confirmation=False, overlap=False, timeout=60):
    config = {"width": 4, "heads": 2, "layers": 1, "context": 8, "vocab": 8}
    model = TinyTransformer(**config)
    state = {key: np.zeros_like(tensor.detach().numpy()) for key, tensor in model.state_dict().items()}
    state["final.weight"].fill(1)
    state["tok.weight"][3] = [-1, -1, 1, 1]
    state["tok.weight"][4] = [1, 1, -1, -1]
    state["output.weight"][2] = [-1, -1, 1, 1]
    state["output.bias"][0] = 1
    np.savez(root / "base.npz", **state)
    state["output.bias"][1] = 2
    np.savez(root / "donor.npz", **state)

    def reference(name):
        return {"path": name, "sha256": file_hash(root / name)}

    for stage, split_token in [("development", 5), ("confirmation", 5 if overlap else 6)]:
        with (root / f"{stage}.jsonl").open("w") as stream:
            for row in range(32):
                prefix = [split_token] + [int(bit) for bit in f"{row:05b}"]
                for group, query, target in [("fix", 4, 0 if stage == "confirmation" and fail_confirmation else 1),
                                             ("keep", 3, 2)]:
                    stream.write(json.dumps({"group": group, "tokens": [*prefix, query, target],
                                             "positions": [6]}) + "\n")

    issue = {
        "version": "wet.issue.v1", "issue": "Fixture next-token regression", "base": reference("base.npz"),
        "model": {"adapter": "tiny-transformer-v1", "config": config},
        "datasets": {stage: reference(f"{stage}.jsonl") for stage in ("development", "confirmation")},
        "gates": [{"group": "fix", "metric": "accuracy", "kind": "improve", "min_delta": 0.1},
                  {"group": "keep", "metric": "accuracy", "kind": "preserve", "min_delta": 0}],
        "budget": {"attempts": 2, "timeout_seconds": timeout, "threads": 1, "batch_size": 8},
    }
    proposal = {
        "version": "wet.proposal.v1", "base_sha256": issue["base"]["sha256"], "author": "Fixture",
        "hypothesis": "Restore the earlier classifier bias.",
        "operations": [{"op": "restore", "tensors": ["output.bias"], "donor": reference("donor.npz")}],
    }
    (root / "issue.json").write_text(json.dumps(issue))
    (root / "proposal.json").write_text(json.dumps(proposal))
    return issue


def cli(root, command, *args, code=0):
    src = Path(__file__).resolve().parents[1] / "src"
    process = subprocess.run(
        [sys.executable, "-m", "auto_weight", command, "--workspace", str(root / "repair"), *args],
        cwd=root, env={**os.environ, "PYTHONPATH": str(src)}, capture_output=True, text=True, timeout=90,
    )
    assert process.returncode == code, process.stdout + process.stderr
    return json.loads(process.stdout or process.stderr)


@pytest.mark.parametrize("fail_confirmation", [False, True])
def test_cli_replays_and_publishes_only_passing_checkpoint(tmp_path, fail_confirmation):
    issue = prepare(tmp_path, fail_confirmation=fail_confirmation)
    assert cli(tmp_path, "init", "--issue", "issue.json")["status"] == "ready"
    dev = cli(tmp_path, "try", "--proposal", "proposal.json")
    assert dev["status"] == "eligible"
    workspace = tmp_path / "repair"
    assert list((workspace / "attempts/001/result").iterdir()) == [workspace / "attempts/001/result/report.json"]
    # A second attempt and confirmation both use the frozen donor.
    (tmp_path / "donor.npz").unlink()
    again = cli(tmp_path, "try", "--proposal", "proposal.json")
    assert again["status"] == "eligible"
    assert again["evaluation"] == dev["evaluation"]
    assert len(list((workspace / "donors").iterdir())) == 1
    confirmation = cli(tmp_path, "finalize", "--attempt", "002")
    verify = cli(tmp_path, "verify")
    accepted = workspace / "confirmation/result/model.npz"
    assert confirmation["evaluation"]["candidate_sha256"] == dev["evaluation"]["candidate_sha256"]
    assert file_hash(tmp_path / "base.npz") == issue["base"]["sha256"]
    assert accepted.exists() == (not fail_confirmation)
    assert cli(tmp_path, "finalize", "--attempt", "001", code=1)["status"] == "error"
    assert cli(tmp_path, "try", "--proposal", "proposal.json", code=1)["status"] == "error"
    if fail_confirmation:
        assert dev["evaluation"]["groups"]["fix"]["baseline"]["accuracy"] == 0
        assert confirmation["evaluation"]["groups"]["fix"]["baseline"]["accuracy"] == 1
        assert verify["confirmation"] == "discarded"
        assert verify["checkpoint"] is None
    else:
        assert verify["confirmation"] == "accepted"
        assert file_hash(accepted) == confirmation["evaluation"]["candidate_sha256"]
        accepted.write_bytes(b"damaged model")
        assert "hash mismatch" in cli(tmp_path, "verify", code=1)["error"]


def test_development_baseline_skips_scoring_on_hit_and_rebuilds_invalid_cache(tmp_path):
    from auto_weight.runner import BASELINE_NAME
    from auto_weight.worker import development_baseline

    staging = tmp_path / "staging"
    staging.mkdir()
    frozen = {"id": "issue"}
    expected = np.array([[1.0, 0.5], [0.0, 2.0]])
    expected_samples = {"fix": ["before"]}

    def score():
        return {"fix": expected}, expected_samples

    def unexpected_score():
        pytest.fail("the baseline scorer ran despite a cache hit")

    development_baseline(tmp_path, frozen, staging, score)
    cache = tmp_path / BASELINE_NAME
    (staging / BASELINE_NAME).replace(cache)
    scores, samples = development_baseline(tmp_path, frozen, staging, unexpected_score)
    np.testing.assert_array_equal(scores["fix"], expected)
    assert samples == expected_samples
    assert not (staging / BASELINE_NAME).exists()

    cache.write_text("{}")
    development_baseline(tmp_path, frozen, staging, score)
    assert (staging / BASELINE_NAME).is_file()


def test_overlapping_confirmation_is_consumed_without_export(tmp_path):
    prepare(tmp_path, overlap=True, fail_confirmation=True)
    cli(tmp_path, "init", "--issue", "issue.json")
    assert cli(tmp_path, "try", "--proposal", "proposal.json")["status"] == "eligible"
    result = cli(tmp_path, "finalize", "--attempt", "001", code=1)
    assert "prompts overlap" in result["error"]
    assert cli(tmp_path, "verify")["checkpoint"] is None
    assert "consumed" in cli(tmp_path, "finalize", "--attempt", "001", code=1)["error"]


def test_modified_base_fails_closed(tmp_path):
    prepare(tmp_path)
    cli(tmp_path, "init", "--issue", "issue.json")
    frozen_base = tmp_path / "repair/base.npz"
    frozen_base.write_bytes(b"changed")
    result = cli(tmp_path, "try", "--proposal", "proposal.json", code=1)
    assert result["status"] == "error"
    assert "frozen input hash mismatch" in result["error"]
    assert not list((tmp_path / "repair").glob(".trial-*"))
    assert not (tmp_path / "repair/confirmation").exists()


def test_timeout_keeps_error_report_and_removes_partial_candidate(tmp_path, monkeypatch):
    from auto_weight import runner

    prepare(tmp_path, timeout=1)
    workspace = tmp_path / "repair"
    runner.initialize(tmp_path / "issue.json", workspace)

    def time_out(args, **kwargs):
        assert kwargs["timeout"] == 1
        (Path(args[5]) / "model.npz").write_bytes(b"partial")
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(runner.subprocess, "run", time_out)
    result = runner.attempt(workspace, tmp_path / "proposal.json")
    assert result["status"] == "error"
    assert "wall time" in result["error"]
    assert not list(workspace.glob(".trial-*"))
    assert runner.verify(workspace)["attempts"] == {"001": "error"}
