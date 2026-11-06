"""Exercise external repair and construction through the installed command."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from auto_weight.checkpoint import RAW_HF_ADAPTER, checkpoint_hash
from auto_weight.contracts import file_hash


EVALUATOR = """
import json
from pathlib import Path
import sys
import numpy as np

request = json.loads(Path(sys.argv[1]).read_text())
artifact = Path(request["artifact"])
if artifact.is_dir():
    from safetensors.numpy import load_file
    kernel = load_file(artifact / "model.safetensors")["kernel"]
else:
    with np.load(artifact, allow_pickle=False) as weights:
        kernel = weights["kernel"]
observations = []
for line in Path(request["cases"]).read_text().splitlines():
    row = json.loads(line)
    inputs = row["inputs"]
    predicted = int(np.argmax(np.asarray(inputs["x"]) @ kernel))
    for seed in request["seeds"]:
        observations.append({"id": row["id"], "seed": seed, "metrics": {
            "accuracy": int(predicted == inputs["label"]),
        }})
if request["config"].get("duplicate"):
    observations[-1] = observations[0]
Path(sys.argv[2]).write_text(json.dumps({
    **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
    "observations": list(reversed(observations)), "work": len(observations),
}))
"""


def ref(path):
    return {"path": str(path), "sha256": file_hash(path)}


def fixture(root, mode, malformed=False, raw_hf=False):
    construct = mode == "construct"
    base = np.zeros((2, 2), dtype="float32") if construct else np.diag([-1, 1]).astype("float32")
    adapter = RAW_HF_ADAPTER if raw_hf else "named-tensors-v1"
    base_path = root / "base.npz"
    if raw_hf:
        pytest.importorskip("torch")
        save_file = pytest.importorskip("safetensors.numpy").save_file
        base_path = root / "base"
        base_path.mkdir()
        save_file({"kernel": base}, base_path / "model.safetensors", metadata={"format": "pt"})
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (base_path / name).write_text("{}")
    else:
        np.savez(base_path, kernel=base)
    base_ref = {"path": str(base_path), "sha256": checkpoint_hash(base_path, adapter)}
    target = np.eye(2, dtype="float32")
    np.savez(root / "patch.npz", kernel=target if construct else target[:1, :1])
    (root / "evaluator.py").write_text(EVALUATOR)
    splits = {"confirmation": 64}
    if not construct:
        splits["development"] = 0
    for split, offset in splits.items():
        rows = []
        for index in range(offset, offset + 32):
            for group, label in (("fix", 0), ("keep", 1)):
                value = 1 + index / 100
                rows.append({"id": f"{group}-{index}", "group": group,
                             "inputs": {"x": [0, value] if label else [value, 0], "label": label}})
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    gates = [{"group": "fix", "metric": "accuracy", "kind": "target", "min_candidate": 1}] if construct else [
        {"group": "fix", "metric": "accuracy", "kind": "improve", "min_delta": 0.5},
        {"group": "keep", "metric": "accuracy", "kind": "preserve", "min_delta": 0},
    ]
    (root / "issue.json").write_text(json.dumps({
        "version": "wet.issue.v2", "issue": "Exercise the external evaluator",
        "model": {"adapter": adapter, "config": {}}, "base": base_ref,
        "datasets": {split: ref(root / f"{split}.jsonl") for split in splits}, "gates": gates,
        "budget": {"attempts": 1, "timeout_seconds": 20, "batch_size": 1, "threads": 1},
        "evaluator": {"script": ref(root / "evaluator.py"), "python": sys.executable, "mode": mode,
                      "metrics": {"accuracy": 1}, "seeds": [1, 2], "config": {"duplicate": malformed},
                      "max_work": 128},
    }))
    (root / "proposal.json").write_text(json.dumps({
        "version": "wet.proposal.v2", "base_sha256": base_ref["sha256"],
        "hypothesis": "Restore the correct class weights", "author": "Fixture",
        "operations": [{"op": "replace", "tensors": ["kernel"],
                        "slices": {} if construct else {"kernel": [[0, 1], [0, 1]]},
                        "payload": ref(root / "patch.npz")}],
    }))


def call(root, command, *args, expected=0):
    src = Path(__file__).resolve().parents[1] / "src"
    proc = subprocess.run([sys.executable, "-m", "auto_weight", command, "--workspace", str(root / "run"), *args],
                          env={**os.environ, "PYTHONPATH": str(src)}, capture_output=True, text=True, timeout=40)
    assert proc.returncode == expected, proc.stdout + proc.stderr
    return json.loads(proc.stdout or proc.stderr)


@pytest.mark.parametrize(("mode", "raw_hf"), [("repair", False), ("construct", False), ("repair", True)])
def test_external_checkpoint_end_to_end(tmp_path, mode, raw_hf):
    fixture(tmp_path, mode, raw_hf=raw_hf)
    call(tmp_path, "init", "--issue", str(tmp_path / "issue.json"))
    proposal = str(tmp_path / "proposal.json")
    if mode == "repair":
        assert call(tmp_path, "baseline")["groups"]["fix"]["baseline"]["accuracy"] == 0
        assert not list((tmp_path / "run/attempts").iterdir())
        trial = call(tmp_path, "try", "--proposal", proposal)
        assert trial["status"] == "eligible"
        final = call(tmp_path, "finalize", "--attempt", trial["attempt"])
        assert final["evaluation"]["candidate_sha256"] == trial["evaluation"]["candidate_sha256"]
    else:
        blocked = call(tmp_path, "try", "--proposal", proposal, expected=1)
        assert "no practice" in blocked["error"]
        final = call(tmp_path, "submit", "--proposal", proposal)
        assert "already consumed" in call(tmp_path, "submit", "--proposal", proposal, expected=1)["error"]
    assert final["status"] == "accepted"
    result = call(tmp_path, "verify")
    assert result["confirmation"] == "accepted"
    if raw_hf:
        from safetensors.numpy import load_file

        model = Path(result["checkpoint"])
        np.testing.assert_array_equal(load_file(model / "model.safetensors")["kernel"], np.eye(2))
        assert (model / "tokenizer.json").read_text() == "{}"
    else:
        with np.load(result["checkpoint"], allow_pickle=False) as weights:
            np.testing.assert_array_equal(weights["kernel"], np.eye(2))


def test_duplicate_external_observations_do_not_pass(tmp_path):
    fixture(tmp_path, "repair", malformed=True)
    call(tmp_path, "init", "--issue", str(tmp_path / "issue.json"))
    trial = call(tmp_path, "try", "--proposal", str(tmp_path / "proposal.json"), expected=1)
    assert "duplicate external observation" in trial["error"]
    assert call(tmp_path, "verify")["checkpoint"] is None
