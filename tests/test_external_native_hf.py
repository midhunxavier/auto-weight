"""Native tensor operations retain the external evaluator's frozen repair lifecycle."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.numpy")
torch_safetensors = pytest.importorskip("safetensors.torch")

from auto_weight.checkpoint import RAW_HF_ADAPTER, checkpoint_hash
from auto_weight.contracts import EditError, file_hash
from auto_weight.evaluator import materialize
from test_external_e2e import call, fixture


def checkpoints(root, sharded=False):
    fixture(root, "repair", raw_hf=True)
    base = root / "base"
    if sharded:
        weights = safetensors.load_file(base / "model.safetensors")
        (base / "model.safetensors").unlink()
        safetensors.save_file(weights, base / "model-00001-of-00002.safetensors")
        safetensors.save_file({"aux": np.ones(3, dtype="float32")}, base / "model-00002-of-00002.safetensors")
        (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "kernel": "model-00001-of-00002.safetensors", "aux": "model-00002-of-00002.safetensors"}}))
        evaluator = root / "evaluator.py"
        evaluator.write_text(evaluator.read_text().replace('artifact / "model.safetensors"',
                                                         'artifact / "model-00001-of-00002.safetensors"'))
    donor = root / "donor"
    shutil.copytree(base, donor)
    filename = "model-00001-of-00002.safetensors" if sharded else "model.safetensors"
    safetensors.save_file({"kernel": np.eye(2, dtype="float32")}, donor / filename)
    issue = json.loads((root / "issue.json").read_text())
    issue["base"]["sha256"] = checkpoint_hash(base, RAW_HF_ADAPTER)
    issue["evaluator"]["script"]["sha256"] = file_hash(root / "evaluator.py")
    (root / "issue.json").write_text(json.dumps(issue))
    return base, donor, filename, issue["base"]["sha256"]


@pytest.mark.parametrize(("kind", "sharded"), [("restore", False), ("blend", True), ("scale", False)])
def test_native_external_operation_is_replayed_and_only_passing_models_are_exported(tmp_path, kind, sharded):
    base, donor, filename, base_sha = checkpoints(tmp_path, sharded)
    operation = {"op": kind, "tensors": ["kernel"]}
    if kind == "scale":
        operation["factor"] = 2
    else:
        operation["donor"] = {"path": "donor", "sha256": checkpoint_hash(donor, RAW_HF_ADAPTER)}
        if kind == "blend":
            operation["alpha"] = 0.75
    proposal = {"version": "wet.proposal.v1", "base_sha256": base_sha,
                "author": "test", "hypothesis": "Test native operations with an independent numeric evaluator.",
                "operations": [operation]}
    (tmp_path / "proposal.json").write_text(json.dumps(proposal))
    call(tmp_path, "init", "--issue", str(tmp_path / "issue.json"))
    trial = call(tmp_path, "try", "--proposal", str(tmp_path / "proposal.json"))
    if kind == "scale":
        assert trial["status"] == "discarded"
        assert not trial["evaluation"]["gates"][0]["passed"]
        assert call(tmp_path, "verify")["checkpoint"] is None
        return
    assert trial["status"] == "eligible"
    donor.rename(tmp_path / "moved-donor")
    final = call(tmp_path, "finalize", "--attempt", trial["attempt"])
    assert final["status"] == "accepted"
    assert final["evaluation"]["candidate_sha256"] == trial["evaluation"]["candidate_sha256"]
    exported = Path(call(tmp_path, "verify")["checkpoint"])
    expected = np.eye(2, dtype="float32") if kind == "restore" else np.diag([0.5, 1]).astype("float32")
    np.testing.assert_array_equal(safetensors.load_file(exported / filename)["kernel"], expected)
    assert checkpoint_hash(base, RAW_HF_ADAPTER) == base_sha
    assert file_hash(exported / "tokenizer.json") == file_hash(base / "tokenizer.json")
    if sharded:
        for name in ("model.safetensors.index.json", "model-00002-of-00002.safetensors"):
            assert file_hash(exported / name) == file_hash(base / name)


@pytest.mark.parametrize("mismatch", ["tokenizer", "architecture", "shape", "dtype"])
def test_incompatible_external_blend_fails_without_export(tmp_path, mismatch):
    base, donor, filename, base_sha = checkpoints(tmp_path)
    if mismatch == "tokenizer":
        (donor / "tokenizer.json").write_text('{"changed":true}')
    elif mismatch == "architecture":
        (donor / "config.json").write_text('{"hidden_size":3}')
    else:
        values = np.ones((3, 3), dtype="float32") if mismatch == "shape" else np.eye(2, dtype="float64")
        safetensors.save_file({"kernel": values}, donor / filename)
    proposal = {"version": "wet.proposal.v1", "base_sha256": base_sha, "author": "test",
        "hypothesis": "An incompatible donor must not produce a published candidate.",
        "operations": [{"op": "blend", "tensors": ["kernel"], "alpha": 0.25,
                        "donor": {"path": "donor", "sha256": checkpoint_hash(donor, RAW_HF_ADAPTER)}}]}
    (tmp_path / "proposal.json").write_text(json.dumps(proposal))
    call(tmp_path, "init", "--issue", str(tmp_path / "issue.json"))
    trial = call(tmp_path, "try", "--proposal", str(tmp_path / "proposal.json"), expected=1)
    assert trial["status"] == "error"
    assert "donor" in trial["error"]
    assert call(tmp_path, "verify")["checkpoint"] is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["blend", "restore", "scale"])
def test_native_operations_round_to_stored_dtype_and_preserve_metadata(tmp_path, dtype, kind):
    base, donor, filename, _ = checkpoints(tmp_path)
    # Around +/-1, these formats have spacing eps. A 3/4-step change must
    # round to the adjacent representable value, rather than stay at +/-1.
    eps = torch.finfo(dtype).eps
    original = torch.tensor([1, -1], dtype=dtype)
    adjacent = torch.tensor([1 + eps, -1 - eps], dtype=dtype)
    metadata = {"format": "pt", "source": "rounding-fixture"}
    torch_safetensors.save_file({"kernel": original}, base / filename, metadata=metadata)
    torch_safetensors.save_file({"kernel": adjacent}, donor / filename)
    base_sha = checkpoint_hash(base, RAW_HF_ADAPTER)
    operation = {"op": kind, "tensors": ["kernel"]}
    if kind == "scale":
        operation["factor"] = 1 + 0.75 * eps
    else:
        operation["donor"] = {"path": "donor", "sha256": checkpoint_hash(donor, RAW_HF_ADAPTER)}
        if kind == "blend":
            operation["alpha"] = 0.75
    candidate = tmp_path / "candidate"
    changes = materialize(base, candidate, {"operations": [operation]}, tmp_path, RAW_HF_ADAPTER)
    actual = torch_safetensors.load_file(candidate / filename)["kernel"]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, adjacent, rtol=0, atol=0)
    assert changes[0]["parameters_changed"] == 2
    assert changes[0]["max_abs_delta"] == eps
    assert checkpoint_hash(base, RAW_HF_ADAPTER) == base_sha
    with torch_safetensors.safe_open(candidate / filename, framework="pt") as archive:
        assert archive.metadata() == metadata


@pytest.mark.parametrize(("kind", "bad_value"), [
    ("blend", float("nan")), ("blend", float("inf")),
    ("restore", float("nan")), ("restore", float("-inf")),
    ("scale", None),
])
def test_nonfinite_native_result_fails_without_export(tmp_path, kind, bad_value):
    base, donor, filename, base_sha = checkpoints(tmp_path)
    operation = {"op": kind, "tensors": ["kernel"]}
    if kind == "scale":
        # The stored base is finite, but the allowed factor overflows FP32.
        values = np.diag([-np.finfo("float32").max / 4, 1]).astype("float32")
        safetensors.save_file({"kernel": values}, base / filename)
        base_sha = checkpoint_hash(base, RAW_HF_ADAPTER)
        issue = json.loads((tmp_path / "issue.json").read_text())
        issue["base"]["sha256"] = base_sha
        (tmp_path / "issue.json").write_text(json.dumps(issue))
        operation["factor"] = 10
    else:
        values = np.eye(2, dtype="float32")
        values[0, 0] = bad_value
        safetensors.save_file({"kernel": values}, donor / filename)
        operation["donor"] = {"path": "donor", "sha256": checkpoint_hash(donor, RAW_HF_ADAPTER)}
        if kind == "blend":
            operation["alpha"] = 0.25
    proposal = {"version": "wet.proposal.v1", "base_sha256": base_sha, "author": "test",
                "hypothesis": "Nonfinite edits must fail before scoring or export.", "operations": [operation]}
    (tmp_path / "proposal.json").write_text(json.dumps(proposal))
    call(tmp_path, "init", "--issue", str(tmp_path / "issue.json"))
    trial = call(tmp_path, "try", "--proposal", str(tmp_path / "proposal.json"), expected=1)
    assert trial["status"] == "error"
    assert "nonfinite" in trial["error"]
    assert call(tmp_path, "verify")["checkpoint"] is None
    assert checkpoint_hash(base, RAW_HF_ADAPTER) == base_sha


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_restore_rejects_nonfinite_base_even_with_finite_donor(tmp_path, bad_value):
    base, donor, filename, _ = checkpoints(tmp_path)
    values = np.eye(2, dtype="float32")
    values[0, 0] = bad_value
    safetensors.save_file({"kernel": values}, base / filename)
    operation = {"op": "restore", "tensors": ["kernel"],
                 "donor": {"path": "donor", "sha256": checkpoint_hash(donor, RAW_HF_ADAPTER)}}
    with pytest.raises(EditError, match="nonfinite edited parameter"):
        materialize(base, tmp_path / "candidate", {"operations": [operation]}, tmp_path, RAW_HF_ADAPTER)
