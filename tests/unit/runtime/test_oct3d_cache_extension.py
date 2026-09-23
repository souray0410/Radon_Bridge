import gzip
import hashlib
import json
from pathlib import Path

import pytest

from radon_bridge.runtime.oct3d_cache_extension import (
    load_contract, load_rows, proposal_digest, run, validate_adoption,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    recipe = {"version": "frozen"}
    manifests = {}
    all_rows = []
    for role, participant in (("train", "a"), ("development", "b")):
        value = {"role": role, "recipe": recipe, "samples": [{"id": participant, "eyes": ["L"],
                 "files": {"oct_volume_3d.npy": participant * 64}}]}
        path = tmp_path / f"{role}.json"; path.write_text(json.dumps(value))
        manifests[role] = {"path": str(path), "sha256": digest(path), "rows": 1}
        all_rows.append({"id": participant, "split": role, "expected_eyes": ["L"],
                         "output_path": f"arrays/{participant}", "eyes": []})
    bundle = tmp_path / "rows.jsonl.gz"
    with gzip.open(bundle, "wt") as stream:
        for row in all_rows: stream.write(json.dumps(row) + "\n")
    contract = {"schema": "oct3d_cache_extension_contract_v1", "operation_id": "oct3d-v1",
        "source_operation_id": "two-d-v2", "source_root": str(tmp_path / "source"),
        "output_root": str(tmp_path / "oct3d"), "raw_root": str(tmp_path / "raw"),
        "bundle": str(bundle), "bundle_sha256": digest(bundle), "manifests": manifests,
        "expected_rows": 2, "recipe": recipe, "owner_adoption_sha256": "0" * 64, "test_access": False}
    adoption = tmp_path / "adoption.json"
    adoption.write_text(json.dumps({"schema": "oct3d_cache_extension_owner_adoption_v1", "state": "accepted",
        "operation_id": "oct3d-v1", "source_operation_id": "two-d-v2", "single_writer": True,
        "source_operation_immutable": True, "test_access": False, "writer_identity": "original-writer",
        "proposal_sha256": proposal_digest(contract)}))
    contract["owner_adoption_sha256"] = digest(adoption)
    path = tmp_path / "contract.json"; path.write_text(json.dumps(contract))
    return path, adoption


def test_contract_requires_separate_output_and_original_owner_adoption(tmp_path):
    contract_path, adoption = fixture(tmp_path)
    contract = load_contract(contract_path)
    assert validate_adoption(contract, adoption)["writer_identity"] == "original-writer"
    assert [row["id"] for row in load_rows(contract)] == ["a", "b"]
    bad = json.loads(contract_path.read_text()); bad["output_root"] = bad["source_root"]
    contract_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="separate"):
        load_contract(contract_path)


def test_identity_drift_and_missing_oct3d_fail_closed(tmp_path):
    contract_path, adoption = fixture(tmp_path)
    contract = load_contract(contract_path)
    adoption.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        validate_adoption(contract, adoption)
    train = Path(contract["manifests"]["train"]["path"])
    value = json.loads(train.read_text()); value["samples"][0]["files"] = {}
    train.write_text(json.dumps(value)); contract["manifests"]["train"]["sha256"] = digest(train)
    with pytest.raises(ValueError, match="lacks OCT3D"):
        load_rows(contract)


def test_adoption_binds_full_proposal_and_state_cursor_is_strict(tmp_path, monkeypatch):
    contract_path, adoption = fixture(tmp_path)
    contract = load_contract(contract_path)
    contract["output_root"] += "-changed"
    with pytest.raises(ValueError, match="not accepted"):
        validate_adoption(contract, adoption)
    contract = load_contract(contract_path)
    output = Path(contract["output_root"]); output.mkdir()
    state = {"schema": "oct3d_cache_extension_state_v1", "operation_id": contract["operation_id"],
             "contract_sha256": digest(contract_path), "owner_adoption_sha256": digest(adoption),
             "next_index": 3, "completed": 3, "state": "paused_finite", "test_access": False}
    (output / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="state identity"):
        run(contract_path, adoption, 1, 1)
