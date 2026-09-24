import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

import pytest

from radon_bridge.runtime.oct3d_cache_extension import (
    load_contract, load_rows, materialize, proposal_digest, run, validate_adoption,
    validate_root_audit, validate_runner, validate_runtime,
)


def test_materialize_uses_frozen_eye_selection_when_extra_oct_is_valid(tmp_path):
    import numpy as np
    from PIL import Image

    raw = tmp_path / "raw"; raw.mkdir()
    right_planes = []
    for side in ("left", "right"):
        with zipfile.ZipFile(raw / f"{side}.zip", "w") as archive:
            for index in range(128):
                pixels = np.array([[0, 10 + index], [20, 30]], dtype=np.uint8)
                payload = io.BytesIO(); Image.fromarray(pixels).save(payload, format="PNG")
                archive.writestr(f"slice_{index}.png", payload.getvalue())
                if side == "right":
                    right_planes.append(np.asarray(Image.fromarray(pixels).resize(
                        (224, 224), Image.Resampling.BILINEAR)))
    expected = np.stack([np.stack(right_planes)[None]])
    encoded = io.BytesIO(); np.save(encoded, expected, allow_pickle=False)
    row = {"id": "a", "split": "train", "output_path": "arrays/a",
           "expected_eyes": ["right"], "oct3d_sha256": hashlib.sha256(encoded.getvalue()).hexdigest(),
           "eyes": [{"eye": side, "oct": {"archive": f"{side}.zip"}} for side in ("left", "right")]}

    receipt = materialize(row, raw, tmp_path / "output", {"version": "frozen"})
    assert receipt["valid_eyes"] == ["right"]
    assert receipt["rejections"] == [{"eye": "left", "reason": "excluded_by_frozen_reference",
                                      "type": "FrozenReferenceExclusion"}]
    assert digest(tmp_path / "output/arrays/a/oct_volume_3d.npy") == row["oct3d_sha256"]


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
    runner = tmp_path / "runner.py"; runner.write_text("# reviewed runner\n")
    raw = tmp_path / "raw"; raw.mkdir(); target = tmp_path / "target"; target.write_text("x")
    (raw / "images").symlink_to(target)
    manifest = tmp_path / "manifest.jsonl"
    stat = target.stat(); identity = json.dumps([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    manifest.write_text(json.dumps({"path": "images", "sha256": digest(target), "bytes": 1}) + "\n")
    verifier = tmp_path / "verify.py"; verifier.write_text("# verifier\n")
    database = tmp_path / "verified.sqlite"
    db = sqlite3.connect(database); db.execute("CREATE TABLE verified(path,expected,actual,identity,bytes)")
    db.execute("INSERT INTO verified VALUES(?,?,?,?,?)", ("images", digest(target), digest(target), identity, 1)); db.commit(); db.close()
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "complete_verified", "root": str(raw),
        "manifest_sha256": digest(manifest), "verified_files": 1, "verified_bytes": 1,
        "test_scientific_access": False}))
    contract = {"schema": "oct3d_cache_extension_contract_v2", "operation_id": "oct3d-v1",
        "source_operation_id": "two-d-v2", "source_root": str(tmp_path / "source"),
        "output_root": str(tmp_path / "oct3d"), "raw_root": str(raw),
        "bundle": str(bundle), "bundle_sha256": digest(bundle), "manifests": manifests,
        "expected_rows": 2, "recipe": recipe,
        "runner": {"path": str(runner), "sha256": digest(runner)},
        "runtime_pins": {"python": ".".join(map(str, sys.version_info[:3])),
                         "numpy": __import__("numpy").__version__, "pillow": __import__("PIL").__version__},
        "root_audit": {"manifest": str(manifest), "manifest_sha256": digest(manifest),
            "status": str(status), "status_sha256": digest(status), "verified_sqlite": str(database),
            "verified_sqlite_sha256": digest(database), "verifier": str(verifier),
            "verifier_sha256": digest(verifier), "expected_files": 1, "expected_bytes": 1},
        "raw_view_mapping": {"images": str(target)},
        "owner_adoption_sha256": "0" * 64, "test_access": False}
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


def test_runner_runtime_and_full_root_identity_are_launch_gates(tmp_path):
    contract_path, _ = fixture(tmp_path); contract = load_contract(contract_path)
    validate_runner(contract); validate_runtime(contract)
    assert validate_root_audit(contract) == {"files": 1, "bytes": 1, "accepted": True}
    Path(contract["runner"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="runner source"):
        validate_runner(contract)
    contract["runtime_pins"]["numpy"] = "drift"
    with pytest.raises(ValueError, match="runtime changed"):
        validate_runtime(contract)
    contract = load_contract(contract_path)
    Path(contract["root_audit"]["status"]).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_root_audit(contract)
