import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from radon_bridge.studies.v5_optimizer_boundary_migration import convert, sha256


def fixture(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    state = {
        "schema": "optimizer_boundary_v1", "identity": "run-identity", "world_size": 1,
        "node_ids": [[0, "input"], [1, "logits"]], "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": {"best": .5, "stall": 0},
        "progress": {"epoch": 2, "offset": 16, "updates": 9, "epoch_seen": 16,
                     "epoch_loss": 4.0, "history": [{"epoch": 1}], "seconds": 2.0},
        "rng": {"torch": torch.get_rng_state(), "cuda": [],
                "numpy": np.random.get_state(), "python": random.getstate()},
    }
    source = tmp_path / "v4.pt"; torch.save(state, source)
    description = {
        "schema": "radon_v4_optimizer_boundary_description_v1", "framework_api": "V4",
        "source_verified": True, "checkpoint_sha256": sha256(source),
        "source_checkpoint_schema": "optimizer_boundary_v1", "identity": state["identity"],
        "node_ids": state["node_ids"], "world_size": 1,
        "writer_source_sha256": "a" * 64, "model_definition_sha256": "b" * 64,
        "scientific_spec_sha256": "c" * 64,
    }
    metadata = tmp_path / "description.json"; metadata.write_text(json.dumps(description))
    lock = tmp_path / "framework.json"
    lock.write_text(json.dumps({"api_version": "V5",
        "upstream_commit": "1287681c08846e11364c81653048435482e772a7",
        "release_wheel_sha256": "c022b4f4b0fa1f29458ad1bf9e0d04f6773e9454ab3bd8c07d416294483aab48"}))
    kwargs = dict(source_sha256=sha256(source), source_description=metadata,
                  description_sha256=sha256(metadata), framework_lock=lock)
    return source, state, metadata, kwargs


def test_conversion_preserves_complete_payload_and_fails_closed(tmp_path):
    source, state, _, kwargs = fixture(tmp_path); before = source.read_bytes()
    receipt = convert(source, tmp_path / "candidate", **kwargs)
    assert source.read_bytes() == before
    current = torch.load(tmp_path / "candidate/checkpoint.pt", weights_only=False)
    assert current.pop("framework_api") == "V5"
    current["schema"] = "optimizer_boundary_v1"
    # Serialization may change storage layout, so compare nested values semantically.
    from radon_bridge.studies.v5_optimizer_boundary_migration import _equal
    assert _equal(current, state)
    assert receipt["dispatch_allowed"] is False and receipt["production_cutover"] is False
    with pytest.raises(FileExistsError):
        convert(source, tmp_path / "candidate", **kwargs)


@pytest.mark.parametrize("mutation", ["source", "description", "rng", "cursor", "extra", "sha", "nonhex"])
def test_rejects_drift_or_incomplete_boundary(tmp_path, mutation):
    source, state, metadata, kwargs = fixture(tmp_path)
    if mutation == "source": source.write_bytes(source.read_bytes() + b"drift")
    elif mutation == "description": metadata.write_text(metadata.read_text() + " ")
    elif mutation in {"rng", "cursor", "extra"}:
        changed = copy.deepcopy(state)
        if mutation == "rng": changed["rng"].pop("cuda")
        elif mutation == "cursor": changed["progress"].pop("offset")
        else: changed["unknown"] = True
        torch.save(changed, source); kwargs["source_sha256"] = sha256(source)
        desc = json.loads(metadata.read_text()); desc["checkpoint_sha256"] = kwargs["source_sha256"]
        metadata.write_text(json.dumps(desc)); kwargs["description_sha256"] = sha256(metadata)
    elif mutation == "sha":
        desc = json.loads(metadata.read_text()); desc["writer_source_sha256"] = "short"
        metadata.write_text(json.dumps(desc)); kwargs["description_sha256"] = sha256(metadata)
    else:
        desc = json.loads(metadata.read_text()); desc["writer_source_sha256"] = "z" * 64
        metadata.write_text(json.dumps(desc)); kwargs["description_sha256"] = sha256(metadata)
    with pytest.raises(ValueError):
        convert(source, tmp_path / "candidate", **kwargs)
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("field,value", [
    ("upstream_commit", "f" * 40),
    ("release_wheel_sha256", "f" * 64),
    ("api_version", "V4"),
])
def test_rejects_nonformal_framework_lock(tmp_path, field, value):
    source, _, _, kwargs = fixture(tmp_path)
    lock = kwargs["framework_lock"]
    record = json.loads(lock.read_text()); record[field] = value
    lock.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Exact V5 framework lock required"):
        convert(source, tmp_path / "candidate", **kwargs)
