"""Offline conversion of one immutable V4 optimizer-boundary checkpoint.

This module is deliberately absent from normal checkpoint loading and dispatch.
It changes only the version envelope.  Model, optimizer, scheduler, progress and
RNG payloads remain byte-for-byte equal after deserialization.  A separate V5
strict-load and next-update replay is required before an output may be admitted.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import torch

from radon_bridge.runtime.host_checkpoint import atomic_save


SOURCE_SCHEMA = "optimizer_boundary_v1"
TARGET_SCHEMA = "optimizer_boundary_v2"
DESCRIPTION_SCHEMA = "radon_v4_optimizer_boundary_description_v1"
CONVERSION_SCHEMA = "radon_v5_optimizer_boundary_conversion_v1"
FORMAL_V5_COMMIT = "1287681c08846e11364c81653048435482e772a7"
FORMAL_V5_WHEEL_SHA256 = "c022b4f4b0fa1f29458ad1bf9e0d04f6773e9454ab3bd8c07d416294483aab48"
PAYLOAD_KEYS = frozenset({
    "identity", "model", "node_ids", "optimizer", "progress", "rng",
    "scheduler", "schema", "world_size",
})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and np.array_equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _validate_source(state, description, source_digest):
    if not isinstance(state, dict) or set(state) != PAYLOAD_KEYS:
        raise ValueError("Unexpected V4 optimizer-boundary fields")
    if state["schema"] != SOURCE_SCHEMA or "framework_api" in state:
        raise ValueError("Expected an unconverted V4 optimizer boundary")
    if state["world_size"] != 1 or not isinstance(state["identity"], str):
        raise ValueError("Unsupported identity or world size")
    if not isinstance(state["model"], dict) or not isinstance(state["node_ids"], list):
        raise ValueError("Incomplete model identity")
    if not isinstance(state["optimizer"], dict) or set(state["optimizer"]) != {"state", "param_groups"}:
        raise ValueError("Incomplete optimizer state")
    if not isinstance(state["scheduler"], dict):
        raise ValueError("Incomplete scheduler state")
    progress = state["progress"]
    required_progress = {"epoch", "offset", "updates", "epoch_seen", "epoch_loss", "history", "seconds"}
    if (not isinstance(progress, dict) or set(progress) != required_progress
            or any(type(progress[key]) is not int or progress[key] < 0
                   for key in ("epoch", "offset", "updates", "epoch_seen"))):
        raise ValueError("Incomplete data cursor or optimizer-boundary progress")
    if not isinstance(state["rng"], dict) or set(state["rng"]) != {"torch", "cuda", "numpy", "python"}:
        raise ValueError("Incomplete random state")
    if not torch.is_tensor(state["rng"]["torch"]) or not isinstance(state["rng"]["cuda"], list):
        raise ValueError("Invalid Torch or CUDA random state")
    expected = {
        "schema": DESCRIPTION_SCHEMA,
        "framework_api": "V4",
        "source_verified": True,
        "checkpoint_sha256": source_digest,
        "source_checkpoint_schema": SOURCE_SCHEMA,
        "writer_source_sha256": description.get("writer_source_sha256"),
        "model_definition_sha256": description.get("model_definition_sha256"),
        "scientific_spec_sha256": description.get("scientific_spec_sha256"),
        "identity": state["identity"],
        "node_ids": state["node_ids"],
        "world_size": 1,
    }
    if any(description.get(key) != value for key, value in expected.items()):
        raise ValueError("V4 source description does not bind the checkpoint identity")
    for key in ("writer_source_sha256", "model_definition_sha256", "scientific_spec_sha256"):
        value = description.get(key)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("Source description lacks a pinned SHA256")


def convert(source, output, *, source_sha256, source_description,
            description_sha256, framework_lock):
    """Create a non-production V5 candidate without mutating the V4 source."""
    source, output = Path(source), Path(output)
    source_description, framework_lock = Path(source_description), Path(framework_lock)
    if sha256(source) != source_sha256 or sha256(source_description) != description_sha256:
        raise ValueError("Source or description hash changed")
    lock = json.loads(framework_lock.read_text())
    if (lock.get("api_version") != "V5"
            or lock.get("upstream_commit") != FORMAL_V5_COMMIT
            or lock.get("release_wheel_sha256") != FORMAL_V5_WHEEL_SHA256):
        raise ValueError("Exact V5 framework lock required")
    description = json.loads(source_description.read_text())
    state = torch.load(source, map_location="cpu", weights_only=False)
    _validate_source(state, description, source_sha256)
    result = copy.deepcopy(state)
    result["schema"] = TARGET_SCHEMA
    result["framework_api"] = "V5"
    if sha256(source) != source_sha256 or sha256(source_description) != description_sha256:
        raise ValueError("Source changed during conversion")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + ".lock")).open("a") as output_lock:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(output)
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
        try:
            atomic_save(stage / "checkpoint.pt", result)
            loaded = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=False)
            payload = copy.deepcopy(loaded)
            payload.pop("framework_api")
            payload["schema"] = SOURCE_SCHEMA
            if not _equal(payload, state):
                raise ValueError("Conversion changed the scientific payload")
            receipt = {
                "schema": CONVERSION_SCHEMA,
                "state": "awaiting_v5_strict_load_and_next_update_replay",
                "source_sha256": source_sha256,
                "description_sha256": description_sha256,
                "target_sha256": sha256(stage / "checkpoint.pt"),
                "source_framework_api": "V4",
                "execution_framework_api": "V5",
                "execution_framework": lock,
                "original_payload_exact": True,
                "dispatch_allowed": False,
                "production_cutover": False,
                "test_access": False,
                "converter_sha256": sha256(__file__),
            }
            (stage / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
            os.rename(stage, output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return receipt
