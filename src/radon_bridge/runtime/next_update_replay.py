"""Strict comparison for an isolated V4-to-V5 optimizer-boundary replay."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_state(left, right, path="root", *, rtol=1e-5, atol=1e-6):
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.shape != right.shape
                or left.dtype != right.dtype):
            raise ValueError("tensor identity " + path)
        if not torch.allclose(left, right, rtol=rtol, atol=atol):
            raise ValueError("tensor value " + path)
        return
    if isinstance(left, np.ndarray):
        if (not isinstance(right, np.ndarray) or left.shape != right.shape
                or left.dtype != right.dtype or not np.array_equal(left, right)):
            raise ValueError("numpy value " + path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("keys " + path)
        for key in left:
            compare_state(left[key], right[key], path + "/" + str(key), rtol=rtol, atol=atol)
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError("length " + path)
        for index, (a, b) in enumerate(zip(left, right)):
            compare_state(a, b, path + "/" + str(index), rtol=rtol, atol=atol)
        return
    if left != right:
        raise ValueError("value " + path)


def compare_attempt(attempt, packet_path, *, gpu_job_id, write_receipt=True):
    attempt = Path(attempt);packet_path = Path(packet_path)
    old = torch.load(attempt / "v4/last.pt", map_location="cpu", weights_only=False)
    new = torch.load(attempt / "v5/last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng", "node_ids", "world_size"):
        compare_state(old[key], new[key], key)
    old_progress = {k: v for k, v in old["progress"].items() if k != "seconds"}
    new_progress = {k: v for k, v in new["progress"].items() if k != "seconds"}
    compare_state(old_progress, new_progress, "progress_without_wall_time")
    if new.get("framework_api") != "V5" or new.get("schema") != "optimizer_boundary_v2":
        raise ValueError("not formal V5")
    packet = json.loads(packet_path.read_text())
    expected = packet["receipt_expectation"]
    receipt = {
        "schema": "radon_v5_exact_next_update_replay_v1",
        "status": "accepted_engineering_only",
        "test_access": False,
        "gpu_job_id": str(gpu_job_id),
        "packet_sha256": sha256(packet_path),
        "source_commit": expected["source_commit"],
        "inputs_sha256": expected["inputs_sha256"],
        "updates": new["progress"]["updates"],
        "v4_checkpoint_sha256": sha256(attempt / "v4/last.pt"),
        "v5_checkpoint_sha256": sha256(attempt / "v5/last.pt"),
        "rtol": 1e-5,
        "atol": 1e-6,
        "dispatch_allowed": False,
    }
    if write_receipt:
        partial = attempt / "receipt.json.partial"
        partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        partial.replace(attempt / "receipt.json")
    return receipt


def validate_sbatch_header(text):
    """Reject directives placed after the first shell command."""
    command_seen = False
    for line in text.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("#SBATCH") and command_seen:
                raise ValueError("SBATCH directive follows a shell command")
            continue
        command_seen = True
    return True

