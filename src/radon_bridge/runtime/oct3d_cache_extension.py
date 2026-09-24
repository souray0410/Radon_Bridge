"""Finite, single-owner OCT3D cache extension for an immutable 2-D cache scope.

This module never edits the source cache.  A separately accepted owner adoption
receipt is required before it can create an output directory or acquire its lock.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
import time
import zipfile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError("unsafe relative path")
    return path.as_posix()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    required = {
        "schema", "operation_id", "source_operation_id", "source_root",
        "output_root", "raw_root", "bundle", "bundle_sha256", "manifests",
        "expected_rows", "recipe", "runner", "runtime_pins", "root_audit",
        "raw_view_mapping", "owner_adoption_sha256", "test_access",
    }
    if set(contract) != required or contract["schema"] != "oct3d_cache_extension_contract_v2":
        raise ValueError("unknown or incomplete extension contract")
    if contract["operation_id"] == contract["source_operation_id"]:
        raise ValueError("extension must have a separate operation identity")
    source = Path(contract["source_root"]).resolve()
    output = Path(contract["output_root"]).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("extension output must be separate from source cache")
    if contract["test_access"] is not False:
        raise ValueError("test access must remain sealed")
    adoption = contract["owner_adoption_sha256"]
    if not isinstance(adoption, str) or len(adoption) != 64 or set(adoption) == {"0"}:
        raise ValueError("an exact accepted owner adoption receipt is required")
    return contract


def validate_runner(contract: dict) -> None:
    runner = contract["runner"]
    if set(runner) != {"path", "sha256"} or sha(Path(runner["path"])) != runner["sha256"]:
        raise ValueError("executable runner source changed")


def validate_runtime(contract: dict) -> dict:
    import numpy
    import PIL
    observed = {"python": ".".join(map(str, sys.version_info[:3])),
                "numpy": numpy.__version__, "pillow": PIL.__version__}
    if observed != contract["runtime_pins"]:
        raise ValueError(f"cache runtime changed: {observed}")
    return observed


def _stat_identity(status) -> str:
    return json.dumps([status.st_dev, status.st_ino, status.st_size,
                       status.st_mtime_ns, status.st_ctime_ns])


def validate_root_audit(contract: dict) -> dict:
    audit = contract["root_audit"]
    required = {"manifest", "manifest_sha256", "status", "status_sha256", "verified_sqlite",
                "verified_sqlite_sha256", "verifier", "verifier_sha256", "expected_files",
                "expected_bytes"}
    if set(audit) != required:
        raise ValueError("incomplete root audit contract")
    for name in ("manifest", "status", "verified_sqlite", "verifier"):
        if sha(Path(audit[name])) != audit[name + "_sha256"]:
            raise ValueError("root audit artifact changed: " + name)
    status = json.loads(Path(audit["status"]).read_text())
    if (status.get("state") != "complete_verified" or status.get("root") != contract["raw_root"]
            or status.get("manifest_sha256") != audit["manifest_sha256"]
            or status.get("verified_files") != audit["expected_files"]
            or status.get("verified_bytes") != audit["expected_bytes"]
            or status.get("test_scientific_access") is not False):
        raise ValueError("root audit completion identity changed")
    raw = Path(contract["raw_root"])
    mapping = contract["raw_view_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("raw-view mapping missing")
    for name, target in mapping.items():
        link = raw / safe_relative(name)
        if not link.is_symlink() or os.readlink(link) != target or str(link.resolve(strict=True)) != target:
            raise ValueError("raw-view mapping changed: " + name)
    database = sqlite3.connect(f"file:{audit['verified_sqlite']}?mode=ro", uri=True)
    try:
        database.execute("BEGIN")
        verified = {row[0]: row[1:] for row in
                    database.execute("SELECT path,expected,actual,identity,bytes FROM verified")}
        database.execute("COMMIT")
    finally:
        database.close()
    if len(verified) != audit["expected_files"]:
        raise ValueError("root audit SQLite row count changed")
    count = total = 0
    with Path(audit["manifest"]).open() as stream:
        for line in stream:
            row = json.loads(line); relative = safe_relative(row["path"])
            saved = verified.get(relative)
            if saved is None:
                raise ValueError("root audit missing verified path")
            expected, actual, identity, size = saved
            target = raw / relative; current = target.stat()
            if (expected != row["sha256"] or actual != row["sha256"]
                    or int(size) != int(row["bytes"]) or current.st_size != int(row["bytes"])
                    or _stat_identity(current) != identity):
                raise ValueError("raw identity changed after full verification")
            count += 1; total += int(row["bytes"])
    if count != audit["expected_files"] or total != audit["expected_bytes"]:
        raise ValueError("root manifest totals changed")
    return {"files": count, "bytes": total, "accepted": True}


def proposal_digest(contract: dict) -> str:
    """Hash the complete proposal without its later owner-receipt reference."""
    proposal = {key: value for key, value in contract.items() if key != "owner_adoption_sha256"}
    encoded = json.dumps(proposal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_adoption(contract: dict, receipt_path: Path) -> dict:
    if sha(receipt_path) != contract["owner_adoption_sha256"]:
        raise ValueError("owner adoption receipt changed")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "schema": "oct3d_cache_extension_owner_adoption_v1",
        "state": "accepted",
        "operation_id": contract["operation_id"],
        "source_operation_id": contract["source_operation_id"],
        "single_writer": True,
        "source_operation_immutable": True,
        "test_access": False,
        "proposal_sha256": proposal_digest(contract),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("owner adoption contract is not accepted")
    if not isinstance(receipt.get("writer_identity"), str) or not receipt["writer_identity"]:
        raise ValueError("writer identity missing")
    return receipt


def _manifest_rows(path: Path, digest: str, role: str, recipe: dict) -> list[dict]:
    if sha(path) != digest:
        raise ValueError(f"{role} manifest SHA changed")
    value = json.loads(path.read_text())
    if value.get("role") != role or value.get("recipe") != recipe:
        raise ValueError(f"{role} manifest identity changed")
    rows = value.get("samples")
    if not isinstance(rows, list):
        raise ValueError(f"{role} samples missing")
    return rows


def load_rows(contract: dict) -> list[dict]:
    bundle = Path(contract["bundle"])
    if sha(bundle) != contract["bundle_sha256"]:
        raise ValueError("frozen raw-row bundle SHA changed")
    references = {}
    expected_counts = {}
    for role in ("train", "development"):
        item = contract["manifests"].get(role, {})
        rows = _manifest_rows(Path(item["path"]), item["sha256"], role, contract["recipe"])
        expected_counts[role] = item["rows"]
        if len(rows) != item["rows"]:
            raise ValueError("manifest row count changed")
        for row in rows:
            if row["id"] in references:
                raise ValueError("participant appears in multiple roles")
            files = row.get("files", {})
            if "oct_volume_3d.npy" not in files:
                raise ValueError("reference lacks OCT3D hash")
            references[row["id"]] = {
                "split": role,
                "eyes": list(row["eyes"]),
                "sha256": files["oct_volume_3d.npy"],
            }
    rows = []
    seen = set()
    with gzip.open(bundle, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            participant = row.get("id")
            reference = references.get(participant)
            if reference is None or participant in seen:
                raise ValueError("bundle participant identity mismatch")
            if row.get("split") != reference["split"] or row.get("expected_eyes") != reference["eyes"]:
                raise ValueError("bundle split or eye identity changed")
            row["output_path"] = safe_relative(row["output_path"])
            row["oct3d_sha256"] = reference["sha256"]
            rows.append(row)
            seen.add(participant)
    if seen != set(references) or len(rows) != contract["expected_rows"]:
        raise ValueError("extension does not cover the exact frozen cohort")
    if sum(1 for row in rows if row["split"] == "train") != expected_counts["train"]:
        raise ValueError("train row count changed")
    return rows


def _slice_order(names: list[str]) -> list[str]:
    def key(name: str) -> int:
        match = re.search(r"_(\d+)\.png$", name)
        if not match:
            raise ValueError("unexpected OCT member name")
        return int(match.group(1))
    return sorted(names, key=key)


def materialize(row: dict, raw_root: Path, output_root: Path, recipe: dict) -> dict:
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    destination = output_root / row["output_path"]
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        volume = destination / "oct_volume_3d.npy"
        if (receipt.get("id") == row["id"] and receipt.get("recipe") == recipe
                and receipt.get("files") == {"oct_volume_3d.npy": row["oct3d_sha256"]}
                and volume.is_file() and sha(volume) == row["oct3d_sha256"]):
            return receipt
        raise ValueError("committed OCT3D extension changed")
    volumes, valid, rejections = [], [], []
    expected = row["expected_eyes"]
    raw_eyes = {eye["eye"]: eye for eye in row["eyes"]}
    if (len(expected) != len(set(expected)) or len(raw_eyes) != len(row["eyes"])
            or not set(expected) <= set(raw_eyes)):
        raise ValueError("frozen eye selection is not present in raw bundle")
    for eye in row["eyes"]:
        if eye["eye"] not in expected:
            rejections.append({"eye": eye["eye"], "reason": "excluded_by_frozen_reference",
                               "type": "FrozenReferenceExclusion"})
    for eye_name in expected:
        eye = raw_eyes[eye_name]
        try:
            archive = raw_root / safe_relative(eye["oct"]["archive"])
            with zipfile.ZipFile(archive) as source:
                names = _slice_order(source.namelist())
                if len(names) != 128:
                    raise ValueError("oct_slice_count_not_128")
                indices = [int(re.search(r"_(\d+)\.png$", name).group(1)) for name in names]
                if indices != list(range(indices[0], indices[0] + 128)):
                    raise ValueError("noncontiguous_oct_indices")
                planes, original_shape = [], None
                for name in names:
                    with Image.open(io.BytesIO(source.read(name))) as image:
                        if original_shape is None:
                            original_shape = image.size
                        if image.size != original_shape:
                            raise ValueError("inconsistent_oct_slice_shapes")
                        planes.append(np.asarray(image.convert("L").resize((224, 224), Image.Resampling.BILINEAR)))
                volume = np.stack(planes)[None]
                if volume.max() == volume.min():
                    raise ValueError("constant_oct_volume")
            volumes.append(volume)
            valid.append(eye["eye"])
        except (ValueError, zipfile.BadZipFile, UnidentifiedImageError) as error:
            rejections.append({"eye": eye["eye"], "reason": str(error), "type": type(error).__name__})
    if valid != expected:
        raise ValueError("valid-eye QC differs from frozen reference")
    destination.mkdir(parents=True, exist_ok=True)
    volume_path = destination / "oct_volume_3d.npy"
    temporary = volume_path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.stack(volumes), allow_pickle=False)
    if sha(temporary) != row["oct3d_sha256"]:
        temporary.unlink()
        raise ValueError("rebuilt OCT3D SHA differs from frozen reference")
    os.replace(temporary, volume_path)
    receipt = {
        "schema": "oct3d_cache_extension_participant_v1", "id": row["id"],
        "split": row["split"], "recipe": recipe, "valid_eyes": valid,
        "rejections": rejections, "files": {"oct_volume_3d.npy": row["oct3d_sha256"]},
        "path": row["output_path"], "test_access": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def run(contract_path: Path, adoption_path: Path, max_participants: int, max_seconds: int) -> dict:
    if max_participants < 1 or max_seconds < 1:
        raise ValueError("finite positive execution budgets required")
    contract = load_contract(contract_path)
    # All executable/runtime/raw identities are checked before output parent,
    # lock, state, or participant files can be created or mutated.
    validate_runner(contract)
    validate_runtime(contract)
    validate_root_audit(contract)
    adoption = validate_adoption(contract, adoption_path)
    rows = load_rows(contract)
    output = Path(contract["output_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / ("." + contract["operation_id"] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=True)
        state_path = output / "state.json"
        state = ({"schema": "oct3d_cache_extension_state_v1", "operation_id": contract["operation_id"],
                  "contract_sha256": sha(contract_path), "owner_adoption_sha256": sha(adoption_path),
                  "next_index": 0, "completed": 0, "state": "initialized", "test_access": False}
                 if not state_path.exists() else json.loads(state_path.read_text()))
        index = state.get("next_index")
        if (state.get("schema") != "oct3d_cache_extension_state_v1"
                or state.get("operation_id") != contract["operation_id"]
                or state.get("test_access") is not False
                or type(index) is not int or not 0 <= index <= len(rows)
                or state.get("completed") != index
                or state.get("contract_sha256") != sha(contract_path)
                or state.get("owner_adoption_sha256") != contract["owner_adoption_sha256"]):
            raise ValueError("extension state identity changed")
        started = time.monotonic()
        limit = min(len(rows), index + max_participants)
        while index < limit and time.monotonic() - started < max_seconds:
            materialize(rows[index], Path(contract["raw_root"]), output, contract["recipe"])
            index += 1
            state.update(next_index=index, completed=index, state="running", writer_identity=adoption["writer_identity"])
            atomic_json(state_path, state)
        state["state"] = "complete_pending_independent_acceptance" if index == len(rows) else "paused_finite"
        state["next_index"] = index
        atomic_json(state_path, state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--owner-adoption", type=Path, required=True)
    parser.add_argument("--max-participants", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.contract, arguments.owner_adoption,
                         arguments.max_participants, arguments.max_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
