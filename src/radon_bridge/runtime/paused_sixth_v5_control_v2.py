"""Crash-recoverable one-run V5 control for the reviewed paused R&B run.

The allocation transaction is durable before submission and every GPU job is
submitted held.  A job is released only after its allocation and finalizer are
both uniquely recoverable from Slurm.  This is a finite operation, not another
dispatcher or refiller.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import tempfile
import time

from radon_bridge.runtime.paused_sixth_v5_handoff import FORMAL_V5_COMMIT, RUN_ID


ACTIVE = {
    "preparing", "allocation_ack_unknown", "allocation_ack_pending", "held",
    "finalizer_ack_unknown", "finalizer_ack_pending", "ready_release",
    "release_ack_unknown", "release_ack_pending", "submitted", "granted", "running",
}
TERMINAL = {"completed", "paused", "failed", "cancelled", "identity_drift", "needs_review"}
RETRYABLE = {"paused", "failed"}
ENV_KEYS = {
    "PYTHONPATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
}
MODULE_NAMES = {
    "radon_bridge.runtime.paused_sixth_v5_control",
    "radon_bridge.runtime.paused_sixth_v5_control_v2",
    "radon_bridge.runtime.paused_sixth_v5_coordinator",
    "radon_bridge.runtime.paused_sixth_v5_handoff",
    "mhd_models.workflows.native",
    "mhd_models.scheduling.policy",
    "mhd_models.scheduling.quota_guard",
    "mhd_models.scheduling.slurm_liveness",
    "mhd_models.runtime.training_state",
    "mhd_framework",
    "mhd_framework.core",
    "mhd_framework.utils",
}
RUNTIME_ROLES = {
    "entrypoint", "implementation", "coordinator", "handoff", "native", "policy", "quota_guard", "liveness",
    "training_state", "framework_init", "framework_core", "framework_utils",
    "allocation_wrapper", "finalizer_wrapper", "python_executable",
}
ROLE_MODULES = {
    "entrypoint": "radon_bridge.runtime.paused_sixth_v5_control",
    "implementation": "radon_bridge.runtime.paused_sixth_v5_control_v2",
    "coordinator": "radon_bridge.runtime.paused_sixth_v5_coordinator",
    "handoff": "radon_bridge.runtime.paused_sixth_v5_handoff",
    "native": "mhd_models.workflows.native",
    "policy": "mhd_models.scheduling.policy",
    "quota_guard": "mhd_models.scheduling.quota_guard",
    "liveness": "mhd_models.scheduling.slurm_liveness",
    "training_state": "mhd_models.runtime.training_state",
    "framework_init": "mhd_framework",
    "framework_core": "mhd_framework.core",
    "framework_utils": "mhd_framework.utils",
}
ACK_RECOVERY_GRACE_SECONDS = 120


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def python_launcher_identity(path):
    """Return the exact logical launcher chain without resolving it away."""
    current = Path(os.path.abspath(path))
    rows, seen = [], set()
    for _ in range(16):
        key = str(current)
        if key in seen:
            raise ValueError("Python launcher symlink loop")
        seen.add(key)
        if current.is_symlink():
            target = os.readlink(current)
            rows.append({"path": key, "kind": "symlink", "target": target})
            current = Path(target) if os.path.isabs(target) else current.parent / target
            current = Path(os.path.abspath(current))
            continue
        if not current.is_file():
            raise ValueError("Python launcher chain is incomplete")
        rows.append({"path": key, "kind": "file", "sha256": sha256(current)})
        return rows
    raise ValueError("Python launcher chain is too deep")


def python_launcher_sha256(path):
    return hashlib.sha256(json.dumps(
        python_launcher_identity(path), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_once(path, value):
    """Create an immutable evidence file and durably publish its directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                         0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def locked_existing(path, *, nonblocking=False):
    path = Path(path)
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    handle = os.fdopen(descriptor, "r+")
    before, after = path.stat(), os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        handle.close()
        raise RuntimeError("account lock inode changed")
    try:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(handle, flags)
    except BaseException:
        handle.close()
        raise
    return handle


def _digest_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact(path, digest, label):
    if sha256(path) != digest:
        raise ValueError(label + " changed")


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def runtime_contract_digest(binding):
    return _digest_json(binding["runtime_contract"])


def validate_binding(path, *, now=None, allow_expired=False):
    """Validate every immutable input, including the deployment-order gate."""
    now = time.time() if now is None else now
    binding = read(path)
    required = {
        "schema", "lease_id", "project", "run_id", "source_commit", "framework_commit",
        "account_lock", "role_policy", "role_policy_sha256", "ownership_overlay",
        "journal", "journal_initial_sha256", "intent", "intent_initial_sha256",
        "claims_root", "expected_claim", "asset_dir", "asset_sha256",
        "v5_spec_sha256", "v5_checkpoint_sha256", "execution_dir", "requested_gpus",
        "account_limit", "worker_cpus", "worker_memory_gib", "expires_at",
        "comment_prefix", "sbatch_command_template", "finalizer_command_template",
        "allocation_wrapper", "finalizer_wrapper", "runtime_pins", "runtime_contract",
        "retry_policy", "deployment_gate", "deployment_gate_sha256",
        "test_access", "dispatch_authorized",
    }
    if set(binding) != required or binding.get("schema") != "radon_paused_sixth_v5_control_v2":
        raise ValueError("unknown paused-sixth V5 control binding")
    if (binding["project"] != "Radon_Bridge" or binding["run_id"] != RUN_ID
            or binding["framework_commit"] != FORMAL_V5_COMMIT
            or not _hex(binding["source_commit"], 40)
            or binding["requested_gpus"] != 1 or binding["account_limit"] != 24
            or binding["test_access"] is not False or binding["dispatch_authorized"] is not True
            or type(binding["worker_cpus"]) is not int or binding["worker_cpus"] < 1
            or type(binding["worker_memory_gib"]) is not int or binding["worker_memory_gib"] < 1
            or type(binding["expires_at"]) not in (int, float)
            or (not allow_expired and binding["expires_at"] <= now)
            or not isinstance(binding["comment_prefix"], str) or not binding["comment_prefix"]):
        raise ValueError("paused-sixth V5 control scope changed")
    for key in ("role_policy_sha256", "journal_initial_sha256", "intent_initial_sha256",
                "asset_sha256", "v5_spec_sha256", "v5_checkpoint_sha256",
                "deployment_gate_sha256"):
        if not _hex(binding[key], 64):
            raise ValueError("invalid digest in " + key)

    _exact(binding["role_policy"], binding["role_policy_sha256"], "role policy")
    _exact(binding["deployment_gate"], binding["deployment_gate_sha256"], "deployment gate")
    gate = read(binding["deployment_gate"])
    gate_keys = {
        "schema", "look_v22_stage2", "look_v22_receipt", "look_v22_receipt_sha256",
        "look_successor_monitor_job_id",
        "policy_proposal_review", "policy_proposal", "policy_proposal_sha256",
        "accepted_role_policy_sha256", "production_dispatch_authorized", "test_access",
    }
    if (set(gate) != gate_keys
            or gate.get("schema") != "radon_paused_sixth_v5_deployment_gate_v1"
            or gate.get("look_v22_stage2") != "accepted"
            or gate.get("policy_proposal_review") != "accepted"
            or gate.get("accepted_role_policy_sha256") != binding["role_policy_sha256"]
            or gate.get("production_dispatch_authorized") is not True
            or gate.get("test_access") is not False
            or not str(gate.get("look_successor_monitor_job_id", "")).isdigit()):
        raise ValueError("LOOK v22 stage2 and policy proposal must precede R&B production dispatch")
    for label in ("look_v22_receipt_sha256", "policy_proposal_sha256"):
        if not _hex(gate.get(label), 64):
            raise ValueError("deployment gate has an invalid digest")
    _exact(gate["look_v22_receipt"], gate["look_v22_receipt_sha256"], "LOOK stage2 receipt")
    _exact(gate["policy_proposal"], gate["policy_proposal_sha256"], "policy proposal")
    stage2 = read(gate["look_v22_receipt"])
    proposal = read(gate["policy_proposal"])
    if (stage2.get("schema") != "look_v5_monitor_stage2_acceptance_v1"
            or stage2.get("status") != "accepted"
            or stage2.get("successor_monitor_job_id") != gate["look_successor_monitor_job_id"]
            or stage2.get("accepted_role_policy_sha256") != binding["role_policy_sha256"]
            or stage2.get("successor_monitor_identity_verified") is not True
            or stage2.get("predecessor_release_verified") is not True
            or stage2.get("test_access") is not False
            or proposal.get("schema") != "radon_paused_sixth_v5_policy_proposal_v1"
            or proposal.get("state") != "accepted"
            or proposal.get("role_policy_sha256") != binding["role_policy_sha256"]):
        raise ValueError("deployment evidence is not accepted")

    retry = binding["retry_policy"]
    if (set(retry) != {"max_attempts", "retryable_states", "walltime"}
            or type(retry["max_attempts"]) is not int or retry["max_attempts"] < 2
            or retry["retryable_states"] != ["paused", "failed"]
            or retry["walltime"] != "48:00:00"):
        raise ValueError("unknown 48-hour continuation policy")

    pins = binding["runtime_pins"]
    if (not isinstance(pins, list) or not pins
            or any(not isinstance(row, dict) or set(row) != {"path", "sha256"} for row in pins)
            or len({row["path"] for row in pins}) != len(pins)):
        raise ValueError("invalid runtime pins")
    pin_map = {row["path"]: row["sha256"] for row in pins}
    for pinned, digest in pin_map.items():
        if not _hex(digest, 64):
            raise ValueError("invalid runtime pin digest")
        _exact(pinned, digest, "runtime source")

    runtime = binding["runtime_contract"]
    if (set(runtime) != {"schema", "python", "python_launcher_sha256",
                         "python_sha256", "python_realpath",
                         "python_version", "torch_version", "framework_api",
                         "framework_commit", "environment", "modules", "roles"}
            or runtime["schema"] != "radon_v5_runtime_contract_v1"
            or runtime["framework_api"] != "V5"
            or runtime["framework_commit"] != FORMAL_V5_COMMIT
            or set(runtime["environment"]) != ENV_KEYS
            or set(runtime["roles"]) != RUNTIME_ROLES
            or {row.get("name") for row in runtime["modules"]} != MODULE_NAMES):
        raise ValueError("formal V5 runtime contract is incomplete")
    if (not Path(runtime["python"]).is_absolute()
            or runtime["python"] != os.path.abspath(runtime["python"])
            or runtime["python_realpath"] != str(Path(runtime["python"]).resolve(strict=True))
            or python_launcher_sha256(runtime["python"]) != runtime["python_launcher_sha256"]
            or pin_map.get(runtime["python"]) != runtime["python_sha256"]):
        raise ValueError("Python executable is not pinned")
    for role, pinned in runtime["roles"].items():
        if pin_map.get(pinned) is None:
            raise ValueError("runtime role is not pinned: " + role)
    if (runtime["roles"]["allocation_wrapper"] != binding["allocation_wrapper"]
            or runtime["roles"]["finalizer_wrapper"] != binding["finalizer_wrapper"]
            or runtime["roles"]["python_executable"] != runtime["python"]):
        raise ValueError("wrapper/runtime role binding changed")
    modules = {row["name"]: row for row in runtime["modules"]}
    for name, row in modules.items():
        if set(row) != {"name", "path", "sha256"} or pin_map.get(row["path"]) != row["sha256"]:
            raise ValueError("module pin is incomplete: " + name)
    for role, module in ROLE_MODULES.items():
        if runtime["roles"][role] != modules[module]["path"]:
            raise ValueError("runtime role/module mapping changed: " + role)

    allocation = binding["sbatch_command_template"]
    finalizer = binding["finalizer_command_template"]
    if (not isinstance(allocation, list) or allocation[:2] != ["sbatch", "--parsable"]
            or "--hold" not in allocation or "--gres=gpu:a100:1" not in allocation
            or "--time=48:00:00" not in allocation
            or allocation.count("--comment={attempt_comment}") != 1
            or allocation.count(binding["allocation_wrapper"]) != 1
            or not isinstance(finalizer, list) or finalizer[:2] != ["sbatch", "--parsable"]
            or finalizer.count("--comment={finalizer_comment}") != 1
            or sum(item.count("{gpu_job_id}") for item in finalizer) < 1
            or finalizer.count(binding["finalizer_wrapper"]) != 1
            or any("--gres=" in item for item in finalizer)
            or any(not isinstance(item, str) or not item for item in allocation + finalizer)):
        raise ValueError("held allocation/finalizer command contract changed")

    asset = Path(binding["asset_dir"])
    _exact(asset / "asset.json", binding["asset_sha256"], "production asset")
    _exact(asset / "spec.json", binding["v5_spec_sha256"], "V5 spec")
    _exact(asset / "last.pt", binding["v5_checkpoint_sha256"], "V5 checkpoint")
    record = read(asset / "asset.json")
    if (record.get("schema") != "radon_paused_sixth_v5_production_asset_v1"
            or record.get("state") != "accepted_for_new_claim"
            or record.get("run_id") != RUN_ID
            or record.get("framework_commit") != FORMAL_V5_COMMIT
            or record.get("spec_sha256") != binding["v5_spec_sha256"]
            or record.get("checkpoint_sha256") != binding["v5_checkpoint_sha256"]):
        raise ValueError("production asset contract changed")
    expected = binding["expected_claim"]
    if (not isinstance(expected, dict) or expected.get("state") != "claimed"
            or expected.get("owner") != "radon-v5-paused-sixth-migration"
            or expected.get("migration_phase") != "source_reserved"
            or expected.get("run_dir") != str(Path(expected.get("run_dir", "")).resolve())):
        raise ValueError("exact migration reservation is required")
    return binding


def verify_runtime(binding):
    """Verify the running interpreter, environment and every imported runtime module."""
    runtime = binding["runtime_contract"]
    if (os.path.abspath(sys.executable) != runtime["python"]
            or str(Path(sys.executable).resolve(strict=True)) != runtime["python_realpath"]
            or python_launcher_sha256(sys.executable) != runtime["python_launcher_sha256"]
            or sha256(runtime["python"]) != runtime["python_sha256"]
            or platform.python_version() != runtime["python_version"]):
        raise RuntimeError("running Python differs from the immutable runtime")
    for key, value in runtime["environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("runtime environment differs: " + key)
    import torch
    if torch.__version__ != runtime["torch_version"]:
        raise RuntimeError("Torch version differs from the immutable runtime")
    observed = {}
    for row in runtime["modules"]:
        module = importlib.import_module(row["name"])
        path = str(Path(module.__file__).resolve())
        if path != row["path"] or sha256(path) != row["sha256"]:
            raise RuntimeError("runtime module differs: " + row["name"])
        observed[row["name"]] = row["sha256"]
    framework = importlib.import_module("mhd_framework")
    if getattr(framework, "__api_version__", None) != "V5":
        raise RuntimeError("runtime is not formal V5")
    return {"schema": "radon_v5_runtime_observation_v1",
            "contract_sha256": runtime_contract_digest(binding),
            "modules": observed, "python": runtime["python"],
            "python_sha256": runtime["python_sha256"], "framework_api": "V5"}


def _policy_and_journal(binding):
    policy = read(binding["role_policy"])
    if policy.get("schema") != "research_gpu_roles_v1" or policy.get("account_ceiling") != 24:
        raise ValueError("unknown role policy")
    project = policy.get("projects", {}).get("Radon_Bridge")
    if (not isinstance(project, dict) or type(project.get("reserved_gpus")) is not int
            or project["reserved_gpus"] < 1
            or project.get("request_journals", []).count(binding["journal"]) != 1):
        raise ValueError("dedicated R&B journal is not uniquely role-registered")
    journal = read(binding["journal"])
    if journal.get("schema") != "radon_paused_sixth_v5_requests_v2" \
            or not isinstance(journal.get("requests"), list):
        raise ValueError("unknown paused-sixth request journal")
    return policy, project, journal


def live_snapshot(binding):
    """Reconcile all registered journals and the signed legacy ownership overlay."""
    from mhd_models.scheduling.quota_guard import snapshot
    snap = snapshot()
    policy = read(binding["role_policy"])
    owners, unresolved = {}, 0
    unresolved_by_project = {}
    for project, body in policy.get("projects", {}).items():
        for path in body.get("request_journals", []):
            journal = read(path)
            if not isinstance(journal.get("requests"), list):
                raise ValueError("registered project journal is malformed")
            for request in journal["requests"]:
                job, state = str(request.get("job_id", "")), str(request.get("state", "")).lower()
                if job.isdigit() and job in snap["jobs"]:
                    if job in owners:
                        raise ValueError("live GPU job has duplicate ownership")
                    owners[job] = project
                elif not job.isdigit() and state in ACTIVE:
                    count = request.get("requested_gpus", 1)
                    if type(count) is not int or count < 1:
                        raise ValueError("unresolved intent has invalid GPU count")
                    unresolved += count
                    unresolved_by_project[project] = unresolved_by_project.get(project, 0) + count
    overlay_path = binding.get("ownership_overlay")
    if overlay_path:
        overlay = read(overlay_path)
        if (overlay.get("schema") != "legacy_native_ownership_overlay_v2"
                or overlay.get("test") is not False
                or overlay.get("role_policy_sha256") != binding["role_policy_sha256"]):
            raise ValueError("legacy ownership overlay does not bind the current role policy")
        for entry in overlay.get("entries", []):
            job = str(entry.get("job_id", ""))
            if entry.get("classification") != "non_project_legacy_native" or job not in snap["jobs"]:
                raise ValueError("ownership overlay contains an unknown job")
            if job in owners:
                raise ValueError("ownership overlay duplicates project ownership")
            owners[job] = "legacy_native_overlay"
    live = {job for job, row in snap["jobs"].items() if row.get("gpus", 0) > 0}
    if live - set(owners):
        raise ValueError("a live GPU job lacks registered ownership")
    return {"account_limit": min(binding["account_limit"], snap["limit"]),
            "account_running_pending": snap["total_gpus"],
            "radon_running_pending": sum(snap["jobs"][job]["gpus"] for job, owner in owners.items()
                                         if owner == "Radon_Bridge"),
            "unresolved_gpu_intents": unresolved,
            "radon_unresolved_gpu_intents": unresolved_by_project.get("Radon_Bridge", 0)}


def _attempt_comment(binding, attempt):
    return f'{binding["comment_prefix"]}-a{attempt:03d}'


def _render(template, replacements):
    result = []
    for item in template:
        for key, value in replacements.items():
            item = item.replace("{" + key + "}", str(value))
        if "{" in item or "}" in item:
            raise ValueError("unresolved command placeholder")
        result.append(item)
    return result


def _persist(binding, journal, intent, row):
    matches = [index for index, value in enumerate(journal["requests"])
               if value.get("attempt_id") == row["attempt_id"]]
    if len(matches) != 1:
        raise RuntimeError("attempt journal identity changed")
    journal["requests"][matches[0]] = dict(row)
    intent["transaction"] = dict(row) if row["state"] in ACTIVE else None
    write(binding["journal"], journal)
    write(binding["intent"], intent)


def _validate_continuation(binding, row):
    continuation = row.get("continuation")
    if (row.get("state") not in RETRYABLE or not isinstance(continuation, dict)
            or continuation.get("schema") != "radon_paused_sixth_v5_continuation_v1"
            or continuation.get("execution_dir") != str(Path(binding["execution_dir"]).resolve())
            or continuation.get("spec_sha256") != binding["v5_spec_sha256"]
            or not _hex(continuation.get("checkpoint_sha256"), 64)
            or set(continuation.get("progress", {})) != {"epoch", "offset", "updates"}
            or any(type(value) is not int or value < 0
                   for value in continuation.get("progress", {}).values())
            or continuation.get("test_access") is not False):
        raise ValueError("terminal attempt lacks an exact resumable V5 continuation")
    _exact(Path(binding["execution_dir"]) / "last.pt", continuation["checkpoint_sha256"],
           "continuation checkpoint")
    _exact(Path(binding["execution_dir"]) / "spec.json", continuation["spec_sha256"],
           "continuation spec")
    return continuation


def _new_attempt(binding, journal, snapshot, project, now):
    rows = journal["requests"]
    if (len({row.get("attempt_id") for row in rows}) != len(rows)
            or any(row.get("lease_id") != binding["lease_id"]
                   or row.get("attempt") != index
                   or row.get("attempt_id") != f'{binding["lease_id"]}/attempt-{index:03d}'
                   for index, row in enumerate(rows, 1))):
        raise RuntimeError("attempt journal sequence changed")
    if rows and any(row.get("state") in ACTIVE for row in rows):
        active = [row for row in rows if row.get("state") in ACTIVE]
        if len(active) != 1:
            raise RuntimeError("multiple active attempts")
        return dict(active[0]), False
    if rows and rows[-1].get("state") == "completed":
        return {"state": "completed"}, False
    if rows:
        _validate_continuation(binding, rows[-1])
    attempt = len(rows) + 1
    if attempt > binding["retry_policy"]["max_attempts"]:
        return {"state": "attempt_limit_reached"}, False
    occupied = snapshot["account_running_pending"] + snapshot["unresolved_gpu_intents"]
    if snapshot["account_limit"] != binding["account_limit"]:
        raise ValueError("account limit differs from binding")
    if occupied + 1 > binding["account_limit"]:
        return {"state": "waiting_account_capacity"}, False
    if (snapshot["radon_running_pending"] + snapshot["radon_unresolved_gpu_intents"] + 1
            > project["reserved_gpus"]):
        return {"state": "waiting_radon_role_capacity"}, False
    comment = _attempt_comment(binding, attempt)
    row = {
        "attempt_id": f'{binding["lease_id"]}/attempt-{attempt:03d}',
        "lease_id": binding["lease_id"], "attempt": attempt, "state": "preparing",
        "attempt_comment": comment, "finalizer_comment": comment + "-finalizer",
        "requested_gpus": 1, "binding_sha256": binding["binding_sha256"],
        "role_policy_sha256": binding["role_policy_sha256"],
        "production_asset_sha256": binding["asset_sha256"],
        "run_dir": binding["expected_claim"]["run_dir"],
        "execution_dir": str(Path(binding["execution_dir"]).resolve()), "created_at": now,
    }
    journal["requests"].append(dict(row))
    return row, True


def _unique_job(lookup, comment, *, kind):
    result = lookup(comment)
    if isinstance(result, list):
        rows, absence_proven = result, False
    elif (isinstance(result, dict) and set(result) == {"jobs", "absence_proven"}
          and isinstance(result["jobs"], list)
          and isinstance(result["absence_proven"], bool)):
        rows, absence_proven = result["jobs"], result["absence_proven"]
    else:
        raise ValueError("invalid Slurm recovery observation")
    if len(rows) > 1:
        raise RuntimeError("Slurm comment is not unique: " + comment)
    if not rows:
        return None, absence_proven
    row = rows[0]
    expected_gpus = 1 if kind == "allocation" else 0
    if (row.get("comment") != comment or row.get("account") != "pi-mengy"
            or row.get("gpus") != expected_gpus or not str(row.get("job_id", "")).isdigit()):
        raise ValueError("recovered Slurm job identity changed")
    return row, absence_proven


def _may_resubmit_unknown(row, *, prefix, absence_proven, now):
    unknown_at = row.get(prefix + "_ack_unknown_at")
    generations = row.get(prefix + "_submission_generations", 0)
    if (type(unknown_at) not in (int, float) or unknown_at > now
            or type(generations) is not int or generations < 1):
        raise RuntimeError(prefix + " acknowledgement recovery metadata changed")
    return absence_proven and now - unknown_at >= ACK_RECOVERY_GRACE_SECONDS


def _dependency_matches(value, job_id):
    value = str(value or "").replace("(unfulfilled)", "")
    return value == "afterany:" + str(job_id)


def _finalizer_proof_path(binding, row):
    identity = hashlib.sha256(row["attempt_id"].encode()).hexdigest()
    return Path(binding["intent"]).parent / "finalizer_proofs" / (identity + ".json")


def _load_finalizer_proof(binding, row):
    path = row.get("finalizer_pending_proof_path")
    digest = row.get("finalizer_pending_proof_sha256")
    if (path != str(_finalizer_proof_path(binding, row)) or not _hex(digest, 64)
            or not Path(path).is_file() or sha256(path) != digest):
        raise RuntimeError("immutable pending-finalizer proof is missing or changed")
    proof = read(path)
    if (proof.get("schema") != "radon_paused_sixth_v5_pending_finalizer_proof_v1"
            or proof.get("binding_sha256") != binding["binding_sha256"]
            or proof.get("attempt_id") != row["attempt_id"]
            or proof.get("allocation_job_id") != row["job_id"]
            or proof.get("finalizer_job_id") != row["finalizer_job_id"]
            or proof.get("account") != "pi-mengy" or proof.get("gpus") != 0
            or proof.get("comment") != row["finalizer_comment"]
            or proof.get("state") != "PENDING"
            or not _dependency_matches(proof.get("dependency"), row["job_id"])):
        raise RuntimeError("immutable pending-finalizer proof identity changed")
    return proof


def _persist_finalizer_proof(binding, row, finalizer, now):
    if (finalizer.get("state") != "PENDING" or finalizer.get("account") != "pi-mengy"
            or finalizer.get("gpus") != 0 or finalizer.get("comment") != row["finalizer_comment"]
            or not _dependency_matches(finalizer.get("dependency"), row["job_id"])):
        raise RuntimeError("pending afterany finalizer identity is unproven")
    path = _finalizer_proof_path(binding, row)
    proof = {"schema": "radon_paused_sixth_v5_pending_finalizer_proof_v1",
             "binding_sha256": binding["binding_sha256"], "attempt_id": row["attempt_id"],
             "allocation_job_id": row["job_id"], "finalizer_job_id": finalizer["job_id"],
             "account": finalizer["account"], "gpus": finalizer["gpus"],
             "comment": finalizer["comment"], "state": finalizer["state"],
             "dependency": finalizer["dependency"], "observed_at": now}
    try:
        write_once(path, proof)
    except FileExistsError:
        # The proof is intentionally immutable.  Recovery may observe the same
        # transaction at a later wall-clock time, so validate the existing
        # identity instead of comparing a newly generated ``observed_at``.
        row["finalizer_pending_proof_path"] = str(path)
        row["finalizer_pending_proof_sha256"] = sha256(path)
        return _load_finalizer_proof(binding, row)
    row["finalizer_pending_proof_path"] = str(path)
    row["finalizer_pending_proof_sha256"] = sha256(path)
    return _load_finalizer_proof(binding, row)


def _validate_finalizer_runtime_identity(row, proof, observed, actual_finalizer, gpu_job_id):
    live_dependency = observed.get("dependency")
    dependency_cleared = live_dependency in {None, "", "(null)"}
    if (proof.get("finalizer_job_id") != actual_finalizer
            or proof.get("allocation_job_id") != str(gpu_job_id)
            or observed.get("job_id") != actual_finalizer or observed.get("state") != "RUNNING"
            or observed.get("account") != "pi-mengy" or observed.get("gpus") != 0
            or observed.get("comment") != row["finalizer_comment"]
            or (not dependency_cleared and not _dependency_matches(live_dependency, gpu_job_id))):
        raise ValueError("zero-GPU afterany finalizer identity is unproven")


def publish_once(binding_path, *, snapshot, submit, lookup, release,
                 runtime_verify=verify_runtime, now=None):
    """Recover or advance one held-first transaction without duplicate submission."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    runtime_verify(binding)
    binding["binding_sha256"] = sha256(binding_path)
    try:
        handle = locked_existing(binding["account_lock"], nonblocking=True)
    except BlockingIOError:
        return {"action": "none", "state": "waiting_account_lock"}
    with handle:
        binding = validate_binding(binding_path, now=now)
        binding["binding_sha256"] = sha256(binding_path)
        _, project, journal = _policy_and_journal(binding)
        intent = read(binding["intent"])
        if intent.get("schema") != "radon_paused_sixth_v5_intent_v2" \
                or set(intent) != {"schema", "transaction"}:
            raise ValueError("unknown paused-sixth transaction intent")
        if not journal["requests"]:
            _exact(binding["journal"], binding["journal_initial_sha256"], "initial request journal")
            _exact(binding["intent"], binding["intent_initial_sha256"], "initial transaction intent")
        active = [row for row in journal["requests"] if row.get("state") in ACTIVE]
        if len(active) > 1:
            raise RuntimeError("multiple active attempts")
        if active and active[0].get("state") in {
                "preparing", "allocation_ack_unknown", "allocation_ack_pending"}:
            recovered, _ = _unique_job(lookup, active[0]["attempt_comment"], kind="allocation")
            if recovered is not None:
                row = active[0]
                if (row.get("allocation_job_id") not in (None, recovered["job_id"])
                        or recovered.get("state") not in {"PENDING", "RUNNING"}
                        or recovered.get("held") is not True):
                    raise RuntimeError("pre-snapshot allocation recovery identity changed")
                row.update(state="held", allocation_job_id=recovered["job_id"],
                           job_id=recovered["job_id"], held_observed_at=now,
                           recovered_before_live_snapshot=True)
                _persist(binding, journal, intent, row)
                journal = read(binding["journal"])
        current = snapshot()
        for key in ("account_limit", "account_running_pending", "radon_running_pending",
                    "unresolved_gpu_intents", "radon_unresolved_gpu_intents"):
            if type(current.get(key)) is not int or current[key] < 0:
                raise ValueError("invalid reconciled capacity snapshot")
        row, created = _new_attempt(binding, journal, current, project, now)
        if row.get("state") not in ACTIVE:
            return {"action": "none", "state": row["state"]}
        if created:
            _persist(binding, journal, intent, row)  # durable before any sbatch
        elif intent.get("transaction", {}).get("attempt_id") != row["attempt_id"]:
            intent["transaction"] = dict(row)
            write(binding["intent"], intent)

        # An allocation that is already released or running belongs to its
        # allocation owner/finalizer.  The publisher must never regress it to
        # a pre-release state during reconciliation.
        if row["state"] in {"submitted", "granted", "running"}:
            return {"action": "none", "state": row["state"],
                    "job_id": row.get("job_id"),
                    "finalizer_job_id": row.get("finalizer_job_id"),
                    "attempt": row["attempt"]}

        allocation, allocation_absent = _unique_job(
            lookup, row["attempt_comment"], kind="allocation")
        if allocation is None:
            if row["state"] == "allocation_ack_pending":
                pending_at = row.get("allocation_ack_pending_at")
                if (allocation_absent and type(pending_at) in (int, float)
                        and now - pending_at >= ACK_RECOVERY_GRACE_SECONDS):
                    row.update(state="identity_drift",
                               identity_error="acknowledged allocation is absent from squeue and sacct")
                    _persist(binding, journal, intent, row)
                    return {"action": "none", "state": row["state"]}
                return {"action": "none", "state": row["state"],
                        "job_id": row.get("allocation_job_id")}
            if (row["state"] == "allocation_ack_unknown"
                    and not _may_resubmit_unknown(
                        row, prefix="allocation", absence_proven=allocation_absent, now=now)):
                return {"action": "none", "state": row["state"]}
            if row["state"] not in {"preparing", "allocation_ack_unknown"}:
                raise RuntimeError("allocation disappeared after durable acknowledgement")
            command = _render(binding["sbatch_command_template"],
                              {"attempt_comment": row["attempt_comment"]})
            row.update(state="preparing", allocation_submit_started_at=now,
                       allocation_submission_generations=
                       row.get("allocation_submission_generations", 0) + 1)
            _persist(binding, journal, intent, row)
            try:
                returned = str(submit(command, timeout=20))
            except BaseException as error:
                row.update(state="allocation_ack_unknown", error_type=type(error).__name__,
                           allocation_ack_unknown_at=now)
                _persist(binding, journal, intent, row)
                return {"action": "none", "state": row["state"]}
            if not returned.isdigit():
                row.update(state="identity_drift", observed_job_id=returned)
                _persist(binding, journal, intent, row)
                return {"action": "none", "state": row["state"]}
            row.update(state="allocation_ack_pending", allocation_job_id=returned,
                       allocation_ack_pending_at=now)
            _persist(binding, journal, intent, row)
            allocation, _ = _unique_job(lookup, row["attempt_comment"], kind="allocation")
            if allocation is None:
                return {"action": "none", "state": "allocation_ack_pending", "job_id": returned}
        if row.get("allocation_job_id") not in (None, allocation["job_id"]):
            raise RuntimeError("sbatch acknowledgement conflicts with recovered allocation")
        if allocation.get("state") not in {"PENDING", "RUNNING"}:
            raise ValueError("held allocation is not live")
        if row.get("state") not in {"ready_release", "release_ack_unknown", "release_ack_pending",
                                    "submitted", "granted", "running"} and not allocation.get("held"):
            raise RuntimeError("allocation ran before journal and finalizer were durable")
        if row["state"] in {"preparing", "allocation_ack_unknown", "allocation_ack_pending"}:
            row.update(state="held", allocation_job_id=allocation["job_id"],
                       job_id=allocation["job_id"], held_observed_at=now)
            _persist(binding, journal, intent, row)

        finalizer, finalizer_absent = _unique_job(
            lookup, row["finalizer_comment"], kind="finalizer")
        if finalizer is None:
            if row["state"] == "finalizer_ack_pending":
                pending_at = row.get("finalizer_ack_pending_at")
                if (finalizer_absent and type(pending_at) in (int, float)
                        and now - pending_at >= ACK_RECOVERY_GRACE_SECONDS):
                    row.update(state="identity_drift",
                               identity_error="acknowledged finalizer is absent from squeue and sacct")
                    _persist(binding, journal, intent, row)
                    return {"action": "none", "state": row["state"]}
                return {"action": "none", "state": row["state"],
                        "job_id": row["job_id"],
                        "finalizer_job_id": row.get("finalizer_job_id")}
            if (row["state"] == "finalizer_ack_unknown"
                    and not _may_resubmit_unknown(
                        row, prefix="finalizer", absence_proven=finalizer_absent, now=now)):
                return {"action": "none", "state": row["state"], "job_id": row["job_id"]}
            if row["state"] not in {"held", "finalizer_ack_unknown"}:
                raise RuntimeError("finalizer disappeared after durable acknowledgement")
            command = _render(binding["finalizer_command_template"], {
                "gpu_job_id": row["job_id"], "finalizer_comment": row["finalizer_comment"]})
            row.update(state="held", finalizer_submit_started_at=now,
                       finalizer_submission_generations=
                       row.get("finalizer_submission_generations", 0) + 1)
            _persist(binding, journal, intent, row)
            try:
                returned = str(submit(command, timeout=20))
            except BaseException as error:
                row.update(state="finalizer_ack_unknown", error_type=type(error).__name__,
                           finalizer_ack_unknown_at=now)
                _persist(binding, journal, intent, row)
                return {"action": "none", "state": row["state"], "job_id": row["job_id"]}
            if not returned.isdigit():
                row.update(state="identity_drift", observed_finalizer_job_id=returned)
                _persist(binding, journal, intent, row)
                return {"action": "none", "state": row["state"]}
            row.update(state="finalizer_ack_pending", finalizer_job_id=returned,
                       finalizer_ack_pending_at=now)
            _persist(binding, journal, intent, row)
            finalizer, _ = _unique_job(lookup, row["finalizer_comment"], kind="finalizer")
            if finalizer is None:
                return {"action": "none", "state": "finalizer_ack_pending",
                        "job_id": row["job_id"], "finalizer_job_id": returned}
        if row.get("finalizer_job_id") not in (None, finalizer["job_id"]):
            raise RuntimeError("finalizer acknowledgement/dependency identity changed")
        if row["state"] in {"held", "finalizer_ack_unknown", "finalizer_ack_pending"}:
            row["finalizer_job_id"] = finalizer["job_id"]
            _persist_finalizer_proof(binding, row, finalizer, now)
            row.update(state="ready_release", finalizer_job_id=finalizer["job_id"],
                       release_ready_at=now)
            _persist(binding, journal, intent, row)

        # Reconciliation may resume after ``ready_release``.  Refuse to
        # release unless the immutable PENDING proof created before the
        # predecessor release is still exact.
        _load_finalizer_proof(binding, row)

        allocation, _ = _unique_job(lookup, row["attempt_comment"], kind="allocation")
        if allocation is None:
            raise RuntimeError("held allocation disappeared before release")
        if allocation.get("held"):
            row["state"] = "release_ack_pending"
            _persist(binding, journal, intent, row)
            try:
                release(row["job_id"])
            except BaseException as error:
                row.update(state="release_ack_unknown", error_type=type(error).__name__)
                _persist(binding, journal, intent, row)
                return {"action": "none", "state": row["state"], "job_id": row["job_id"]}
            allocation, _ = _unique_job(lookup, row["attempt_comment"], kind="allocation")
            if allocation is None or allocation.get("held"):
                return {"action": "none", "state": "release_ack_pending", "job_id": row["job_id"]}
        row.update(state="submitted", submitted_at=now)
        _persist(binding, journal, intent, row)
        return {"action": "none", "state": "submitted", "job_id": row["job_id"],
                "finalizer_job_id": row["finalizer_job_id"], "attempt": row["attempt"]}


def grant_allocation(binding_path, *, job_id, observe_allocation,
                     runtime_verify=verify_runtime, now=None):
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    binding["binding_sha256"] = sha256(binding_path)
    runtime_verify(binding)
    with locked_existing(binding["account_lock"]):
        validate_binding(binding_path, now=now)
        _, _, journal = _policy_and_journal(binding)
        rows = [row for row in journal["requests"] if row.get("job_id") == str(job_id)]
        if len(rows) != 1 or rows[0].get("state") not in {"submitted", "granted", "release_ack_pending", "release_ack_unknown"}:
            raise ValueError("submitted V5 request identity changed")
        row = rows[0]
        _load_finalizer_proof(binding, row)
        observed = observe_allocation(str(job_id))
        if (observed.get("job_id") != str(job_id) or observed.get("state") != "RUNNING"
                or observed.get("account") != "pi-mengy" or observed.get("gpus") != 1
                or observed.get("held") is not False
                or observed.get("comment") != row["attempt_comment"]):
            raise ValueError("Slurm allocation grant identity is unproven")
        if row["state"] != "granted":
            row.update(state="granted", granted_at=now)
            intent = read(binding["intent"])
            _persist(binding, journal, intent, row)
        return row


def prepare_execution(binding, row, *, now=None):
    now = time.time() if now is None else now
    execution, asset = Path(binding["execution_dir"]), Path(binding["asset_dir"])
    if execution.exists():
        manifest = read(execution / "migration.json")
        if (manifest.get("binding_sha256") != binding["binding_sha256"]
                or manifest.get("spec_sha256") != binding["v5_spec_sha256"]
                or sha256(execution / "spec.json") != binding["v5_spec_sha256"]):
            raise RuntimeError("existing V5 execution revision conflicts")
        if row["attempt"] == 1:
            _exact(execution / "last.pt", binding["v5_checkpoint_sha256"], "initial V5 checkpoint")
        else:
            _validate_continuation(binding, row["previous_attempt"])
        return manifest
    if row["attempt"] != 1:
        raise RuntimeError("continuation execution directory disappeared")
    execution.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=execution.name + ".", dir=execution.parent))
    try:
        for name in ("last.pt", "spec.json"):
            (stage / name).write_bytes((asset / name).read_bytes())
        manifest = {"schema": "radon_paused_sixth_v5_execution_revision_v2",
                    "run_id": RUN_ID, "framework_commit": FORMAL_V5_COMMIT,
                    "binding_sha256": binding["binding_sha256"],
                    "source_v4_run": binding["expected_claim"]["run_dir"],
                    "production_asset_sha256": binding["asset_sha256"],
                    "spec_sha256": binding["v5_spec_sha256"],
                    "initial_checkpoint_sha256": binding["v5_checkpoint_sha256"],
                    "created_at": now, "test_access": False}
        write(stage / "migration.json", manifest)
        os.rename(stage, execution)
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage)
    return manifest


def _expected_claim(binding, journal, row):
    if row["attempt"] == 1:
        return binding["expected_claim"]
    previous = journal["requests"][row["attempt"] - 2]
    if previous.get("state") not in RETRYABLE or not isinstance(previous.get("terminal_claim"), dict):
        raise RuntimeError("previous V5 attempt did not publish a terminal claim")
    return previous["terminal_claim"]


def attest_process(*, pid, job_id, step, expected_argv, binding, proc_root="/proc"):
    """Parent-side process proof; no command field is trusted from the worker."""
    root = Path(proc_root) / str(pid)
    argv = (root / "cmdline").read_bytes().split(b"\0")
    if argv and argv[-1] == b"":
        argv.pop()
    observed_argv = [item.decode() for item in argv]
    if observed_argv != expected_argv:
        raise ValueError("/proc command differs from parent-fixed worker argv")
    executable = str(Path(os.readlink(root / "exe")).resolve())
    if executable != binding["runtime_contract"]["python_realpath"]:
        raise ValueError("worker executable differs from pinned Python")
    cgroup = (root / "cgroup").read_text()
    if (re.search(rf"(?:job[_-]?){re.escape(str(job_id))}(?:/|$)", cgroup) is None
            or re.search(rf"(?:step[_-]?){re.escape(str(step))}(?:/|$)", cgroup) is None):
        raise ValueError("worker PID is not in the observed Slurm job/step cgroup")
    environ = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environ[key.decode()] = value.decode()
    expected_env = binding["runtime_contract"]["environment"]
    if any(environ.get(key) != value for key, value in expected_env.items()):
        raise ValueError("worker process environment differs from pinned runtime")
    return {"schema": "radon_v5_parent_process_attestation_v1", "pid": int(pid),
            "job_id": str(job_id), "step": str(step),
            "argv_sha256": _digest_json(observed_argv), "executable": executable,
            "cgroup_sha256": hashlib.sha256(cgroup.encode()).hexdigest(),
            "environment_sha256": _digest_json({key: environ[key] for key in sorted(expected_env)})}


def claim_running_step(binding_path, *, claims, observe_job, attest, step_record,
                       owner_contract, expected_owner_contract, now=None):
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    binding["binding_sha256"] = sha256(binding_path)
    record, contract = read(step_record), read(owner_contract)
    if contract != expected_owner_contract:
        raise ValueError("worker changed the parent owner contract")
    if (contract.get("schema") != "radon_paused_sixth_v5_owner_contract_v1"
            or contract.get("binding_sha256") != binding["binding_sha256"]
            or contract.get("runtime_contract_sha256") != runtime_contract_digest(binding)
            or contract.get("argv_sha256") != _digest_json(contract.get("argv"))
            or not _hex(contract.get("nonce"), 64)
            or record.get("schema") != "radon_paused_sixth_v5_step_v2"
            or record.get("owner_contract_sha256") != sha256(owner_contract)
            or record.get("nonce") != contract.get("nonce")
            or record.get("job_id") != contract.get("job_id")
            or not str(record.get("step", "")) or type(record.get("pid")) is not int):
        raise ValueError("worker record is not bound to the parent owner contract")
    with locked_existing(binding["account_lock"]):
        validate_binding(binding_path, now=now)
        _, _, journal = _policy_and_journal(binding)
        matches = [row for row in journal["requests"] if row.get("attempt_id") == contract.get("attempt_id")]
        if (len(matches) != 1 or matches[0].get("job_id") != record["job_id"]
                or matches[0].get("state") != "granted"):
            raise ValueError("granted attempt identity is missing")
        row = matches[0]
        observed = observe_job(record["job_id"], str(record["step"]))
        if (observed.get("job_id") != record["job_id"] or observed.get("step") != str(record["step"])
                or observed.get("job_state") != "RUNNING" or observed.get("step_state") != "RUNNING"
                or observed.get("account") != "pi-mengy" or observed.get("gpus") != 1):
            raise ValueError("independent Slurm allocation/step identity is unproven")
        process = attest(pid=record["pid"], job_id=record["job_id"], step=str(record["step"]),
                         expected_argv=contract["argv"], binding=binding)
        if process.get("schema") != "radon_v5_parent_process_attestation_v1":
            raise ValueError("parent process attestation is invalid")
        if row["attempt"] > 1:
            row["previous_attempt"] = journal["requests"][row["attempt"] - 2]
        prepare_execution(binding, row, now=now)
        expected = _expected_claim(binding, journal, row)
        def transition(old):
            if old != expected:
                raise RuntimeError("paused-sixth claim changed before V5 step binding")
            return {**old, "state": "running", "owner": "radon-v5-production-" + record["job_id"],
                    "job_id": record["job_id"], "step": str(record["step"]),
                    "generation": old["generation"] + 1, "updated_at": now,
                    "spec_sha256": binding["v5_spec_sha256"], "framework_commit": FORMAL_V5_COMMIT,
                    "production_asset_sha256": binding["asset_sha256"],
                    "execution_dir": str(Path(binding["execution_dir"]).resolve()),
                    "binding_sha256": binding["binding_sha256"],
                    "owner_contract_sha256": sha256(owner_contract),
                    "process_attestation_sha256": _digest_json(process),
                    "previous": {key: old.get(key) for key in
                                 ("state", "owner", "job_id", "step", "generation", "spec_sha256")}}
        claim = claims.mutate(expected["run_dir"], transition)
        row.update(state="running", step=str(record["step"]), running_at=now,
                   owner_contract_sha256=sha256(owner_contract),
                   process_attestation=process, claim_generation=claim["generation"])
        intent = read(binding["intent"])
        _persist(binding, journal, intent, row)
        return claim


def _continuation(binding, state):
    execution = Path(binding["execution_dir"])
    status = read(execution / "status.json")
    if (state not in RETRYABLE or status.get("state") != state
            or status.get("test_used") is not False
            or not (execution / "last.pt").is_file() or not (execution / "spec.json").is_file()
            or sha256(execution / "spec.json") != binding["v5_spec_sha256"]):
        raise ValueError("terminal worker did not publish a safe resumable V5 boundary")
    checkpoint = __import__("torch").load(execution / "last.pt", map_location="cpu", weights_only=False)
    progress = {key: checkpoint.get("progress", {}).get(key) for key in ("epoch", "offset", "updates")}
    if (checkpoint.get("schema") != "optimizer_boundary_v2"
            or checkpoint.get("framework_api") != "V5"
            or progress != {key: status.get(key) for key in ("epoch", "offset", "updates")}
            or any(type(value) is not int or value < 0 for value in progress.values())):
        raise ValueError("terminal checkpoint/status boundary is inconsistent")
    return {"schema": "radon_paused_sixth_v5_continuation_v1",
            "execution_dir": str(execution.resolve()), "checkpoint_sha256": sha256(execution / "last.pt"),
            "spec_sha256": binding["v5_spec_sha256"], "progress": progress, "test_access": False}


def worker(binding_path, step_record, gate, owner_contract, nonce):
    binding = validate_binding(binding_path)
    runtime = verify_runtime(binding)
    contract = read(owner_contract)
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    if (not job.isdigit() or not step or contract.get("job_id") != job
            or contract.get("nonce") != nonce
            or contract.get("binding_sha256") != sha256(binding_path)
            or contract.get("runtime_contract_sha256") != runtime["contract_sha256"]):
        raise ValueError("worker lacks its exact parent allocation contract")
    record = {"schema": "radon_paused_sixth_v5_step_v2", "job_id": job, "step": step,
              "pid": os.getpid(), "nonce": nonce, "owner_contract_sha256": sha256(owner_contract),
              "runtime_observation_sha256": _digest_json(runtime), "time": time.time()}
    write(step_record, record)
    deadline = time.monotonic() + 300
    while not Path(gate).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("V5 claim gate was not published")
        time.sleep(1)
    gate_value = read(gate)
    if (gate_value.get("schema") != "radon_paused_sixth_v5_claim_gate_v2"
            or gate_value.get("job_id") != job or gate_value.get("step") != step
            or gate_value.get("nonce") != nonce
            or gate_value.get("owner_contract_sha256") != sha256(owner_contract)):
        raise ValueError("V5 claim gate identity changed")
    from mhd_models.workflows.native import run
    return run(str(Path(binding["execution_dir"]) / "spec.json"), binding["execution_dir"], mode="train")


def _fields(text):
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def _gpu_count(fields):
    tres = fields.get("ReqTRES", "") + "," + fields.get("AllocTRES", "")
    return max([int(match.group(1)) for match in
                (re.fullmatch(r"gres/gpu(?::[^=,]+)?=(\d+)", item) for item in tres.split(","))
                if match] or [0])


def observe_running_step(job_id, step_id):
    job = _fields(subprocess.check_output(["scontrol", "show", "job", str(job_id), "-o"],
                                          text=True, timeout=20))
    step = _fields(subprocess.check_output(["scontrol", "show", "step", f"{job_id}.{step_id}", "-o"],
                                           text=True, timeout=20))
    return {"job_id": str(job.get("JobId", "")), "step": str(step.get("StepId", "")).split(".", 1)[-1],
            "job_state": job.get("JobState"), "step_state": step.get("State"),
            "account": job.get("Account"), "gpus": _gpu_count(job)}


def observe_running_allocation(job_id):
    fields = _fields(subprocess.check_output(["scontrol", "show", "job", str(job_id), "-o"],
                                             text=True, timeout=20))
    return {"job_id": str(fields.get("JobId", "")), "state": fields.get("JobState"),
            "account": fields.get("Account"), "gpus": _gpu_count(fields),
            "comment": fields.get("Comment"), "held": fields.get("Reason") == "JobHeldUser"}


def allocation_owner(binding_path, *, popen=subprocess.Popen, process_attestor=attest_process, now=None):
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    runtime = verify_runtime(binding)
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise ValueError("allocation owner lacks numeric Slurm job identity")
    row = grant_allocation(binding_path, job_id=job, observe_allocation=observe_running_allocation, now=now)
    root = Path(binding["execution_dir"]).parent / f'owner_{job}_a{row["attempt"]:03d}'
    root.mkdir(parents=True, exist_ok=True)
    step_record, gate, contract_path = root / "step.json", root / "claim_gate.json", root / "owner.json"
    nonce = secrets.token_hex(32)
    argv = [binding["runtime_contract"]["python"], "-m",
            "radon_bridge.runtime.paused_sixth_v5_control", "--binding", str(Path(binding_path).resolve()),
            "--worker", "--step-record", str(step_record.resolve()), "--gate", str(gate.resolve()),
            "--owner-contract", str(contract_path.resolve()), "--nonce", nonce]
    contract = {"schema": "radon_paused_sixth_v5_owner_contract_v1",
                "binding_sha256": sha256(binding_path), "runtime_contract_sha256": runtime["contract_sha256"],
                "attempt_id": row["attempt_id"], "job_id": job, "nonce": nonce, "argv": argv,
                "argv_sha256": _digest_json(argv), "created_at": now}
    write(contract_path, contract)
    command = ["srun", "--jobid=" + job, "--overlap", "--exact", "--nodes=1", "--ntasks=1",
               "--gpus=1", "--cpus-per-task=" + str(binding["worker_cpus"]),
               "--mem=" + str(binding["worker_memory_gib"]) + "G", "--unbuffered", *argv]
    log = (root / "worker.log").open("x")
    child = popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 300
        while not step_record.exists():
            if child.poll() is not None:
                raise RuntimeError("V5 worker exited before publishing its step record")
            if time.monotonic() >= deadline:
                raise TimeoutError("V5 worker step record timed out")
            time.sleep(1)
        record = read(step_record)
        from mhd_models.scheduling.policy import Claims
        claims = Claims(binding["claims_root"])
        claim = claim_running_step(
            binding_path, claims=claims, step_record=step_record, owner_contract=contract_path,
            expected_owner_contract=contract, observe_job=observe_running_step,
            attest=process_attestor, now=now)
        write(gate, {"schema": "radon_paused_sixth_v5_claim_gate_v2", "job_id": job,
                     "step": record["step"], "nonce": nonce,
                     "owner_contract_sha256": sha256(contract_path),
                     "claim_generation": claim["generation"], "time": time.time()})
        code = child.wait()
        from mhd_models.scheduling.slurm_liveness import step_presence
        if step_presence(job, str(record["step"])) is not False:
            claims.update(binding["expected_claim"]["run_dir"], claim["owner"],
                          state="liveness_needs_review")
            raise RuntimeError("V5 worker step death is unproven")
        status = read(Path(binding["execution_dir"]) / "status.json")
        if code == 0 and status.get("state") == "completed":
            from mhd_models.runtime.training_state import verify_completion
            verify_completion(binding["execution_dir"], read(Path(binding["execution_dir"]) / "spec.json"))
            state, continuation = "completed", None
        elif status.get("state") in RETRYABLE:
            state, continuation = status["state"], _continuation(binding, status["state"])
        else:
            state, continuation = "needs_review", None
        if state in {"completed", "paused", "failed"}:
            terminal_claim = claims.release(binding["expected_claim"]["run_dir"], claim["owner"],
                                            state, step_dead=True)
        else:
            terminal_claim = claims.update(binding["expected_claim"]["run_dir"], claim["owner"],
                                           state="liveness_needs_review")
        with locked_existing(binding["account_lock"]):
            _, _, journal = _policy_and_journal(binding)
            matches = [item for item in journal["requests"] if item.get("attempt_id") == row["attempt_id"]]
            if len(matches) != 1 or matches[0].get("state") != "running":
                raise RuntimeError("R&B V5 journal changed before terminal publication")
            matches[0].update(state=state, exit_code=code, finished_at=time.time(),
                              accepted=(state == "completed"), terminal_claim=terminal_claim)
            if continuation is not None:
                matches[0]["continuation"] = continuation
            intent = read(binding["intent"])
            _persist(binding, journal, intent, matches[0])
        return code if state != "needs_review" else 1
    finally:
        log.close()


def finalize_afterany(binding_path, *, gpu_job_id, observe_terminal, observe_finalizer,
                      claims_factory=None, finalizer_job_id=None,
                      runtime_verify=verify_runtime, now=None):
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now, allow_expired=True)
    runtime_verify(binding)
    binding["binding_sha256"] = sha256(binding_path)
    actual_finalizer = os.environ.get("SLURM_JOB_ID", "") if finalizer_job_id is None else str(finalizer_job_id)
    if not actual_finalizer.isdigit():
        raise ValueError("finalizer lacks its own Slurm job identity")
    with locked_existing(binding["account_lock"]):
        _, _, journal = _policy_and_journal(binding)
        rows = [row for row in journal["requests"] if row.get("job_id") == str(gpu_job_id)]
        if len(rows) != 1 or rows[0].get("finalizer_job_id") != actual_finalizer:
            raise ValueError("finalizer journal identity changed")
        row = rows[0]
        proof = _load_finalizer_proof(binding, row)
        finalizer = observe_finalizer(actual_finalizer)
        _validate_finalizer_runtime_identity(
            row, proof, finalizer, actual_finalizer, str(gpu_job_id))
        if row.get("state") in TERMINAL:
            return row
        terminal = observe_terminal(str(gpu_job_id))
        if (terminal.get("job_id") != str(gpu_job_id)
                or terminal.get("state") not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                                                  "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
                or not isinstance(terminal.get("exit_code"), str)):
            raise ValueError("GPU allocation terminal identity is unproven")
        if claims_factory is None:
            from mhd_models.scheduling.policy import Claims
            claims_factory = Claims
        claims = claims_factory(binding["claims_root"])
        claim = read(claims.path(binding["expected_claim"]["run_dir"]))
        expected = _expected_claim(binding, journal, row)
        owner = "radon-v5-production-" + str(gpu_job_id)
        execution = Path(binding["execution_dir"])
        status = read(execution / "status.json") if (execution / "status.json").exists() else {}
        continuation, accepted = None, False
        if (terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
                and status.get("state") == "completed" and (execution / "accepted.json").is_file()):
            from mhd_models.runtime.training_state import verify_completion
            verify_completion(execution, read(execution / "spec.json"))
            state, accepted = "completed", True
        elif status.get("state") in RETRYABLE:
            state = status["state"]
            continuation = _continuation(binding, state)
        else:
            state = "needs_review"
        if claim.get("owner") == owner and claim.get("job_id") == str(gpu_job_id):
            if state in {"completed", "paused", "failed"}:
                terminal_claim = claims.release(expected["run_dir"], owner, state, step_dead=True)
            else:
                terminal_claim = claims.update(expected["run_dir"], owner, state="liveness_needs_review")
        elif claim == expected:
            terminal_claim = claim
        else:
            raise RuntimeError("claim is neither the prior boundary nor this V5 allocation")
        row.update(state=state, accepted=accepted, slurm_state=terminal["state"],
                   exit_code=terminal["exit_code"], finalized_at=now, terminal_claim=terminal_claim)
        if continuation is not None:
            row["continuation"] = continuation
        intent = read(binding["intent"])
        _persist(binding, journal, intent, row)
        return row


def lookup_slurm_jobs(comment):
    queue_output = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%a|%b|%k|%r"],
        text=True, timeout=20)
    accounting_output = subprocess.check_output(
        ["sacct", "-S", "now-2hours", "-nPX", "-u", os.environ["USER"], "-o",
         "JobIDRaw,State,Account,ReqTRES,Comment"], text=True, timeout=20)
    rows = {}
    for line in queue_output.splitlines():
        fields = line.split("|", 5)
        if len(fields) != 6 or fields[4] != comment:
            continue
        gres = fields[3]
        match = re.search(r"gpu(?::[^:,(]+)?:([0-9]+)", gres)
        gpus = int(match.group(1)) if match else 0
        detail = _fields(subprocess.check_output(
            ["scontrol", "show", "job", fields[0], "-o"], text=True, timeout=20))
        rows[fields[0]] = {"job_id": fields[0], "state": fields[1], "account": fields[2],
                           "gpus": gpus, "comment": fields[4],
                           "held": fields[5] == "JobHeldUser",
                           "dependency": detail.get("Dependency", "")}
    for line in accounting_output.splitlines():
        fields = line.split("|", 4)
        if len(fields) != 5 or fields[4] != comment or not fields[0].isdigit():
            continue
        match = re.search(r"gres/gpu(?::[^=,]+)?=([0-9]+)", fields[3])
        recovered = {"job_id": fields[0], "state": fields[1].split()[0],
                     "account": fields[2], "gpus": int(match.group(1)) if match else 0,
                     "comment": fields[4], "held": False, "dependency": ""}
        if fields[0] not in rows:
            rows[fields[0]] = recovered
    return {"jobs": [rows[key] for key in sorted(rows)], "absence_proven": True}


def observe_terminal(job_id):
    output = subprocess.check_output(
        ["sacct", "-j", str(job_id), "-nP", "-X", "-o", "JobIDRaw,State,ExitCode"],
        text=True, timeout=20)
    matches = [line.split("|") for line in output.splitlines()
               if line.split("|", 1)[0] == str(job_id)]
    if len(matches) != 1:
        raise ValueError("ambiguous allocation terminal record")
    return {"job_id": matches[0][0], "state": matches[0][1].split()[0],
            "exit_code": matches[0][2]}


def observe_finalizer(job_id):
    fields = _fields(subprocess.check_output(
        ["scontrol", "show", "job", str(job_id), "-o"], text=True, timeout=20))
    return {"job_id": str(fields.get("JobId", "")), "state": fields.get("JobState"),
            "account": fields.get("Account"), "gpus": _gpu_count(fields),
            "comment": fields.get("Comment"),
            "dependency": fields.get("Dependency", "")}


def submit_sbatch(command, *, timeout):
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("sbatch failed: " + result.stderr.strip())
    return result.stdout.strip().split(";", 1)[0]


def release_job(job_id):
    subprocess.run(["scontrol", "release", str(job_id)], check=True,
                   text=True, capture_output=True, timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--publisher", action="store_true")
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--allocation-owner", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--gpu-job-id")
    parser.add_argument("--step-record")
    parser.add_argument("--gate")
    parser.add_argument("--owner-contract")
    parser.add_argument("--nonce")
    args = parser.parse_args()
    if sum((args.worker, args.publisher, args.runtime_preflight,
            args.allocation_owner, args.finalize)) != 1:
        parser.error("choose exactly one execution mode")
    if args.runtime_preflight:
        binding = validate_binding(args.binding)
        print(json.dumps(verify_runtime(binding), sort_keys=True))
        return
    if args.worker:
        if not all((args.step_record, args.gate, args.owner_contract, args.nonce)):
            parser.error("worker requires step, gate, owner contract and nonce")
        raise SystemExit(worker(args.binding, args.step_record, args.gate,
                                args.owner_contract, args.nonce))
    if args.publisher:
        binding = validate_binding(args.binding)
        result = publish_once(args.binding,
                              snapshot=lambda: live_snapshot(binding), submit=submit_sbatch,
                              lookup=lookup_slurm_jobs, release=release_job)
        print(json.dumps(result, sort_keys=True))
        return
    if args.allocation_owner:
        raise SystemExit(allocation_owner(args.binding))
    if args.finalize:
        if not args.gpu_job_id:
            parser.error("finalizer requires GPU job id")
        result = finalize_afterany(args.binding, gpu_job_id=args.gpu_job_id,
                                   observe_terminal=observe_terminal,
                                   observe_finalizer=observe_finalizer)
        print(json.dumps(result, sort_keys=True))
        return
    parser.error("choose publisher, runtime preflight, allocation owner, finalizer or worker")


if __name__ == "__main__":
    main()
