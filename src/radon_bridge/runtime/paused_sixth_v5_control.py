"""One-shot production control for the reviewed paused formal-V5 run.

This is intentionally a finite adapter, not a dispatcher or refiller.  It writes
one request through the established account lock, and the granted allocation
starts only the immutable V5 asset named by the binding.  The historical V4 run
is never used as a loader fallback and is never overwritten: its claim is the
continuity anchor while the V5 execution writes to a separate revision directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from radon_bridge.runtime.paused_sixth_v5_handoff import (
    FORMAL_V5_COMMIT,
    RUN_ID,
)


TERMINAL = {"completed", "paused", "failed", "cancelled", "identity_drift", "needs_review"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def _exact(path, digest, label):
    if sha256(path) != digest:
        raise ValueError(label + " changed")


def validate_binding(path, *, now=None, allow_expired=False):
    now = time.time() if now is None else now
    binding = read(path)
    required = {
        "schema", "lease_id", "project", "run_id", "source_commit", "framework_commit",
        "source_file", "source_file_sha256", "runtime_pins", "account_lock", "role_policy",
        "role_policy_sha256", "ownership_overlay", "journal", "journal_initial_sha256",
        "intent", "intent_initial_sha256", "claims_root", "expected_claim",
        "asset_dir", "asset_sha256", "v5_spec_sha256", "v5_checkpoint_sha256",
        "execution_dir", "python", "worker_cpus", "worker_memory_gib",
        "requested_gpus", "account_limit", "expires_at", "sbatch_command",
        "slurm_comment", "finalizer_command_template", "test_access", "dispatch_authorized",
    }
    if set(binding) != required or binding.get("schema") != "radon_paused_sixth_v5_control_v1":
        raise ValueError("unknown paused-sixth V5 control binding")
    if (binding["project"] != "Radon_Bridge" or binding["run_id"] != RUN_ID
            or binding["framework_commit"] != FORMAL_V5_COMMIT
            or not isinstance(binding["source_commit"], str) or len(binding["source_commit"]) != 40
            or binding["requested_gpus"] != 1 or binding["account_limit"] != 24
            or binding["test_access"] is not False
            or binding["dispatch_authorized"] is not True
            or type(binding["worker_cpus"]) is not int or binding["worker_cpus"] < 1
            or type(binding["worker_memory_gib"]) is not int or binding["worker_memory_gib"] < 1
            or not isinstance(binding["slurm_comment"], str) or not binding["slurm_comment"]
            or type(binding["expires_at"]) not in (int, float)
            or (not allow_expired and binding["expires_at"] <= now)):
        raise ValueError("paused-sixth V5 control scope changed")
    if not re.fullmatch(r"[0-9a-f]{40}", binding["source_commit"]):
        raise ValueError("invalid source commit")
    for key in ("source_file_sha256", "role_policy_sha256", "journal_initial_sha256",
                "intent_initial_sha256", "asset_sha256", "v5_spec_sha256",
                "v5_checkpoint_sha256"):
        value = binding[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid digest in " + key)
    _exact(binding["source_file"], binding["source_file_sha256"], "control source")
    pins = binding["runtime_pins"]
    if (not isinstance(pins, list) or not pins
            or len({row.get("path") for row in pins if isinstance(row, dict)}) != len(pins)
            or not any(row.get("path") == binding["source_file"] and
                       row.get("sha256") == binding["source_file_sha256"] for row in pins)):
        raise ValueError("runtime source pins are incomplete")
    for row in pins:
        if set(row) != {"path", "sha256"}:
            raise ValueError("invalid runtime source pin")
        _exact(row["path"], row["sha256"], "runtime source")
    _exact(binding["role_policy"], binding["role_policy_sha256"], "role policy")
    asset = Path(binding["asset_dir"])
    _exact(asset / "asset.json", binding["asset_sha256"], "production asset")
    _exact(asset / "spec.json", binding["v5_spec_sha256"], "V5 spec")
    _exact(asset / "last.pt", binding["v5_checkpoint_sha256"], "V5 checkpoint")
    asset_record = read(asset / "asset.json")
    if (asset_record.get("schema") != "radon_paused_sixth_v5_production_asset_v1"
            or asset_record.get("state") != "accepted_for_new_claim"
            or asset_record.get("run_id") != RUN_ID
            or asset_record.get("framework_commit") != FORMAL_V5_COMMIT
            or asset_record.get("spec_sha256") != binding["v5_spec_sha256"]
            or asset_record.get("checkpoint_sha256") != binding["v5_checkpoint_sha256"]):
        raise ValueError("production asset contract changed")
    command = binding["sbatch_command"]
    finalizer = binding["finalizer_command_template"]
    if (not isinstance(command, list) or not command or command[0] != "sbatch"
            or "--parsable" not in command or "--gres=gpu:a100:1" not in command
            or not any(x == "--time=48:00:00" for x in command)
            or command.count("--comment=" + binding["slurm_comment"]) != 1
            or not all(isinstance(x, str) and x for x in command)
            or not isinstance(finalizer, list) or not finalizer or finalizer[0] != "sbatch"
            or "--parsable" not in finalizer
            or sum(x.count("{gpu_job_id}") for x in finalizer) < 1
            or any("--gres=" in x for x in finalizer)):
        raise ValueError("Slurm command contract changed")
    expected = binding["expected_claim"]
    if (not isinstance(expected, dict)
            or expected.get("state") != "claimed"
            or expected.get("owner") != "radon-v5-paused-sixth-migration"
            or expected.get("migration_phase") != "source_reserved"
            or expected.get("run_dir") != str(Path(expected.get("run_dir", "")).resolve())):
        raise ValueError("migration reservation is not the expected claim")
    return binding


def pins_digest(binding):
    payload = sorted(binding["runtime_pins"], key=lambda row: row["path"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    if journal.get("schema") != "radon_paused_sixth_v5_requests_v1" or not isinstance(journal.get("requests"), list):
        raise ValueError("unknown paused-sixth request journal")
    return policy, project, journal


def live_snapshot(binding):
    """Reconcile Slurm with every role journal and the signed legacy overlay."""
    from mhd_models.scheduling.quota_guard import snapshot

    snap = snapshot()
    policy = read(binding["role_policy"])
    owners = {}
    unresolved = 0
    for project, body in policy.get("projects", {}).items():
        for path in body.get("request_journals", []):
            journal = read(path)
            if not isinstance(journal.get("requests"), list):
                raise ValueError("registered project journal is malformed")
            for request in journal["requests"]:
                state = str(request.get("state", "")).lower()
                job = str(request.get("job_id", ""))
                if job.isdigit():
                    if job in snap["jobs"]:
                        if job in owners:
                            raise ValueError("live GPU job has duplicate ownership")
                        owners[job] = project
                elif state not in TERMINAL and state not in {"owner_exited"}:
                    count = request.get("requested_gpus", 1)
                    if type(count) is not int or count < 1:
                        raise ValueError("unresolved intent has invalid GPU count")
                    unresolved += count
    overlay_path = binding.get("ownership_overlay")
    if overlay_path:
        overlay = read(overlay_path)
        if (overlay.get("schema") != "legacy_native_ownership_overlay_v2"
                or overlay.get("test") is not False
                or overlay.get("role_policy_sha256") != binding["role_policy_sha256"]):
            raise ValueError("legacy ownership overlay does not bind current role policy")
        for entry in overlay.get("entries", []):
            job = str(entry.get("job_id", ""))
            if entry.get("classification") != "non_project_legacy_native" or job not in snap["jobs"]:
                raise ValueError("overlay contains an unknown live job")
            if job in owners:
                raise ValueError("overlay duplicates project ownership")
            owners[job] = "legacy_native_overlay"
    live_gpu = {job for job, row in snap["jobs"].items() if row.get("gpus", 0) > 0}
    missing = live_gpu - set(owners)
    if missing:
        raise ValueError("live GPU job lacks a registered owner: " + ",".join(sorted(missing)))
    return {
        "account_limit": min(binding["account_limit"], snap["limit"]),
        "account_running_pending": snap["total_gpus"],
        "radon_running_pending": sum(snap["jobs"][job]["gpus"] for job, owner in owners.items()
                                     if owner == "Radon_Bridge"),
        "unresolved_gpu_intents": unresolved,
    }


def decide(binding, snapshot, journal):
    if snapshot.get("account_limit") != binding["account_limit"]:
        raise ValueError("account limit differs from binding")
    for key in ("account_running_pending", "radon_running_pending", "unresolved_gpu_intents",
                "radon_reserved_gpus"):
        if type(snapshot.get(key)) is not int or snapshot[key] < 0:
            raise ValueError("invalid reconciled account snapshot")
    rows = [row for row in journal["requests"] if row.get("lease_id") == binding["lease_id"]]
    if len(rows) > 1:
        raise ValueError("duplicate paused-sixth lease")
    if rows:
        return {"action": "none", "state": "terminal_" + rows[0]["state"] if rows[0].get("state") in TERMINAL else "already_submitted"}
    if snapshot["radon_reserved_gpus"] < 1:
        raise ValueError("R&B role has no production entitlement")
    occupied = snapshot["account_running_pending"] + snapshot["unresolved_gpu_intents"]
    if occupied + 1 > binding["account_limit"]:
        return {"action": "none", "state": "waiting_account_capacity"}
    if snapshot["radon_running_pending"] + 1 > snapshot["radon_reserved_gpus"]:
        return {"action": "none", "state": "waiting_radon_role_capacity"}
    return {"action": "submit_once", "state": "ready"}


def publish_once(binding_path, *, snapshot, submit, now=None):
    """Write one durable intent and submit one project allocation under the shared lock."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    try:
        handle = locked_existing(binding["account_lock"], nonblocking=True)
    except BlockingIOError:
        return {"action": "none", "state": "waiting_account_lock"}
    with handle:
        binding = validate_binding(binding_path, now=now)
        _, project, journal = _policy_and_journal(binding)
        intent = read(binding["intent"])
        if intent.get("schema") != "radon_paused_sixth_v5_intent_v1":
            raise ValueError("unknown paused-sixth submission intent")
        if intent.get("attempt") is not None:
            return {"action": "none", "state": "submission_intent_needs_review"}
        if journal["requests"]:
            return decide(binding, snapshot(), journal)
        _exact(binding["journal"], binding["journal_initial_sha256"], "initial request journal")
        _exact(binding["intent"], binding["intent_initial_sha256"], "initial submission intent")
        current = snapshot()
        current["radon_reserved_gpus"] = project["reserved_gpus"]
        decision = decide(binding, current, journal)
        if decision["action"] != "submit_once":
            return decision
        entry = {
            "lease_id": binding["lease_id"], "state": "intent", "requested_gpus": 1,
            "binding_sha256": sha256(binding_path), "role_policy_sha256": binding["role_policy_sha256"],
            "production_asset_sha256": binding["asset_sha256"], "run_dir": binding["expected_claim"]["run_dir"],
            "execution_dir": str(Path(binding["execution_dir"]).resolve()), "time": now,
        }
        intent["attempt"] = dict(entry)
        write(binding["intent"], intent)
        job_id = str(submit(binding["sbatch_command"], timeout=20))
        if not job_id.isdigit():
            entry.update(state="identity_drift", observed_job_id=job_id)
            intent["attempt"] = dict(entry)
            write(binding["intent"], intent)
            return {"action": "none", "state": "identity_drift"}
        entry.update(state="submitted", job_id=job_id)
        journal["requests"].append(dict(entry))
        intent["attempt"] = dict(entry)
        write(binding["journal"], journal)
        write(binding["intent"], intent)
        final = [x.replace("{gpu_job_id}", job_id) for x in binding["finalizer_command_template"]]
        try:
            finalizer_id = str(submit(final, timeout=20))
        except BaseException as error:
            entry.update(state="finalizer_submission_needs_review", error_type=type(error).__name__)
            journal["requests"][-1] = dict(entry)
            intent["attempt"] = dict(entry)
            write(binding["journal"], journal)
            write(binding["intent"], intent)
            raise
        if not finalizer_id.isdigit():
            entry.update(state="finalizer_identity_drift", observed_finalizer_job_id=finalizer_id)
        else:
            entry.update(finalizer_job_id=finalizer_id)
        journal["requests"][-1] = dict(entry)
        intent["attempt"] = dict(entry)
        write(binding["journal"], journal)
        write(binding["intent"], intent)
        return {"action": "none", "state": entry["state"], "job_id": job_id,
                "finalizer_job_id": entry.get("finalizer_job_id")}


def grant_allocation(binding_path, *, job_id, observe_allocation, now=None):
    """Turn exactly one submitted request into a scheduler-proven grant."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    with locked_existing(binding["account_lock"]):
        validate_binding(binding_path, now=now)
        _, _, journal = _policy_and_journal(binding)
        rows = [row for row in journal["requests"] if row.get("lease_id") == binding["lease_id"]]
        if len(rows) != 1 or rows[0].get("job_id") != str(job_id):
            raise ValueError("submitted V5 request identity changed")
        row = rows[0]
        if row.get("state") == "granted":
            intent = read(binding["intent"])
            intent["attempt"] = dict(row)
            write(binding["intent"], intent)
            return row
        if row.get("state") != "submitted":
            raise ValueError("only a submitted V5 request may be granted")
        observed = observe_allocation(str(job_id))
        if (observed.get("job_id") != str(job_id) or observed.get("state") != "RUNNING"
                or observed.get("account") != "pi-mengy" or observed.get("gpus") != 1
                or observed.get("comment") != binding["slurm_comment"]):
            raise ValueError("Slurm allocation grant identity is unproven")
        row.update(state="granted", granted_at=now, slurm_comment=binding["slurm_comment"])
        write(binding["journal"], journal)
        intent = read(binding["intent"])
        intent["attempt"] = dict(row)
        write(binding["intent"], intent)
        return row


def prepare_execution(binding, *, now=None):
    """Create the V5 execution revision without mutating the historical V4 run."""
    now = time.time() if now is None else now
    execution = Path(binding["execution_dir"])
    asset = Path(binding["asset_dir"])
    if execution.exists():
        manifest = read(execution / "migration.json")
        if (manifest.get("binding_sha256") != binding["binding_sha256"]
                or sha256(execution / "last.pt") != binding["v5_checkpoint_sha256"]
                or sha256(execution / "spec.json") != binding["v5_spec_sha256"]):
            raise RuntimeError("existing V5 execution revision conflicts")
        return manifest
    execution.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=execution.name + ".", dir=execution.parent))
    try:
        for name in ("last.pt", "spec.json"):
            source = asset / name
            target = stage / name
            target.write_bytes(source.read_bytes())
        manifest = {
            "schema": "radon_paused_sixth_v5_execution_revision_v1",
            "run_id": RUN_ID, "framework_commit": FORMAL_V5_COMMIT,
            "binding_sha256": binding["binding_sha256"],
            "source_v4_run": binding["expected_claim"]["run_dir"],
            "production_asset_sha256": binding["asset_sha256"],
            "spec_sha256": binding["v5_spec_sha256"],
            "checkpoint_sha256": binding["v5_checkpoint_sha256"],
            "created_at": now, "test_access": False,
        }
        write(stage / "migration.json", manifest)
        os.rename(stage, execution)
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage)
    return manifest


def claim_running_step(binding_path, *, claims, observe_job, step_record, now=None):
    """Bind journal, allocation, step command, V5 asset and claim in one lock epoch."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    binding["binding_sha256"] = sha256(binding_path)
    record = read(step_record)
    expected_command = hashlib.sha256(json.dumps(record.get("argv"), separators=(",", ":")).encode()).hexdigest()
    if (record.get("schema") != "radon_paused_sixth_v5_step_v1"
            or record.get("binding_sha256") != binding["binding_sha256"]
            or record.get("source_file_sha256") != binding["source_file_sha256"]
            or record.get("runtime_pins_digest") != pins_digest(binding)
            or record.get("production_asset_sha256") != binding["asset_sha256"]
            or record.get("command_sha256") != expected_command
            or str(record.get("job_id", "")) == "" or str(record.get("step", "")) == ""):
        raise ValueError("worker step record is not bound to the immutable V5 command")
    with locked_existing(binding["account_lock"]):
        validate_binding(binding_path, now=now)
        _, _, journal = _policy_and_journal(binding)
        matches = [row for row in journal["requests"] if row.get("lease_id") == binding["lease_id"]]
        if (len(matches) != 1 or matches[0].get("job_id") != str(record["job_id"])
                or matches[0].get("state") != "granted"):
            raise ValueError("granted request identity is missing")
        observed = observe_job(str(record["job_id"]), str(record["step"]))
        if (observed.get("job_id") != str(record["job_id"])
                or observed.get("step") != str(record["step"])
                or observed.get("job_state") != "RUNNING" or observed.get("step_state") != "RUNNING"
                or observed.get("account") != "pi-mengy" or observed.get("gpus") != 1
                or observed.get("command_sha256") != record["command_sha256"]):
            raise ValueError("Slurm allocation/step command identity is unproven")
        binding["binding_sha256"] = sha256(binding_path)
        prepare_execution(binding, now=now)
        old_expected = binding["expected_claim"]
        def transition(old):
            if old != old_expected:
                raise RuntimeError("paused-sixth migration reservation changed")
            return {
                **old,
                "state": "running", "owner": "radon-v5-production-" + str(record["job_id"]),
                "job_id": str(record["job_id"]), "step": str(record["step"]),
                "generation": old["generation"] + 1, "updated_at": now,
                "spec_sha256": binding["v5_spec_sha256"],
                "framework_commit": FORMAL_V5_COMMIT,
                "production_asset_sha256": binding["asset_sha256"],
                "execution_dir": str(Path(binding["execution_dir"]).resolve()),
                "binding_sha256": binding["binding_sha256"],
                "runtime_pins_digest": record["runtime_pins_digest"],
                "command_sha256": record["command_sha256"],
                "previous": {k: old.get(k) for k in ("state", "owner", "job_id", "step", "generation", "spec_sha256")},
            }
        claim = claims.mutate(old_expected["run_dir"], transition)
        matches[0].update(state="running", granted_at=now, step=str(record["step"]),
                          command_sha256=record["command_sha256"],
                          claim_generation=claim["generation"],
                          v5_spec_sha256=binding["v5_spec_sha256"])
        write(binding["journal"], journal)
        return claim


def worker(binding_path, step_record, gate):
    binding = validate_binding(binding_path)
    job = os.environ.get("SLURM_JOB_ID", "")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not job.isdigit() or not step:
        raise ValueError("worker lacks Slurm job/step identity")
    argv = [binding["python"], "-m", "radon_bridge.runtime.paused_sixth_v5_control",
            "--binding", str(Path(binding_path).resolve()), "--worker",
            "--step-record", str(Path(step_record).resolve()), "--gate", str(Path(gate).resolve())]
    record = {
        "schema": "radon_paused_sixth_v5_step_v1", "job_id": job, "step": step,
        "binding_sha256": sha256(binding_path), "source_file_sha256": binding["source_file_sha256"],
        "runtime_pins_digest": pins_digest(binding),
        "production_asset_sha256": binding["asset_sha256"], "argv": argv,
        "command_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
        "pid": os.getpid(), "time": time.time(),
    }
    write(step_record, record)
    deadline = time.monotonic() + 300
    while not Path(gate).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("V5 claim gate was not published")
        time.sleep(1)
    gate_value = read(gate)
    if (gate_value.get("schema") != "radon_paused_sixth_v5_claim_gate_v1"
            or gate_value.get("job_id") != job or gate_value.get("step") != step
            or gate_value.get("binding_sha256") != sha256(binding_path)):
        raise ValueError("V5 claim gate identity changed")
    from mhd_models.workflows.native import run
    return run(str(Path(binding["execution_dir"]) / "spec.json"), binding["execution_dir"], mode="train")


def _fields(text):
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def observe_running_step(job_id, step_id, command_sha256):
    job = _fields(subprocess.check_output(
        ["scontrol", "show", "job", str(job_id), "-o"], text=True, timeout=20))
    step = _fields(subprocess.check_output(
        ["scontrol", "show", "step", f"{job_id}.{step_id}", "-o"], text=True, timeout=20))
    tres = job.get("ReqTRES", "") + "," + job.get("AllocTRES", "")
    gpus = max([int(match.group(1)) for match in
                (re.fullmatch(r"gres/gpu(?::[^=,]+)?=(\d+)", x) for x in tres.split(","))
                if match] or [0])
    return {
        "job_id": str(job.get("JobId", "")), "step": str(step.get("StepId", "")).split(".", 1)[-1],
        "job_state": job.get("JobState"), "step_state": step.get("State"),
        "account": job.get("Account"), "gpus": gpus, "command_sha256": command_sha256,
    }


def observe_running_allocation(job_id):
    job = _fields(subprocess.check_output(
        ["scontrol", "show", "job", str(job_id), "-o"], text=True, timeout=20))
    tres = job.get("ReqTRES", "") + "," + job.get("AllocTRES", "")
    gpus = max([int(match.group(1)) for match in
                (re.fullmatch(r"gres/gpu(?::[^=,]+)?=(\d+)", x) for x in tres.split(","))
                if match] or [0])
    return {"job_id": str(job.get("JobId", "")), "state": job.get("JobState"),
            "account": job.get("Account"), "gpus": gpus, "comment": job.get("Comment")}


def allocation_owner(binding_path, *, popen=subprocess.Popen, now=None):
    """Own the granted allocation and run the one exact V5 training revision."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now)
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise ValueError("allocation owner lacks numeric Slurm job identity")
    grant_allocation(binding_path, job_id=job, observe_allocation=observe_running_allocation, now=now)
    root = Path(binding["execution_dir"]).parent / ("owner_" + job)
    root.mkdir(parents=True, exist_ok=True)
    step_record = root / "step.json"
    gate = root / "claim_gate.json"
    command = [
        "srun", "--jobid=" + job, "--overlap", "--exact", "--nodes=1", "--ntasks=1",
        "--gpus=1", "--cpus-per-task=" + str(binding["worker_cpus"]),
        "--mem=" + str(binding["worker_memory_gib"]) + "G", "--unbuffered",
        binding["python"], "-m", "radon_bridge.runtime.paused_sixth_v5_control",
        "--binding", str(Path(binding_path).resolve()), "--worker",
        "--step-record", str(step_record.resolve()), "--gate", str(gate.resolve()),
    ]
    log = (root / "worker.log").open("x")
    child = popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 300
        while not step_record.exists():
            if child.poll() is not None:
                raise RuntimeError("V5 worker exited before publishing its Slurm step")
            if time.monotonic() >= deadline:
                raise TimeoutError("V5 worker step record timed out")
            time.sleep(1)
        record = read(step_record)
        from mhd_models.scheduling.policy import Claims
        claims = Claims(binding["claims_root"])
        claim = claim_running_step(
            binding_path, claims=claims, step_record=step_record,
            observe_job=lambda actual_job, actual_step: observe_running_step(
                actual_job, actual_step, record["command_sha256"]), now=now)
        write(gate, {"schema": "radon_paused_sixth_v5_claim_gate_v1",
                     "job_id": job, "step": record["step"],
                     "binding_sha256": sha256(binding_path),
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
            terminal = "completed"
        elif code == 75 and status.get("state") == "paused":
            terminal = "paused"
        else:
            terminal = "failed"
        claims.release(binding["expected_claim"]["run_dir"], claim["owner"], terminal, step_dead=True)
        with locked_existing(binding["account_lock"]):
            _, _, journal = _policy_and_journal(binding)
            rows = [row for row in journal["requests"] if row.get("lease_id") == binding["lease_id"]]
            if len(rows) != 1 or rows[0].get("job_id") != job:
                raise RuntimeError("R&B V5 journal changed before terminal publication")
            rows[0].update(state=terminal, exit_code=code, finished_at=time.time(),
                           accepted=(terminal == "completed"))
            write(binding["journal"], journal)
        return code
    finally:
        log.close()


def finalize_afterany(binding_path, *, gpu_job_id, finalizer_job_id, observe_terminal,
                      claims_factory=None, now=None):
    """Close an unexpectedly terminated allocation without guessing step liveness."""
    now = time.time() if now is None else now
    binding = validate_binding(binding_path, now=now, allow_expired=True)
    with locked_existing(binding["account_lock"]):
        _, _, journal = _policy_and_journal(binding)
        rows = [row for row in journal["requests"] if row.get("lease_id") == binding["lease_id"]]
        if len(rows) != 1 or rows[0].get("job_id") != str(gpu_job_id):
            raise ValueError("finalizer GPU journal identity changed")
        row = rows[0]
        if row.get("finalizer_job_id") != str(finalizer_job_id):
            raise ValueError("afterany finalizer identity changed")
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
        owner = "radon-v5-production-" + str(gpu_job_id)
        execution = Path(binding["execution_dir"])
        status = read(execution / "status.json") if (execution / "status.json").exists() else {}
        accepted = False
        state = "failed"
        if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0" \
                and status.get("state") == "completed" and (execution / "accepted.json").is_file():
            from mhd_models.runtime.training_state import verify_completion
            verify_completion(execution, read(execution / "spec.json"))
            state = "completed"
            accepted = True
        elif status.get("state") == "paused":
            state = "paused"
        if claim.get("owner") == owner and claim.get("job_id") == str(gpu_job_id):
            claims.release(binding["expected_claim"]["run_dir"], owner, state, step_dead=True)
        elif claim != binding["expected_claim"]:
            raise RuntimeError("claim is neither the migration reservation nor this V5 allocation")
        row.update(state=state, accepted=accepted, slurm_state=terminal["state"],
                   exit_code=terminal["exit_code"], finalized_at=now)
        write(binding["journal"], journal)
        intent = read(binding["intent"])
        intent["attempt"] = dict(row)
        write(binding["intent"], intent)
        return row


def observe_terminal(job_id):
    result = subprocess.check_output(
        ["sacct", "-j", str(job_id), "-nP", "-X", "-o", "JobIDRaw,State,ExitCode"],
        text=True, timeout=20)
    rows = [line.split("|") for line in result.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) >= 3 and row[0] == str(job_id)]
    if len(matches) != 1:
        raise ValueError("ambiguous GPU allocation terminal record")
    return {"job_id": matches[0][0], "state": matches[0][1].split()[0], "exit_code": matches[0][2]}


def submit_sbatch(command, *, timeout):
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("sbatch failed: " + result.stderr.strip())
    return result.stdout.strip().split(";", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--publisher", action="store_true")
    parser.add_argument("--allocation-owner", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--gpu-job-id")
    parser.add_argument("--finalizer-job-id")
    parser.add_argument("--step-record")
    parser.add_argument("--gate")
    args = parser.parse_args()
    if args.worker:
        if not args.step_record or not args.gate:
            parser.error("--worker requires --step-record and --gate")
        raise SystemExit(worker(args.binding, args.step_record, args.gate))
    if args.publisher:
        result = publish_once(args.binding, snapshot=lambda: live_snapshot(validate_binding(args.binding)),
                              submit=submit_sbatch)
        print(json.dumps(result, sort_keys=True))
        return
    if args.allocation_owner:
        raise SystemExit(allocation_owner(args.binding))
    if args.finalize:
        if not args.gpu_job_id or not args.finalizer_job_id:
            parser.error("--finalize requires --gpu-job-id and --finalizer-job-id")
        result = finalize_afterany(args.binding, gpu_job_id=args.gpu_job_id,
                                   finalizer_job_id=args.finalizer_job_id,
                                   observe_terminal=observe_terminal)
        print(json.dumps(result, sort_keys=True))
        return
    parser.error("choose --publisher, --allocation-owner, --finalize, or --worker")


if __name__ == "__main__":
    main()
