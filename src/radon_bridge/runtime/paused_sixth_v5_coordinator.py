"""Idempotent one-shot coordinator for the reviewed paused-sixth V5 cutover.

The coordinator is deliberately finite.  It installs one reviewed role-policy
transition, reserves one historical source, freezes one V5 asset, builds one
immutable runtime binding and invokes the crash-recoverable held-first
publisher.  A durable phase record is written before the first mutation and
every phase is recoverable from its own immutable postcondition.
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
import shlex
import shutil
import subprocess
import tempfile
import time

from radon_bridge.runtime import paused_sixth_v5_control_v2 as control
from radon_bridge.runtime import paused_sixth_v5_handoff as handoff


SCHEMA = "radon_paused_sixth_v5_coordinator_contract_v1"
STATE_SCHEMA = "radon_paused_sixth_v5_coordinator_state_v1"
PHASES = ("prepared", "policy_installed", "claim_reserved", "asset_published",
          "binding_ready", "publisher_invoked")
HEX64 = set("0123456789abcdef")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _fsync_dir(path):
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial",
                                               dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _immutable_json(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"immutable artifact conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial",
                                               dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise RuntimeError(f"immutable artifact conflicts: {path}")
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _locked(path):
    descriptor = os.open(Path(path), os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    handle = os.fdopen(descriptor, "r+")
    before, after = Path(path).stat(), os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        handle.close()
        raise RuntimeError("account lock inode changed")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def validate_contract(path):
    value = _read(path)
    required = {
        "schema", "root", "state", "account_lock", "role_policy",
        "initial_policy_sha256", "temporary_policy", "temporary_policy_sha256",
        "steady_policy", "steady_policy_sha256", "rollback_requires_future_look_stage2",
        "policy_intent", "policy_transition_receipt", "look_stage2",
        "look_stage2_sha256", "look_successor_job_id", "review_receipts",
        "policy_proposal", "policy_proposal_sha256", "run", "claims_root",
        "claim_path", "source_checkpoint", "source_spec", "reservation_receipt",
        "asset_dir", "conversion_receipt", "converted_checkpoint", "packet",
        "independent_recheck", "attempt_checkpoint", "attempt_spec", "source_root",
        "mhd_models_root", "framework_root", "python", "ownership_overlay",
        "allocation_wrapper", "finalizer_wrapper", "deployment_gate", "binding",
        "execution_dir", "native_sources", "runtime_environment", "worker_cpus",
        "worker_memory_gib", "source_commit", "expires_at", "test_access"
    }
    if set(value) != required or value.get("schema") != SCHEMA:
        raise ValueError("unknown coordinator contract")
    if value.get("test_access") is not False or not str(value["look_successor_job_id"]).isdigit():
        raise ValueError("invalid coordinator scope")
    for key in ("initial_policy_sha256", "temporary_policy_sha256",
                "steady_policy_sha256", "look_stage2_sha256", "policy_proposal_sha256"):
        if not _hex64(value[key]):
            raise ValueError("invalid digest: " + key)
    reviews = value["review_receipts"]
    if not isinstance(reviews, list) or len(reviews) < 4:
        raise ValueError("all independent review receipts, including the coordinator, are required")
    for row in reviews:
        if set(row) != {"path", "sha256"} or not _hex64(row["sha256"]):
            raise ValueError("invalid review receipt")
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError("review receipt changed")
    for key in ("temporary_policy", "steady_policy", "look_stage2", "policy_proposal"):
        digest_key = key + "_sha256"
        if _sha(value[key]) != value[digest_key]:
            raise ValueError(key + " changed")
    if value["rollback_requires_future_look_stage2"] is not True:
        raise ValueError("cap9 rollback must remain gated by a future LOOK stage2")
    stage2 = _read(value["look_stage2"])
    if (stage2.get("schema") != "look_v5_monitor_stage2_acceptance_v1"
            or stage2.get("status") != "accepted"
            or stage2.get("successor_monitor_job_id") != value["look_successor_job_id"]
            or stage2.get("accepted_role_policy_sha256") != value["temporary_policy_sha256"]
            or stage2.get("successor_monitor_identity_verified") is not True
            or stage2.get("predecessor_release_verified") is not True
            or stage2.get("test_access") is not False):
        raise ValueError("real LOOK successor stage2 is not accepted")
    proposal = _read(value["policy_proposal"])
    if (proposal.get("schema") != "radon_paused_sixth_v5_policy_proposal_v1"
            or proposal.get("state") != "accepted"
            or proposal.get("role_policy_sha256") != value["temporary_policy_sha256"]):
        raise ValueError("R&B policy proposal is not accepted")
    if not isinstance(value["expires_at"], (int, float)) or value["expires_at"] <= time.time():
        raise ValueError("coordinator contract expired")
    if (not isinstance(value["runtime_environment"], dict)
            or set(value["runtime_environment"]) != control.ENV_KEYS
            or any(not isinstance(item, str) for item in value["runtime_environment"].values())
            or type(value["worker_cpus"]) is not int or value["worker_cpus"] < 1
            or type(value["worker_memory_gib"]) is not int or value["worker_memory_gib"] < 1
            or not isinstance(value["source_commit"], str) or len(value["source_commit"]) != 40):
        raise ValueError("invalid immutable runtime inputs")
    if (not isinstance(value["native_sources"], dict)
            or any(not str(job).isdigit() or not isinstance(source, str)
                   for job, source in value["native_sources"].items())):
        raise ValueError("invalid legacy-native provenance map")
    root = Path(value["root"]).resolve()
    for key in ("state", "policy_intent", "policy_transition_receipt",
                "reservation_receipt", "asset_dir", "ownership_overlay",
                "allocation_wrapper", "finalizer_wrapper", "deployment_gate",
                "binding", "execution_dir"):
        if root not in Path(value[key]).resolve().parents:
            raise ValueError(key + " escapes coordinator root")
    return value


def _initial_state(contract_path, contract):
    return {"schema": STATE_SCHEMA, "phase": "prepared", "pending": None,
            "contract": str(Path(contract_path).resolve()),
            "contract_sha256": _sha(contract_path), "history": [],
            "test_access": False}


def _state(contract_path, contract):
    path = Path(contract["state"])
    initial = _initial_state(contract_path, contract)
    if not path.exists():
        _immutable_json(path, initial)
    value = _read(path)
    if (value.get("schema") != STATE_SCHEMA
            or value.get("contract") != initial["contract"]
            or value.get("contract_sha256") != initial["contract_sha256"]
            or value.get("phase") not in PHASES or value.get("test_access") is not False):
        raise ValueError("coordinator state changed")
    expected_pending = None
    index = PHASES.index(value["phase"])
    if index + 1 < len(PHASES):
        expected_pending = PHASES[index + 1]
    if value.get("pending") not in {None, expected_pending}:
        raise ValueError("coordinator pending phase is inconsistent")
    return value


def _mark_pending(contract, state, phase):
    if PHASES.index(phase) != PHASES.index(state["phase"]) + 1:
        raise RuntimeError("invalid next coordinator phase")
    if state.get("pending") == phase:
        return state
    value = dict(state); value["pending"] = phase
    _atomic_json(contract["state"], value)
    return value


def _advance_state(contract, state, phase, evidence):
    if PHASES.index(phase) < PHASES.index(state["phase"]):
        return state
    if PHASES.index(phase) > PHASES.index(state["phase"]) + 1:
        raise RuntimeError("coordinator phase skipped")
    if phase != state["phase"]:
        state = dict(state)
        state["phase"] = phase
        state["pending"] = None
        state["history"] = [*state["history"], {"phase": phase, "evidence": evidence}]
        _atomic_json(contract["state"], state)
    return state


def _install_policy(contract, snapshot, now, fault=None):
    fault = (lambda phase: None) if fault is None else fault
    intent_value = {"schema": "radon_paused_sixth_v5_policy_install_intent_v1",
                    "from_sha256": contract["initial_policy_sha256"],
                    "to_sha256": contract["temporary_policy_sha256"],
                    "look_stage2": contract["look_stage2"],
                    "look_stage2_sha256": contract["look_stage2_sha256"],
                    "look_successor_job_id": contract["look_successor_job_id"],
                    "test_access": False}
    with _locked(contract["account_lock"]):
        live = _sha(contract["role_policy"])
        if live not in {contract["initial_policy_sha256"], contract["temporary_policy_sha256"]}:
            raise RuntimeError("live role policy is neither predecessor nor recoverable target")
        if not Path(contract["policy_intent"]).exists():
            if live != contract["initial_policy_sha256"]:
                raise RuntimeError("policy changed without a durable install intent")
            current = snapshot(contract)
            required = {"account_running_pending", "radon_running_pending",
                        "unresolved_gpu_intents", "radon_unresolved_gpu_intents",
                        "all_live_owned_once"}
            if set(current) != required or current["all_live_owned_once"] is not True:
                raise ValueError("invalid preinstall capacity snapshot")
            if (current["account_running_pending"] > 23
                    or current["radon_running_pending"] > 9
                    or current["unresolved_gpu_intents"] != 0
                    or current["radon_unresolved_gpu_intents"] != 0):
                raise RuntimeError("temporary cap10 transition lacks capacity")
            intent_value["snapshot"] = current
            intent_value["created_at"] = now
            _immutable_json(contract["policy_intent"], intent_value)
        else:
            existing = _read(contract["policy_intent"])
            if any(existing.get(key) != value for key, value in intent_value.items()):
                raise RuntimeError("policy install intent conflicts")
        if live == contract["initial_policy_sha256"]:
            raw = Path(contract["temporary_policy"]).read_bytes()
            target = Path(contract["role_policy"])
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, target); _fsync_dir(target.parent)
                fault("after_policy_replace")
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        if _sha(contract["role_policy"]) != contract["temporary_policy_sha256"]:
            raise RuntimeError("temporary policy installation failed")
        receipt = {"schema": "radon_paused_sixth_v5_policy_transition_v1",
                   "previous_sha256": contract["initial_policy_sha256"],
                   "installed_sha256": contract["temporary_policy_sha256"],
                   "intent": contract["policy_intent"],
                   "intent_sha256": _sha(contract["policy_intent"]),
                   "look_stage2_sha256": contract["look_stage2_sha256"],
                   "installed_at": now, "test_access": False}
        if Path(contract["policy_transition_receipt"]).exists():
            old = _read(contract["policy_transition_receipt"])
            for key in ("schema", "previous_sha256", "installed_sha256", "intent",
                        "intent_sha256", "look_stage2_sha256", "test_access"):
                if old.get(key) != receipt.get(key):
                    raise RuntimeError("policy transition receipt conflicts")
        else:
            _immutable_json(contract["policy_transition_receipt"], receipt)
    return {"policy_sha256": contract["temporary_policy_sha256"],
            "transition_receipt_sha256": _sha(contract["policy_transition_receipt"])}


def _observe_old_job(job_id):
    output = subprocess.check_output(
        ["sacct", "-j", str(job_id), "-nP", "-X", "-o", "JobIDRaw,State,ExitCode"],
        text=True, timeout=20)
    rows = [line.split("|") for line in output.splitlines()
            if line.split("|", 1)[0] == str(job_id)]
    if len(rows) != 1:
        raise ValueError("old allocation terminal record is ambiguous")
    return {"job_id": rows[0][0], "state": rows[0][1].split()[0], "exit_code": rows[0][2]}


def _reserve(contract, now):
    from mhd_models.scheduling.policy import Claims
    def precondition():
        if _sha(contract["role_policy"]) != contract["temporary_policy_sha256"]:
            raise RuntimeError("temporary policy changed before source reservation")
        current = _ownership(contract)
        if (current["account_running_pending"] > 23
                or current["radon_running_pending"] > 9
                or current["unresolved_gpu_intents"] != 0
                or current["radon_unresolved_gpu_intents"] != 0
                or current["all_live_owned_once"] is not True):
            raise RuntimeError("capacity or ownership changed before source reservation")
    return handoff.reserve_actual_failed_claim(
        run=contract["run"], claims=Claims(contract["claims_root"]),
        claim_path=contract["claim_path"], source_checkpoint=contract["source_checkpoint"],
        source_spec=contract["source_spec"], account_lock=contract["account_lock"],
        observe_job=_observe_old_job, reservation_receipt=contract["reservation_receipt"], now=now,
        locked_precondition=precondition)


def _asset(contract):
    asset = Path(contract["asset_dir"])
    if asset.exists():
        record = _read(asset / "asset.json")
        if (record.get("schema") != "radon_paused_sixth_v5_production_asset_v1"
                or record.get("run_id") != handoff.RUN_ID
                or record.get("checkpoint_sha256") != _sha(asset / "last.pt")
                or record.get("spec_sha256") != _sha(asset / "spec.json")
                or record.get("checkpoint_sha256") != handoff.QUALIFIED_CHECKPOINT_SHA256
                or record.get("spec_sha256") != handoff.QUALIFIED_SPEC_SHA256
                or record.get("source_checkpoint_sha256") != handoff.SOURCE_CHECKPOINT_SHA256
                or record.get("source_spec_sha256") != handoff.SOURCE_SPEC_SHA256
                or record.get("framework_commit") != handoff.FORMAL_V5_COMMIT):
            raise RuntimeError("existing production asset conflicts")
        return record
    return handoff.publish_production_asset(
        output=asset, reservation_receipt=contract["reservation_receipt"],
        source_checkpoint=contract["source_checkpoint"], source_spec=contract["source_spec"],
        conversion_receipt=contract["conversion_receipt"],
        converted_checkpoint=contract["converted_checkpoint"], packet=contract["packet"],
        independent_recheck=contract["independent_recheck"],
        attempt_checkpoint=contract["attempt_checkpoint"], attempt_spec=contract["attempt_spec"])


def _gpu_count(tres):
    if tres in {"", "N/A", "(null)"}:
        return 0
    matches = re.findall(r"(?:^|,)gres/gpu(?::[^:,]+)?:(\d+)(?:,|$)", tres)
    if len(matches) != 1:
        raise ValueError("ambiguous GPU TRES: " + tres)
    return int(matches[0])


def _scheduler_inventory():
    listing = subprocess.check_output(
        ["squeue", "-u", os.environ["USER"], "-t", "R,PD", "-h",
         "-o", "%A|%T|%b"], text=True, timeout=20)
    jobs = {}
    for line in listing.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].isdigit():
            raise ValueError("malformed Slurm inventory")
        jobs[fields[0]] = {"state": fields[1], "gpus": _gpu_count(fields[2])}
    return jobs


def _journal_ownership(policy, jobs):
    owners, unresolved = {}, {}
    for project, body in policy.get("projects", {}).items():
        for path in body.get("request_journals", []):
            journal = _read(path)
            if not isinstance(journal.get("requests"), list):
                raise ValueError("malformed registered journal")
            for row in journal["requests"]:
                job, state = str(row.get("job_id", "")), str(row.get("state", "")).lower()
                if job.isdigit() and job in jobs and jobs[job]["gpus"]:
                    if job in owners:
                        raise ValueError("duplicate live GPU owner: " + job)
                    owners[job] = project
                elif not job.isdigit() and state in control.ACTIVE:
                    count = row.get("requested_gpus", 1)
                    if type(count) is not int or count < 1:
                        raise ValueError("invalid unresolved GPU intent")
                    unresolved[project] = unresolved.get(project, 0) + count
    return owners, unresolved


def _native_entry(job_id, journal_path):
    journal = _read(journal_path)
    matches = [row for row in journal.get("requests", [])
               if str(row.get("job_id")) == str(job_id)
               and row.get("state") in {"pending", "granted"}]
    if len(matches) != 1:
        raise ValueError("legacy native source is not unique: " + str(job_id))
    output = subprocess.check_output(["scontrol", "show", "job", "-o", str(job_id)],
                                     text=True, timeout=20).strip()
    fields = dict(token.split("=", 1) for token in shlex.split(output) if "=" in token)
    if (fields.get("JobState") not in {"PENDING", "RUNNING"}
            or fields.get("UserId", "").split("(")[0] != os.environ["USER"]
            or fields.get("Account") != "pi-mengy"
            or "gres/gpu:a100=1" not in fields.get("AllocTRES", "")):
        raise ValueError("legacy native Slurm identity changed: " + str(job_id))
    request = matches[0]
    request_sha = hashlib.sha256(json.dumps(
        request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"job_id": str(job_id), "request_journal": str(journal_path),
            "journal_limit": journal.get("limit"), "request_entry_sha256": request_sha,
            "request_entry": request, "journal_state": request["state"],
            "slurm": {"state": fields["JobState"], "user": os.environ["USER"],
                      "account": fields["Account"], "alloc_tres": fields["AllocTRES"],
                      "command": fields.get("Command"), "start_time": fields.get("StartTime"),
                      "node_list": fields.get("NodeList")},
            "classification": "non_project_legacy_native"}


def _ownership(contract, *, make_overlay=False, now=None):
    jobs = _scheduler_inventory()
    policy = _read(contract["role_policy"])
    owners, unresolved = _journal_ownership(policy, jobs)
    entries = []
    for job, source in contract["native_sources"].items():
        if job in jobs and jobs[job]["gpus"]:
            if job in owners:
                raise ValueError("legacy native job also has project owner: " + job)
            owners[job] = "legacy_native_overlay"
            if make_overlay:
                entries.append(_native_entry(job, source))
    live = {job for job, row in jobs.items() if row["gpus"]}
    if live != set(owners):
        raise ValueError("live GPU ownership mismatch: " + repr(sorted(live - set(owners))))
    result = {"account_running_pending": sum(row["gpus"] for row in jobs.values()),
              "radon_running_pending": sum(jobs[job]["gpus"] for job, owner in owners.items()
                                           if owner == "Radon_Bridge"),
              "unresolved_gpu_intents": sum(unresolved.values()),
              "radon_unresolved_gpu_intents": unresolved.get("Radon_Bridge", 0),
              "all_live_owned_once": True}
    if make_overlay:
        import datetime
        current = datetime.datetime.fromtimestamp(
            time.time() if now is None else now, datetime.timezone.utc)
        expiry = datetime.datetime.fromtimestamp(contract["expires_at"], datetime.timezone.utc)
        overlay = {"schema": "legacy_native_ownership_overlay_v2",
                   "checked_at_utc": current.isoformat().replace("+00:00", "Z"),
                   "expires_at_utc": expiry.isoformat().replace("+00:00", "Z"),
                   "test": False, "role_policy": contract["role_policy"],
                   "role_policy_sha256": contract["temporary_policy_sha256"],
                   "native_max_gpus": 0, "entries": entries,
                   "live_intersection_rule": "Fresh Slurm PENDING/RUNNING first; each live job must occur in exactly one project journal or this overlay.",
                   "semantics": {"creates_project_entitlement": False,
                                 "permits_new_native_request": False,
                                 "on_job_exit_or_source_change": "fail_closed_and_resign_under_same_account_lock"}}
        result["overlay"] = overlay
    return result


def _immutable_text(path, value, mode=0o700):
    path = Path(path)
    raw = value.encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("immutable wrapper conflicts: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        try: os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw: raise RuntimeError("wrapper race")
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _module_contract(contract):
    if str(Path(os.sys.executable).resolve()) != str(Path(contract["python"]).resolve()):
        raise RuntimeError("coordinator must use the pinned production Python")
    observed = {}
    for name in sorted(control.MODULE_NAMES):
        module = importlib.import_module(name)
        path = str(Path(module.__file__).resolve())
        expected_root = (contract["source_root"] if name.startswith("radon_bridge.")
                         else contract["mhd_models_root"] if name.startswith("mhd_models.")
                         else contract["framework_root"])
        if Path(expected_root).resolve() not in Path(path).parents:
            raise RuntimeError("runtime module escaped its pinned source root: " + name)
        observed[name] = {"name": name, "path": path, "sha256": _sha(path)}
    roles = {role: observed[module]["path"] for role, module in control.ROLE_MODULES.items()}
    roles.update({"allocation_wrapper": contract["allocation_wrapper"],
                  "finalizer_wrapper": contract["finalizer_wrapper"],
                  "python_executable": str(Path(contract["python"]).resolve())})
    return {"schema": "radon_v5_runtime_contract_v1",
            "python": str(Path(contract["python"]).resolve()),
            "python_sha256": _sha(contract["python"]),
            "python_realpath": str(Path(contract["python"]).resolve(strict=True)),
            "python_version": platform.python_version(),
            "torch_version": importlib.import_module("torch").__version__,
            "framework_api": "V5", "framework_commit": handoff.FORMAL_V5_COMMIT,
            "environment": contract["runtime_environment"],
            "modules": [observed[name] for name in sorted(observed)], "roles": roles}


def _build_binding(contract, now):
    if Path(contract["binding"]).exists():
        return _require_binding(contract)
    reservation = _read(contract["reservation_receipt"])
    asset = _read(Path(contract["asset_dir"]) / "asset.json")
    with _locked(contract["account_lock"]):
        if _sha(contract["role_policy"]) != contract["temporary_policy_sha256"]:
            raise RuntimeError("temporary policy is not installed")
        owned = _ownership(contract, make_overlay=True, now=now)
        if (owned["account_running_pending"] > 23
                or owned["radon_running_pending"] > 9
                or owned["unresolved_gpu_intents"] != 0
                or owned["radon_unresolved_gpu_intents"] != 0
                or owned["all_live_owned_once"] is not True):
            raise RuntimeError("capacity or ownership changed before binding construction")
        _immutable_json(contract["ownership_overlay"], owned["overlay"])
        stage2 = _read(contract["look_stage2"])
        gate = {"schema": "radon_paused_sixth_v5_deployment_gate_v1",
                "look_v22_stage2": "accepted", "look_v22_receipt": contract["look_stage2"],
                "look_v22_receipt_sha256": contract["look_stage2_sha256"],
                "look_successor_monitor_job_id": contract["look_successor_job_id"],
                "policy_proposal_review": "accepted",
                "policy_proposal": contract["policy_proposal"],
                "policy_proposal_sha256": contract["policy_proposal_sha256"],
                "accepted_role_policy_sha256": contract["temporary_policy_sha256"],
                "production_dispatch_authorized": True, "test_access": False}
        if stage2["successor_monitor_job_id"] != gate["look_successor_monitor_job_id"]:
            raise RuntimeError("LOOK stage2 identity changed")
        _immutable_json(contract["deployment_gate"], gate)
        exports = "\n".join("export %s=%s" % (key, shlex.quote(value))
                            for key, value in sorted(contract["runtime_environment"].items()))
        python = shlex.quote(str(Path(contract["python"]).resolve()))
        binding = shlex.quote(str(Path(contract["binding"]).resolve()))
        allocation = ("#!/bin/bash\nset -euo pipefail\n" + exports + "\nexec " + python
                      + " -m radon_bridge.runtime.paused_sixth_v5_control --binding "
                      + binding + " --allocation-owner\n")
        finalizer = ("#!/bin/bash\nset -euo pipefail\n" + exports
                     + "\n[[ $# -eq 1 && $1 =~ ^[0-9]+$ ]]\nexec " + python
                     + " -m radon_bridge.runtime.paused_sixth_v5_control --binding "
                     + binding + " --finalize --gpu-job-id \"$1\"\n")
        _immutable_text(contract["allocation_wrapper"], allocation)
        _immutable_text(contract["finalizer_wrapper"], finalizer)
        runtime = _module_contract(contract)
        pin_paths = {row["path"]: row["sha256"] for row in runtime["modules"]}
        for path in (contract["allocation_wrapper"], contract["finalizer_wrapper"],
                     runtime["python"]):
            pin_paths[path] = _sha(path)
        pins = [{"path": path, "sha256": pin_paths[path]} for path in sorted(pin_paths)]
        value = {"schema": "radon_paused_sixth_v5_control_v2",
                 "lease_id": "rb-paused-sixth-v5-20260924", "project": "Radon_Bridge",
                 "run_id": handoff.RUN_ID, "source_commit": contract["source_commit"],
                 "framework_commit": handoff.FORMAL_V5_COMMIT,
                 "account_lock": contract["account_lock"], "role_policy": contract["role_policy"],
                 "role_policy_sha256": contract["temporary_policy_sha256"],
                 "ownership_overlay": contract["ownership_overlay"],
                 "journal": str(Path(contract["root"]) / "requests.json"),
                 "journal_initial_sha256": _sha(Path(contract["root"]) / "requests.json"),
                 "intent": str(Path(contract["root"]) / "intent.json"),
                 "intent_initial_sha256": _sha(Path(contract["root"]) / "intent.json"),
                 "claims_root": contract["claims_root"], "expected_claim": reservation["claim"],
                 "asset_dir": contract["asset_dir"],
                 "asset_sha256": _sha(Path(contract["asset_dir"]) / "asset.json"),
                 "v5_spec_sha256": asset["spec_sha256"],
                 "v5_checkpoint_sha256": asset["checkpoint_sha256"],
                 "execution_dir": contract["execution_dir"], "requested_gpus": 1,
                 "account_limit": 24, "worker_cpus": contract["worker_cpus"],
                 "worker_memory_gib": contract["worker_memory_gib"],
                 "expires_at": contract["expires_at"],
                 "comment_prefix": "rbv5-715620-28fe66b4",
                 "sbatch_command_template": ["sbatch", "--parsable", "--hold",
                     "--account=pi-mengy", "--gres=gpu:a100:1", "--cpus-per-task=16",
                     "--mem=160G", "--time=48:00:00", "--comment={attempt_comment}",
                     contract["allocation_wrapper"]],
                 "finalizer_command_template": ["sbatch", "--parsable",
                     "--account=pi-mengy", "--cpus-per-task=1", "--mem=4G", "--time=00:30:00",
                     "--dependency=afterany:{gpu_job_id}", "--comment={finalizer_comment}",
                     contract["finalizer_wrapper"], "{gpu_job_id}"],
                 "allocation_wrapper": contract["allocation_wrapper"],
                 "finalizer_wrapper": contract["finalizer_wrapper"],
                 "runtime_pins": pins, "runtime_contract": runtime,
                 "retry_policy": {"max_attempts": 3,
                                  "retryable_states": ["paused", "failed"],
                                  "walltime": "48:00:00"},
                 "deployment_gate": contract["deployment_gate"],
                 "deployment_gate_sha256": _sha(contract["deployment_gate"]),
                 "test_access": False, "dispatch_authorized": True}
        _immutable_json(contract["binding"], value)
    return _require_binding(contract)


def _require_binding(contract):
    if not Path(contract["binding"]).is_file():
        raise RuntimeError("immutable runtime binding has not been generated")
    value = control.validate_binding(contract["binding"])
    if (value["role_policy_sha256"] != contract["temporary_policy_sha256"]
            or value["deployment_gate"] != contract["deployment_gate"]
            or value["ownership_overlay"] != contract["ownership_overlay"]
            or value["asset_dir"] != contract["asset_dir"]):
        raise RuntimeError("runtime binding is outside coordinator contract")
    return value


class ProductionOperations:
    """Real operations; injection points exist only for zero-GPU fault tests."""
    def __init__(self): self.fault = lambda phase: None
    def snapshot(self, contract):
        return _ownership(contract)

    def install_policy(self, contract, now):
        return _install_policy(contract, self.snapshot, now, fault=self.fault)
    def reserve(self, contract, now): return _reserve(contract, now)
    def publish_asset(self, contract): return _asset(contract)
    def require_binding(self, contract): return _build_binding(contract, time.time())
    def publisher(self, contract):
        return control.publish_once(
            contract["binding"], snapshot=lambda: control.live_snapshot(_read(contract["binding"])),
            submit=control.submit_sbatch, lookup=control.lookup_slurm_jobs,
            release=control.release_job)


def advance(contract_path, *, operations=None, fault=None, now=None):
    """Advance every recoverable phase once, or fail closed at the first gap."""
    operations = ProductionOperations() if operations is None else operations
    fault = (lambda phase: None) if fault is None else fault
    if hasattr(operations, "fault"):
        operations.fault = fault
    now = time.time() if now is None else now
    contract = validate_contract(contract_path)
    state = _state(contract_path, contract)

    if PHASES.index(state["phase"]) < PHASES.index("policy_installed"):
        state = _mark_pending(contract, state, "policy_installed")
    evidence = operations.install_policy(contract, now)
    if PHASES.index(state["phase"]) < PHASES.index("policy_installed"):
        fault("after_policy_mutation")
        state = _advance_state(contract, state, "policy_installed", evidence)
    if PHASES.index(state["phase"]) < PHASES.index("claim_reserved"):
        state = _mark_pending(contract, state, "claim_reserved")
    reservation = operations.reserve(contract, now)
    if PHASES.index(state["phase"]) < PHASES.index("claim_reserved"):
        fault("after_claim_reservation")
        evidence = {"reservation_receipt_sha256": _sha(contract["reservation_receipt"]),
                    "claim_generation": reservation["claim"]["generation"]}
        state = _advance_state(contract, state, "claim_reserved", evidence)
    if PHASES.index(state["phase"]) < PHASES.index("asset_published"):
        state = _mark_pending(contract, state, "asset_published")
    asset = operations.publish_asset(contract)
    if PHASES.index(state["phase"]) < PHASES.index("asset_published"):
        fault("after_asset_publication")
        evidence = {"asset_sha256": _sha(Path(contract["asset_dir"]) / "asset.json"),
                    "checkpoint_sha256": asset["checkpoint_sha256"],
                    "spec_sha256": asset["spec_sha256"]}
        state = _advance_state(contract, state, "asset_published", evidence)
    if PHASES.index(state["phase"]) < PHASES.index("binding_ready"):
        state = _mark_pending(contract, state, "binding_ready")
    binding = operations.require_binding(contract)
    if PHASES.index(state["phase"]) < PHASES.index("binding_ready"):
        fault("after_binding_validation")
        evidence = {"binding_sha256": _sha(contract["binding"]),
                    "role_policy_sha256": binding["role_policy_sha256"]}
        state = _advance_state(contract, state, "binding_ready", evidence)
    if PHASES.index(state["phase"]) < PHASES.index("publisher_invoked"):
        state = _mark_pending(contract, state, "publisher_invoked")
        result = operations.publisher(contract); fault("after_publisher_invocation")
        evidence = {"publisher_result": result,
                    "journal_sha256": _sha(_read(contract["binding"])["journal"])}
        state = _advance_state(contract, state, "publisher_invoked", evidence)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    print(json.dumps(advance(args.contract), sort_keys=True))


if __name__ == "__main__":
    main()
