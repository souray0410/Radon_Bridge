"""One-shot V5 qualification lease through the existing account lock.

This is an adapter for the established R&B dispatcher journal, not a daemon or
refiller.  A separately accepted role-policy receipt must grant the lease.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time


HEX = re.compile(r"[0-9a-f]{64}")
TERMINAL = {"completed", "failed", "expired", "identity_drift"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def validate(lease, now=None):
    now = time.time() if now is None else now
    required = {
        "schema", "lease_id", "project", "mode", "requested_gpus",
        "packet_sha256", "expires_at", "test_access", "account_limit",
        "return_entitlement", "role_policy_sha256", "control_sha256",
    }
    if set(lease) != required or lease["schema"] != "radon_v5_qualification_lease_v1":
        raise ValueError("Unknown qualification lease")
    if (lease["project"] != "Radon_Bridge" or lease["mode"] != "qualification"
            or lease["requested_gpus"] != 1 or lease["test_access"] is not False
            or lease["account_limit"] != 24):
        raise ValueError("Lease is outside the bounded R&B qualification scope")
    if (not isinstance(lease["lease_id"], str) or not lease["lease_id"]
            or not HEX.fullmatch(lease["packet_sha256"])
            or not HEX.fullmatch(lease["role_policy_sha256"])
            or not HEX.fullmatch(lease["control_sha256"])):
        raise ValueError("Lease identity is incomplete")
    returned = lease["return_entitlement"]
    if returned != {"project": "Liu", "gpus": 2}:
        raise ValueError("Lease must preserve and return the Liu two-GPU entitlement")
    if type(lease["expires_at"]) not in (int, float) or lease["expires_at"] <= now:
        raise ValueError("Qualification lease expired")
    return lease


def decide(lease, account, journal, *, now=None):
    now = time.time() if now is None else now
    validate(lease, now)
    if account.get("limit") != 24 or type(account.get("total_gpus")) is not int:
        raise ValueError("Unknown account snapshot")
    if account["total_gpus"] > 24:
        raise ValueError("Account already exceeds its GPU limit")
    entries = [r for r in journal.get("requests", []) if r.get("lease_id") == lease["lease_id"]]
    if len(entries) > 1:
        raise ValueError("Duplicate qualification lease journal identity")
    if entries:
        state = entries[0].get("state")
        if state in TERMINAL:
            return {"action": "none", "state": "terminal_" + state}
        return {"action": "none", "state": "already_submitted"}
    if account["total_gpus"] + 1 > 24:
        return {"action": "none", "state": "waiting_account_capacity"}
    return {"action": "submit_once", "state": "qualification_ready"}


def publish_once(*, lease_path, role_policy_path, control_path, packet_path,
                 account_lock, journal_path, snapshot, submit, now=None):
    """Submit once after revalidating every identity under the shared lock."""
    from radon_bridge.runtime.state import file_sha256
    now = time.time() if now is None else now
    lock = Path(account_lock)
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"action": "none", "state": "waiting_account_lock"}
        lease = validate(read(lease_path), now)
        if file_sha256(role_policy_path) != lease["role_policy_sha256"]:
            raise ValueError("Role policy changed")
        if file_sha256(control_path) != lease["control_sha256"]:
            raise ValueError("Refiller control changed")
        control = read(control_path)
        if control.get("stop_future_requests") is not True:
            raise ValueError("Legacy refiller must remain paused during qualification")
        if file_sha256(packet_path) != lease["packet_sha256"]:
            raise ValueError("Qualification packet changed")
        journal = read(journal_path)
        if journal.get("schema") != "radon_bridge_workflow_requests_v1":
            raise ValueError("Unknown R&B dispatcher journal")
        decision = decide(lease, snapshot(), journal, now=now)
        if decision["action"] != "submit_once":
            return decision
        packet = read(packet_path)
        command = packet.get("command")
        if (packet.get("schema") != "radon_v5_next_update_packet_v1"
                or packet.get("test_access") is not False
                or not isinstance(command, list) or not command
                or not all(isinstance(x, str) and x for x in command)):
            raise ValueError("Invalid qualification packet")
        entry = {"lease_id": lease["lease_id"], "packet_sha256": lease["packet_sha256"],
                 "state": "intent", "requested_gpus": 1, "time": now}
        journal["requests"].append(entry); write(journal_path, journal)
        job_id = str(submit(command))
        if not job_id.isdigit():
            entry.update(state="identity_drift", observed_job_id=job_id)
        else:
            entry.update(state="submitted", job_id=job_id)
        write(journal_path, journal)
        return {"action": "none", "state": entry["state"], "job_id": entry.get("job_id")}


def terminal_transition(entry, *, slurm_state, exit_code):
    if entry.get("state") != "submitted" or not str(entry.get("job_id", "")).isdigit():
        raise ValueError("Only an identified submitted lease may terminate")
    state = slurm_state.split()[0]
    if state == "COMPLETED" and exit_code == "0:0":
        entry.update(state="completed", result="qualification_finished_return_entitlement")
    elif state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}:
        entry.update(state="failed", result="qualification_failed_return_entitlement",
                     slurm_state=state, exit_code=exit_code)
    else:
        raise ValueError("Slurm state is not terminal")
    return entry
