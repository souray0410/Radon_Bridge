"""One-shot V5 qualification lease through the existing account lock.

This is an adapter for the established R&B dispatcher journal, not a daemon or
refiller.  A separately accepted role-policy receipt must grant the lease.
"""
import fcntl
import copy
import json
import os
from pathlib import Path
import re
import subprocess
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
        "previous_role_policy_sha256",
        "account_lock_inode", "account_lock_device", "account_lock_ctime_ns",
        "journal_initial_sha256", "intent_initial_sha256",
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
            or not HEX.fullmatch(lease["previous_role_policy_sha256"])
            or not HEX.fullmatch(lease["control_sha256"])
            or not HEX.fullmatch(lease["journal_initial_sha256"])
            or not HEX.fullmatch(lease["intent_initial_sha256"])
            or any(type(lease[k]) is not int or lease[k] <= 0 for k in
                   ("account_lock_inode", "account_lock_device", "account_lock_ctime_ns"))):
        raise ValueError("Lease identity is incomplete")
    returned = lease["return_entitlement"]
    if returned != {"project": "Uncertainty_Lab", "gpus": 2}:
        raise ValueError("Lease must preserve and return the Uncertainty_Lab two-GPU entitlement")
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
                 role_policy_proposal_path, policy_transition_receipt_path,
                 account_lock, journal_path, intent_path, snapshot, submit, now=None):
    """Submit once after revalidating every identity under the shared lock."""
    from radon_bridge.runtime.state import file_sha256
    now = time.time() if now is None else now
    lock = Path(account_lock)
    try:
        descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError as error:
        raise ValueError("Established account lock is missing") from error
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"action": "none", "state": "waiting_account_lock"}
        lease = validate(read(lease_path), now)
        lock_stat = os.fstat(handle.fileno())
        if (lock_stat.st_ino, lock_stat.st_dev, lock_stat.st_ctime_ns) != (
                lease["account_lock_inode"], lease["account_lock_device"],
                lease["account_lock_ctime_ns"]):
            raise ValueError("Account lock identity changed")
        live_policy_sha = file_sha256(role_policy_path)
        proposal_sha = file_sha256(role_policy_proposal_path)
        if proposal_sha != lease["role_policy_sha256"]:
            raise ValueError("Role policy proposal changed")
        transition_path = Path(policy_transition_receipt_path)
        if live_policy_sha == lease["previous_role_policy_sha256"]:
            if transition_path.exists():
                raise ValueError("Unexpected prior role transition receipt")
            current=read(role_policy_path);expected=copy.deepcopy(current)
            journals=expected.get("projects",{}).get("Uncertainty_Lab",{}).get("request_journals")
            if not isinstance(journals,list) or str(journal_path) in journals:
                raise ValueError("Borrowed journal transition is not unique")
            journals.append(str(journal_path))
            if read(role_policy_proposal_path)!=expected:
                raise ValueError("Proposal must only register the borrowed journal under Uncertainty_Lab")
            install_required=True
        elif live_policy_sha == lease["role_policy_sha256"]:
            if not transition_path.is_file():raise ValueError("Missing role transition receipt")
            receipt=read(transition_path)
            if (receipt.get("schema")!="radon_v5_role_policy_transition_v1"
                    or receipt.get("lease_id")!=lease["lease_id"]
                    or receipt.get("previous_sha256")!=lease["previous_role_policy_sha256"]
                    or receipt.get("installed_sha256")!=lease["role_policy_sha256"]):
                raise ValueError("Role transition receipt changed")
            install_required=False
        else:
            raise ValueError("Role policy changed")
        if file_sha256(control_path) != lease["control_sha256"]:
            raise ValueError("Refiller control changed")
        control = read(control_path)
        if control.get("stop_future_requests") is not True:
            raise ValueError("Legacy refiller must remain paused during qualification")
        if file_sha256(packet_path) != lease["packet_sha256"]:
            raise ValueError("Qualification packet changed")
        journal = read(journal_path)
        if journal.get("schema") != "radon_v5_qualification_requests_v1":
            raise ValueError("Qualification requires its dedicated role-registered journal")
        if not journal.get("requests") and file_sha256(journal_path) != lease["journal_initial_sha256"]:
            raise ValueError("Initial qualification journal changed")
        decision = decide(lease, snapshot(), journal, now=now)
        if decision["action"] != "submit_once":
            return decision
        intent = read(intent_path)
        if intent.get("schema") != "radon_v5_qualification_intent_v1":
            raise ValueError("Unknown qualification intent journal")
        if intent.get("attempt") is not None:
            return {"action": "none", "state": "submission_intent_needs_review"}
        if file_sha256(intent_path) != lease["intent_initial_sha256"]:
            raise ValueError("Initial qualification intent changed")
        packet = read(packet_path)
        command = packet.get("command")
        finalizer = packet.get("finalizer_command_template")
        if (packet.get("schema") != "radon_v5_next_update_packet_v1"
                or packet.get("test_access") is not False
                or packet.get("requested_gpus") != 1 or packet.get("submit_timeout_seconds") != 20
                or not isinstance(command, list) or not command
                or not all(isinstance(x, str) and x for x in command)
                or command[0] != "sbatch" or "--parsable" not in command
                or not any(x == "--gres=gpu:a100:1" for x in command)
                or not isinstance(finalizer,list) or not finalizer
                or finalizer[0]!="sbatch" or "--parsable" not in finalizer
                or sum(x.count("{gpu_job_id}") for x in finalizer)!=1
                or any("--gres=" in x for x in finalizer)):
            raise ValueError("Invalid qualification packet")
        expectation = packet.get("receipt_expectation")
        if (not isinstance(expectation, dict)
                or set(expectation) != {"attempt_root", "source_commit", "inputs_sha256"}
                or not isinstance(expectation["attempt_root"], str)
                or not expectation["attempt_root"]
                or not isinstance(expectation["source_commit"], str)
                or not expectation["source_commit"]
                or not isinstance(expectation["inputs_sha256"], dict)
                or not expectation["inputs_sha256"]
                or any(not isinstance(k, str) or not HEX.fullmatch(v)
                       for k, v in expectation["inputs_sha256"].items())):
            raise ValueError("Qualification receipt expectation is incomplete")
        if install_required:
            # All static, journal and capacity gates passed. Install the exact
            # reviewed bytes immediately before the durable intent and sbatch.
            proposal = Path(role_policy_proposal_path).read_bytes()
            target = Path(role_policy_path)
            fd, temporary = tempfile.mkstemp(prefix=target.name+".",suffix=".partial",dir=target.parent)
            try:
                with os.fdopen(fd,"wb") as stream:
                    stream.write(proposal);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,target)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            receipt={"schema":"radon_v5_role_policy_transition_v1",
                "lease_id":lease["lease_id"],"previous_sha256":live_policy_sha,
                "installed_sha256":file_sha256(target),"time":now}
            if receipt["installed_sha256"]!=lease["role_policy_sha256"]:
                raise ValueError("Installed role policy bytes changed")
            write(transition_path,receipt)
        entry = {"lease_id": lease["lease_id"], "packet_sha256": lease["packet_sha256"],
                 "state": "intent", "requested_gpus": 1, "time": now}
        intent["attempt"] = entry; write(intent_path, intent)
        job_id = str(submit(command, timeout=20))
        if not job_id.isdigit():
            entry.update(state="identity_drift", observed_job_id=job_id)
            write(intent_path, intent)
        else:
            entry.update(state="gpu_submitted_pending_finalizer",job_id=job_id)
            # Register the numeric GPU identity immediately.  The role-budget
            # reader does not need a finalizer id, and this closes the window in
            # which a live GPU could be absent from the borrowed UL journal.
            journal["requests"].append(dict(entry));write(journal_path,journal)
            write(intent_path, intent)
            final_command=[x.replace("{gpu_job_id}",job_id) for x in finalizer]
            try:
                finalizer_id=str(submit(final_command,timeout=20))
            except BaseException as error:
                entry.update(state="finalizer_submission_needs_review",
                             finalizer_error_type=type(error).__name__)
                journal["requests"][-1]=dict(entry);write(journal_path,journal)
                write(intent_path,intent)
                raise
            if not finalizer_id.isdigit():
                entry.update(state="finalizer_identity_drift",observed_finalizer_job_id=finalizer_id)
                journal["requests"][-1]=dict(entry);write(journal_path,journal)
                write(intent_path,intent)
            else:
                entry.update(state="submitted",finalizer_job_id=finalizer_id)
                journal["requests"][-1]=dict(entry); write(journal_path,journal)
                write(intent_path,intent)
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


def finalize_once(*, lease_path, account_lock, journal_path, intent_path,
                  finalizer_job_id, gpu_job_id, slurm_state, exit_code,
                  packet_path, qualification_receipt_path, status_path, now=None):
    """Close the registered lease from its zero-GPU afterany finalizer."""
    now=time.time() if now is None else now;lease=read(lease_path)
    validate(lease,min(now,lease.get("expires_at",now)-1))
    descriptor=os.open(account_lock,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(descriptor,"r+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        st=os.fstat(handle.fileno())
        if (st.st_ino,st.st_dev,st.st_ctime_ns)!=(lease["account_lock_inode"],
                lease["account_lock_device"],lease["account_lock_ctime_ns"]):
            raise ValueError("Account lock identity changed")
        journal=read(journal_path);matches=[e for e in journal.get("requests",[])
            if e.get("lease_id")==lease["lease_id"]]
        if len(matches)!=1:raise ValueError("Qualification journal identity changed")
        entry=matches[0]
        if entry.get("job_id")!=str(gpu_job_id) or entry.get("finalizer_job_id")!=str(finalizer_job_id):
            raise ValueError("Qualification/finalizer job identity changed")
        if entry.get("state") in TERMINAL:return entry
        if entry.get("state")!="submitted":raise ValueError("Qualification is not submitted")
        if slurm_state.split()[0]=="COMPLETED" and exit_code=="0:0":
            from radon_bridge.runtime.state import file_sha256
            if file_sha256(packet_path)!=lease["packet_sha256"]:
                raise ValueError("Qualification packet changed before finalization")
            packet=read(packet_path);expected=packet.get("receipt_expectation",{})
            expected_path=(Path(expected.get("attempt_root", ""))/
                           ("attempt_"+str(gpu_job_id))/"receipt.json")
            if Path(qualification_receipt_path)!=expected_path:
                raise ValueError("Qualification receipt attempt path changed")
            receipt=read(qualification_receipt_path)
            if (receipt.get("schema")!="radon_v5_exact_next_update_replay_v1"
                    or receipt.get("status")!="accepted_engineering_only"
                    or receipt.get("dispatch_allowed") is not False
                    or receipt.get("gpu_job_id")!=str(gpu_job_id)
                    or receipt.get("packet_sha256")!=lease["packet_sha256"]
                    or receipt.get("source_commit")!=expected.get("source_commit")
                    or receipt.get("inputs_sha256")!=expected.get("inputs_sha256")):
                raise ValueError("Qualification completion receipt is not accepted")
        terminal_transition(entry,slurm_state=slurm_state,exit_code=exit_code)
        intent=read(intent_path);intent["attempt"]=dict(entry)
        write(journal_path,journal);write(intent_path,intent)
        status={"schema":"radon_v5_qualification_status_v1","lease_id":lease["lease_id"],
            "state":entry["state"],"gpu_job_id":str(gpu_job_id),
            "finalizer_job_id":str(finalizer_job_id),"updated_at":now,
            "test_access":False,"dispatch_allowed":False,
            "next_gate":"independent replay review before a formal V5 production claim"}
        write(status_path,status)
        return entry


def submit_sbatch(command, *, timeout):
    """Bound the scheduler RPC while the account lock protects admission."""
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("sbatch failed: " + result.stderr.strip())
    return result.stdout.strip().split(";", 1)[0]
