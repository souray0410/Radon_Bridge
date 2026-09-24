"""One-time, fail-closed V4 to formal-V5 production handoff.

This is an operator migration tool, never part of normal checkpoint loading.  It
reserves a stopped V4 execution under the existing claim lock, freezes its last
optimizer boundary, and only permits a later V5 claim when an independently
granted GPU job and the current role-budget evidence are supplied.

The early-stop helper deliberately holds the *one run's* claim lock while the
legacy worker observes ``pause.json``.  The frozen legacy dispatcher therefore
cannot release and immediately reacquire the run in the gap between worker exit
and migration reservation.  It does not hold the account lock while waiting.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from radon_bridge.runtime.state import atomic_write_json, file_sha256


FORMAL_V5_COMMIT = "1287681c08846e11364c81653048435482e772a7"
FORMAL_V5_WHEEL = "c022b4f4b0fa1f29458ad1bf9e0d04f6773e9454ab3bd8c07d416294483aab48"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED"}


def read(path):
    return json.loads(Path(path).read_text())


def _lock_existing(path):
    """Open an existing lock path without silently creating a replacement."""
    path = Path(path)
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    handle = os.fdopen(descriptor, "r+")
    before = path.stat()
    after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        handle.close()
        raise RuntimeError("Lock inode changed during open")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _budget_ok(path, expected_sha256, *, granted_job=None):
    if file_sha256(path) != expected_sha256:
        raise ValueError("Role-budget receipt changed")
    value = read(path)
    if (value.get("schema") != "radon_v5_role_budget_gate_v1"
            or value.get("project") != "Radon_Bridge"
            or value.get("test_access") is not False
            or value.get("account_cap") != 24
            or value.get("running_plus_pending", 25) > value.get("account_cap", 0)
            or value.get("role_policy_sha256") != value.get("observed_role_policy_sha256")):
        raise ValueError("A current Radon_Bridge role slot is not proven")
    if granted_job is None:
        if (value.get("project_slot_available") is not True
                or value["running_plus_pending"] >= value["account_cap"]):
            raise ValueError("A current Radon_Bridge role slot is not proven")
    elif str(granted_job) not in {str(item) for item in value.get("registered_job_ids", [])}:
        raise ValueError("Granted V5 job is absent from the role ledger")
    return value


def _terminal_job(job, observe_job):
    row = observe_job(str(job))
    if (row.get("job_id") != str(job) or row.get("state") not in TERMINAL
            or row.get("exit_code") is None):
        raise ValueError("Prior V4 job terminal identity is unproven")
    return row


def _reserve_failed_claim(old, *, run, spec_sha256, old_owner, old_job,
                          old_generation, migration_owner, now):
    expected = {"run_dir": str(Path(run).resolve()), "spec_sha256": spec_sha256,
                "owner": old_owner, "job_id": str(old_job), "generation": old_generation,
                "state": "failed"}
    if any(old.get(key) != value for key, value in expected.items()):
        raise ValueError("Failed V4 claim is not the reviewed source")
    # Use the existing scheduler's blocking state.  Frozen V4 dispatchers do not
    # understand a new claim-state name and would otherwise consider it eligible.
    return dict(old, owner=migration_owner, job_id="migration-cpu", step=None,
                state="claimed", migration_phase="v5_migration_reserved",
                generation=old_generation + 1,
                updated_at=now, previous={key: old.get(key) for key in
                    ("state", "owner", "job_id", "step", "generation")})


def reserve_paused_failed(*, run, spec_path, checkpoint_path, claims,
                          expected_spec_sha256, expected_checkpoint_sha256,
                          expected_owner, expected_job, expected_generation,
                          migration_owner, account_lock, role_budget_path,
                          role_budget_sha256, observe_job, receipt_path, now=None):
    """Close one reviewed failed claim into an exclusive CPU migration owner."""
    now = time.time() if now is None else now
    run = Path(run).resolve(); status = read(run / "status.json")
    if status.get("state") != "paused" or status.get("test_used") is not False:
        raise ValueError("Only a checkpointed non-test paused V4 run can migrate")
    if file_sha256(spec_path) != expected_spec_sha256:
        raise ValueError("Scientific specification changed")
    if file_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("V4 optimizer boundary changed")
    terminal = _terminal_job(expected_job, observe_job)
    budget = _budget_ok(role_budget_path, role_budget_sha256)
    with _lock_existing(account_lock):
        # Recheck mutable inputs immediately before the unique owner transition.
        if (read(run / "status.json").get("state") != "paused"
                or file_sha256(checkpoint_path) != expected_checkpoint_sha256):
            raise RuntimeError("Paused boundary changed at reservation")
        token = claims.mutate(run, lambda old: _reserve_failed_claim(
            old, run=run, spec_sha256=expected_spec_sha256,
            old_owner=expected_owner, old_job=expected_job,
            old_generation=expected_generation, migration_owner=migration_owner,
            now=now))
        receipt = {"schema": "radon_v5_paused_migration_reservation_v1",
                   "state": "v5_migration_reserved", "run_dir": str(run),
                   "spec_sha256": expected_spec_sha256,
                   "checkpoint_sha256": expected_checkpoint_sha256,
                   "old_job_terminal": terminal, "claim": token,
                   "role_budget_sha256": role_budget_sha256,
                   "role_policy_sha256": budget["role_policy_sha256"],
                   "framework_commit": FORMAL_V5_COMMIT,
                   "framework_wheel_sha256": FORMAL_V5_WHEEL,
                   "created_at": now, "dispatch_allowed": False,
                   "test_access": False}
        if Path(receipt_path).exists():
            if read(receipt_path) != receipt:
                raise FileExistsError("Different migration reservation already exists")
        else:
            atomic_write_json(receipt, receipt_path)
    return receipt


def freeze_reserved_boundary(*, reservation_receipt, source_checkpoint,
                             source_spec, output):
    """Copy immutable inputs after reservation; never overwrite an existing freeze."""
    receipt = read(reservation_receipt); output = Path(output)
    if (receipt.get("schema") != "radon_v5_paused_migration_reservation_v1"
            or receipt.get("state") != "v5_migration_reserved"
            or receipt.get("dispatch_allowed") is not False):
        raise ValueError("Migration reservation is not accepted")
    if (file_sha256(source_checkpoint) != receipt["checkpoint_sha256"]
            or file_sha256(source_spec) != receipt["spec_sha256"]):
        raise ValueError("Reserved V4 source changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        shutil.copyfile(source_checkpoint, stage / "last.v4.pt")
        shutil.copyfile(source_spec, stage / "spec.v4.json")
        frozen = {"schema": "radon_v5_frozen_v4_boundary_v1",
                  "reservation_sha256": file_sha256(reservation_receipt),
                  "checkpoint_sha256": file_sha256(stage / "last.v4.pt"),
                  "spec_sha256": file_sha256(stage / "spec.v4.json"),
                  "source_run_dir": receipt["run_dir"], "test_access": False}
        if (frozen["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or frozen["spec_sha256"] != receipt["spec_sha256"]):
            raise RuntimeError("Frozen V4 copy differs")
        atomic_write_json(frozen, stage / "freeze.json")
        os.rename(stage, output)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return frozen


def claim_v5_gpu(*, run, claims, migration_owner, expected_generation,
                 job_id, step, observe_job, role_budget_path,
                 role_budget_sha256, account_lock, gpu_acceptance_path,
                 gpu_acceptance_sha256, production_asset_path,
                 production_asset_sha256, now=None):
    """Transfer a reserved migration to one already granted, accepted V5 GPU owner."""
    now = time.time() if now is None else now
    if file_sha256(gpu_acceptance_path) != gpu_acceptance_sha256:
        raise ValueError("V5 GPU acceptance changed")
    accepted = read(gpu_acceptance_path)
    if (accepted.get("schema") != "radon_v5_exact_next_update_replay_v1"
            or accepted.get("accepted") is not True
            or accepted.get("framework_commit") != FORMAL_V5_COMMIT
            or accepted.get("test_access") is not False):
        raise ValueError("Exact formal-V5 GPU next-update acceptance required")
    if file_sha256(production_asset_path) != production_asset_sha256:
        raise ValueError("Formal-V5 production asset receipt changed")
    asset = read(production_asset_path)
    if (asset.get("schema") != "radon_v5_production_asset_v1"
            or asset.get("state") != "accepted"
            or asset.get("framework_commit") != FORMAL_V5_COMMIT
            or asset.get("strict_cpu_load") is not True
            or asset.get("eligible_for_formal_selection") is not True
            or asset.get("test_access") is not False
            or accepted.get("production_checkpoint_sha256") != asset.get("checkpoint_sha256")):
        raise ValueError("Accepted V5 replay does not bind the production asset")
    row = observe_job(str(job_id))
    if (row.get("job_id") != str(job_id) or row.get("state") != "RUNNING"
            or row.get("gpus") != 1 or row.get("account") != "pi-mengy"):
        raise ValueError("Exact granted one-GPU job is unproven")
    _budget_ok(role_budget_path, role_budget_sha256, granted_job=job_id)
    with _lock_existing(account_lock):
        _budget_ok(role_budget_path, role_budget_sha256, granted_job=job_id)
        def transition(old):
            if (old.get("owner") != migration_owner
                    or old.get("state") != "claimed"
                    or old.get("migration_phase") != "v5_migration_reserved"
                    or old.get("generation") != expected_generation):
                raise RuntimeError("Migration owner changed before V5 claim")
            return dict(old, owner="radon-v5-production-" + str(job_id),
                        job_id=str(job_id), step=str(step), state="claimed",
                        generation=expected_generation + 1, updated_at=now,
                        framework_commit=FORMAL_V5_COMMIT,
                        production_asset_sha256=production_asset_sha256,
                        previous={key: old.get(key) for key in
                            ("state", "owner", "job_id", "step", "generation")})
        return claims.mutate(run, transition)


def early_pause_to_reservation(*, run, claims, expected_claim,
                               pause_path, pause_receipt, observe_step,
                               checkpoint_path, migration_owner, timeout,
                               poll_interval=2, now=time.time, sleep=time.sleep):
    """Request a boundary pause while excluding legacy release/reacquire.

    The caller must already have installed a reviewed V5 handoff registry entry
    and suppressed new V4 demand.  No signal or cancellation is issued here.
    """
    run = Path(run).resolve(); claim_path = Path(claims.path(run))
    lock_path = claim_path.with_suffix(".lock")
    deadline = now() + timeout
    with _lock_existing(lock_path):
        current = read(claim_path)
        if any(current.get(k) != v for k, v in expected_claim.items()):
            raise ValueError("Live V4 claim identity changed")
        if Path(pause_path).exists():
            if read(pause_path) != pause_receipt:
                raise FileExistsError("Different pause request already exists")
        else:
            atomic_write_json(pause_receipt, pause_path)
        while True:
            step = observe_step(str(current["job_id"]), str(current["step"]))
            status = read(run / "status.json")
            if step.get("state") in TERMINAL:
                if status.get("state") != "paused":
                    raise RuntimeError("Worker died without publishing paused boundary")
                digest = file_sha256(checkpoint_path)
                migrated = dict(current, owner=migration_owner, job_id="migration-cpu",
                    step=None, state="claimed", migration_phase="v5_migration_reserved",
                    generation=current["generation"] + 1, updated_at=now(),
                    previous={key: current.get(key) for key in
                        ("state", "owner", "job_id", "step", "generation")})
                atomic_write_json(migrated, claim_path)
                return {"state": "v5_migration_reserved", "checkpoint_sha256": digest,
                        "old_step_terminal": step, "claim": migrated,
                        "test_access": False}
            if now() >= deadline:
                raise TimeoutError("V4 worker did not reach a complete pause boundary")
            sleep(poll_interval)
