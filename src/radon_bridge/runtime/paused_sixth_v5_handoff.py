"""Concrete handoff for the one reviewed paused R&B run.

This module consumes the artifacts produced by the existing converter and the
real A100 exact-next-update attempt.  It is deliberately not a generic V4
compatibility loader.  A production asset is immutable and a later claim can
only bind an already granted Slurm allocation observed through the scheduler
and its registered request journal.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from radon_bridge.runtime.state import atomic_write_json, file_sha256


RUN_ID = "2026_09_12_11_23_31_715620"
FORMAL_V5_COMMIT = "1287681c08846e11364c81653048435482e772a7"
SOURCE_CHECKPOINT_SHA256 = "792748f7a1c0c4896ca84617b1f9b9712a18392d279dc70dd42b8b76a2c1c0f5"
SOURCE_SPEC_SHA256 = "6c62039f7cf82a49d060c527a771b684ee49cd8e3d72906ef38fb2e2f1322be5"
QUALIFIED_CHECKPOINT_SHA256 = "bb5438f72b17a3ee4189106d04f22d50bcacb3fb1b11114a47fe8a56233b1dac"
QUALIFIED_SPEC_SHA256 = "36e941c0629023cef8e6571207849b94c53a9649816deec1efc8aa94dd8c53dc"
PACKET_SHA256 = "9ed543018902173d582e8203077f7c7d180b6526a6dc6fe47c6d640947a17670"
OLD_JOB_ID = "52347024"
OLD_OWNER = "radon-workflow-52347024"
OLD_GENERATION = 110
OLD_CLAIM_SHA256 = "bbabe60c58ca2cfabda41ebf61d8f9c7bc946673bfe3ccc196f24235a32a9d58"
PAUSED_STATUS_SHA256 = "ce6fe3d10edebb5433b53aa5475284d76a2e377fc6c063d29a3184cb126ae348"


def _read(path):
    return json.loads(Path(path).read_text())


def _locked_existing(path):
    path = Path(path)
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    handle = os.fdopen(fd, "r+")
    before, after = path.stat(), os.fstat(fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        handle.close()
        raise RuntimeError("lock inode changed")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _exact_file(path, digest, label):
    if file_sha256(path) != digest:
        raise ValueError(label + " changed")


def reserve_actual_failed_claim(*, run, claims, claim_path, source_checkpoint,
                                source_spec, account_lock, observe_job,
                                reservation_receipt, now):
    """Reserve only the reviewed failed sixth claim before producing assets."""
    run = Path(run).resolve()
    expected = {
        "run_dir": str(run), "spec_sha256": SOURCE_SPEC_SHA256,
        "owner": OLD_OWNER, "job_id": OLD_JOB_ID, "generation": OLD_GENERATION,
        "state": "failed", "step": "1",
    }
    _exact_file(run / "status.json", PAUSED_STATUS_SHA256, "paused status")
    status = _read(run / "status.json")
    if (status.get("state") != "paused" or status.get("test_used") is not False
            or status.get("updates") != 26904):
        raise ValueError("run is not the reviewed paused optimizer boundary")
    _exact_file(source_checkpoint, SOURCE_CHECKPOINT_SHA256, "source checkpoint")
    _exact_file(source_spec, SOURCE_SPEC_SHA256, "source spec")
    terminal = observe_job(OLD_JOB_ID)
    if (terminal.get("job_id") != OLD_JOB_ID or terminal.get("state") != "COMPLETED"
            or terminal.get("exit_code") != "0:0"):
        raise ValueError("old allocation terminal identity is unproven")
    with _locked_existing(account_lock):
        exact_claim = _read(claim_path)
        if (_read(run / "status.json") != status
                or file_sha256(source_checkpoint) != SOURCE_CHECKPOINT_SHA256):
            raise RuntimeError("paused source changed inside account lock")
        if file_sha256(claim_path) == OLD_CLAIM_SHA256:
            if any(exact_claim.get(key) != value for key, value in expected.items()):
                raise ValueError("failed claim fields are not the reviewed source")
            def transition(old):
                if old != exact_claim:
                    raise RuntimeError("claim changed before reservation")
                return dict(old, state="claimed", owner="radon-v5-paused-sixth-migration",
                            job_id="migration-cpu", step=None, generation=OLD_GENERATION + 1,
                            migration_phase="source_reserved", updated_at=now,
                            previous={k: old[k] for k in
                                      ("state", "owner", "job_id", "step", "generation")})
            reserved = claims.mutate(run, transition)
        else:
            reserved = exact_claim
            if (reserved.get("state") != "claimed"
                    or reserved.get("owner") != "radon-v5-paused-sixth-migration"
                    or reserved.get("job_id") != "migration-cpu"
                    or reserved.get("step") is not None
                    or reserved.get("generation") != OLD_GENERATION + 1
                    or reserved.get("migration_phase") != "source_reserved"
                    or reserved.get("previous") != {k: expected[k] for k in
                                                     ("state", "owner", "job_id", "step", "generation")}):
                raise RuntimeError("claim is neither original nor recoverable reservation")
        receipt = {"schema":"radon_paused_sixth_v5_reservation_v1",
                   "run_id":RUN_ID,"claim":reserved,
                   "source_checkpoint_sha256":SOURCE_CHECKPOINT_SHA256,
                   "source_spec_sha256":SOURCE_SPEC_SHA256,
                   "old_job_terminal":terminal,"test_access":False,
                   "dispatch_allowed":False}
        reservation_receipt = Path(reservation_receipt)
        if reservation_receipt.exists():
            if _read(reservation_receipt) != receipt:
                raise RuntimeError("reservation receipt conflicts")
        else:
            atomic_write_json(receipt, reservation_receipt)
        return receipt


def validate_real_chain(*, source_checkpoint, source_spec, conversion_receipt,
                        converted_checkpoint, packet, independent_recheck,
                        attempt_checkpoint, attempt_spec):
    """Validate the actual 26904->26905 evidence without writing anything."""
    _exact_file(source_checkpoint, SOURCE_CHECKPOINT_SHA256, "source checkpoint")
    _exact_file(source_spec, SOURCE_SPEC_SHA256, "source spec")
    _exact_file(packet, PACKET_SHA256, "qualification packet")
    _exact_file(attempt_checkpoint, QUALIFIED_CHECKPOINT_SHA256, "A100 V5 checkpoint")
    _exact_file(attempt_spec, QUALIFIED_SPEC_SHA256, "A100 V5 spec")
    conversion = _read(conversion_receipt)
    if (conversion.get("schema") != "v4_to_v5_checkpoint_candidate_v1"
            or conversion.get("state") != "awaiting_numerical_replay"
            or conversion.get("source_sha256") != SOURCE_CHECKPOINT_SHA256
            or conversion.get("target_sha256") != file_sha256(converted_checkpoint)
            or conversion.get("tensor_values_dtypes_preserved") is not True
            or conversion.get("target_framework", {}).get("api") != "V5"
            or conversion.get("target_framework", {}).get("commit") != FORMAL_V5_COMMIT
            or conversion.get("original_source_modified") is not False
            or conversion.get("training_repeated") is not False
            or conversion.get("dispatch_authorized") is not False):
        raise ValueError("conversion receipt does not bind the reviewed source")
    packet_value = _read(packet)
    inputs = packet_value.get("receipt_expectation", {}).get("inputs_sha256", {})
    if (packet_value.get("schema") != "radon_v5_next_update_packet_v1"
            or packet_value.get("test_access") is not False
            or packet_value.get("dispatch_allowed") is not False
            or inputs.get("v4_checkpoint") != SOURCE_CHECKPOINT_SHA256
            or inputs.get("v4_spec") != SOURCE_SPEC_SHA256
            or inputs.get("v5_checkpoint") != file_sha256(converted_checkpoint)
            or inputs.get("v5_spec") != QUALIFIED_SPEC_SHA256):
        raise ValueError("qualification packet does not bind conversion")
    recheck = _read(independent_recheck)
    checked = recheck.get("independent_read_only_recheck", {})
    scheduler = recheck.get("scheduler", {})
    if (recheck.get("schema") != "radon_v5_next_update_independent_recheck_v1"
            or recheck.get("state") != "accepted_engineering_only"
            or recheck.get("source_run_id") != RUN_ID
            or recheck.get("test_access") is not False
            or recheck.get("dispatch_allowed") is not False
            or scheduler.get("gpu_job_id") != "52387276"
            or checked.get("v5_output_checkpoint_sha256") != QUALIFIED_CHECKPOINT_SHA256
            or checked.get("progress") != {"epoch": 8, "offset": 21568, "updates": 26905}
            or checked.get("errors") != 0 or checked.get("tensor_exact") != 1093
            or checked.get("numpy_rng_exact") != 1):
        raise ValueError("independent A100 recheck is not the reviewed acceptance")
    state = torch.load(attempt_checkpoint, map_location="cpu", weights_only=False)
    if (state.get("schema") != "optimizer_boundary_v2"
            or state.get("framework_api") != "V5"
            or {k: state.get("progress", {}).get(k) for k in ("epoch", "offset", "updates")}
               != checked["progress"]):
        raise ValueError("qualified checkpoint is not the accepted V5 boundary")
    return {"conversion": conversion, "packet": packet_value, "recheck": recheck}


def publish_production_asset(*, output, reservation_receipt, **chain):
    """Freeze update 26905 as the sole resumable V5 asset (O_EXCL semantics)."""
    evidence = validate_real_chain(**chain)
    reservation = _read(reservation_receipt)
    if (reservation.get("schema") != "radon_paused_sixth_v5_reservation_v1"
            or reservation.get("run_id") != RUN_ID
            or reservation.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or reservation.get("dispatch_allowed") is not False):
        raise ValueError("exact paused-sixth reservation required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        shutil.copyfile(chain["attempt_checkpoint"], stage / "last.pt")
        shutil.copyfile(chain["attempt_spec"], stage / "spec.json")
        receipt = {
            "schema": "radon_paused_sixth_v5_production_asset_v1",
            "state": "accepted_for_new_claim",
            "run_id": RUN_ID,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_spec_sha256": SOURCE_SPEC_SHA256,
            "checkpoint_sha256": file_sha256(stage / "last.pt"),
            "spec_sha256": file_sha256(stage / "spec.json"),
            "progress": {"epoch": 8, "offset": 21568, "updates": 26905},
            "framework_commit": FORMAL_V5_COMMIT,
            "conversion_receipt_sha256": file_sha256(chain["conversion_receipt"]),
            "qualification_packet_sha256": PACKET_SHA256,
            "independent_recheck_sha256": file_sha256(chain["independent_recheck"]),
            "reservation_receipt_sha256": file_sha256(reservation_receipt),
            "qualification_gpu_job_id": evidence["recheck"]["scheduler"]["gpu_job_id"],
            "strict_cpu_load": True,
            "test_access": False,
            "dispatch_allowed": False,
        }
        atomic_write_json(receipt, stage / "asset.json")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


def _journal_grant(path, *, job_id, run_dir, asset_sha256):
    journal = _read(path)
    matches = []
    for request in journal.get("requests", []):
        if (str(request.get("job_id")) == str(job_id)
                and request.get("state") == "granted"):
            matches.append(request)
    if len(matches) != 1:
        raise ValueError("job must have exactly one granted request")
    request = matches[0]
    metadata = request.get("metadata", {})
    if (metadata.get("project") != "Radon_Bridge"
            or metadata.get("run_dir") != str(Path(run_dir).resolve())
            or metadata.get("production_asset_sha256") != asset_sha256
            or request.get("requested_gpus") != 1):
        raise ValueError("grant is not bound to the paused-sixth asset")
    return request


def claim_granted_v5(*, run, claims, asset_dir, expected_claim,
                     journal_path, journal_sha256, job_id, observe_job,
                     account_lock, now):
    """Atomically bind the real scheduler-observed step; caller cannot supply it."""
    asset_dir = Path(asset_dir)
    asset_path = asset_dir / "asset.json"
    asset = _read(asset_path)
    asset_sha = file_sha256(asset_path)
    if (asset.get("schema") != "radon_paused_sixth_v5_production_asset_v1"
            or asset.get("state") != "accepted_for_new_claim"
            or asset.get("run_id") != RUN_ID
            or asset.get("checkpoint_sha256") != file_sha256(asset_dir / "last.pt")
            or asset.get("spec_sha256") != file_sha256(asset_dir / "spec.json")
            or asset.get("framework_commit") != FORMAL_V5_COMMIT):
        raise ValueError("production asset is not accepted")
    _exact_file(journal_path, journal_sha256, "registered request journal")
    grant = _journal_grant(journal_path, job_id=job_id, run_dir=run,
                           asset_sha256=asset_sha)
    observed = observe_job(str(job_id))
    steps = observed.get("steps", [])
    if (observed.get("job_id") != str(job_id) or observed.get("state") != "RUNNING"
            or observed.get("account") != "pi-mengy" or observed.get("gpus") != 1
            or len(steps) != 1 or steps[0].get("state") != "RUNNING"):
        raise ValueError("unique running Slurm job/step is unproven")
    step = str(steps[0].get("step"))
    if not step or step == "None":
        raise ValueError("Slurm step identity missing")
    with _locked_existing(account_lock):
        _exact_file(journal_path, journal_sha256, "registered request journal")
        _journal_grant(journal_path, job_id=job_id, run_dir=run,
                       asset_sha256=asset_sha)
        current = observe_job(str(job_id))
        if current != observed:
            raise RuntimeError("Slurm identity changed inside account lock")
        def transition(old):
            if any(old.get(k) != v for k, v in expected_claim.items()):
                raise RuntimeError("paused-sixth claim changed")
            return dict(old, state="claimed", owner="radon-v5-production-" + str(job_id),
                        job_id=str(job_id), step=step,
                        generation=old["generation"] + 1, updated_at=now,
                        framework_commit=FORMAL_V5_COMMIT,
                        production_asset_path=str(asset_path.resolve()),
                        production_asset_sha256=asset_sha,
                        registered_journal_path=str(Path(journal_path).resolve()),
                        registered_journal_sha256=journal_sha256,
                        grant_id=grant.get("request_id"),
                        previous={k: old.get(k) for k in
                                  ("state", "owner", "job_id", "step", "generation")})
        return claims.mutate(run, transition)
