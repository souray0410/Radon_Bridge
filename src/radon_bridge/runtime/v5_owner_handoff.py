"""Fail-closed reservation for a natural V4-owner to formal-V5 handoff.

The registry is consumed by the existing dispatcher.  Arming it never pauses a
worker or edits a claim; it only prevents a released ``paused`` run from being
claimed again as V4 while the migration owner freezes the released boundary.
"""
import fcntl
import json
from pathlib import Path
import time

from radon_bridge.runtime.state import atomic_write_json, file_sha256


FORMAL_V5_COMMIT = "1287681c08846e11364c81653048435482e772a7"
FORMAL_V5_WHEEL = "c022b4f4b0fa1f29458ad1bf9e0d04f6773e9454ab3bd8c07d416294483aab48"
ACTIVE = {"armed_waiting_owner_release", "boundary_claimed", "converted",
          "cpu_accepted", "gpu_qualification_pending", "gpu_accepted",
          "production_claim_pending"}


def read(path):
    return json.loads(Path(path).read_text())


def validate(registry):
    if (registry.get("schema") != "radon_v4_to_v5_owner_handoffs_v1"
            or registry.get("test_access") is not False
            or not isinstance(registry.get("entries"), list)):
        raise ValueError("Unknown V5 owner-handoff registry")
    seen = set()
    for row in registry["entries"]:
        required = {"run_dir", "spec_sha256", "old_owner", "old_job_id",
                    "old_claim_generation", "state", "framework_commit",
                    "framework_wheel_sha256", "dispatch_allowed"}
        if set(row) != required or row["run_dir"] in seen:
            raise ValueError("Invalid or duplicate V5 owner-handoff entry")
        seen.add(row["run_dir"])
        if (row["framework_commit"] != FORMAL_V5_COMMIT
                or row["framework_wheel_sha256"] != FORMAL_V5_WHEEL
                or row["dispatch_allowed"] is not False
                or row["state"] not in ACTIVE | {"cancelled", "production_handed_off"}
                or not str(row["old_job_id"]).isdigit()
                or type(row["old_claim_generation"]) is not int):
            raise ValueError("V5 handoff identity is incomplete")
    return registry


def protected(task, claims, registry_path):
    if not registry_path:
        return False
    registry = validate(read(registry_path))
    run = str(Path(task["run_dir"]).resolve())
    rows = [row for row in registry["entries"] if row["run_dir"] == run]
    if not rows or rows[0]["state"] not in ACTIVE:
        return False
    row = rows[0];claim = read(claims.path(run))
    if row["spec_sha256"] != task["spec_sha256"]:
        raise ValueError("Armed V5 handoff specification changed")
    # Before natural release the exact old owner must still be visible.  After
    # release its identity must remain in either the record or its previous link.
    old_matches = (claim.get("owner") == row["old_owner"] and
                   claim.get("job_id") == row["old_job_id"] and
                   claim.get("generation") == row["old_claim_generation"])
    previous = claim.get("previous", {})
    migration_matches = (str(claim.get("owner", "")).startswith("radon-v5-migration-")
                         and claim.get("generation") == row["old_claim_generation"] + 1
                         and previous.get("job_id") == row["old_job_id"])
    if not (old_matches or migration_matches):
        raise ValueError("Armed V5 handoff claim identity drift")
    return True


def arm_once(*, registry_path, account_lock, claims, task, expected_owner,
             expected_job_id, expected_generation, deployment_receipt_path,
             required_dispatcher_commit, now=None):
    """Arm before release; requires proof the live dispatcher reads this guard."""
    now = time.time() if now is None else now
    receipt = read(deployment_receipt_path)
    if (receipt.get("schema") != "radon_v5_handoff_guard_deployment_v1"
            or receipt.get("active") is not True
            or receipt.get("source_commit") != required_dispatcher_commit
            or receipt.get("registry_path") != str(Path(registry_path))
            or receipt.get("test_access") is not False):
        raise ValueError("Active guarded dispatcher deployment is unproven")
    with Path(account_lock).open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        registry = validate(read(registry_path))
        run = str(Path(task["run_dir"]).resolve())
        if any(row["run_dir"] == run for row in registry["entries"]):
            raise ValueError("V5 handoff already registered")
        claim = read(claims.path(run))
        expected = {"owner": expected_owner, "job_id": str(expected_job_id),
                    "generation": expected_generation, "state": "running",
                    "spec_sha256": task["spec_sha256"]}
        if any(claim.get(key) != value for key, value in expected.items()):
            raise ValueError("Live V4 claim is not the reviewed owner")
        registry["entries"].append({
            "run_dir": run, "spec_sha256": task["spec_sha256"],
            "old_owner": expected_owner, "old_job_id": str(expected_job_id),
            "old_claim_generation": expected_generation,
            "state": "armed_waiting_owner_release",
            "framework_commit": FORMAL_V5_COMMIT,
            "framework_wheel_sha256": FORMAL_V5_WHEEL,
            "dispatch_allowed": False})
        atomic_write_json(registry, registry_path)
        return {"state": "armed_waiting_owner_release", "run_dir": run,
                "armed_at": now, "registry_sha256": file_sha256(registry_path)}
