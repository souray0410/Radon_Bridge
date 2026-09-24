import json
from pathlib import Path

import pytest

from radon_bridge.runtime import paused_sixth_v5_coordinator as c


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def contract(tmp_path):
    root = tmp_path / "operation"; root.mkdir()
    policy = tmp_path / "policy.json"; initial = {"schema": "research_gpu_roles_v1"}
    proposed = root / "proposed.json"; target = {"schema": "research_gpu_roles_v1", "target": True}
    steady = root / "steady.json"
    dump(policy, initial); dump(proposed, target); dump(steady, {"schema": "research_gpu_roles_v1"})
    stage2 = root / "look.json"
    proposal = root / "proposal.json"
    dump(stage2, {"schema": "look_v5_monitor_stage2_acceptance_v1", "status": "accepted",
                  "successor_monitor_job_id": "800", "accepted_role_policy_sha256": c._sha(proposed),
                  "successor_monitor_identity_verified": True,
                  "predecessor_release_verified": True, "test_access": False})
    # The accepted policy digest is fixed after writing the target policy.
    stage = json.loads(stage2.read_text()); stage["accepted_role_policy_sha256"] = c._sha(proposed)
    # proposal below must bind the same digest.
    dump(proposal, {"schema": "radon_paused_sixth_v5_policy_proposal_v1",
                    "state": "accepted", "role_policy_sha256": c._sha(proposed)})
    reviews = []
    for index in range(4):
        receipt = root / f"review{index}.json"; dump(receipt, {"accepted": True, "n": index})
        reviews.append({"path": str(receipt), "sha256": c._sha(receipt)})
    lock = tmp_path / "account.lock"; lock.write_text("")
    values = {"schema": c.SCHEMA, "root": str(root), "state": str(root / "state.json"),
      "account_lock": str(lock), "role_policy": str(policy),
      "initial_policy_sha256": c._sha(policy), "temporary_policy": str(proposed),
      "temporary_policy_sha256": c._sha(proposed),
      "steady_policy": str(steady), "steady_policy_sha256": c._sha(steady),
      "rollback_requires_future_look_stage2": True,
      "policy_intent": str(root / "policy_intent.json"),
      "policy_transition_receipt": str(root / "policy_receipt.json"),
      "look_stage2": str(stage2), "look_stage2_sha256": c._sha(stage2),
      "look_successor_job_id": "800", "review_receipts": reviews,
      "policy_proposal": str(proposal), "policy_proposal_sha256": c._sha(proposal),
      "run": str(tmp_path / "run"), "claims_root": str(tmp_path / "claims"),
      "claim_path": str(tmp_path / "claim.json"), "source_checkpoint": str(tmp_path / "source.pt"),
      "source_spec": str(tmp_path / "source.json"),
      "reservation_receipt": str(root / "reservation.json"), "asset_dir": str(root / "asset"),
      "conversion_receipt": str(tmp_path / "conversion.json"),
      "converted_checkpoint": str(tmp_path / "converted.pt"), "packet": str(tmp_path / "packet.json"),
      "independent_recheck": str(tmp_path / "recheck.json"),
      "attempt_checkpoint": str(tmp_path / "attempt.pt"), "attempt_spec": str(tmp_path / "attempt.json"),
      "source_root": str(tmp_path / "source_root"), "mhd_models_root": str(tmp_path / "models"),
      "framework_root": str(tmp_path / "framework"), "python": "/usr/bin/python3",
      "ownership_overlay": str(root / "overlay.json"),
      "allocation_wrapper": str(root / "allocation.sbatch"),
      "finalizer_wrapper": str(root / "finalizer.sbatch"),
      "deployment_gate": str(root / "deployment_gate.json"), "binding": str(root / "binding.json"),
      "execution_dir": str(root / "execution"), "native_sources": {},
      "runtime_environment": {key: "" for key in c.control.ENV_KEYS},
      "worker_cpus": 16, "worker_memory_gib": 160, "source_commit": "1" * 40,
      "expires_at": 99999999999, "test_access": False}
    # Correct the circular stage2/proposal fixture by binding both to target policy.
    stage = json.loads(stage2.read_text()); stage["accepted_role_policy_sha256"] = values["temporary_policy_sha256"]
    dump(stage2, stage); values["look_stage2_sha256"] = c._sha(stage2)
    proposal_value = json.loads(proposal.read_text()); proposal_value["role_policy_sha256"] = values["temporary_policy_sha256"]
    dump(proposal, proposal_value); values["policy_proposal_sha256"] = c._sha(proposal)
    path = root / "contract.json"; dump(path, values)
    return path, values


class FakeOperations:
    def __init__(self):
        self.mutations = {name: 0 for name in ("policy", "claim", "asset", "binding", "request")}

    def install_policy(self, value, now):
        if not Path(value["policy_transition_receipt"]).exists():
            self.mutations["policy"] += 1
            dump(value["policy_transition_receipt"], {"installed": value["temporary_policy_sha256"]})
        return {"policy_sha256": value["temporary_policy_sha256"],
                "transition_receipt_sha256": c._sha(value["policy_transition_receipt"])}

    def reserve(self, value, now):
        if not Path(value["reservation_receipt"]).exists():
            self.mutations["claim"] += 1
            dump(value["reservation_receipt"], {"claim": {"generation": 111}})
        return json.loads(Path(value["reservation_receipt"]).read_text())

    def publish_asset(self, value):
        root = Path(value["asset_dir"])
        if not root.exists():
            self.mutations["asset"] += 1; root.mkdir()
            (root / "last.pt").write_bytes(b"v5"); (root / "spec.json").write_text("{}")
            dump(root / "asset.json", {"checkpoint_sha256": c._sha(root / "last.pt"),
                 "spec_sha256": c._sha(root / "spec.json")})
        return json.loads((root / "asset.json").read_text())

    def require_binding(self, value):
        if not Path(value["binding"]).exists():
            self.mutations["binding"] += 1
            dump(value["binding"], {"role_policy_sha256": value["temporary_policy_sha256"],
                                    "journal": str(Path(value["root"]) / "requests.json")})
            dump(Path(value["root"]) / "requests.json", {"requests": []})
        return json.loads(Path(value["binding"]).read_text())

    def publisher(self, value):
        journal = Path(value["root"]) / "requests.json"; body = json.loads(journal.read_text())
        if not body["requests"]:
            self.mutations["request"] += 1; body["requests"].append({"job_id": "900", "state": "submitted"}); dump(journal, body)
        return {"state": "submitted", "job_id": "900", "finalizer_job_id": "901"}


@pytest.mark.parametrize("point", ["after_policy_mutation", "after_claim_reservation",
                                    "after_asset_publication", "after_binding_validation",
                                    "after_publisher_invocation"])
def test_crash_after_each_phase_recovers_without_duplicate_mutation(tmp_path, point):
    path, _ = contract(tmp_path); operations = FakeOperations(); fired = {"value": False}
    def fault(phase):
        if phase == point and not fired["value"]:
            fired["value"] = True
            raise RuntimeError("injected crash")
    with pytest.raises(RuntimeError, match="injected crash"):
        c.advance(path, operations=operations, fault=fault, now=10)
    state = c.advance(path, operations=operations, now=11)
    assert state["phase"] == "publisher_invoked"
    assert operations.mutations == {"policy": 1, "claim": 1, "asset": 1,
                                    "binding": 1, "request": 1}
    assert [row["phase"] for row in state["history"]] == list(c.PHASES[1:])


def test_policy_replace_crash_recovers_receipt_without_second_replace(tmp_path):
    path, value = contract(tmp_path); calls = {"snapshot": 0, "fault": 0}
    def snapshot(_):
        calls["snapshot"] += 1
        return {"account_running_pending": 22, "radon_running_pending": 9,
                "unresolved_gpu_intents": 0, "radon_unresolved_gpu_intents": 0,
                "all_live_owned_once": True}
    def fault(phase):
        calls["fault"] += 1
        raise RuntimeError(phase)
    with pytest.raises(RuntimeError, match="after_policy_replace"):
        c._install_policy(value, snapshot, 10, fault=fault)
    assert c._sha(value["role_policy"]) == value["temporary_policy_sha256"]
    assert not Path(value["policy_transition_receipt"]).exists()
    result = c._install_policy(value, snapshot, 11)
    assert result["policy_sha256"] == value["temporary_policy_sha256"]
    assert Path(value["policy_transition_receipt"]).is_file()
    assert calls == {"snapshot": 1, "fault": 1}


def test_contract_rejects_future_look_receipt_without_exact_successor(tmp_path):
    path, value = contract(tmp_path)
    stage2 = json.loads(Path(value["look_stage2"]).read_text())
    stage2["successor_monitor_job_id"] = "801"; dump(value["look_stage2"], stage2)
    value["look_stage2_sha256"] = c._sha(value["look_stage2"]); dump(path, value)
    with pytest.raises(ValueError, match="real LOOK successor"):
        c.validate_contract(path)
