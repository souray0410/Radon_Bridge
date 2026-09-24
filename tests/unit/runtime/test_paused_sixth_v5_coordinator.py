import json
import importlib
import os
from pathlib import Path
import sys

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
    coordinator_sha = c._sha(c.__file__)
    modules = []
    for name in sorted(c.control.MODULE_NAMES):
        module_path = str(Path(importlib.import_module(name).__file__).resolve())
        modules.append({"name": name, "path": module_path, "sha256": c._sha(module_path)})
    runtime_environment = {key: os.environ.get(key, "") for key in c.control.ENV_KEYS}
    python = os.path.abspath(sys.executable)
    python_realpath = str(Path(sys.executable).resolve(strict=True))
    review_values = {
      "coordinator_code": {"schema": c.REVIEW_KINDS["coordinator_code"],
          "candidate": {"implementation_commit": "1" * 40,
                        "coordinator_sha256": coordinator_sha,
                        "pre_mutation_modules_sha256": c._json_sha(modules),
                        "python": python,
                        "python_realpath": python_realpath,
                        "python_launcher_sha256": c.control.python_launcher_sha256(python),
                        "python_sha256": c._sha(sys.executable),
                        "runtime_environment_sha256": c._json_sha(runtime_environment)},
          "verdict": {"code": "GO"}, "test_access": False},
    }
    reviews = []
    fixtures = Path(__file__).resolve().parents[2] / "fixtures/runtime/paused_sixth_v5_reviews"
    for kind in c.REVIEW_KINDS:
        receipt = root / c.REVIEW_FILENAMES[kind]
        if kind == "coordinator_code":
            dump(receipt, review_values[kind])
        else:
            receipt.write_bytes((fixtures / c.REVIEW_FILENAMES[kind]).read_bytes())
        reviews.append({"kind": kind, "path": str(receipt), "sha256": c._sha(receipt)})
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
      "source_root": str(Path(c.__file__).resolve().parents[2]),
      "mhd_models_root": str(Path(importlib.import_module("mhd_models").__file__).resolve().parents[1]),
      "framework_root": str(Path(importlib.import_module("mhd_framework").__file__).resolve().parents[1]),
      "python": python, "python_realpath": python_realpath,
      "python_launcher_sha256": c.control.python_launcher_sha256(python),
      "ownership_overlay": str(root / "overlay.json"),
      "allocation_wrapper": str(root / "allocation.sbatch"),
      "finalizer_wrapper": str(root / "finalizer.sbatch"),
      "deployment_gate": str(root / "deployment_gate.json"),
      "binding_intent": str(root / "binding_intent.json"), "binding": str(root / "binding.json"),
      "execution_dir": str(root / "execution"), "native_sources": {},
      "runtime_environment": runtime_environment,
      "worker_cpus": 16, "worker_memory_gib": 160, "source_commit": "1" * 40,
      "coordinator_source": str(Path(c.__file__).resolve()),
      "coordinator_source_sha256": coordinator_sha,
      "python_sha256": c._sha(sys.executable),
      "pre_mutation_modules": modules,
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
    second = tmp_path / "second"; second.mkdir()
    path, value = contract(second)
    stage2 = json.loads(Path(value["look_stage2"]).read_text())
    stage2["successor_monitor_job_id"] = "801"; dump(value["look_stage2"], stage2)
    value["look_stage2_sha256"] = c._sha(value["look_stage2"]); dump(path, value)
    with pytest.raises(ValueError, match="real LOOK successor"):
        c.validate_contract(path)


def test_contract_rejects_duplicate_or_contract_selected_review_receipts(tmp_path):
    second = tmp_path / "second"; second.mkdir()
    path, value = contract(second)
    first = value["review_receipts"][0]
    value["review_receipts"] = [dict(first, kind=kind) for kind in c.REVIEW_KINDS]
    dump(path, value)
    with pytest.raises(ValueError, match="exact and distinct"):
        c.validate_contract(path)

    path, value = contract(tmp_path)
    row = value["review_receipts"][0]
    row["verdict_path"] = "accepted"
    row["accepted_value"] = True
    dump(path, value)
    with pytest.raises(ValueError, match="all independent review receipts"):
        c.validate_contract(path)


def test_contract_rejects_wrong_review_schema_commit_and_no_go(tmp_path):
    for mutation in ("schema", "commit", "verdict"):
        case = tmp_path / mutation; case.mkdir()
        path, value = contract(case)
        row = next(item for item in value["review_receipts"]
                   if item["kind"] == "coordinator_code")
        receipt = json.loads(Path(row["path"]).read_text())
        if mutation == "schema":
            receipt["schema"] = "arbitrary_acceptance_v1"
        elif mutation == "commit":
            receipt["candidate"]["implementation_commit"] = "2" * 40
        else:
            receipt["verdict"]["code"] = "NO_GO"
        dump(row["path"], receipt); row["sha256"] = c._sha(row["path"]); dump(path, value)
        with pytest.raises(ValueError, match="verdict is not accepted"):
            c.validate_contract(path)


def test_contract_rejects_non_whitelisted_review_path(tmp_path):
    path, value = contract(tmp_path)
    row = next(item for item in value["review_receipts"] if item["kind"] == "coordinator_code")
    renamed = Path(row["path"]).with_name("arbitrary-go.json")
    renamed.write_bytes(Path(row["path"]).read_bytes())
    row["path"] = str(renamed); dump(path, value)
    with pytest.raises(ValueError, match="path is not whitelisted"):
        c.validate_contract(path)


def test_contract_rejects_unreviewed_or_changed_pre_mutation_runtime(tmp_path):
    path, value = contract(tmp_path)
    row = next(item for item in value["pre_mutation_modules"]
               if item["name"] == "radon_bridge.runtime.paused_sixth_v5_coordinator")
    row["sha256"] = "0" * 64
    # Even a rewritten receipt cannot bless bytes that differ from the running module.
    review = next(item for item in value["review_receipts"]
                  if item["kind"] == "coordinator_code")
    receipt = json.loads(Path(review["path"]).read_text())
    receipt["candidate"]["pre_mutation_modules_sha256"] = c._json_sha(value["pre_mutation_modules"])
    dump(review["path"], receipt); review["sha256"] = c._sha(review["path"]); dump(path, value)
    with pytest.raises(ValueError, match="pre-mutation module is not pinned"):
        c.validate_contract(path)


def test_control_runtime_contract_requires_coordinator_role():
    assert "radon_bridge.runtime.paused_sixth_v5_coordinator" in c.control.MODULE_NAMES
    assert c.control.ROLE_MODULES["coordinator"] == "radon_bridge.runtime.paused_sixth_v5_coordinator"


@pytest.mark.parametrize("point", ["after_binding_intent", "after_ownership_overlay",
                                    "after_deployment_gate", "after_allocation_wrapper",
                                    "after_finalizer_wrapper", "after_binding_write"])
def test_binding_internal_crash_recovers_same_immutable_bytes(tmp_path, monkeypatch, point):
    _, value = contract(tmp_path)
    Path(value["role_policy"]).write_bytes(Path(value["temporary_policy"]).read_bytes())
    dump(value["reservation_receipt"], {"claim": {"generation": 111}})
    asset = Path(value["asset_dir"]); asset.mkdir()
    dump(asset / "asset.json", {"checkpoint_sha256": "a" * 64, "spec_sha256": "b" * 64})
    dump(Path(value["root"]) / "requests.json", {"requests": []})
    dump(Path(value["root"]) / "intent.json", {"attempts": []})
    Path(value["allocation_wrapper"]).parent.mkdir(parents=True, exist_ok=True)

    snapshot = {"account_running_pending": 22, "radon_running_pending": 9,
                "unresolved_gpu_intents": 0, "radon_unresolved_gpu_intents": 0,
                "all_live_owned_once": True}
    def ownership(_, make_overlay=False, now=None):
        result = dict(snapshot)
        if make_overlay:
            result["overlay"] = {"schema": "legacy_native_ownership_overlay_v2",
                                 "checked_at": now, "entries": [], "test": False}
        return result
    runtime = {"schema": "radon_v5_runtime_contract_v1", "python": value["python"],
               "python_launcher_sha256": value["python_launcher_sha256"],
               "python_sha256": value["python_sha256"],
               "python_realpath": value["python_realpath"],
               "python_version": "test", "torch_version": "test", "framework_api": "V5",
               "framework_commit": c.handoff.FORMAL_V5_COMMIT,
               "environment": value["runtime_environment"], "modules": [],
               "roles": {"allocation_wrapper": value["allocation_wrapper"],
                         "finalizer_wrapper": value["finalizer_wrapper"],
                         "python_executable": value["python"]}}
    monkeypatch.setattr(c, "_ownership", ownership)
    monkeypatch.setattr(c, "_module_contract", lambda _: runtime)
    monkeypatch.setattr(c, "_require_binding", lambda contract: json.loads(Path(contract["binding"]).read_text()))
    fired = {"value": False}
    def fault(name):
        if name == point and not fired["value"]:
            fired["value"] = True
            raise RuntimeError("internal binding crash")
    with pytest.raises(RuntimeError, match="internal binding crash"):
        c._build_binding(value, 100, fault=fault)
    intent = json.loads(Path(value["binding_intent"]).read_text())
    result = c._build_binding(value, 200)
    assert intent["created_at"] == 100
    assert json.loads(Path(value["binding_intent"]).read_text()) == intent
    assert result["source_commit"] == value["source_commit"]


def test_binding_wrappers_preserve_reviewed_virtualenv_launcher(tmp_path, monkeypatch):
    _, value = contract(tmp_path)
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    value["python"] = str(launcher.absolute())
    value["python_realpath"] = str(Path(sys.executable).resolve(strict=True))
    value["python_launcher_sha256"] = c.control.python_launcher_sha256(launcher)
    value["python_sha256"] = c._sha(launcher)
    Path(value["role_policy"]).write_bytes(Path(value["temporary_policy"]).read_bytes())
    dump(value["reservation_receipt"], {"claim": {"generation": 111}})
    asset = Path(value["asset_dir"]); asset.mkdir()
    dump(asset / "asset.json", {"checkpoint_sha256": "a" * 64, "spec_sha256": "b" * 64})
    dump(Path(value["root"]) / "requests.json", {"requests": []})
    dump(Path(value["root"]) / "intent.json", {"attempts": []})
    snapshot = {"account_running_pending": 22, "radon_running_pending": 9,
                "unresolved_gpu_intents": 0, "radon_unresolved_gpu_intents": 0,
                "all_live_owned_once": True,
                "overlay": {"schema": "legacy_native_ownership_overlay_v2",
                            "checked_at": 10, "entries": [], "test": False}}
    monkeypatch.setattr(c, "_ownership", lambda *_, **__: dict(snapshot))
    monkeypatch.setattr(c, "_module_contract", lambda _: {
        "schema": "radon_v5_runtime_contract_v1", "python": value["python"],
        "python_launcher_sha256": value["python_launcher_sha256"],
        "python_sha256": value["python_sha256"], "python_realpath": value["python_realpath"],
        "python_version": "test", "torch_version": "test", "framework_api": "V5",
        "framework_commit": c.handoff.FORMAL_V5_COMMIT,
        "environment": value["runtime_environment"], "modules": [],
        "roles": {"allocation_wrapper": value["allocation_wrapper"],
                  "finalizer_wrapper": value["finalizer_wrapper"],
                  "python_executable": value["python"]}})
    monkeypatch.setattr(c, "_require_binding",
                        lambda contract: json.loads(Path(contract["binding"]).read_text()))
    c._build_binding(value, 10)
    allocation = Path(value["allocation_wrapper"]).read_text()
    finalizer = Path(value["finalizer_wrapper"]).read_text()
    assert value["python"] in allocation and value["python"] in finalizer
    assert "--runtime-preflight" in allocation
    if value["python"] != value["python_realpath"]:
        assert value["python_realpath"] not in allocation
        assert value["python_realpath"] not in finalizer
