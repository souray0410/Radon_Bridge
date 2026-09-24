import hashlib
import json
from pathlib import Path

from radon_bridge.runtime import paused_sixth_v5_control as c


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Claims:
    def __init__(self, value, root=None):
        self.value = value
        self.root = Path(root) if root is not None else None

    def mutate(self, run, fn):
        self.value = fn(dict(self.value))
        return self.value

    def path(self, run):
        return self.root / "claim.json"

    def release(self, run, owner, state, *, step_dead):
        assert step_dead
        self.value = {**self.value, "state": state}
        c.write(self.path(run), self.value)
        return self.value


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "control.py"
    source.write_text("immutable")
    lock = tmp_path / "account.lock"
    lock.write_text("")
    journal = tmp_path / "requests.json"
    dump(journal, {"schema": "radon_paused_sixth_v5_requests_v1", "requests": []})
    intent = tmp_path / "intent.json"
    dump(intent, {"schema": "radon_paused_sixth_v5_intent_v1", "attempt": None})
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "last.pt").write_bytes(b"v5-checkpoint")
    (asset / "spec.json").write_text("{}")
    asset_record = {
        "schema": "radon_paused_sixth_v5_production_asset_v1",
        "state": "accepted_for_new_claim", "run_id": c.RUN_ID,
        "framework_commit": c.FORMAL_V5_COMMIT,
        "checkpoint_sha256": digest(asset / "last.pt"),
        "spec_sha256": digest(asset / "spec.json"),
    }
    dump(asset / "asset.json", asset_record)
    run = tmp_path / "historical_v4" / c.RUN_ID
    run.mkdir(parents=True)
    expected_claim = {
        "run_dir": str(run.resolve()), "spec_sha256": "legacy-spec",
        "owner": "radon-v5-paused-sixth-migration", "job_id": "migration-cpu",
        "step": None, "generation": 111, "state": "claimed",
        "migration_phase": "source_reserved", "updated_at": 1,
    }
    policy = tmp_path / "policy.json"
    dump(policy, {"schema": "research_gpu_roles_v1", "account_ceiling": 24,
                  "native_max_gpus": 0, "projects": {"Radon_Bridge": {
                  "reserved_gpus": 14, "request_journals": [str(journal)]}}})
    binding = tmp_path / "binding.json"
    value = {
        "schema": "radon_paused_sixth_v5_control_v1", "lease_id": "rb-paused-sixth-v5",
        "project": "Radon_Bridge", "run_id": c.RUN_ID, "source_commit": "1" * 40,
        "framework_commit": c.FORMAL_V5_COMMIT,
        "source_file": str(source), "source_file_sha256": digest(source),
        "runtime_pins": [{"path": str(source), "sha256": digest(source)}],
        "account_lock": str(lock), "role_policy": str(policy), "role_policy_sha256": digest(policy),
        "ownership_overlay": None, "journal": str(journal), "journal_initial_sha256": digest(journal),
        "intent": str(intent), "intent_initial_sha256": digest(intent), "claims_root": str(tmp_path / "claims"),
        "expected_claim": expected_claim, "asset_dir": str(asset), "asset_sha256": digest(asset / "asset.json"),
        "v5_spec_sha256": digest(asset / "spec.json"), "v5_checkpoint_sha256": digest(asset / "last.pt"),
        "execution_dir": str(tmp_path / "v5_execution"), "python": "/python", "worker_cpus": 14,
        "worker_memory_gib": 128, "requested_gpus": 1, "account_limit": 24, "expires_at": 9999,
        "slurm_comment": "rb-paused-sixth-v5-test",
        "sbatch_command": ["sbatch", "--parsable", "--gres=gpu:a100:1", "--time=48:00:00",
                           "--comment=rb-paused-sixth-v5-test", "run.sbatch"],
        "finalizer_command_template": ["sbatch", "--parsable", "--dependency=afterany:{gpu_job_id}", "final.sbatch"],
        "test_access": False, "dispatch_authorized": True,
    }
    dump(binding, value)
    return binding, value


def test_writer_submits_once_and_registers_finalizer(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    ids = iter(["700", "701"])
    snapshot = lambda: {"account_limit": 24, "account_running_pending": 22,
                        "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    result = c.publish_once(binding, snapshot=snapshot,
                            submit=lambda command, timeout: next(ids), now=10)
    assert result == {"action": "none", "state": "submitted", "job_id": "700", "finalizer_job_id": "701"}
    row = c.read(value["journal"])["requests"][0]
    assert row["job_id"] == "700" and row["finalizer_job_id"] == "701"
    again = c.publish_once(binding, snapshot=snapshot,
                           submit=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()), now=11)
    assert again["state"] == "submission_intent_needs_review"


def test_writer_waits_for_account_or_role_capacity(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    account = lambda: {"account_limit": 24, "account_running_pending": 24,
                       "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    assert c.publish_once(binding, snapshot=account, submit=None, now=10)["state"] == "waiting_account_capacity"
    # Recreate pristine intent/journal because a waiting decision does not mutate either.
    assert c.read(value["intent"])["attempt"] is None
    role = lambda: {"account_limit": 24, "account_running_pending": 20,
                    "radon_running_pending": 14, "unresolved_gpu_intents": 0}
    assert c.publish_once(binding, snapshot=role, submit=None, now=10)["state"] == "waiting_radon_role_capacity"


def test_running_step_requires_explicit_scheduler_grant(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    ids = iter(["700", "701"])
    snapshot = lambda: {"account_limit": 24, "account_running_pending": 22,
                        "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    c.publish_once(binding, snapshot=snapshot, submit=lambda command, timeout: next(ids), now=10)
    record = tmp_path / "step.json"
    argv = ["worker"]
    command_sha = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    dump(record, {"schema": "radon_paused_sixth_v5_step_v1", "job_id": "700", "step": "batch",
                  "binding_sha256": digest(binding), "source_file_sha256": value["source_file_sha256"],
                  "runtime_pins_digest": c.pins_digest(value),
                  "production_asset_sha256": value["asset_sha256"], "argv": argv,
                  "command_sha256": command_sha})
    claims = Claims(value["expected_claim"])
    import pytest
    with pytest.raises(ValueError, match="granted request identity is missing"):
        c.claim_running_step(binding, claims=claims, step_record=record,
            observe_job=lambda *_args: (_ for _ in ()).throw(AssertionError()), now=11)
    assert claims.value == value["expected_claim"]
    assert not Path(value["execution_dir"]).exists()


def test_claim_binds_v5_spec_asset_command_and_separate_execution(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    ids = iter(["700", "701"])
    snapshot = lambda: {"account_limit": 24, "account_running_pending": 22,
                        "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    c.publish_once(binding, snapshot=snapshot, submit=lambda command, timeout: next(ids), now=10)
    c.grant_allocation(binding, job_id="700", now=11,
        observe_allocation=lambda job: {"job_id": job, "state": "RUNNING", "account": "pi-mengy",
                                        "gpus": 1, "comment": value["slurm_comment"]})
    record = tmp_path / "step.json"
    argv = ["/python", "-m", "radon_bridge.runtime.paused_sixth_v5_control",
            "--binding", str(binding.resolve()), "--worker", "--step-record", str(record.resolve()),
            "--gate", str((tmp_path / "gate.json").resolve())]
    dump(record, {"schema": "radon_paused_sixth_v5_step_v1", "job_id": "700", "step": "batch",
                  "binding_sha256": digest(binding), "source_file_sha256": value["source_file_sha256"],
                  "runtime_pins_digest": c.pins_digest(value),
                  "production_asset_sha256": value["asset_sha256"], "argv": argv,
                  "command_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()})
    observed = {"job_id": "700", "step": "batch", "job_state": "RUNNING", "step_state": "RUNNING",
                "account": "pi-mengy", "gpus": 1, "command_sha256": c.read(record)["command_sha256"]}
    claims = Claims(value["expected_claim"])
    claim = c.claim_running_step(binding, claims=claims, step_record=record,
                                 observe_job=lambda job, step: dict(observed), now=12)
    assert claim["spec_sha256"] == value["v5_spec_sha256"]
    assert claim["production_asset_sha256"] == value["asset_sha256"]
    assert claim["execution_dir"] == str(Path(value["execution_dir"]).resolve())
    assert Path(value["execution_dir"], "migration.json").is_file()
    assert not Path(value["expected_claim"]["run_dir"], "last.pt").exists()
    row = c.read(value["journal"])["requests"][0]
    assert row["state"] == "running" and row["step"] == "batch"


def test_claim_rejects_scheduler_command_drift_without_execution_or_claim(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    ids = iter(["700", "701"])
    snapshot = lambda: {"account_limit": 24, "account_running_pending": 22,
                        "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    c.publish_once(binding, snapshot=snapshot, submit=lambda command, timeout: next(ids), now=10)
    c.grant_allocation(binding, job_id="700", now=11,
        observe_allocation=lambda job: {"job_id": job, "state": "RUNNING", "account": "pi-mengy",
                                        "gpus": 1, "comment": value["slurm_comment"]})
    record = tmp_path / "step.json"
    argv = ["worker"]
    command_sha = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    dump(record, {"schema": "radon_paused_sixth_v5_step_v1", "job_id": "700", "step": "batch",
                  "binding_sha256": digest(binding), "source_file_sha256": value["source_file_sha256"],
                  "runtime_pins_digest": c.pins_digest(value),
                  "production_asset_sha256": value["asset_sha256"], "argv": argv, "command_sha256": command_sha})
    claims = Claims(value["expected_claim"])
    import pytest
    with pytest.raises(ValueError, match="Slurm allocation/step command"):
        c.claim_running_step(binding, claims=claims, step_record=record,
            observe_job=lambda job, step: {"job_id": job, "step": step, "job_state": "RUNNING",
            "step_state": "RUNNING", "account": "pi-mengy", "gpus": 1, "command_sha256": "0" * 64}, now=12)
    assert claims.value == value["expected_claim"]
    assert not Path(value["execution_dir"]).exists()


def test_finalizer_closes_failed_request_but_preserves_migration_reservation(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path, monkeypatch)
    ids = iter(["700", "701"])
    snapshot = lambda: {"account_limit": 24, "account_running_pending": 22,
                        "radon_running_pending": 8, "unresolved_gpu_intents": 0}
    c.publish_once(binding, snapshot=snapshot, submit=lambda command, timeout: next(ids), now=10)
    claim_path = Path(value["claims_root"]) / "claim.json"
    claim_path.parent.mkdir()
    claims = Claims(value["expected_claim"], value["claims_root"])
    c.write(claims.path(value["expected_claim"]["run_dir"]), value["expected_claim"])
    row = c.finalize_afterany(
        binding, gpu_job_id="700", finalizer_job_id="701",
        observe_terminal=lambda job: {"job_id": job, "state": "FAILED", "exit_code": "1:0"},
        claims_factory=lambda root: claims,
        now=10000)
    assert row["state"] == "failed" and row["accepted"] is False
    assert c.read(claims.path(value["expected_claim"]["run_dir"])) == value["expected_claim"]


def test_observe_running_step_parses_typed_and_generic_gpu_tres(monkeypatch):
    outputs = iter([
        "JobId=700 JobState=RUNNING Account=pi-mengy ReqTRES=cpu=16,gres/gpu:a100=1,gres/gpu=1,mem=160G",
        "StepId=700.batch State=RUNNING TRES=cpu=14,gres/gpu:a100=1,gres/gpu=1,mem=128G",
    ])
    monkeypatch.setattr(c.subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    observed = c.observe_running_step("700", "batch", "a" * 64)
    assert observed["gpus"] == 1 and observed["step"] == "batch"
