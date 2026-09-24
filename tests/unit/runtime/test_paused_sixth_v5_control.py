import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import pytest

from radon_bridge.runtime import paused_sixth_v5_control_v2 as c


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Claims:
    def __init__(self, value, root=None):
        self.value = dict(value)
        self.root = Path(root) if root is not None else None

    def mutate(self, run, fn):
        self.value = fn(dict(self.value))
        return self.value

    def path(self, run):
        return self.root / "claim.json"

    def release(self, run, owner, state, *, step_dead):
        assert step_dead and owner == self.value["owner"]
        self.value = {**self.value, "state": state}
        c.write(self.path(run), self.value)
        return self.value

    def update(self, run, owner, **values):
        assert owner == self.value["owner"]
        self.value.update(values)
        c.write(self.path(run), self.value)
        return self.value


class Scheduler:
    def __init__(self, journal):
        self.journal = Path(journal)
        self.jobs = {}
        self.next_id = 700
        self.events = []
        self.crash = None
        self.accept_on_crash = True

    def submit(self, command, timeout):
        assert timeout == 20
        assert c.read(self.journal)["requests"][-1]["state"] in {"preparing", "held"}
        comment = next(item.split("=", 1)[1] for item in command if item.startswith("--comment="))
        finalizer = comment.endswith("-finalizer")
        job_id = str(self.next_id)
        self.next_id += 1
        row = {"job_id": job_id, "state": "PENDING", "account": "pi-mengy",
               "gpus": 0 if finalizer else 1, "comment": comment,
               "held": False if finalizer else True, "dependency": ""}
        if finalizer:
            row["dependency"] = next(item.split("=", 1)[1] for item in command
                                     if item.startswith("--dependency="))
        self.events.append(("submit", comment))
        if self.crash == ("finalizer" if finalizer else "allocation"):
            self.crash = None
            if self.accept_on_crash:
                self.jobs[comment] = row
            raise TimeoutError("injected acknowledgement loss")
        self.jobs[comment] = row
        return job_id

    def lookup(self, comment):
        jobs = [dict(self.jobs[comment])] if comment in self.jobs else []
        return {"jobs": jobs, "absence_proven": True}

    def release(self, job_id):
        matches = [row for row in self.jobs.values() if row["job_id"] == job_id]
        assert len(matches) == 1 and matches[0]["held"] is True
        matches[0]["held"] = False
        self.events.append(("release", job_id))


def fixture(tmp_path):
    lock = tmp_path / "account.lock"
    lock.write_text("")
    journal = tmp_path / "requests.json"
    dump(journal, {"schema": "radon_paused_sixth_v5_requests_v2", "requests": []})
    intent = tmp_path / "intent.json"
    dump(intent, {"schema": "radon_paused_sixth_v5_intent_v2", "transaction": None})
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
                  "projects": {"Radon_Bridge": {"reserved_gpus": 14,
                  "request_journals": [str(journal)]}}})
    stage2 = tmp_path / "look_stage2.json"
    dump(stage2, {"status": "accepted"})
    proposal = tmp_path / "policy_proposal.json"
    dump(proposal, {"schema": "radon_paused_sixth_v5_policy_proposal_v1",
                    "state": "accepted", "role_policy_sha256": digest(policy)})
    gate = tmp_path / "deployment_gate.json"
    dump(gate, {"schema": "radon_paused_sixth_v5_deployment_gate_v1",
                "look_v21_stage2": "accepted", "look_v21_receipt": str(stage2),
                "look_v21_receipt_sha256": digest(stage2),
                "policy_proposal_review": "accepted", "policy_proposal": str(proposal),
                "policy_proposal_sha256": digest(proposal),
                "accepted_role_policy_sha256": digest(policy),
                "production_dispatch_authorized": True, "test_access": False})
    sources = {}
    for index, name in enumerate(sorted(c.MODULE_NAMES)):
        path = tmp_path / f"module_{index}.py"
        path.write_text(name)
        sources[name] = str(path.resolve())
    allocation_wrapper = tmp_path / "allocation.sbatch"
    finalizer_wrapper = tmp_path / "finalizer.sbatch"
    allocation_wrapper.write_text("allocation")
    finalizer_wrapper.write_text("finalizer")
    python = str(Path(sys.executable).resolve())
    pins = ([{"path": path, "sha256": digest(path)} for path in sources.values()]
            + [{"path": str(allocation_wrapper.resolve()), "sha256": digest(allocation_wrapper)},
               {"path": str(finalizer_wrapper.resolve()), "sha256": digest(finalizer_wrapper)},
               {"path": python, "sha256": digest(python)}])
    modules = [{"name": name, "path": path, "sha256": digest(path)}
               for name, path in sources.items()]
    roles = {role: sources[module] for role, module in c.ROLE_MODULES.items()}
    roles.update({"allocation_wrapper": str(allocation_wrapper.resolve()),
                  "finalizer_wrapper": str(finalizer_wrapper.resolve()),
                  "python_executable": python})
    runtime = {"schema": "radon_v5_runtime_contract_v1", "python": python,
               "python_sha256": digest(python), "python_realpath": python,
               "python_version": platform.python_version(), "torch_version": "test",
               "framework_api": "V5", "framework_commit": c.FORMAL_V5_COMMIT,
               "environment": {key: os.environ.get(key, "") for key in c.ENV_KEYS},
               "modules": modules, "roles": roles}
    binding = tmp_path / "binding.json"
    value = {
        "schema": "radon_paused_sixth_v5_control_v2", "lease_id": "rb-paused-sixth-v5",
        "project": "Radon_Bridge", "run_id": c.RUN_ID, "source_commit": "1" * 40,
        "framework_commit": c.FORMAL_V5_COMMIT, "account_lock": str(lock),
        "role_policy": str(policy), "role_policy_sha256": digest(policy),
        "ownership_overlay": None, "journal": str(journal),
        "journal_initial_sha256": digest(journal), "intent": str(intent),
        "intent_initial_sha256": digest(intent), "claims_root": str(tmp_path / "claims"),
        "expected_claim": expected_claim, "asset_dir": str(asset),
        "asset_sha256": digest(asset / "asset.json"), "v5_spec_sha256": digest(asset / "spec.json"),
        "v5_checkpoint_sha256": digest(asset / "last.pt"),
        "execution_dir": str((tmp_path / "v5_execution").resolve()), "requested_gpus": 1,
        "account_limit": 24, "worker_cpus": 14, "worker_memory_gib": 128,
        "expires_at": 99999999999, "comment_prefix": "rb-v5-test",
        "sbatch_command_template": ["sbatch", "--parsable", "--hold", "--gres=gpu:a100:1",
                                    "--time=48:00:00", "--comment={attempt_comment}",
                                    str(allocation_wrapper.resolve())],
        "finalizer_command_template": ["sbatch", "--parsable",
                                       "--dependency=afterany:{gpu_job_id}",
                                       "--comment={finalizer_comment}",
                                       str(finalizer_wrapper.resolve())],
        "allocation_wrapper": str(allocation_wrapper.resolve()),
        "finalizer_wrapper": str(finalizer_wrapper.resolve()),
        "runtime_pins": pins, "runtime_contract": runtime,
        "retry_policy": {"max_attempts": 3, "retryable_states": ["paused", "failed"],
                         "walltime": "48:00:00"},
        "deployment_gate": str(gate), "deployment_gate_sha256": digest(gate),
        "test_access": False, "dispatch_authorized": True,
    }
    dump(binding, value)
    return binding, value


def capacity():
    return {"account_limit": 24, "account_running_pending": 22,
            "radon_running_pending": 8, "unresolved_gpu_intents": 0,
            "radon_unresolved_gpu_intents": 0}


def publish(binding, scheduler, now=10):
    return c.publish_once(binding, snapshot=capacity, submit=scheduler.submit,
                          lookup=scheduler.lookup, release=scheduler.release,
                          runtime_verify=lambda binding: {}, now=now)


def test_held_first_transaction_is_durable_before_submit_and_release(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    result = publish(binding, scheduler)
    assert result["state"] == "submitted" and result["job_id"] == "700"
    assert scheduler.events == [("submit", "rb-v5-test-a001"),
                                ("submit", "rb-v5-test-a001-finalizer"),
                                ("release", "700")]
    row = c.read(value["journal"])["requests"][0]
    assert row["state"] == "submitted" and row["allocation_submission_generations"] == 1
    assert row["finalizer_submission_generations"] == 1
    again = publish(binding, scheduler, now=11)
    assert again["state"] == "submitted" and len(scheduler.events) == 3


def test_allocation_ack_loss_recovers_accepted_job_without_duplicate(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    scheduler.crash = "allocation"
    assert publish(binding, scheduler)["state"] == "allocation_ack_unknown"
    assert publish(binding, scheduler, now=11)["state"] == "submitted"
    assert [event for event in scheduler.events if event[1] == "rb-v5-test-a001"] == [
        ("submit", "rb-v5-test-a001")]


def test_ack_unknown_waits_for_grace_and_proven_absence_before_resubmit(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    scheduler.crash = "allocation"
    scheduler.accept_on_crash = False
    assert publish(binding, scheduler, now=10)["state"] == "allocation_ack_unknown"
    assert publish(binding, scheduler, now=100)["state"] == "allocation_ack_unknown"
    assert len(scheduler.events) == 1
    assert publish(binding, scheduler, now=131)["state"] == "submitted"
    assert len([event for event in scheduler.events
                if event[0] == "submit" and not event[1].endswith("finalizer")]) == 2


def test_finalizer_ack_loss_recovers_before_releasing_gpu(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    scheduler.crash = "finalizer"
    first = publish(binding, scheduler)
    assert first["state"] == "finalizer_ack_unknown"
    assert scheduler.jobs["rb-v5-test-a001"]["held"] is True
    assert publish(binding, scheduler, now=11)["state"] == "submitted"
    assert len([event for event in scheduler.events if event[1].endswith("finalizer")]) == 1


def test_binding_requires_complete_runtime_and_accepted_deployment_order(tmp_path):
    binding, value = fixture(tmp_path)
    c.validate_binding(binding, now=1)
    broken = c.read(binding)
    broken["runtime_contract"]["modules"] = broken["runtime_contract"]["modules"][:-1]
    dump(binding, broken)
    with pytest.raises(ValueError, match="runtime contract is incomplete"):
        c.validate_binding(binding, now=1)
    dump(binding, value)
    gate = c.read(value["deployment_gate"])
    gate["look_v21_stage2"] = "pending"
    dump(value["deployment_gate"], gate)
    value["deployment_gate_sha256"] = digest(value["deployment_gate"])
    dump(binding, value)
    with pytest.raises(ValueError, match="LOOK v21 stage2"):
        c.validate_binding(binding, now=1)


def _submitted(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    publish(binding, scheduler)
    scheduler.jobs["rb-v5-test-a001"].update(state="RUNNING", held=False)
    c.grant_allocation(binding, job_id="700", observe_allocation=lambda job: {
        **scheduler.jobs["rb-v5-test-a001"], "job_id": job},
        runtime_verify=lambda binding: {}, now=11)
    return binding, value, scheduler


def test_claim_uses_parent_fixed_argv_nonce_and_independent_attestation(tmp_path):
    binding, value, _ = _submitted(tmp_path)
    owner = tmp_path / "owner.json"
    record = tmp_path / "step.json"
    nonce = "a" * 64
    argv = [value["runtime_contract"]["python"], "-m", "fixed.worker", "--nonce", nonce]
    contract = {"schema": "radon_paused_sixth_v5_owner_contract_v1",
                "binding_sha256": digest(binding),
                "runtime_contract_sha256": c.runtime_contract_digest(value),
                "attempt_id": "rb-paused-sixth-v5/attempt-001", "job_id": "700",
                "nonce": nonce, "argv": argv, "argv_sha256": c._digest_json(argv),
                "created_at": 11}
    dump(owner, contract)
    dump(record, {"schema": "radon_paused_sixth_v5_step_v2", "job_id": "700",
                  "step": "batch", "pid": 123, "nonce": nonce,
                  "owner_contract_sha256": digest(owner),
                  "runtime_observation_sha256": "0" * 64, "time": 12})
    claims = Claims(value["expected_claim"])
    seen = {}

    def attest(**kwargs):
        seen.update(kwargs)
        return {"schema": "radon_v5_parent_process_attestation_v1"}

    claim = c.claim_running_step(
        binding, claims=claims, step_record=record, owner_contract=owner,
        expected_owner_contract=contract,
        observe_job=lambda job, step: {"job_id": job, "step": step,
            "job_state": "RUNNING", "step_state": "RUNNING", "account": "pi-mengy", "gpus": 1},
        attest=attest, now=12)
    assert seen["expected_argv"] == argv and seen["pid"] == 123
    assert claim["framework_commit"] == c.FORMAL_V5_COMMIT
    assert Path(value["execution_dir"], "migration.json").is_file()


def test_worker_cannot_rewrite_parent_owner_contract(tmp_path):
    binding, value, _ = _submitted(tmp_path)
    owner = tmp_path / "owner.json"
    record = tmp_path / "step.json"
    nonce = "b" * 64
    expected = {"schema": "radon_paused_sixth_v5_owner_contract_v1",
                "binding_sha256": digest(binding),
                "runtime_contract_sha256": c.runtime_contract_digest(value),
                "attempt_id": "rb-paused-sixth-v5/attempt-001", "job_id": "700",
                "nonce": nonce, "argv": ["fixed"],
                "argv_sha256": c._digest_json(["fixed"]), "created_at": 11}
    forged = {**expected, "argv": ["forged"], "argv_sha256": c._digest_json(["forged"])}
    dump(owner, forged)
    dump(record, {"schema": "radon_paused_sixth_v5_step_v2", "job_id": "700",
                  "step": "batch", "pid": 123, "nonce": nonce,
                  "owner_contract_sha256": digest(owner),
                  "runtime_observation_sha256": "0" * 64, "time": 12})
    claims = Claims(value["expected_claim"])
    with pytest.raises(ValueError, match="parent owner contract"):
        c.claim_running_step(
            binding, claims=claims, step_record=record, owner_contract=owner,
            expected_owner_contract=expected,
            observe_job=lambda *_: (_ for _ in ()).throw(AssertionError()),
            attest=lambda **_: (_ for _ in ()).throw(AssertionError()), now=12)
    assert claims.value == value["expected_claim"]
    assert not Path(value["execution_dir"]).exists()


def test_parent_process_attestation_reads_proc_cmdline_exe_cgroup_and_environment(tmp_path):
    _, value = fixture(tmp_path)
    root = tmp_path / "proc" / "123"
    root.mkdir(parents=True)
    argv = [value["runtime_contract"]["python"], "-m", "worker"]
    (root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
    (root / "exe").symlink_to(value["runtime_contract"]["python_realpath"])
    (root / "cgroup").write_text("0::/slurm/job_700/step_batch/\n")
    environment = value["runtime_contract"]["environment"]
    (root / "environ").write_bytes(
        b"\0".join(f"{key}={item}".encode() for key, item in environment.items()) + b"\0")
    observed = c.attest_process(pid=123, job_id="700", step="batch", expected_argv=argv,
                                binding=value, proc_root=tmp_path / "proc")
    assert observed["schema"] == "radon_v5_parent_process_attestation_v1"
    (root / "cmdline").write_bytes(b"forged\0")
    with pytest.raises(ValueError, match="parent-fixed"):
        c.attest_process(pid=123, job_id="700", step="batch", expected_argv=argv,
                         binding=value, proc_root=tmp_path / "proc")


def test_retry_uses_same_run_and_exact_paused_checkpoint(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    publish(binding, scheduler)
    row = c.read(value["journal"])["requests"][0]
    value_with_sha = dict(value, binding_sha256=digest(binding))
    c.prepare_execution(value_with_sha, row, now=11)
    execution = Path(value["execution_dir"])
    (execution / "last.pt").write_bytes(b"continued")
    terminal_claim = {**value["expected_claim"], "state": "paused", "generation": 113}
    row.update(state="paused", terminal_claim=terminal_claim,
               continuation={"schema": "radon_paused_sixth_v5_continuation_v1",
                             "execution_dir": str(execution.resolve()),
                             "checkpoint_sha256": digest(execution / "last.pt"),
                             "spec_sha256": value["v5_spec_sha256"],
                             "progress": {"epoch": 1, "offset": 2, "updates": 3},
                             "test_access": False})
    dump(value["journal"], {"schema": "radon_paused_sixth_v5_requests_v2", "requests": [row]})
    dump(value["intent"], {"schema": "radon_paused_sixth_v5_intent_v2", "transaction": None})
    scheduler = Scheduler(value["journal"])
    scheduler.next_id = 702
    result = publish(binding, scheduler, now=200)
    assert result["state"] == "submitted" and result["attempt"] == 2
    retry = c.read(value["journal"])["requests"][1]
    assert retry["run_dir"] == value["expected_claim"]["run_dir"]
    assert retry["execution_dir"] == value["execution_dir"]


def test_finalizer_rejects_wrong_account_gpu_or_dependency(tmp_path, monkeypatch):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    publish(binding, scheduler)
    claim_path = Path(value["claims_root"]) / "claim.json"
    claim_path.parent.mkdir()
    c.write(claim_path, value["expected_claim"])
    journal = c.read(value["journal"])
    journal["requests"][0]["state"] = "paused"
    dump(value["journal"], journal)
    with pytest.raises(ValueError, match="finalizer identity"):
        c.finalize_afterany(
            binding, gpu_job_id="700", finalizer_job_id="701",
            observe_terminal=lambda job: {"job_id": job, "state": "FAILED", "exit_code": "1:0"},
            observe_finalizer=lambda job: {"job_id": job, "state": "RUNNING",
                "account": "wrong", "gpus": 1, "comment": "rb-v5-test-a001-finalizer",
                "dependency": "afterany:999"}, claims_factory=lambda root: Claims(
                    value["expected_claim"], value["claims_root"]),
            runtime_verify=lambda binding: {}, now=20)


def test_capacity_wait_does_not_mutate_journal(tmp_path):
    binding, value = fixture(tmp_path)
    scheduler = Scheduler(value["journal"])
    full = lambda: {"account_limit": 24, "account_running_pending": 24,
                    "radon_running_pending": 8, "unresolved_gpu_intents": 0,
                    "radon_unresolved_gpu_intents": 0}
    result = c.publish_once(binding, snapshot=full, submit=scheduler.submit,
                            lookup=scheduler.lookup, release=scheduler.release,
                            runtime_verify=lambda binding: {}, now=10)
    assert result["state"] == "waiting_account_capacity"
    assert c.read(value["journal"])["requests"] == []
    role_full = lambda: {"account_limit": 24, "account_running_pending": 20,
                         "radon_running_pending": 13, "unresolved_gpu_intents": 1,
                         "radon_unresolved_gpu_intents": 1}
    result = c.publish_once(binding, snapshot=role_full, submit=scheduler.submit,
                            lookup=scheduler.lookup, release=scheduler.release,
                            runtime_verify=lambda binding: {}, now=11)
    assert result["state"] == "waiting_radon_role_capacity"
    assert c.read(value["journal"])["requests"] == []


def test_ibex_recovery_does_not_request_unsupported_sacct_dependency(monkeypatch):
    commands = []

    def output(command, **kwargs):
        commands.append(command)
        return ""

    monkeypatch.setattr(c.subprocess, "check_output", output)
    monkeypatch.setenv("USER", "mengh")
    assert c.lookup_slurm_jobs("absent") == {"jobs": [], "absence_proven": True}
    sacct = next(command for command in commands if command[0] == "sacct")
    assert "Dependency" not in sacct[-1]


def test_finalizer_dependency_is_read_from_live_scontrol(monkeypatch):
    monkeypatch.setattr(c.subprocess, "check_output", lambda command, **kwargs:
        "JobId=701 JobState=RUNNING Account=pi-mengy ReqTRES=cpu=1,mem=2G "
        "Comment=rb-finalizer Dependency=afterany:700")
    assert c.observe_finalizer("701") == {
        "job_id": "701", "state": "RUNNING", "account": "pi-mengy", "gpus": 0,
        "comment": "rb-finalizer", "dependency": "afterany:700"}
