import json
from pathlib import Path
import pytest
from radon_bridge.runtime import project_dispatch as m
from radon_bridge.runtime.state import file_sha256


def fixture(tmp_path,peak=104.017):
    p=tmp_path/'full.json'
    p.write_text(json.dumps(dict(status='accepted',spec_sha256='identity',peak_step_memory_gib=peak)))
    task=dict(execution='native',run_dir=str(tmp_path/'run'),spec_sha256='identity')
    config=dict(worker_memory_gib=104,native_profile_references={'identity':dict(path=str(p),sha256=file_sha256(p))})
    return task,config,p


def test_known_failed_envelope_waits_before_probe_and_keeps_evidence(tmp_path):
    task,cfg,path=fixture(tmp_path)
    digest=file_sha256(path)
    hold=m.resource_wait(task,cfg)
    assert hold['state']=='waiting_resource_configuration'
    assert hold['peak_step_memory_gib']==104.017
    assert m.resource_wait(task,dict(cfg,worker_memory_gib=106)) is None
    assert file_sha256(path)==digest


def test_allocation_budget_independent_of_worker_cap(tmp_path):
    task,cfg,path=fixture(tmp_path)
    assert m.resource_wait(task,dict(cfg,worker_memory_gib=106,other_reserved_memory_gib=6))


def test_corrupt_resource_evidence_is_not_a_cache_miss(tmp_path):
    task,cfg,path=fixture(tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):m.resource_wait(task,cfg)


def test_failed_resource_configuration_does_not_block_other_work(tmp_path,monkeypatch):
    task,cfg,path=fixture(tmp_path)
    other=dict(execution='radon',run_dir='other')
    monkeypatch.setattr(m,'work',lambda _:[task,other])
    monkeypatch.setattr(m,'eligible',lambda *args:True)
    assert m.admissible_work(dict(cfg,output=str(tmp_path)),object())==[other]
    assert json.loads((tmp_path/'resource_admission.json').read_text())['ready']==1
