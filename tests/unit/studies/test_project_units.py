import json
from pathlib import Path
import pytest
from radon_bridge.studies import project_units as u
from radon_bridge.studies.research_matrix import arms
from radon_bridge.runtime.state import file_sha256,stable_hash


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp_path):
    spec=dict(arms=arms('glaucoma'),seed=3416,disease='glaucoma',model={'name':'resnet50'})
    path=write(tmp_path/'case.json',spec);root=tmp_path/'case'
    return spec,path,root,u.compile_units(path,root)


def test_fifty_five_arms_share_case_but_are_individually_claimable(tmp_path):
    spec,path,root,tasks=fixture(tmp_path)
    assert len(spec['arms'])==55 and len(tasks)==58
    assert len({t['run_dir'] for t in tasks})==58
    assert len({t['id'] for t in tasks})==58
    assert u.compile_units(path,root)==tasks
    assert [u.read(t['spec'])['unit'] for t in tasks if u.unit_ready(t)]==['prepare']


def test_core_peers_parallel_and_host_aug_waits_only_for_host(tmp_path,monkeypatch):
    spec,path,root,tasks=fixture(tmp_path)
    monkeypatch.setattr(u,'verify_prepared',lambda *a:None)
    checked=[]
    def verify(s,r,n):
        checked.append(n)
        if n!='mmtm256':raise FileNotFoundError(n)
    monkeypatch.setattr(u,'verify_training',verify)
    ready=[u.read(t['spec'])['unit'] for t in tasks if u.unit_ready(t)]
    assert all('arm_'+n in ready for n in u.CORE)
    assert 'arm_mmtm256_plus_radon' in ready
    assert 'arm_attention256_plus_radon' not in ready
    assert 'report_core' not in ready and 'report_full' not in ready


def test_core_completion_does_not_certify_full_case(tmp_path,monkeypatch):
    spec,path,root,tasks=fixture(tmp_path)
    monkeypatch.setattr(u,'verify_prepared',lambda *a:None)
    def verify(s,r,n):
        if n not in u.CORE:raise FileNotFoundError(n)
    monkeypatch.setattr(u,'verify_training',verify)
    ready=[u.read(t['spec'])['unit'] for t in tasks if u.unit_ready(t)]
    assert 'report_core' in ready and 'report_full' not in ready


def test_receipt_tampering_and_scope_change_fail_closed(tmp_path):
    spec,path,root,tasks=fixture(tmp_path)
    write(root/'prepared.json',dict(state='accepted',identity=stable_hash(spec),test_access=False,files={}))
    with pytest.raises(ValueError,match='Incomplete'):u.unit_ready(tasks[1])
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):u.load_unit(u.read(tasks[0]['spec']))


def test_unchanged_case_and_arm_identity_when_split(tmp_path):
    spec,path,root,tasks=fixture(tmp_path)
    bases={'fixed_svd_channel':{}}
    write(root/'bases/accepted.json',{'bases':bases})
    arm=spec['arms'][1]
    expected=stable_hash(dict(case=stable_hash(spec),arm=arm,bases=bases,host_best_sha256=None))
    assert u.arm_identity(spec,root,arm)==expected
    assert u.read(path)==spec


def test_unit_status_does_not_overwrite_other_units(tmp_path,monkeypatch):
    from radon_bridge.studies import project_case
    spec,path,root,tasks=fixture(tmp_path)
    task=tasks[1];unit=u.read(task['spec'])
    monkeypatch.setattr(u,'prerequisites',lambda *a:None)
    monkeypatch.setattr(u,'verify_unit',lambda *a:{'state':'accepted'})
    monkeypatch.setattr(project_case,'execute',lambda *a,**kw:{'state':'accepted'})
    u.execute(unit,task['run_dir'],'cpu')
    assert u.read(Path(task['run_dir'])/'status.json')['state']=='completed'
    assert not (root/'status.json').exists()
    assert not (root/'units/arm_svd_radon/status.json').exists()
