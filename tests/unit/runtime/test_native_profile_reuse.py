import copy
import json
import pytest
from radon_bridge.runtime.state import file_sha256
from radon_bridge.runtime.native_profile_reuse import prior, qualify


def fixture(tmp_path):
    hw=dict(name='A100', total_bytes=80*1024**3, torch='2.8',cuda='12.8',cudnn=1,driver='570')
    def receipt(name,full,checkpoint):
        root=tmp_path/name;root.mkdir()
        files={}
        for n in ('boundary.pt','expected.pt','resumed.pt'):
            p=root/n;p.write_bytes(b'fixture');files[n]=file_sha256(p)
        r=dict(status='accepted',spec_sha256='spec',hardware=hw,test_used=False,
            formal_updates=0,max_eyes=2,full_state_replay_exact=True,fresh_train_batches=True,
            warmup_updates=5,measured_updates=20,step_memory_observed=True,
            full_development=full,full_train_read=full,files=files,peak_gpu_gib=39,
            peak_step_memory_gib=100,checkpoint_source_sha256=checkpoint)
        p=root/'accepted.json';p.write_text(json.dumps(r));return p
    old=receipt('old',True,'old_checkpoint');now=receipt('now',False,'new_checkpoint')
    ref=dict(path=str(old),sha256=file_sha256(old))
    kwargs=dict(current=now,spec_sha='spec',hardware=hw,checkpoint_sha='new_checkpoint',
        total_gpu_gib=80,other_gpu_gib=1,allocated_ram_gib=128,other_ram_gib=2,
        allocated_cpus=16,worker_cpus=14,worker_ram_gib=120,output=tmp_path/'joined.json')
    return ref,kwargs


def test_reuse_keeps_fresh_checkpoint_probe_and_raw_receipt(tmp_path):
    ref,k=fixture(tmp_path);raw=file_sha256(k['current'])
    assert prior(ref,k['spec_sha'],k['hardware'])==ref
    result=qualify(ref,**k)
    assert result['static_coverage_reused'] and result['checkpoint_sha256']=='new_checkpoint'
    assert file_sha256(k['current'])==raw
    assert not json.loads(k['current'].read_text())['full_development']


@pytest.mark.parametrize('key,value', [('spec_sha','different'),('checkpoint_sha','old_checkpoint'),
    ('hardware',{}),('other_gpu_gib',42),('other_ram_gib',20),('allocated_ram_gib',0),('worker_cpus',17),('other_ram_gib',float('nan'))])
def test_unknown_changed_or_insufficient_envelope_rejects(tmp_path,key,value):
    ref,k=fixture(tmp_path);k[key]=value
    with pytest.raises(ValueError):qualify(ref,**k)
    assert not k['output'].exists()


def test_corruption_and_incomplete_source_rejected(tmp_path):
    ref,k=fixture(tmp_path)
    p=k['current'].parent/'resumed.pt';p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='file changed'):qualify(ref,**k)
    old=__import__('pathlib').Path(ref['path']);d=json.loads(old.read_text());d['full_train_read']=False;old.write_text(json.dumps(d));ref['sha256']=file_sha256(old)
    with pytest.raises(ValueError,match='Whole-cohort'):prior(ref,k['spec_sha'],k['hardware'])


def test_live_envelope_uses_slurm_limits_and_rejects_scientific_companion(monkeypatch):
    from types import SimpleNamespace
    from radon_bridge.runtime.native_profile_reuse import live_envelope
    import subprocess
    gpu=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _:None,
        mem_get_info=lambda _: (79*1024**3,80*1024**3)))
    steps=['StepId=42.1 State=RUNNING CPUs=14 TRES=cpu=14,mem=100G',
           'StepId=42.0 State=RUNNING CPUs=1 TRES=cpu=1,mem=2G']
    def output(command,**_):
        return 'AllocTRES=cpu=16,mem=128G' if 'job' in command else '\n'.join(steps)
    monkeypatch.setattr(subprocess,'check_output',output)
    row=live_envelope(gpu,'42','1')
    assert row['allocated_ram_gib']==128 and row['other_ram_gib']==2
    assert row['worker_ram_gib']==100
    assert row['allocated_cpus']==15 and row['worker_cpus']==14
    steps.append('StepId=42.2 State=RUNNING CPUs=1 TRES=cpu=1,mem=6G')
    with pytest.raises(ValueError,match='concurrent scientific'):live_envelope(gpu,'42','1')


@pytest.mark.parametrize('old_peak,current_peak,accepted', [
    (105, 50, False), (90, 50, True), (50, 101, False), (100, 50, True)])
def test_reused_full_peak_must_fit_actual_worker_step(tmp_path, old_peak, current_peak, accepted):
    ref, k = fixture(tmp_path)
    from pathlib import Path
    for path, peak in ((Path(ref['path']), old_peak), (k['current'], current_peak)):
        row = json.loads(path.read_text())
        row['peak_step_memory_gib'] = peak
        path.write_text(json.dumps(row))
    ref['sha256'] = file_sha256(ref['path'])
    k['worker_ram_gib'] = 100
    if accepted:
        result = qualify(ref, **k)
        assert result['limits']['worker_ram_gib'] == 100
        assert result['limits']['allocated_ram_gib'] == 128
    else:
        with pytest.raises(ValueError, match='Worker step memory limit'):
            qualify(ref, **k)
        assert not k['output'].exists()


@pytest.mark.parametrize('value', [None, '100', True, 0, -1, float('nan'), float('inf'), 129])
def test_unknown_or_inconsistent_worker_limit_rejects(tmp_path, value):
    ref, k = fixture(tmp_path)
    k['worker_ram_gib'] = value
    with pytest.raises(ValueError):
        qualify(ref, **k)
    assert not k['output'].exists()


@pytest.mark.parametrize('memory', ['', 'mem=NaNG', 'mem=infG', 'mem=0G', 'mem=-1G', 'mem=unknown', 'mem=129G'])
def test_live_envelope_requires_finite_positive_worker_memory(monkeypatch, memory):
    from radon_bridge.runtime.native_profile_reuse import live_envelope
    import subprocess
    def output(command, **_):
        if 'job' in command:
            return 'AllocTRES=cpu=16,mem=128G'
        return f'StepId=42.1 State=RUNNING CPUs=14 TRES=cpu=14,{memory}'
    monkeypatch.setattr(subprocess, 'check_output', output)
    with pytest.raises(ValueError):
        live_envelope(None, '42', '1')


@pytest.mark.parametrize('key', ['peak_step_memory_gib', 'peak_gpu_gib'])
@pytest.mark.parametrize('value', [None, True, '50', 0, -1, float('nan'), float('inf')])
def test_unknown_measured_peak_rejects(tmp_path, key, value):
    ref, k = fixture(tmp_path)
    row = json.loads(k['current'].read_text())
    row[key] = value
    k['current'].write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Unknown measured resource peak'):
        qualify(ref, **k)
    assert not k['output'].exists()


def test_hardware_change_recomputes_full_profile_but_corruption_does_not(tmp_path):
    from radon_bridge.runtime.native_profile_reuse import matching_reference
    ref,k=fixture(tmp_path)
    assert matching_reference(ref,k['spec_sha'],k['hardware'])==ref
    assert matching_reference(ref,k['spec_sha'],dict(k['hardware'],driver='different')) is None
    from pathlib import Path
    Path(ref['path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        matching_reference(ref,k['spec_sha'],dict(k['hardware'],driver='different'))


def test_operational_worker_memory_preserves_allocation_reserve():
    from radon_bridge.runtime.project_dispatch import worker_memory_gib
    assert worker_memory_gib({})==100
    assert worker_memory_gib({'worker_memory_gib':104})==104
    for bad in (True,0,39,107,128,'104'):
        with pytest.raises(ValueError):worker_memory_gib({'worker_memory_gib':bad})
