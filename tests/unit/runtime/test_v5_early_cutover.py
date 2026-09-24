import hashlib
import json
from pathlib import Path

import pytest

from radon_bridge.runtime import v5_early_cutover as c


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)); return path


class Claims:
    def __init__(self, root, run, claim):
        self.root=Path(root); self._path=self.root/'claim.json'; write(self._path,claim)
        self._path.with_suffix('.lock').touch()
    def path(self, run): return self._path
    def mutate(self, run, fn):
        value=fn(json.loads(self._path.read_text()));write(self._path,value);return value


def fixture(tmp_path, state='failed'):
    run=tmp_path/'run';run.mkdir();spec=run/'spec.json';spec.write_text('{}')
    checkpoint=run/'last.pt';checkpoint.write_bytes(b'checkpoint')
    status={'state':'paused','test_used':False,'updates':10};write(run/'status.json',status)
    spec_sha=c.file_sha256(spec); checkpoint_sha=c.file_sha256(checkpoint)
    claim={'run_dir':str(run.resolve()),'spec_sha256':spec_sha,'owner':'old','job_id':'123',
           'step':'1','generation':7,'state':state}
    claims=Claims(tmp_path/'claims',run,claim)
    account=tmp_path/'account.lock';account.touch()
    budget={'schema':'radon_v5_role_budget_gate_v1','project':'Radon_Bridge','test_access':False,
            'account_cap':24,'running_plus_pending':20,'project_slot_available':True,
            'registered_job_ids':[], 'role_policy_sha256':'a'*64,
            'observed_role_policy_sha256':'a'*64}
    budget_path=write(tmp_path/'budget.json',budget)
    return run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget_path


def terminal(job):return {'job_id':job,'state':'COMPLETED','exit_code':'0:0'}


def test_failed_paused_reservation_and_freeze(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path)
    receipt=tmp_path/'reservation.json'
    value=c.reserve_paused_failed(run=run,spec_path=spec,checkpoint_path=checkpoint,
        claims=claims,expected_spec_sha256=spec_sha,expected_checkpoint_sha256=checkpoint_sha,
        expected_owner='old',expected_job='123',expected_generation=7,
        migration_owner='radon-v5-migration-test',account_lock=account,
        role_budget_path=budget,role_budget_sha256=c.file_sha256(budget),
        observe_job=terminal,receipt_path=receipt,now=1)
    assert value['dispatch_allowed'] is False
    claimed=json.loads(claims.path(run).read_text())
    assert claimed['state']=='claimed'
    assert claimed['migration_phase']=='v5_migration_reserved'
    frozen=c.freeze_reserved_boundary(reservation_receipt=receipt,
        source_checkpoint=checkpoint,source_spec=spec,output=tmp_path/'frozen')
    assert frozen['checkpoint_sha256']==checkpoint_sha
    assert (tmp_path/'frozen/last.v4.pt').read_bytes()==b'checkpoint'


def test_reservation_rejects_nonterminal_or_budget_drift(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path)
    args=dict(run=run,spec_path=spec,checkpoint_path=checkpoint,claims=claims,
        expected_spec_sha256=spec_sha,expected_checkpoint_sha256=checkpoint_sha,
        expected_owner='old',expected_job='123',expected_generation=7,
        migration_owner='radon-v5-migration-test',account_lock=account,
        role_budget_path=budget,role_budget_sha256=c.file_sha256(budget),
        receipt_path=tmp_path/'reservation.json')
    with pytest.raises(ValueError,match='terminal'):
        c.reserve_paused_failed(**args,observe_job=lambda j:{'job_id':j,'state':'RUNNING','exit_code':None})
    bad=json.loads(budget.read_text());bad['running_plus_pending']=24;write(budget,bad)
    args['role_budget_sha256']=c.file_sha256(budget)
    with pytest.raises(ValueError,match='role slot'):
        c.reserve_paused_failed(**args,observe_job=terminal)


def test_exactly_once_and_source_drift_fail_closed(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path)
    kwargs=dict(run=run,spec_path=spec,checkpoint_path=checkpoint,claims=claims,
        expected_spec_sha256=spec_sha,expected_checkpoint_sha256=checkpoint_sha,
        expected_owner='old',expected_job='123',expected_generation=7,
        migration_owner='radon-v5-migration-test',account_lock=account,
        role_budget_path=budget,role_budget_sha256=c.file_sha256(budget),observe_job=terminal,
        receipt_path=tmp_path/'reservation.json',now=1)
    c.reserve_paused_failed(**kwargs)
    with pytest.raises(ValueError,match='reviewed source'):
        c.reserve_paused_failed(**kwargs)
    output=tmp_path/'existing';output.mkdir()
    with pytest.raises(FileExistsError):
        c.freeze_reserved_boundary(reservation_receipt=kwargs['receipt_path'],
            source_checkpoint=checkpoint,source_spec=spec,output=output)


def test_early_pause_holds_claim_lock_and_transitions_after_terminal(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path,state='running')
    clock=[0];calls=[]
    def observe(job,step):
        calls.append((job,step))
        if len(calls)==1:return {'job_id':job,'step':step,'state':'RUNNING'}
        write(run/'status.json',{'state':'paused','test_used':False,'updates':11})
        return {'job_id':job,'step':step,'state':'COMPLETED','exit_code':'75:0'}
    result=c.early_pause_to_reservation(run=run,claims=claims,
        expected_claim={'owner':'old','job_id':'123','step':'1','generation':7,
                        'state':'running','spec_sha256':spec_sha},
        pause_path=run/'pause.json',pause_receipt={'reason':'v5_early_cutover','token':'x'},
        observe_step=observe,checkpoint_path=checkpoint,
        migration_owner='radon-v5-migration-test',timeout=5,poll_interval=1,
        now=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
    assert result['state']=='v5_migration_reserved'
    assert read(run/'pause.json')['token']=='x'
    assert read(claims.path(run))['generation']==8
    assert read(claims.path(run))['state']=='claimed'


def read(path):return json.loads(Path(path).read_text())


def test_early_pause_worker_death_without_paused_checkpoint_fails(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path,state='running')
    write(run/'status.json',{'state':'training','test_used':False})
    with pytest.raises(RuntimeError,match='without publishing'):
        c.early_pause_to_reservation(run=run,claims=claims,
            expected_claim={'owner':'old','job_id':'123','step':'1','generation':7,
                            'state':'running','spec_sha256':spec_sha},
            pause_path=run/'pause.json',pause_receipt={'reason':'v5_early_cutover'},
            observe_step=lambda j,s:{'job_id':j,'step':s,'state':'FAILED','exit_code':'1:0'},
            checkpoint_path=checkpoint,migration_owner='migration',timeout=1)


def test_v5_gpu_claim_requires_exact_acceptance_and_granted_job(tmp_path):
    run,spec,checkpoint,spec_sha,checkpoint_sha,claims,account,budget=fixture(tmp_path,state='failed')
    claim=read(claims.path(run));claim.update(owner='migration',state='claimed',
        migration_phase='v5_migration_reserved',generation=8)
    write(claims.path(run),claim)
    asset=write(tmp_path/'asset.json',{'schema':'radon_v5_production_asset_v1','state':'accepted',
        'framework_commit':c.FORMAL_V5_COMMIT,'strict_cpu_load':True,
        'eligible_for_formal_selection':True,'checkpoint_sha256':'b'*64,'test_access':False})
    acceptance=write(tmp_path/'gpu.json',{'schema':'radon_v5_exact_next_update_replay_v1',
        'accepted':True,'framework_commit':c.FORMAL_V5_COMMIT,
        'production_checkpoint_sha256':'b'*64,'test_access':False})
    budget_body=read(budget);budget_body.update(running_plus_pending=24,
        project_slot_available=False,registered_job_ids=['999']);write(budget,budget_body)
    result=c.claim_v5_gpu(run=run,claims=claims,migration_owner='migration',expected_generation=8,
        job_id='999',step='1',observe_job=lambda j:{'job_id':j,'state':'RUNNING','gpus':1,'account':'pi-mengy'},
        role_budget_path=budget,role_budget_sha256=c.file_sha256(budget),account_lock=account,
        gpu_acceptance_path=acceptance,gpu_acceptance_sha256=c.file_sha256(acceptance),
        production_asset_path=asset,production_asset_sha256=c.file_sha256(asset),now=2)
    assert result['owner']=='radon-v5-production-999'
    assert result['framework_commit']==c.FORMAL_V5_COMMIT
