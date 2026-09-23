import json
import types
import pytest

from radon_bridge.runtime import v5_owner_handoff as h


class Claims:
    def __init__(self, root, claim):self.root=root;self.claim=claim
    def path(self, run):return self.root/'claim.json'


def write(path, value):path.write_text(json.dumps(value));return path


def fixture(tmp_path):
    run=tmp_path/'run';run.mkdir();task={'run_dir':str(run),'spec_sha256':'a'*64}
    claim={'owner':'old-owner','job_id':'123','generation':7,'state':'running','spec_sha256':'a'*64}
    claims=Claims(tmp_path,claim);write(claims.path(run),claim)
    registry=write(tmp_path/'registry.json',{'schema':'radon_v4_to_v5_owner_handoffs_v1','test_access':False,'entries':[]})
    lock=tmp_path/'account.lock';lock.touch()
    deployment=write(tmp_path/'deployment.json',{'schema':'radon_v5_handoff_guard_deployment_v1','active':True,
        'source_commit':'new','registry_path':str(registry),'test_access':False})
    return task,claims,registry,lock,deployment


def test_arm_blocks_old_dispatcher_reclaim_before_and_after_release(tmp_path):
    task,claims,registry,lock,deployment=fixture(tmp_path)
    h.arm_once(registry_path=registry,account_lock=lock,claims=claims,task=task,
        expected_owner='old-owner',expected_job_id='123',expected_generation=7,
        deployment_receipt_path=deployment,required_dispatcher_commit='new',now=1)
    assert h.protected(task,claims,registry)
    write(claims.path(task['run_dir']),{'owner':'radon-v5-migration-op','job_id':'cpu-op',
        'generation':8,'state':'claimed','spec_sha256':'a'*64,'previous':{'job_id':'123','state':'paused'}})
    assert h.protected(task,claims,registry)


def test_arm_rejects_wrong_owner_or_unproved_guard(tmp_path):
    task,claims,registry,lock,deployment=fixture(tmp_path)
    with pytest.raises(ValueError,match='reviewed owner'):
        h.arm_once(registry_path=registry,account_lock=lock,claims=claims,task=task,
            expected_owner='wrong',expected_job_id='123',expected_generation=7,
            deployment_receipt_path=deployment,required_dispatcher_commit='new')
    d=json.loads(deployment.read_text());d['active']=False;write(deployment,d)
    with pytest.raises(ValueError,match='unproven'):
        h.arm_once(registry_path=registry,account_lock=lock,claims=claims,task=task,
            expected_owner='old-owner',expected_job_id='123',expected_generation=7,
            deployment_receipt_path=deployment,required_dispatcher_commit='new')


def test_identity_drift_fails_closed(tmp_path):
    task,claims,registry,lock,deployment=fixture(tmp_path)
    h.arm_once(registry_path=registry,account_lock=lock,claims=claims,task=task,
        expected_owner='old-owner',expected_job_id='123',expected_generation=7,
        deployment_receipt_path=deployment,required_dispatcher_commit='new')
    c=json.loads(claims.path(task['run_dir']).read_text());c['job_id']='999';write(claims.path(task['run_dir']),c)
    with pytest.raises(ValueError,match='identity drift'):h.protected(task,claims,registry)
