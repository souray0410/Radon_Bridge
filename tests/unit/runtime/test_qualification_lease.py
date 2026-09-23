import fcntl
import json
import pytest

from radon_bridge.runtime import qualification_lease as q
from radon_bridge.runtime.state import file_sha256


def fixture(tmp_path, now=100):
    role=tmp_path/'role.json';role.write_text('{}\n')
    control=tmp_path/'control.json';control.write_text('{"stop_future_requests":true}\n')
    packet=tmp_path/'packet.json';packet.write_text(json.dumps({
        'schema':'radon_v5_next_update_packet_v1','test_access':False,'command':['salloc','one']})+'\n')
    lease={'schema':'radon_v5_qualification_lease_v1','lease_id':'rb-once','project':'Radon_Bridge',
        'mode':'qualification','requested_gpus':1,'packet_sha256':file_sha256(packet),'expires_at':now+60,
        'test_access':False,'account_limit':24,'return_entitlement':{'project':'Liu','gpus':2},
        'role_policy_sha256':file_sha256(role),'control_sha256':file_sha256(control)}
    leasep=tmp_path/'lease.json';leasep.write_text(json.dumps(lease))
    journal=tmp_path/'requests.json';journal.write_text(json.dumps({'schema':'radon_bridge_workflow_requests_v1','requests':[]}))
    return lease,dict(lease_path=leasep,role_policy_path=role,control_path=control,packet_path=packet,
        account_lock=tmp_path/'account.lock',journal_path=journal)


def test_ready_only_below_global_limit(tmp_path):
    lease,_=fixture(tmp_path)
    assert q.decide(lease,{'limit':24,'total_gpus':23},{'requests':[]},now=100)['action']=='submit_once'
    assert q.decide(lease,{'limit':24,'total_gpus':24},{'requests':[]},now=100)['state']=='waiting_account_capacity'


def test_lease_preserves_liu_entitlement_and_expiry(tmp_path):
    lease,_=fixture(tmp_path);lease['return_entitlement']['gpus']=1
    with pytest.raises(ValueError,match='Liu'):q.validate(lease,100)
    lease,_=fixture(tmp_path);lease['expires_at']=100
    with pytest.raises(ValueError,match='expired'):q.validate(lease,100)


def test_publish_rechecks_lock_policy_packet_and_journal(tmp_path):
    _,paths=fixture(tmp_path);submitted=[]
    result=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd:submitted.append(cmd) or '123',now=100)
    assert result=={'action':'none','state':'submitted','job_id':'123'} and len(submitted)==1
    again=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd:pytest.fail('duplicate submit'),now=101)
    assert again['state']=='already_submitted'


def test_busy_shared_lock_does_not_snapshot_or_submit(tmp_path):
    _,paths=fixture(tmp_path)
    with paths['account_lock'].open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=q.publish_once(**paths,snapshot=lambda:pytest.fail('snapshot'),
            submit=lambda cmd:pytest.fail('submit'),now=100)
    assert result['state']=='waiting_account_lock'


@pytest.mark.parametrize('field,pattern',[('role_policy_path','Role policy'),('control_path','Refiller control'),('packet_path','packet')])
def test_identity_drift_fails_closed(tmp_path,field,pattern):
    _,paths=fixture(tmp_path);paths[field].write_text('changed\n')
    with pytest.raises(ValueError,match=pattern):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda cmd:'123',now=100)


def test_unidentified_submission_is_terminal_and_not_retried(tmp_path):
    _,paths=fixture(tmp_path)
    result=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda cmd:'unknown',now=100)
    assert result['state']=='identity_drift'
    assert q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd:pytest.fail('retry'),now=101)['state']=='terminal_identity_drift'


def test_terminal_states_return_entitlement():
    complete=q.terminal_transition({'state':'submitted','job_id':'12'},slurm_state='COMPLETED',exit_code='0:0')
    assert complete['state']=='completed' and 'return_entitlement' in complete['result']
    failed=q.terminal_transition({'state':'submitted','job_id':'13'},slurm_state='OUT_OF_MEMORY',exit_code='0:125')
    assert failed['state']=='failed' and failed['slurm_state']=='OUT_OF_MEMORY'
