import fcntl
import json
import pytest

from radon_bridge.runtime import qualification_lease as q
from radon_bridge.runtime.state import file_sha256


def fixture(tmp_path, now=100):
    tmp_path.mkdir(parents=True,exist_ok=True)
    policy={'schema':'research_gpu_roles_v1','account_ceiling':24,'native_max_gpus':0,
        'projects':{'Uncertainty_Lab':{'reserved_gpus':2,'request_journals':[]}}}
    role=tmp_path/'role.json';role.write_text(json.dumps(policy)+'\n')
    control=tmp_path/'control.json';control.write_text('{"stop_future_requests":true}\n')
    packet=tmp_path/'packet.json';packet.write_text(json.dumps({
        'schema':'radon_v5_next_update_packet_v1','test_access':False,'requested_gpus':1,
        'submit_timeout_seconds':20,'command':['sbatch','--parsable','--gres=gpu:a100:1','one.sbatch'],
        'finalizer_command_template':['sbatch','--parsable','--dependency=afterany:{gpu_job_id}','final.sbatch']})+'\n')
    lock=tmp_path/'account.lock';lock.touch()
    journal=tmp_path/'requests.json';journal.write_text(json.dumps({'schema':'radon_v5_qualification_requests_v1','requests':[]}))
    policy['projects']['Uncertainty_Lab']['request_journals'].append(str(journal))
    proposed=tmp_path/'role.proposed.json';proposed.write_text(json.dumps(policy)+'\n')
    intent=tmp_path/'intent.json';intent.write_text(json.dumps({'schema':'radon_v5_qualification_intent_v1','attempt':None}))
    lease={'schema':'radon_v5_qualification_lease_v1','lease_id':'rb-once','project':'Radon_Bridge',
        'mode':'qualification','requested_gpus':1,'packet_sha256':file_sha256(packet),'expires_at':now+60,
        'test_access':False,'account_limit':24,'return_entitlement':{'project':'Uncertainty_Lab','gpus':2},
        'role_policy_sha256':file_sha256(proposed),'previous_role_policy_sha256':file_sha256(role),
        'control_sha256':file_sha256(control)}
    st=lock.stat();lease.update(account_lock_inode=st.st_ino,account_lock_device=st.st_dev,
        account_lock_ctime_ns=st.st_ctime_ns,journal_initial_sha256=file_sha256(journal),
        intent_initial_sha256=file_sha256(intent))
    leasep=tmp_path/'lease.json';leasep.write_text(json.dumps(lease))
    return lease,dict(lease_path=leasep,role_policy_path=role,role_policy_proposal_path=proposed,
        policy_transition_receipt_path=tmp_path/'transition.json',control_path=control,packet_path=packet,
        account_lock=lock,journal_path=journal,intent_path=intent)


def test_ready_only_below_global_limit(tmp_path):
    lease,_=fixture(tmp_path)
    assert q.decide(lease,{'limit':24,'total_gpus':23},{'requests':[]},now=100)['action']=='submit_once'
    assert q.decide(lease,{'limit':24,'total_gpus':24},{'requests':[]},now=100)['state']=='waiting_account_capacity'


def test_lease_preserves_liu_entitlement_and_expiry(tmp_path):
    lease,_=fixture(tmp_path);lease['return_entitlement']['gpus']=1
    with pytest.raises(ValueError,match='Uncertainty_Lab'):q.validate(lease,100)
    lease,_=fixture(tmp_path);lease['expires_at']=100
    with pytest.raises(ValueError,match='expired'):q.validate(lease,100)


def test_publish_rechecks_lock_policy_packet_and_journal(tmp_path):
    _,paths=fixture(tmp_path);submitted=[]
    result=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd,timeout:submitted.append((cmd,timeout)) or str(122+len(submitted)),now=100)
    assert result=={'action':'none','state':'submitted','job_id':'123'} and len(submitted)==2
    again=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd,timeout:pytest.fail('duplicate submit'),now=101)
    assert again['state']=='already_submitted'


def test_busy_shared_lock_does_not_snapshot_or_submit(tmp_path):
    _,paths=fixture(tmp_path)
    with paths['account_lock'].open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=q.publish_once(**paths,snapshot=lambda:pytest.fail('snapshot'),
            submit=lambda cmd,timeout:pytest.fail('submit'),now=100)
    assert result['state']=='waiting_account_lock'


@pytest.mark.parametrize('field,pattern',[('role_policy_proposal_path','Role policy proposal'),('control_path','Refiller control'),('packet_path','packet')])
def test_identity_drift_fails_closed(tmp_path,field,pattern):
    _,paths=fixture(tmp_path);paths[field].write_text('changed\n')
    with pytest.raises(ValueError,match=pattern):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda cmd,timeout:'123',now=100)


def test_unidentified_submission_is_terminal_and_not_retried(tmp_path):
    _,paths=fixture(tmp_path)
    result=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda cmd,timeout:'unknown',now=100)
    assert result['state']=='identity_drift'
    assert q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda cmd,timeout:pytest.fail('retry'),now=101)['state']=='submission_intent_needs_review'


def test_scheduler_timeout_keeps_intent_and_never_retries(tmp_path):
    _,paths=fixture(tmp_path);calls=[]
    def timeout(command,timeout):
        calls.append((command,timeout));raise TimeoutError('ambiguous scheduler RPC')
    with pytest.raises(TimeoutError):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=timeout,now=100)
    assert len(calls)==1 and json.loads(paths['intent_path'].read_text())['attempt']['state']=='intent'
    again=q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda c,timeout:pytest.fail('retry'),now=101)
    assert again['state']=='submission_intent_needs_review'


def test_failed_static_gate_does_not_install_role_policy(tmp_path):
    lease,paths=fixture(tmp_path);before=paths['role_policy_path'].read_bytes()
    paths['packet_path'].write_text('changed\n')
    with pytest.raises(ValueError,match='packet'):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
            submit=lambda c,timeout:pytest.fail('submit'),now=100)
    assert paths['role_policy_path'].read_bytes()==before
    assert not paths['policy_transition_receipt_path'].exists()


def test_missing_or_replaced_account_lock_fails_closed(tmp_path):
    _,paths=fixture(tmp_path);paths['account_lock'].unlink()
    with pytest.raises(ValueError,match='missing'):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda c,timeout:'1',now=100)
    lease,paths=fixture(tmp_path/'other');paths['account_lock'].unlink();paths['account_lock'].touch()
    with pytest.raises(ValueError,match='identity'):
        q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},submit=lambda c,timeout:'1',now=100)


def test_terminal_states_return_entitlement():
    complete=q.terminal_transition({'state':'submitted','job_id':'12'},slurm_state='COMPLETED',exit_code='0:0')
    assert complete['state']=='completed' and 'return_entitlement' in complete['result']
    failed=q.terminal_transition({'state':'submitted','job_id':'13'},slurm_state='OUT_OF_MEMORY',exit_code='0:125')
    assert failed['state']=='failed' and failed['slurm_state']=='OUT_OF_MEMORY'


def test_afterany_finalizer_closes_journal_and_keeps_dispatch_closed(tmp_path):
    lease,paths=fixture(tmp_path);submitted=[]
    q.publish_once(**paths,snapshot=lambda:{'limit':24,'total_gpus':19},
        submit=lambda c,timeout:submitted.append(c) or str(122+len(submitted)),now=100)
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'schema':'radon_v5_exact_next_update_replay_v1',
        'status':'accepted_engineering_only','dispatch_allowed':False}))
    status=tmp_path/'status.json'
    result=q.finalize_once(lease_path=paths['lease_path'],account_lock=paths['account_lock'],
        journal_path=paths['journal_path'],intent_path=paths['intent_path'],finalizer_job_id='124',
        gpu_job_id='123',slurm_state='COMPLETED',exit_code='0:0',qualification_receipt_path=receipt,
        status_path=status,now=101)
    assert result['state']=='completed'
    assert json.loads(status.read_text())['dispatch_allowed'] is False
