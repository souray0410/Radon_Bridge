import hashlib
import json
from pathlib import Path

import pytest
import torch

from radon_bridge.runtime import paused_sixth_v5_handoff as h


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value))


def patch_constants(monkeypatch, source, spec, packet, checkpoint, v5spec):
    monkeypatch.setattr(h, "SOURCE_CHECKPOINT_SHA256", sha(source))
    monkeypatch.setattr(h, "SOURCE_SPEC_SHA256", sha(spec))
    monkeypatch.setattr(h, "PACKET_SHA256", sha(packet))
    monkeypatch.setattr(h, "QUALIFIED_CHECKPOINT_SHA256", sha(checkpoint))
    monkeypatch.setattr(h, "QUALIFIED_SPEC_SHA256", sha(v5spec))


def chain(tmp_path, monkeypatch):
    source=tmp_path/'source.pt'; spec=tmp_path/'source.json'; converted=tmp_path/'converted.pt'
    source.write_bytes(b'v4'); spec.write_text('{}'); converted.write_bytes(b'v5-converted')
    state={'schema':'optimizer_boundary_v2','framework_api':'V5',
           'progress':{'epoch':8,'offset':21568,'updates':26905}}
    attempt=tmp_path/'attempt.pt'; torch.save(state, attempt)
    v5spec=tmp_path/'v5spec.json'; v5spec.write_text('{}')
    packet=tmp_path/'packet.json'
    write(packet, {'schema':'radon_v5_next_update_packet_v1','test_access':False,
      'dispatch_allowed':False,'receipt_expectation':{'inputs_sha256':{
      'v4_checkpoint':sha(source),'v4_spec':sha(spec),'v5_checkpoint':sha(converted),
      'v5_spec':sha(v5spec)}}})
    patch_constants(monkeypatch, source, spec, packet, attempt, v5spec)
    conversion=tmp_path/'conversion.json'
    write(conversion, {'schema':'v4_to_v5_checkpoint_candidate_v1',
      'state':'awaiting_numerical_replay','source_sha256':sha(source),
      'target_sha256':sha(converted),'tensor_values_dtypes_preserved':True,
      'target_framework':{'api':'V5','commit':h.FORMAL_V5_COMMIT},
      'original_source_modified':False,'training_repeated':False,'dispatch_authorized':False})
    recheck=tmp_path/'recheck.json'
    write(recheck, {'schema':'radon_v5_next_update_independent_recheck_v1',
      'state':'accepted_engineering_only','source_run_id':h.RUN_ID,'test_access':False,
      'dispatch_allowed':False,'scheduler':{'gpu_job_id':'52387276'},
      'independent_read_only_recheck':{'v5_output_checkpoint_sha256':sha(attempt),
      'progress':{'epoch':8,'offset':21568,'updates':26905},'errors':0,
      'tensor_exact':1093,'numpy_rng_exact':1}})
    return dict(source_checkpoint=source,source_spec=spec,conversion_receipt=conversion,
      converted_checkpoint=converted,packet=packet,independent_recheck=recheck,
      attempt_checkpoint=attempt,attempt_spec=v5spec)


def test_real_chain_publishes_resumable_update_26905(tmp_path, monkeypatch):
    evidence=chain(tmp_path, monkeypatch)
    reservation=tmp_path/'reservation.json'
    write(reservation,{'schema':'radon_paused_sixth_v5_reservation_v1','run_id':h.RUN_ID,
      'source_checkpoint_sha256':h.SOURCE_CHECKPOINT_SHA256,'dispatch_allowed':False})
    receipt=h.publish_production_asset(output=tmp_path/'asset',reservation_receipt=reservation, **evidence)
    assert receipt['state']=='accepted_for_new_claim'
    assert torch.load(tmp_path/'asset/last.pt',weights_only=False)['progress']['updates']==26905


def test_reserve_binds_exact_failed_claim_and_terminal_job(tmp_path, monkeypatch):
    run=tmp_path/'run'; run.mkdir(); source=run/'last.pt'; spec=run/'spec.json'
    source.write_bytes(b'v4'); spec.write_text('{}')
    monkeypatch.setattr(h,'SOURCE_CHECKPOINT_SHA256',sha(source))
    monkeypatch.setattr(h,'SOURCE_SPEC_SHA256',sha(spec))
    claim={'run_dir':str(run.resolve()),'spec_sha256':sha(spec),'owner':h.OLD_OWNER,
      'job_id':h.OLD_JOB_ID,'generation':h.OLD_GENERATION,'state':'failed','step':'1'}
    claim_path=tmp_path/'claim.json'; write(claim_path,claim)
    write(run/'status.json',{'state':'paused','test_used':False,'updates':26904})
    monkeypatch.setattr(h,'OLD_CLAIM_SHA256',sha(claim_path))
    monkeypatch.setattr(h,'PAUSED_STATUS_SHA256',sha(run/'status.json'))
    lock=tmp_path/'account.lock'; lock.write_text(''); claims=Claims(claim)
    result=h.reserve_actual_failed_claim(run=run,claims=claims,claim_path=claim_path,
      source_checkpoint=source,source_spec=spec,account_lock=lock,now=2,
      reservation_receipt=tmp_path/'reservation.json',
      observe_job=lambda job:{'job_id':job,'state':'COMPLETED','exit_code':'0:0'})
    assert result['claim']['migration_phase']=='source_reserved'
    assert result['claim']['generation']==111


@pytest.mark.parametrize('field', ['packet','independent_recheck','attempt_checkpoint'])
def test_chain_fails_closed_on_any_drift(tmp_path, monkeypatch, field):
    evidence=chain(tmp_path, monkeypatch); Path(evidence[field]).write_bytes(b'drift')
    reservation=tmp_path/'reservation.json'
    write(reservation,{'schema':'radon_paused_sixth_v5_reservation_v1','run_id':h.RUN_ID,
      'source_checkpoint_sha256':h.SOURCE_CHECKPOINT_SHA256,'dispatch_allowed':False})
    with pytest.raises((ValueError, json.JSONDecodeError)):
        h.publish_production_asset(output=tmp_path/'asset',reservation_receipt=reservation, **evidence)
    assert not (tmp_path/'asset').exists()


class Claims:
    def __init__(self, value): self.value=value
    def mutate(self, run, fn): self.value=fn(dict(self.value)); return self.value


def test_claim_derives_step_from_scheduler_and_binds_grant(tmp_path):
    run=tmp_path/'run'; run.mkdir(); asset=tmp_path/'asset'; asset.mkdir()
    (asset/'last.pt').write_bytes(b'checkpoint'); (asset/'spec.json').write_text('{}')
    write(asset/'asset.json',{'schema':'radon_paused_sixth_v5_production_asset_v1',
      'state':'accepted_for_new_claim','run_id':h.RUN_ID,
      'checkpoint_sha256':sha(asset/'last.pt'),'spec_sha256':sha(asset/'spec.json'),
      'framework_commit':h.FORMAL_V5_COMMIT})
    journal=tmp_path/'requests.json'
    write(journal,{'requests':[{'request_id':'r1','job_id':'77','state':'granted',
      'requested_gpus':1,'metadata':{'project':'Radon_Bridge','run_dir':str(run.resolve()),
      'production_asset_sha256':sha(asset/'asset.json')}}]})
    lock=tmp_path/'account.lock'; lock.write_text('')
    old={'state':'failed','owner':'old','job_id':'12','step':'1','generation':110,
         'run_dir':str(run.resolve()),'spec_sha256':'s'}; claims=Claims(old)
    observed={'job_id':'77','state':'RUNNING','account':'pi-mengy','gpus':1,
              'steps':[{'step':'batch','state':'RUNNING'}]}
    out=h.claim_granted_v5(run=run,claims=claims,asset_dir=asset,
      expected_claim=old,journal_path=journal,journal_sha256=sha(journal),job_id='77',
      observe_job=lambda job:dict(observed),account_lock=lock,now=1)
    assert out['step']=='batch' and out['generation']==111
    assert out['production_asset_sha256']==sha(asset/'asset.json')


def test_claim_rejects_callerless_or_ambiguous_live_identity(tmp_path):
    # Signature deliberately has no step argument: only the scheduler supplies it.
    assert 'step' not in h.claim_granted_v5.__code__.co_varnames[:h.claim_granted_v5.__code__.co_argcount]
