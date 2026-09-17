import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from radon_bridge.studies import semester as s
from radon_bridge.runtime.state import file_sha256


def test_matrix_counts_and_independent_host_dependencies():
    m=s.manifest();p=m['positions']
    assert len(p)==len({r['id'] for r in p})==309
    assert sum(r['arm']['id'] in s.CORE for r in p)==162
    assert sum(r['seed']==3416 for r in p)==103
    for r in p:
        assert bool(r['replication_gate'])==(r['seed']!=3416)
        assert len(r['dependencies'])==1+bool(r['arm'].get('host'))
    assert m['future']['six_network_architecture_arrangements']==729
    assert m['future']['prerequisite_for_eye_core'] is False
    assert m['test_access'] is False


def test_heartbeat_never_hides_stall_and_real_progress_resets():
    obs={'a':dict(state='running',optimizer_steps=10,updated_at=1)}
    clock=s.track_progress({},obs,100)
    obs['a']['updated_at']=100+49*3600
    later=s.track_progress(clock,obs,100+49*3600)
    assert later['a']['stalled']
    obs['a']['optimizer_steps']=11
    assert not s.track_progress(later,obs,100+49*3600)['a']['stalled']
    with pytest.raises(ValueError,match='future'):s.track_progress(clock,obs | {'a':{'state':'running','optimizer_steps':10}},99)


def test_case_counts_are_not_metrics_or_test_permission():
    m=s.manifest();p=[r for r in m['positions'] if r['group']=='cataract/resnet50' and r['seed']==3416]
    obs={r['id']:dict(state='accepted') for r in p}
    partial=s.weekly_report(m,dict(list(obs.items())[:5]),{},100)
    assert not partial['complete_core_seed_groups']
    complete=s.weekly_report(m,obs,partial,101)
    assert len(complete['complete_core_seed_groups'])==1
    assert len(complete['newly_accepted_positions'])==1
    assert not complete['test_release_authorized'] and not complete['ranking_allowed']
    assert complete['complete_core_seed_groups'][0]['evidence_scope']=='preliminary_single_seed'


def test_stage_never_accepts_failure_or_raw_receipt_string():
    m=s.manifest();obs={r['id']:dict(state='accepted') for r in m['positions']}
    assert not s.stage_ready(m,obs,{},lambda *a:True)
    assert not s.stage_ready(m,obs,{},lambda *a:False)
    obs[next(iter(obs))]['state']='needs_review'
    assert not s.stage_ready(m,obs,{},lambda *a:True)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return s.reference(path)


def test_pilot_release_requires_technical_receipts_not_favorable_score(tmp_path):
    gates={k:write(tmp_path/(k+'.json'),dict(state='accepted',test_access=False)) for k in s.TECHNICAL_GATES}
    value=dict(schema='radon_pilot_acceptance_v1',group='g',seed=3416,test_access=False,
               accepted_positions=['a'],gates=gates,mean_difference=-.2)
    ref=write(tmp_path/'pilot.json',value)
    assert s.pilot_release('g',ref,['a'],lambda x:True)
    with pytest.raises(ValueError):s.pilot_release('g',ref,['a','b'],lambda x:True)
    with pytest.raises(ValueError):s.pilot_release('g',ref,['a'],lambda x:False)
    (tmp_path/'matched_report.json').write_text('{}')
    with pytest.raises(ValueError):s.pilot_release('g',ref,['a'],lambda x:True)


def test_weekly_deadlines_use_riyadh_and_are_not_daily_delivery():
    def stamp(text):return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo('Asia/Riyadh')).timestamp()
    assert [e['kind'] for e in s.due_events(stamp('2026-09-19T20:00'))]==['snapshot']
    assert not s.due_events(stamp('2026-09-20T06:44'))
    assert [e['kind'] for e in s.due_events(stamp('2026-09-20T06:45'))]==['draft']
    assert s.due_events(stamp('2026-09-20T09:45'))[-1]['kind']=='refresh_and_deliver'
    assert not s.due_events(stamp('2026-09-21T09:45'))


def test_monitor_does_not_accept_unverified_imports(tmp_path):
    ref=write(tmp_path/'observations.json',dict(schema='radon_stage_observations_v1',test_access=False,
        observations={'fake':dict(state='accepted')}))
    config=dict(schema='radon_semester_monitor_v1',output=str(tmp_path/'out'),test_access=False,observation_imports=[ref])
    with pytest.raises(ValueError,match='live owner verifier'):s.reconcile(config)


def test_monitor_unbound_is_waiting_and_restart_keeps_progress_clock(tmp_path):
    cfg=dict(schema='radon_semester_monitor_v1',output=str(tmp_path),test_access=False)
    a=s.reconcile(cfg);clock=json.loads((tmp_path/'progress_clock.json').read_text())
    b=s.reconcile(cfg);again=json.loads((tmp_path/'progress_clock.json').read_text())
    assert a['accepted_training_positions']==b['accepted_training_positions']==0
    assert clock==again
    assert len(list(tmp_path.glob('positions.csv')))==1


@pytest.mark.parametrize('architecture,oct_model', [
    ('resnet50','resnet50'), ('densenet121','monai_densenet121_3d'),
    ('swin_b','swin_unetr_encoder_3d')])
def test_individual_seed_dispatch_does_not_wait_for_other_parent_seeds(tmp_path,monkeypatch,architecture,oct_model):
    from radon_bridge.studies import project_orders as po
    root=tmp_path/'projects';catalog=write(tmp_path/'catalog.json',dict(candidates=[dict(model=n) for n in sorted({architecture,oct_model})]))
    gate=write(tmp_path/'gate.json',dict(status='accepted',execution_contract='radon_independent_units_v1'))
    config=dict(project=dict(output=str(root),runtime_gate=gate,training={}),catalog=catalog,
                protocol={'sha256':'protocol'},source_pins=[])
    selected=[]
    for role in ('cfp','oct'):
        parent=tmp_path/role
        write(parent/'spec.json',dict(training=dict(seed=3416)))
        selected.append(dict(state='waiting_replications',selected=dict(run_dir=str(parent)),replicas=[dict(state='awaiting_native_acceptance',run_dir='absent')]))
    groups={f'cataract/{model}/{track}':g for model,track,g in zip((architecture,oct_model),('cfp_2d','oct_volume_3d'),selected)}
    def materialize(source,dest,spec,verify):
        target=dest/Path(source).name;write(target/'selected_artifact.json',{'verified':True});return target
    monkeypatch.setattr(po,'materialize_selected',materialize)
    monkeypatch.setitem(sys.modules,'radon_bridge.analysis.project_rollup',SimpleNamespace(summarize=lambda t,r:dict(accepted=0,complete=False)))
    reserved=[]
    def reserve(root,namespace,key,spec,**kwargs):
        reserved.append(key);run=root/key.replace('/','_');run.mkdir(parents=True,exist_ok=True);return run
    status=po.advance(config,groups,lambda *a:True,reserve)
    assert reserved==[f'cataract/{architecture}/seed3416']
    assert status['tasks']==9  # shared preparation, six arms, core and full reports
    assert status['cases']==1
    assert status['groups'][f'cataract/{architecture}']['3417']=='waiting_this_seed_parents'
    po.advance(config,groups,lambda *a:True,reserve)
    assert len(list((root/'bindings').glob('*.json')))==1


def test_pilot_case_cannot_be_released_from_forged_acceptance(tmp_path,monkeypatch):
    from radon_bridge.studies import project_orders as po
    from radon_bridge.runtime.state import stable_hash
    spec=write(tmp_path/'spec.json',dict(seed=3416))
    write(tmp_path/'bindings'/(stable_hash('g/seed3416')+'.json'),dict(spec=spec['path'],spec_sha256=spec['sha256'],run_dir=str(tmp_path/'run')))
    assert not po.pilot_accepted(tmp_path,'g')
    write(tmp_path/'run/accepted.json',dict(state='accepted'))
    assert not po.pilot_accepted(tmp_path,'g')
    write(tmp_path/'run/packages/core/accepted.json',dict(state='accepted'))
    def reject(*args):raise ValueError('Missing full scientific evidence')
    monkeypatch.setitem(sys.modules,'radon_bridge.studies.project_units',SimpleNamespace(verify_core=reject))
    with pytest.raises(ValueError,match='scientific'):po.pilot_accepted(tmp_path,'g')


def test_weekly_delta_uses_prior_week_not_previous_monitor_tick(tmp_path,monkeypatch):
    # Monday belongs to the following Sunday's report; prior delivery is baseline.
    now=datetime(2026,9,21,12,tzinfo=ZoneInfo('Asia/Riyadh')).timestamp()
    monkeypatch.setattr(s.time,'time',lambda:now)
    key=s.manifest()['positions'][0]['id']
    write(tmp_path/'weekly/2026-09-20/refresh_and_deliver.json',dict(report=dict(accepted_positions=[key])))
    cfg=dict(schema='radon_semester_monitor_v1',output=str(tmp_path),test_access=False)
    result=s.reconcile(cfg)
    assert result['previous_week_baseline'].endswith('2026-09-20/refresh_and_deliver.json')
    assert result['withdrawn_acceptances']==[key]
