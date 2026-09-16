"""Finite stage references and evidence gates; never submits GPUs or reads test.

A position is not a new execution. Bind existing runs explicitly; use their own
acceptors before publishing observations in this contract. Scheduling remains
owned by the existing project/native claim managers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import csv
import fcntl
import logging
import signal
import json
import time

from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash
from radon_bridge.studies.research_matrix import arms, comparisons

VERSION = 'radon_semester_20260916_v1'
DISEASES = ('cataract', 'glaucoma', 'macular_degeneration')
ARCHITECTURES = ('resnet50', 'densenet121', 'swin_b')
SEEDS = (3416, 3417, 3418)
CORE = ('continue', 'svd_radon', 'svd_resample', 'svd_self', 'mmtm256', 'attention256')
TECHNICAL_GATES = ('data_and_parent_audit', 'runtime_and_recovery',
                   'matched_diagnostics', 'matched_report')
STATES = {'waiting', 'running', 'paused', 'needs_review', 'accepted',
          'infeasible_before_performance'}


def manifest():
    positions = []
    definitions = []
    for disease in DISEASES:
        for architecture in ARCHITECTURES:
            group = disease + '/' + architecture
            for seed in SEEDS:
                for arm in arms(disease, architecture):
                    positions.append(dict(id=f'{group}/seed{seed}/{arm["id"]}',
                        group=group, disease=disease, architecture=architecture,
                        seed=seed, arm=arm,
                        dependencies=[f'{group}/seed{seed}/parents'] +
                        ([f'{group}/seed{seed}/{arm["host"]}'] if arm.get('host') else []),
                        replication_gate=None if seed == 3416 else f'{group}/pilot_acceptance',
                        execution_id=None))
            definitions.extend(dict(group=group, **c) for c in comparisons(disease, architecture))
    assert len(positions) == 309
    return dict(schema=VERSION, stage='eye_core', positions=positions,
        counts=dict(core=162,reference_mechanism_additional=147,training_positions=309,
                    registered_executions=0),
        comparisons=definitions, test_access=False,
        transfer_levels=['method_retraining','parameter_adaptation','frozen_representation'],
        future=dict(six_network_architecture_arrangements=729, retained=True,
                    prerequisite_for_eye_core=False),
        supplements='independent locked manifests; not implicitly counted or dispatched',
        cardiac_gate=['eye_core_acceptance','semantic_data_acceptance','user_task_protocol_lock'],
        eye_heart_gate=['cardiac_acceptance','clinical_question_lock','paired_time_audit'],
        deadline='2026-12-31', task_scope='classification',
        allocation=dict(minimum_hours=48,default_hours=48,generic_gpu_floor=2),
        weekly=dict(timezone='Asia/Riyadh',meeting='Sunday 10:00',
                    snapshot='Saturday',draft='Sunday 06:45',refresh='Sunday 09:45',
                    delivery='Sunday 09:55',test_access=False))


def reference(path):
    return dict(path=str(path), sha256=file_sha256(Path(path)))


def checked(ref):
    if set(ref) != {'path','sha256'} or file_sha256(Path(ref['path'])) != ref['sha256']:
        raise ValueError('Evidence checksum mismatch')
    return json.loads(Path(ref['path']).read_text())


def pilot_release(group, receipt, required_positions, verify):
    """Technical acceptance only, including adverse scientific results."""
    if not receipt:
        return False
    r = checked(receipt)
    if (r.get('schema') != 'radon_pilot_acceptance_v1' or r.get('group') != group or
        r.get('seed') != 3416 or r.get('test_access') is not False or
        r.get('accepted_positions') != sorted(required_positions) or
        set(r.get('gates', {})) != set(TECHNICAL_GATES)):
        raise ValueError('Incomplete or mismatched pilot acceptance')
    for name, ref in r['gates'].items():
        evidence = checked(ref)
        if evidence.get('state') != 'accepted' or evidence.get('test_access') is not False:
            raise ValueError('Pilot technical gate not accepted: '+name)
    if verify(r) is not True:
        raise ValueError('Live scientific acceptance required')
    return True


def stage_ready(plan, observations, gates, verify):
    """No vacuous completion, raw state strings, or failure-as-exclusion."""
    expected = {p['id'] for p in plan['positions']}
    if not expected or set(observations) != expected:
        return False
    for key, record in observations.items():
        state = record.get('state')
        if state not in ('accepted','infeasible_before_performance'):
            return False
        if verify(key, record) is not True:
            return False
        if state == 'infeasible_before_performance' and not record.get('preperformance_receipt'):
            return False
    for name in ('data_and_parent_audit','comparison_lock','complete_statistics','supplement_coverage'):
        if name not in gates:
            return False
        receipt = checked(gates[name])
        if receipt.get('state') != 'accepted' or receipt.get('test_access') is not False:
            return False
    return True


def track_progress(previous, observations, now):
    """Heartbeat writes alone never reset the 48-hour scientific-progress clock."""
    result = {}
    for key, row in observations.items():
        if row.get('state') not in STATES:
            raise ValueError('Unknown observation state')
        signature = stable_hash({k:row.get(k) for k in
            ('state','epoch','optimizer_steps','participant_offset','artifact_sha256','stage')})
        old = previous.get(key,{})
        changed = old.get('progress_sha256') != signature
        last = now if changed else old.get('last_progress_at', now)
        if last > now:
            raise ValueError('Progress timestamp is in the future')
        stalled = now-last >= 48*3600 and row['state'] not in ('accepted','infeasible_before_performance')
        result[key] = dict(progress_sha256=signature, last_progress_at=last,
            stalled=stalled, owner='ukb_research_maintenance',
            next_action=('inspect live step, log/history, data and resource bottleneck; '
                         'version and validate bounded repair; preserve healthy workers') if stalled else None)
    return result


def weekly_report(plan, observations, previous, now):
    expected = {p['id']:p for p in plan['positions']}
    if set(observations)-set(expected):
        raise ValueError('Observation outside locked stage')
    rows = []
    for key,p in expected.items():
        r = observations.get(key, {'state':'waiting'})
        if r.get('state') not in STATES:
            raise ValueError('Unknown state')
        rows.append(dict(position=key,group=p['group'],seed=p['seed'],arm=p['arm']['id'],
                         state=r['state']))
    accepted = {r['position'] for r in rows if r['state']=='accepted'}
    old_accepted = set(previous.get('accepted_positions', []))
    complete = []
    for group in sorted({r['group'] for r in rows}):
        for seed in SEEDS:
            batch=[r for r in rows if r['group']==group and r['seed']==seed and r['arm'] in CORE]
            if len(batch)==6 and all(r['state']=='accepted' for r in batch):
                complete.append(dict(group=group,seed=seed,
                    evidence_scope='preliminary_single_seed' if seed==3416 else 'replication_seed'))
    return dict(schema='radon_weekly_evidence_v1', updated_at=now, test_access=False,
        planned_training_positions=len(rows), accepted_training_positions=len(accepted),
        newly_accepted_positions=sorted(accepted-old_accepted),
        withdrawn_acceptances=sorted(old_accepted-accepted),
        accepted_positions=sorted(accepted), complete_core_seed_groups=complete, rows=rows,
        ranking_allowed=False, missing='Only owner-verified matched result bundles may provide metrics; '
        'positions/queues/GPU counts are not scientific findings.',
        test_release_authorized=False, stage_complete=False)


def due_events(now):
    local = datetime.fromtimestamp(now, ZoneInfo('Asia/Riyadh'))
    sunday = local.date() + timedelta(days=(6-local.weekday())%7)
    events=[]
    if local.weekday()==5:
        events.append(dict(id=f'{sunday}/saturday_snapshot', kind='snapshot'))
    if local.weekday()==6 and local.hour>=6 and (local.hour>6 or local.minute>=45):
        events.append(dict(id=f'{sunday}/sunday_draft',kind='draft'))
    if local.weekday()==6 and (local.hour>9 or local.hour==9 and local.minute>=45):
        events.append(dict(id=f'{sunday}/sunday_delivery',kind='refresh_and_deliver'))
    return events



def import_cases(queue_path, plan):
    """Explicit import of v1 case records; owner verifier remains authoritative."""
    queue=json.loads(Path(queue_path).read_text())
    if queue.get('schema')!='radon_bridge_project_work_feed_v1' or queue.get('test_access') is not False:
        raise ValueError('Unsealed project feed')
    positions={p['id']:p for p in plan['positions']}; observations={}; issues=[]
    for task in queue['tasks']:
        case_observations={}
        try:
            spec=checked(dict(path=task['spec'],sha256=task['spec_sha256']))
            if spec.get('test_access') is not False:raise ValueError('Unsealed case')
            root=Path(task['run_dir'])
            state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
            accepted=False
            if (root/'accepted.json').exists():
                from radon_bridge.studies.project_case import verify_case
                verify_case(root,spec);accepted=True
            for arm in spec['arms']:
                key=f"{spec['disease']}/{spec['model']['name']}/seed{spec['seed']}/{arm['id']}"
                if key not in positions:continue
                if arm!=positions[key]['arm']:raise ValueError('Scientific arm mismatch')
                if key in observations or key in case_observations:raise ValueError('Multiple executions need explicit reuse resolution')
                ap=root/'arms'/arm['id']/'status.json'
                a=json.loads(ap.read_text()) if ap.exists() else {}
                raw=a.get('state',state.get('state','waiting'))
                observed='accepted' if accepted else raw if raw in STATES else 'waiting'
                if observed=='accepted' and not accepted:observed='waiting'
                case_observations[key]=dict(state=observed,epoch=a.get('epoch'),
                    optimizer_steps=a.get('optimizer_steps',a.get('updates')),
                    participant_offset=a.get('participant_offset'),
                    stage=a.get('stage',state.get('stage')),
                    artifact_sha256=file_sha256(root/'accepted.json') if accepted else None)
            observations.update(case_observations)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
            issues.append(dict(task=task.get('id'),reason=str(error)))
    return observations,issues

def reconcile(config):
    """CPU observation/report producer, not another training scheduler."""
    if config.get('schema') != 'radon_semester_monitor_v1' or config.get('test_access') is not False:
        raise ValueError('Unknown or unsealed monitor')
    root=Path(config['output']); root.mkdir(parents=True,exist_ok=True)
    plan=manifest(); observations={}; issues=[]
    # Explicit observation imports have already run their project-specific verifier.
    # A report never treats an imported state string as scientific acceptance.
    for ref in config.get('observation_imports',[]):
        bundle=checked(ref)
        if bundle.get('schema')!='radon_stage_observations_v1' or bundle.get('test_access') is not False:
            raise ValueError('Observation import contract mismatch')
        for key,row in bundle['observations'].items():
            if key in observations: raise ValueError('Duplicate observation')
            if row['state'] in ('accepted','infeasible_before_performance'):
                raise ValueError('Accepted results require live owner verifier; use explicit evidence importer')
            observations[key]=row
    for feed in config.get('project_feeds',[]):
        imported, errors=import_cases(feed,plan)
        if set(imported)&set(observations):raise ValueError('Duplicate case mapping')
        observations.update(imported);issues.extend(errors)
    now=time.time()
    baseline=Path(config['previous_week_snapshot']) if config.get('previous_week_snapshot') else None
    previous=json.loads(baseline.read_text()) if baseline and baseline.exists() else {}
    progress=json.loads((root/'progress_clock.json').read_text()) if (root/'progress_clock.json').exists() else {}
    all_observations={p['id']:observations.get(p['id'],dict(state='waiting')) for p in plan['positions']}
    dependency_rows={}
    for ref in config.get('parent_catalogs',[]):
        catalog=checked(ref)
        for entry in catalog['candidates']:
            run=Path(entry['run_dir']);path=run/'status.json'
            if not path.exists():continue
            native=json.loads(path.read_text())
            key='parent/'+run.name
            raw=native.get('state','waiting')
            # Native completion is not project/scientific acceptance here.
            dependency_rows[key]=dict(state=raw if raw in STATES-{'accepted'} else 'waiting',
                epoch=native.get('epoch'),optimizer_steps=native.get('optimizer_steps',native.get('updates')),
                participant_offset=native.get('participant_offset',native.get('offset')),stage=native.get('stage'))
    clock=track_progress(progress,{**all_observations,**dependency_rows},now)
    report=weekly_report(plan,observations,previous,now)
    atomic_write_json(plan,root/'stage_manifest.json')
    atomic_write_json(clock,root/'progress_clock.json')
    atomic_write_json(report,root/'weekly_latest.json')
    atomic_write_json(dict(updated_at=now,incidents=[dict(position=k,**v) for k,v in clock.items() if v['stalled']],
        issues=issues,test_access=False),root/'incidents.json')
    for event in due_events(now):
        dest=root/'weekly'/event['id'].split('/')[0]
        dest.mkdir(parents=True,exist_ok=True)
        target=dest/(event['kind']+'.json')
        if not target.exists(): atomic_write_json(dict(event=event,report=report),target)
    with (root/'positions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['position','group','seed','arm','state'])
        writer.writeheader();writer.writerows(report['rows'])
    text=['# Radon_Bridge 周报证据入口','',
          f"计划位置 {len(plan['positions'])}；已验收 {report['accepted_training_positions']}。",
          '未绑定、未验收和技术等待均不能当性能结果。test保持封存。',
          '完整六网络729安排保留后续；当前首阶段不等待整个大矩阵。',
          '每周六快照；周日06:45英文PPTX/中文讲稿，09:45刷新、09:55前交付。',
          '完整匹配结果、反例、老师问题、48小时阻塞与下周承诺必须分开说明。',
          '当前自动生成任务附表和报告事件；幻灯片须由既有KAUST流程生成并渲染验收。']
    (root/'README.zh-CN.md').write_text('\n'.join(text)+'\n')
    report['import_issues']=issues
    report['parent_progress']=dependency_rows
    atomic_write_json(report,root/'weekly_latest.json')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=900)
    a=p.parse_args()
    if a.manifest: print(json.dumps(manifest(),indent=2));return
    if not a.config:p.error('--config or --manifest is required')
    if a.interval<60:p.error('interval must be at least 60 seconds')
    config=json.loads(Path(a.config).read_text());root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while running:
            try:
                report=reconcile(config)
                atomic_write_json(dict(state='active',updated_at=time.time(),
                    accepted=report['accepted_training_positions'],issues=report['import_issues'],
                    test_access=False),root/'monitor_status.json')
            except Exception as error:
                atomic_write_json(dict(state='needs_review',updated_at=time.time(),error=repr(error),
                    test_access=False),root/'monitor_status.json')
                logging.exception('Semester evidence reconciliation failed; workers untouched')
                if not a.watch:raise
            if not a.watch:break
            deadline=time.monotonic()+a.interval
            while running and time.monotonic()<deadline:time.sleep(min(1,max(0,deadline-time.monotonic())))

if __name__=='__main__':main()
