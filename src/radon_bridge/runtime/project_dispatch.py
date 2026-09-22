"""Automatic real work dispatch for selected native replications and Radon_Bridge cases.

Shares the account submission lock and native run claims. Never cancels another
allocation, never steals a live claim, and never submits when no runnable work
exists. Each granted single-GPU allocation immediately executes useful work.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def source_binding(spec,config):
    if 'workflows/native.py' not in spec.get('trainer_source_sha256',{}):
        raise ValueError('Current package native source required')
    if spec.get('framework',{}).get('api')!='V5':
        raise ValueError('Current V5 specification required; convert historical runs explicitly')
    framework = config['framework_pythonpaths'][spec['framework']['commit']]
    if any(not (Path(framework)/'mhd_framework'/name).is_file() or
           file_sha256(Path(framework)/'mhd_framework'/name)!=digest
           for name,digest in spec['framework']['source_sha256'].items()):
        raise ValueError('Original native framework snapshot changed')
    for source in config['native_sources']:
        if all((Path(source)/name).is_file() and file_sha256(Path(source)/name)==digest
               for name,digest in spec['trainer_source_sha256'].items()):
            return dict(source=source,pythonpath=str(Path(source).parent)+':'+framework+':'+config['dependency_pythonpath'])
    raise ValueError('No immutable source matches the selected native specification')


def work(config):
    """Fresh finite feeds; a model is never selected here from its performance."""
    result=[];feed=read(config['project_feed'])
    if feed:
        if feed.get('schema')!='radon_bridge_project_work_feed_v1' or feed.get('test_access') is not False:raise ValueError('Unsealed project feed')
        for task in feed['tasks']:
            kind=task.get('execution','radon')
            if kind not in ('radon','radon_unit'):raise ValueError('Unknown project execution kind')
            result.append(dict(task,execution=kind))
    native_paths=[config['native_feed']]+config.get('additional_native_feeds',[])
    for native_path in native_paths:
        native=read(native_path)
        if not native:continue
        if native.get('schema')!='radon_bridge_native_work_feed_v1' or native.get('test_access') is not False:raise ValueError('Unsealed replication feed')
        for item in native['queues']:
            if file_sha256(Path(item['queue']))!=item['queue_sha256']:raise ValueError('Replication queue changed')
            for task in read(item['queue'])['tasks']:
                result.append(dict(task,execution='native'))
    unique={}
    for task in result:
        if file_sha256(Path(task['spec']))!=task['spec_sha256']:raise ValueError('Work specification changed')
        run=str(Path(task['run_dir']).resolve())
        if run in unique:
            previous=unique[run]
            if any(previous[k]!=task[k] for k in ('spec_sha256','execution')):
                raise ValueError('Conflicting duplicate work')
            continue  # Same immutable run may be referenced by several studies.
        unique[run]=task
    result=list(unique.values())
    if config.get('weekly_delivery_policy'):
        from radon_bridge.runtime.weekly_delivery import filter_tasks
        result,_=filter_tasks(result,config['weekly_delivery_policy'])
    return sorted(result,key=lambda t:(t['execution']=='native',t.get('priority',0)))


def eligible(task,claims,*,reservation_token=None):
    state=read(claims.path(task['run_dir']));status=read(Path(task['run_dir'])/'status.json')
    if reservation_token is not None:
        if state.get('state')!='claimed' or any(state.get(k)!=reservation_token.get(k)
                for k in ('owner','generation','spec_sha256','job_id')):
            raise RuntimeError('Priority reservation ownership changed')
    elif state.get('state') in ('claimed','running','failed','completed','liveness_needs_review'):return False
    if status.get('state') in ('needs_review','needs_review_epoch_cap','failed','completed'):return False
    if (Path(task['run_dir'])/'accepted.json').exists():
        spec=read(task['spec'])
        if task['execution']=='native':
            from mhd_models.runtime.training_state import verify_completion
            verify_completion(task['run_dir'],spec)
        elif task['execution']=='radon_unit':
            from radon_bridge.studies.project_units import verify_unit
            verify_unit(task['run_dir'],spec)
        else:
            from radon_bridge.studies.project_case import verify_case
            verify_case(task['run_dir'],spec)
        return False
    if task.get('execution')=='radon_unit':
        from radon_bridge.studies.project_units import unit_ready
        try:return unit_ready(task)
        except (OSError,ValueError,KeyError,TypeError) as error:
            atomic_write_json(dict(state='needs_review',error=repr(error),spec_sha256=task['spec_sha256'],
                test_access=False),Path(task['run_dir'])/'admission_error.json')
            return False
    return True


def resource_wait(task,config):
    """Check recorded whole-lifecycle peaks before spending a worker probe."""
    if task.get('execution')!='native':return None
    canonical=Path(task['run_dir'])/'resource_qualification/full_reference.json'
    references=[read(canonical),config.get('native_profile_references',{}).get(task['spec_sha256'])]
    peaks=[]
    for reference in references:
        if not reference:continue
        path=Path(reference['path'])
        if file_sha256(path)!=reference['sha256']:
            raise ValueError('Historical resource receipt changed')
        row=read(path)
        import math
        peak=row.get('peak_step_memory_gib')
        if (row.get('spec_sha256')!=task['spec_sha256'] or row.get('status')!='accepted'
                or type(peak) not in (int,float) or not math.isfinite(peak) or peak<=0):
            raise ValueError('Invalid historical resource identity or peak')
        peaks.append(peak)
    if not peaks:return None  # Unknown resources still require the full bounded probe.
    peak=max(peaks);worker=worker_memory_gib(config)
    allocation=config.get('allocation_memory_gib',128)
    other=config.get('other_reserved_memory_gib',2)
    if (type(allocation) not in (int,float) or type(other) not in (int,float)
            or not math.isfinite(allocation) or not math.isfinite(other)
            or allocation<=0 or other<0):raise ValueError('Unknown allocation memory envelope')
    if peak>worker or other+peak>.85*allocation:
        return dict(state='waiting_resource_configuration',peak_step_memory_gib=peak,
                    worker_memory_gib=worker,allocation_memory_gib=allocation,
                    other_reserved_memory_gib=other,test_access=False)
    return None


def admissible_work(config,claims,reservation=None):
    candidates=[];waiting=[]
    for task in work(config):
        reserved=reservation is not None and task['run_dir']==reservation[0]['run_dir']
        if reserved:
            if not eligible(task,claims,reservation_token=reservation[1]):continue
        elif not eligible(task,claims):continue
        try:hold=resource_wait(task,config)
        except (OSError,ValueError,KeyError) as error:
            hold=dict(state='resource_evidence_needs_review',error=repr(error),test_access=False)
        if hold:
            waiting.append(dict(run=task['run_dir'],**hold));continue
        candidates.append(task)
    if config.get('output'):
        atomic_write_json(dict(ready=len(candidates),waiting=waiting,updated_at=time.time(),
                               test_access=False),Path(config['output'])/'resource_admission.json')
    return candidates


def reserve_priority(qualified, claims, owner, job, native):
    """Claim before pausing; a failed handover leaves native work untouched."""
    from mhd_models.scheduling.project_priority import request_pause,sha
    target,profile,identity,root=qualified
    token=claims.acquire(target['run_dir'],target['spec_sha256'],owner,job)
    receipt=None
    try:
        receipt=request_pause(target,native['run_dir'],profile,sha(profile),identity,root)
        claims.update(target['run_dir'],owner,priority_handover=receipt,
                      source_native_run=native['run_dir'])
    except Exception:
        claims.release(target['run_dir'],owner,'paused',step_dead=True)
        pause=Path(native['run_dir'])/'pause.json'
        if receipt is not None and read(pause)==receipt:
            pause.unlink(missing_ok=True)
        raise
    return target,token,native


def allocation_command(config_path,name,python):
    return ['salloc','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=16',
        '--mem=128G','--gres=gpu:a100:1','--constraint=gpu_a100','--time=48:00:00',
        '--job-name='+name,python,'-m','radon_bridge.runtime.project_dispatch','--config',str(config_path),'--allocation-owner']



def failed_before_submission(record):
    return (record.get('state')=='owner_exited' and not record.get('job_id') and
        Path(record.get('log','/nonexistent')).is_file() and
        Path(record['log']).read_text().strip()=='salloc: error: no controlling terminal: please set --no-shell')


def reconcile_expired(config):
    from mhd_models.scheduling.policy import Claims
    from mhd_models.scheduling.quota_guard import snapshot
    claims=Claims(config['claims']);snap=snapshot();events=[]
    for task in work(config):
        run=Path(task['run_dir']);record=read(claims.path(run))
        if not str(record.get('owner','')).startswith('radon-workflow-'):continue
        if record.get('state') not in ('running','claimed','liveness_needs_review'):continue
        job=str(record['job_id'])
        if job in snap['jobs']:continue
        # Accounting and the control plane must agree; age alone grants nothing.
        text=subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State'],text=True,timeout=20)
        rows=[r.split('|') for r in text.splitlines() if r.split('|')[0]==job]
        if len(rows)!=1:continue
        state=rows[0][1].split()[0]
        if state not in ('TIMEOUT','PREEMPTED','NODE_FAIL'):continue
        if task['execution']=='radon_unit':
            from radon_bridge.studies.project_units import load_unit
            unit=read(task['spec']);_,case_root=load_unit(unit);name=unit['unit']
            if name.startswith('arm_'):checkpoint=case_root/'arms'/name[4:]/'last.pt'
            elif name=='prepare':checkpoint=case_root/'bases/last.pt'
            else:checkpoint=case_root/'prepared.json'
        else:checkpoint=run/'last.pt' if task['execution']=='native' else next(iter(sorted(run.glob('arms/*/last.pt'))),run/'bases/last.pt')
        if not checkpoint.is_file():continue
        claims.release(run,record['owner'],'paused',step_dead=True)
        event=dict(run=str(run),allocation=job,terminal_state=state,checkpoint_sha256=file_sha256(checkpoint),
                   next_step='full_resource_resume_preflight_before_formal_continuation')
        events.append(event)
    if events:atomic_write_json(dict(events=events,time=time.time()),Path(config['output'])/('recovery_'+str(time.time_ns())+'.json'))
    return events

def submit_one(config,path,journal):
    from mhd_models.scheduling.quota_guard import snapshot
    from mhd_models.scheduling.renewal import job_from_log
    from mhd_models.scheduling.policy import Claims
    claims=Claims(config['claims'])
    candidates=admissible_work(config,claims)
    if not candidates:return 'waiting_dependencies'
    lock=Path(config['account_submission_lock'])
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return 'waiting_submission_lock'
        snap=snapshot()
        # Include every existing allocation and durable unconfirmed request.
        for entry in journal['requests']:
            if not entry.get('job_id'):return 'submission_identity_needs_review'
        for pattern in config['submission_journal_globs']:
            import glob
            for prior in glob.glob(pattern,recursive=True):
                value=read(prior)
                if not isinstance(value.get('requests'),list):raise ValueError('Unknown prior journal')
                if any(not str(r.get('job_id','')).isdigit() and not failed_before_submission(r) for r in value['requests']):
                    return 'prior_submission_identity_needs_review'
        active=sum(str(e['job_id']) in snap['jobs'] for e in journal['requests'])
        from mhd_models.scheduling.project_priority import project_limit
        maximum=project_limit(config,config['maximum_workflow_allocations'],'Radon_Bridge')
        if active>=maximum:return 'workflow_allocations_active'
        if snap['total_gpus']>=snap['limit']:return 'waiting_account_capacity'
        name='radon_bridge_auto_'+str(time.time_ns());log=Path(config['output'])/(name+'.log')
        entry=dict(name=name,state='intent',log=str(log),time=time.time())
        journal['requests'].append(entry);atomic_write_json(journal,Path(config['output'])/'requests.json')
        command=allocation_command(path,name,config['python']);entry['command']=command
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        entry.update(pid=child.pid,host=socket.gethostname(),state='submitted_waiting_identity')
        atomic_write_json(journal,Path(config['output'])/'requests.json')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=job_from_log(log.read_text())
            if job:
                entry.update(job_id=job,state='submitted');break
            if child.poll() is not None:
                entry.update(state='identity_needs_review',returncode=child.returncode);break
            time.sleep(.5)
        atomic_write_json(journal,Path(config['output'])/'requests.json')
    return entry['state']


def daemon(config_path):
    config=read(config_path);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if not 1<=config['maximum_workflow_allocations']<=2:raise ValueError('Workflow allocation limit')
    with open('/dev/tty','rb'):pass
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (out/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=read(out/'requests.json',dict(schema='radon_bridge_workflow_requests_v1',requests=[]))
        while not (out/'stop.json').exists():
            try:
                if config.get('session_guard_receipts'):
                    from mhd_models.scheduling.project_priority import verify_session_guards
                    verify_session_guards(config)
                reconcile_expired(config)
                state=submit_one(config,config_path,journal);error=None
            except Exception as exc:state='needs_review';error=repr(exc)
            atomic_write_json(dict(state=state,error=error,updated_at=time.time(),pid=os.getpid(),test_access=False),out/'status.json')
            time.sleep(60)


def allocation_owner(config_path):
    config=read(config_path);job=os.environ['SLURM_JOB_ID']
    text=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20)
    fields=dict(x.split('=',1) for x in text.split() if '=' in x)
    end=datetime.fromisoformat(fields['EndTime']).timestamp()
    environment=os.environ.copy();environment['RADON_ALLOCATION_END']=str(end)
    subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--gpus=1',
        '--cpus-per-task=1','--mem=2G','--unbuffered',config['python'],'-m','radon_bridge.runtime.project_dispatch',
        '--config',str(config_path),'--gpu-owner'],env=environment,check=True)


def worker_memory_gib(config):
    """Operational step allowance; never changes model microbatch or precision."""
    memory=config.get('worker_memory_gib',100)
    # A128GiB allocation retains15% including the2GiB allocation owner.
    if type(memory) is not int or not 40<=memory<=106:
        raise ValueError('Worker RAM must fit the128GiB allocation reserve')
    return memory


def gpu_owner(config_path):
    import torch
    from mhd_models.scheduling.policy import Claims
    from mhd_models.scheduling.slurm_liveness import step_presence
    config=read(config_path);job=os.environ['SLURM_JOB_ID'];end=float(os.environ['RADON_ALLOCATION_END'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<78*1024**3:raise ValueError('A10080 single-device binding required')
    root=Path(config['output'])/job;root.mkdir(exist_ok=True);claims=Claims(config['claims']);owner='radon-workflow-'+job
    reservation=None;resume_native=None
    while time.time()<end-1800:
        candidates=admissible_work(config,claims,reservation=reservation)
        if reservation and not any(t['run_dir']==reservation[0]['run_dir'] for t in candidates):
            claims.release(reservation[0]['run_dir'],owner,'paused',step_dead=True)
            resume_native=reservation[2];reservation=None
        if not candidates:break
        fallback=next((t for t in candidates if resume_native and t['run_dir']==resume_native['run_dir']),None)
        task=reservation[0] if reservation else fallback or candidates[0]
        run=Path(task['run_dir']);spec=read(task['spec'])
        try:
            if reservation:
                token=reservation[1];resume_native=reservation[2];reservation=None
            else:
                token=claims.acquire(run,task['spec_sha256'],owner,job)
                if fallback:resume_native=None
        except RuntimeError:continue
        attempt=root/(run.name+'_'+str(token['generation']));attempt.mkdir()
        record=attempt/'step.json';environment=os.environ.copy()
        command=['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--gpus=1',
            '--cpus-per-task=14','--mem='+str(worker_memory_gib(config))+'G','--unbuffered',config['python'],'-m','radon_bridge.runtime.project_dispatch',
            '--config',str(config_path),'--execute',task['spec'],'--run',str(run),
            '--kind',task['execution'],'--record',str(record)]
        (run/'pause.json').unlink(missing_ok=True)
        try:
            with (attempt/'worker.log').open('x') as log:
                child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        except OSError as error:
            claims.release(run,owner,'failed',step_dead=True)
            atomic_write_json(dict(reason='worker_launch_failed',run=str(run),error=repr(error),
                time=time.time()),attempt/'failure.json')
            continue
        step=None
        priority=None
        if config.get('project_priority_enabled'):
            from mhd_models.scheduling.project_priority import PriorityProbe
            priority=PriorityProbe(config_path,config,job,attempt,task,'radon_bridge')
        while child.poll() is None:
            r=read(record)
            if r.get('step') and step is None:
                step=r['step'];claims.update(run,owner,state='running',step=step)
            if priority is not None:
                try:
                    qualified=priority.poll([t for t in admissible_work(config,claims) if t['execution']!='native'])
                    if qualified is not None and reservation is None:
                        reservation=reserve_priority(qualified,claims,owner,job,task)
                except Exception as exc:atomic_write_json(dict(state='priority_deferred',error=repr(exc)),attempt/'priority_error.json')
            if time.time()>end-900:
                atomic_write_json(dict(reason='allocation_expiry_checkpoint'),run/'pause.json')
            atomic_write_json(dict(state='running',task=task['id'],run=str(run),step=step,updated_at=time.time()),root/'status.json')
            time.sleep(10)
        if priority is not None:priority.finish()
        # A terminated client is not proof that its remote step is dead.
        step=step or read(record).get('step')
        if step is None or step_presence(job,step) is not False:
            claims.update(run,owner,state='liveness_needs_review');raise RuntimeError('Cannot prove worker step exit')
        if child.returncode not in (0,75):
            claims.release(run,owner,'failed',step_dead=True)
            atomic_write_json(dict(reason='worker_failed',run=str(run),job=job,step=step,
                returncode=child.returncode,log=str(attempt/'worker.log'),time=time.time()),attempt/'failure.json')
            if reservation:
                claims.release(reservation[0]['run_dir'],owner,'paused',step_dead=True)
                reservation=None
            continue
        state=read(run/'status.json').get('state')
        if state=='completed':
            if task['execution']=='native':
                from mhd_models.runtime.training_state import verify_completion
                verify_completion(run,spec)
            elif task['execution']=='radon_unit':
                from radon_bridge.studies.project_units import verify_unit
                verify_unit(run,spec)
            else:
                from radon_bridge.studies.project_case import verify_case
                verify_case(run,spec)
            claims.release(run,owner,'completed',step_dead=True)
        elif state=='paused':claims.release(run,owner,'paused',step_dead=True)
        else:claims.release(run,owner,'failed',step_dead=True)
    if reservation:claims.release(reservation[0]['run_dir'],owner,'paused',step_dead=True)
    atomic_write_json(dict(state='owner_finished',updated_at=time.time()),root/'status.json')


def execute_work(config_path,spec_path,run,kind,record):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    config=read(config_path);spec=read(spec_path);run=Path(run)
    step=dict(job_id=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],pid=os.getpid(),run=str(run),time=time.time())
    atomic_write_json(step,Path(record))
    profile_root=Path(record).parent/'profile'
    environment=os.environ.copy()
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if kind=='radon' and config.get('legacy_project_pythonpath'):
        environment['PYTHONPATH']=config['legacy_project_pythonpath']
        os.execvpe(config['python'],[config['python'],'-m','radon_bridge.runtime.project_dispatch','--config',str(config_path),
            '--execute',str(spec_path),'--run',str(run),'--kind','radon','--record',str(record)],environment)
    if kind=='native':
        binding=source_binding(spec,config)
        environment['PYTHONPATH']=binding['pythonpath']
        from radon_bridge.runtime.native_profile_reuse import prior,qualify,live_envelope,hardware_identity,matching_reference
        spec_sha=file_sha256(Path(spec_path));hardware=hardware_identity(torch)
        canonical=run/'resource_qualification/full_reference.json'
        reference=read(canonical) or config.get('native_profile_references',{}).get(spec_sha)
        reuse=matching_reference(reference,spec_sha,hardware)
        if reuse:live_envelope(torch,step['job_id'],step['step'])
        command=[config['python'],str(Path(config['native_profile_source'])/'scheduling/profile_native.py'),
            '--source',binding['source'],'--spec',str(spec_path),'--output',str(profile_root),
            '--fresh-train-batches']
        if not reuse:command.extend(['--full-development','--full-train-read'])
        checkpoint=run/'last.pt'
        checkpoint_sha=file_sha256(checkpoint) if checkpoint.exists() else None
        if checkpoint.exists(): command.extend(['--checkpoint',str(checkpoint)])
        subprocess.run(command,env=environment,check=True)
        from mhd_models.scheduling.prepared_owner import complete_profile
        receipt=read(profile_root/'accepted.json')
        if reuse:
            if (file_sha256(checkpoint) if checkpoint.exists() else None)!=checkpoint_sha:
                raise ValueError('Formal checkpoint changed during qualification')
            qualified=qualify(reuse,profile_root/'accepted.json',spec_sha,hardware,checkpoint_sha,
                output=profile_root/'requalification.json',**live_envelope(torch,step['job_id'],step['step']))
            receipt=dict(receipt,peak_gpu_gib=qualified['peak_gpu_gib'])
        else:
            if not complete_profile(receipt):raise ValueError('Native full resource profile rejected')
            reference=dict(path=str(profile_root/'accepted.json'),sha256=file_sha256(profile_root/'accepted.json'))
            prior(reference,spec_sha,hardware)
            canonical.parent.mkdir(parents=True,exist_ok=True)
            atomic_write_json(reference,canonical)
        # A single exclusive workflow worker; the remaining allocation RAM is reserved.
        if receipt.get('peak_gpu_gib',float('inf'))>torch.cuda.get_device_properties(0).total_memory/1024**3:raise ValueError('Native GPU reserve failed')
        command=[config['python'],'-m','mhd_models.workflows.native','--spec',str(spec_path),'--output',str(run),'--mode','train']
        os.execvpe(config['python'],command,environment)
    elif kind=='radon_unit':
        from radon_bridge.studies.project_units import load_unit,execute
        case,case_root=load_unit(spec)
        # Preparation validates the full real pair before publishing shared bases.
        # Training arms retain their own full update/recovery resource profile.
        if spec['unit']=='prepare':
            subprocess.run([config['python'],'-m','radon_bridge.runtime.project_dispatch','--config',str(config_path),
                '--profile-project',spec['case_spec'],'--run',str(profile_root)],env=environment,check=True)
            receipt=read(profile_root/'accepted.json')
            if receipt.get('status')!='accepted' or receipt.get('case_identity')!=stable_hash(case):
                raise ValueError('Project unit resource receipt mismatch')
        total=torch.cuda.get_device_properties(0).total_memory
        from mhd_models.scheduling.gpu_budget import configure_allocator
        configure_allocator()
        execute(spec,run)
    else:
        # Profiling runs in a subprocess so its optimizer, CUDA cache and limits
        # cannot leak into the formal model's RNG or memory accounting.
        subprocess.run([config['python'],'-m','radon_bridge.runtime.project_dispatch','--config',str(config_path),
            '--profile-project',str(spec_path),'--run',str(profile_root)],env=environment,check=True)
        receipt=read(profile_root/'accepted.json')
        if receipt.get('status')!='accepted' or receipt.get('case_identity')!=stable_hash(spec):raise ValueError('Project resource receipt mismatch')
        from radon_bridge.studies.project_case import execute
        total=torch.cuda.get_device_properties(0).total_memory
        from mhd_models.scheduling.gpu_budget import configure_allocator
        configure_allocator()
        execute(spec,run)


def standby(config_path):
    config=read(config_path);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    work(config)
    for row in config['source_pins']:
        if file_sha256(Path(row['path']))!=row['sha256']:raise ValueError('Dispatcher snapshot changed')
    gate=Path(config.get('handover_gate_directory',str(out)));gate.mkdir(parents=True,exist_ok=True)
    atomic_write_json(dict(state='ready',pid=os.getpid(),config=str(config_path),time=time.time()),gate/'handover_ready.json')
    while not (gate/'handover_armed.json').exists():time.sleep(2)
    daemon(config_path)


def configure_control_imports(config):
    if config.get('legacy_project_pythonpath'):
        raise ValueError('Historical execution settings require independent migration')
    if config.get('control_source'):
        import mhd_models.scheduling as scheduling
        expected=Path(config['control_source']).resolve()/'scheduling'
        if Path(scheduling.__file__).resolve().parent!=expected:
            raise ValueError('Installed current scheduling package differs from control source')
    for row in config.get('control_source_pins',[]):
        if file_sha256(Path(row['path']))!=row['sha256']:
            raise ValueError('Management snapshot changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--standby',action='store_true');p.add_argument('--allocation-owner',action='store_true');p.add_argument('--gpu-owner',action='store_true')
    p.add_argument('--execute');p.add_argument('--kind',choices=['native','radon','radon_unit']);p.add_argument('--record');p.add_argument('--run');p.add_argument('--profile-project')
    a=p.parse_args()
    configure_control_imports(read(a.config))
    if a.standby:standby(a.config)
    elif a.allocation_owner:allocation_owner(a.config)
    elif a.gpu_owner:gpu_owner(a.config)
    elif a.execute:execute_work(a.config,a.execute,a.run,a.kind,a.record)
    elif a.profile_project:
        import torch
        from radon_bridge.studies.project_profile import profile
        profile(read(a.profile_project),a.run,torch.device('cuda:0'))
    else:daemon(a.config)

if __name__=='__main__':main()
