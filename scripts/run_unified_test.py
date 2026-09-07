"""Durable preflight -> locked test -> diagnostics -> statistics pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from radonbridge.artifacts import SOURCE, ARCHIVE, sha256
from radonbridge.unified_evaluation import verify_lock
from scripts.geometry_evidence import read, write
from scripts.gpu_allocation import Allocation, DEFAULT_PATH, memory_snapshot
from scripts.run_integer_experiment import devices


def preflight_jobs(jobs):
    chosen=[next(j for j in jobs if j['model'].get('factory')=='PilotGraph_independent_parent_pair')]
    chosen.append(max((j for j in jobs if j['kind']=='model'),key=lambda j:len(j['model'].get('configuration',{}).get('bridges',[]))))
    for family in ('mmtm','cross_attention'):
        matches=[j for j in jobs if j['kind']=='component' and j['model']['configuration']['bridges'][0].get('family')==family]
        if not matches and family=='cross_attention':
            matches=[j for j in jobs if j['kind']=='component' and j['model']['configuration']['bridges'][0].get('family')=='attention']
        if not matches:raise ValueError('Missing host preflight family '+family)
        chosen.append(matches[0])
    for basis in ('fixed_svd_channel','fixed_random_orthogonal_channel'):
        chosen.append(max((j for j in jobs if j['kind']=='pairing' and j['model']['configuration']['bridges'][0]['compression']==basis),
                          key=lambda j:j['model']['configuration']['bridges'][0]['rho']))
    assert len(chosen)==6 and len({j['job_id'] for j in chosen})==6
    return chosen


def queue(jobs,split,args,commit,stop):
    root=args.output/split;root.mkdir(parents=True,exist_ok=True,mode=0o700)
    registration=dict(source_commit=commit,lock_sha256=sha256(args.lock/'candidate_lock.json'),split=split,job_ids=[j['job_id'] for j in jobs])
    if (root/'registration.json').exists():
        if read(root/'registration.json')!=registration:raise ValueError('Queue source or manifest changed; use new versioned attempt')
    else:write(root/'registration.json',registration)
    allocation=Allocation(args.allocation,[],required=True);active={};accepted={};pending=[];failed=[]
    ledger=read(root/'ledger.json') if (root/'ledger.json').exists() else []
    for j in jobs:
        p=root/'jobs'/j['job_id'];a=p/'acceptance.json'
        if a.exists():
            entry=read(a);s=read(entry['summary_path'])
            assert sha256(entry['summary_path'])==entry['summary_sha256'] and s['state']=='accepted' and s['source_commit']==commit
            assert s['job_id']==j['job_id'] and s['split']==split and s['candidate_lock_sha256']==registration['lock_sha256']
            for name,digest in s['prediction_files'].items():assert sha256(Path(entry['summary_path']).parent/name)==digest
            accepted[j['job_id']]=entry
        elif list(p.glob('attempt_*/failure.json')) and not args.retry_failed:failed.append(j['job_id'])
        else:pending.append(j)
    last=0;allocation_value={}
    def status(state):
        write(root/'status.json',dict(state=state,split=split,total=len(jobs),accepted=len(accepted),failed=failed,
             pending=len(pending),active=[dict(job_id=k,gpu=v['gpu'],pid=v['process'].pid) for k,v in active.items()],
             allocation=allocation_value,controller_pid=os.getpid(),source_commit=commit,updated_at=time.time()))
    try:
        while pending or active:
            info=devices();allocation_value,_=allocation.refresh(info)
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
            _,project_mem,project_pids=memory_snapshot(raw,active,info)
            for key,w in list(active.items()):
                p=w['process']
                if project_mem.get(w['gpu'],0)>10240:w['error']='Project memory exceeds 10 GiB'
                if (stop[0] or w.get('error')) and p.poll() is None and not w.get('stop_at'):
                    p.terminate();w['stop_at']=time.monotonic()
                if p.poll() is None and w.get('stop_at') and time.monotonic()-w['stop_at']>60:os.killpg(p.pid,signal.SIGKILL)
                if p.poll() is None:continue
                w['log'].close();path=w['directory']/'summary.json'
                good=p.returncode==0 and path.exists() and not w.get('error')
                if good:
                    s=read(path);good=(s['state']=='accepted' and s['job_id']==key and s['split']==split and s['source_commit']==commit
                         and s['peak_reserved_mib']<=10240 and s['parameters_BN_gradients_RNG_preserved'] and s['strict_model_load'])
                if good:
                    entry=dict(summary_path=str(path),summary_sha256=sha256(path));write(path.parent.parent/'acceptance.json',entry);accepted[key]=entry
                else:
                    failed.append(key);write(w['directory']/'failure.json',dict(error=w.get('error','Worker did not pass'),exit_code=p.returncode,
                        worker_failure=read(w['directory']/'failure.json') if (w['directory']/'failure.json').exists() else None))
                ledger.append(dict(job_id=key,gpu=w['gpu'],seconds=time.monotonic()-w['started'],accepted=good,directory=str(w['directory'])))
                write(root/'ledger.json',ledger);del active[key]
            if not failed and not stop[0]:
                occupied={w['gpu'] for w in active.values()}
                for gpu in allocation_value['selected_gpu_indices']:
                    if not pending or gpu in occupied or project_pids.get(gpu) or info[gpu]['free']<10240:continue
                    if min(shutil.disk_usage(SOURCE).free,shutil.disk_usage(ARCHIVE).free)<100*1024**3:
                        allocation_value['storage_error']='100 GiB safety margin';break
                    j=pending.pop(0);directory=root/'jobs'/j['job_id']/f'attempt_{time.time_ns()}'
                    directory.mkdir(parents=True,mode=0o700);write(directory/'record.json',j);log=(directory/'worker.log').open('w')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RB_EVALUATION_COMMIT=commit,CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    p=subprocess.Popen([sys.executable,'-m',getattr(args,'worker_module','radonbridge.unified_evaluation'),'--record',str(directory/'record.json'),
                        '--lock',str(args.lock),'--output',str(directory),'--split',split],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[j['job_id']]=dict(process=p,gpu=gpu,log=log,directory=directory,started=time.monotonic())
            if time.monotonic()-last>5:status('needs_attention_draining' if failed else 'running' if active else 'waiting_for_resources');last=time.monotonic()
            if (failed or stop[0]) and not active:break
            time.sleep(2)
        write(root/'accepted_jobs.json',accepted);complete=len(accepted)==len(jobs) and not failed
        status('complete' if complete else 'needs_attention')
        if not complete:raise RuntimeError(split+' queue incomplete; see status and preserved attempts')
        return accepted
    finally:
        for w in active.values():
            if w['process'].poll() is None:w['process'].terminate()
        for w in active.values():
            try:w['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(w['process'].pid,signal.SIGKILL);w['process'].wait()
            w['log'].close()


def run(args):
    args.output.mkdir(parents=True,exist_ok=True,mode=0o700);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with (SOURCE/'.active.lock').open('a') as lock,(args.output/'controller.lock').open('a') as own:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Use immutable committed source')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();os.environ['RB_EVALUATION_COMMIT']=commit
        candidate=verify_lock(args.lock,'validation');jobs=read(args.lock/'jobs.json')
        preflights=preflight_jobs(jobs)
        def state(phase):write(args.output/'status.json',dict(state=phase,controller_pid=os.getpid(),source_commit=commit,lock_directory=str(args.lock),updated_at=time.time()))
        state('development_preflight');accepted=queue(preflights,'validation',args,commit,stop)
        if stop[0]:raise InterruptedError('Stop requested before test')
        # All gates are checked again immediately before the first test access.
        for key in ('lineage_status','replay_status'):
            evidence=candidate[key]
            if sha256(evidence['path'])!=evidence['sha256']:raise ValueError('Acceptance evidence changed: '+key)
        for path,digest in read(args.lock/'source_manifests.json').items():
            if sha256(path)!=digest:raise ValueError('Source manifest changed after comparison registration')
        authorization=dict(test_inference_permitted=True,source_commit=commit,candidate_lock_sha256=sha256(args.lock/'candidate_lock.json'),
            preflight_summaries={v['summary_path']:v['summary_sha256'] for v in accepted.values()},
            scope=candidate['scientific_scope'],training_and_development_acceptance_complete=True,no_test_selection=True)
        auth=args.lock/'test_authorization.json'
        if auth.exists():
            if read(auth)!=authorization:raise ValueError('Test authorization source changed')
        else:write(auth,authorization)
        gate=SOURCE/'runs/2026_09_06_14_05_08/unified_test_gate.json'
        if not (args.lock/'previous_gate.json').exists():write(args.lock/'previous_gate.json',read(gate))
        write(gate,dict(state='locked_unified_test_authorized',test_prediction_permitted=True,authorization_path=str(auth),
              authorization_sha256=sha256(auth),scope=candidate['scientific_scope'],updated_at=time.time()))
        state('test_inference_and_diagnostics');queue(jobs,'test',args,commit,stop)
        state('statistics_and_report')
        subprocess.run([sys.executable,'scripts/report_unified_test.py','--lock',str(args.lock),'--run',str(args.output)],check=True)
        state('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allocation',type=Path,default=DEFAULT_PATH);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException:
        import traceback
        write(a.output/'failure.json',dict(state='needs_attention',error=traceback.format_exc(),updated_at=time.time()))
        raise
