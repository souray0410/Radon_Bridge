"""Finite workstation execution owner; no Slurm submissions or test access."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from radon_bridge.studies.cohort_case import sha, write_json


def verified(path,config):
    if not path.exists():return False
    a=json.loads(path.read_text())
    if a.get('configuration')!=config or a.get('test_used') is not False:raise ValueError('Acceptance mismatch')
    if not a.get('converged_by_policy'):raise ValueError('No plateau acceptance')
    for name,digest in a['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Artifact changed: '+name)
    return True



def profile_verified(path,config,grouped=False):
    path=Path(path)
    if not path.exists():return False
    value=json.loads(path.read_text())
    if (value.get('passed') is not True or value.get('test_used') is not False or value.get('formal_updates')!=0
            or value.get('configuration')!=config or value.get('checkpoint_update_exact') is not True
            or value.get('node_ids_preserved') is not True or value.get('autograd_equivalence') is not True
            or value.get('full_development_participants')!=296):
        raise ValueError('Invalid resource profile')
    if sha(path.parent/'resume.pt')!=value.get('resume_sha256'):raise ValueError('Resource profile resume changed')
    if grouped:
        from radon_bridge.studies.cohort_grouped import validate_grouped_profile_structure
        validate_grouped_profile_structure(value.get('grouped_structure'),config)
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--devices',nargs='+',type=int,required=True)
    args=parser.parse_args();root=Path(args.root)
    lock=(root/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=json.loads((root/'queue.json').read_text())
    if q.get('test_used') is not False:raise ValueError('Unsealed cohort queue')
    if q.get('study_kind')=='grouped_linear':
        from radon_bridge.studies.cohort_grouped import validate_queue
        validate_queue(root,q)
    elif q.get('schema')!='radon_small_cohort_core_v1':raise ValueError('Unknown cohort queue contract')
    cases=q['cases'];active={};failed={};complete=set();profiles={}
    if (root/'status.json').exists():
        old=json.loads((root/'status.json').read_text())
        for worker in old.get('active',{}).values():
            try:os.kill(worker['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Previous worker still alive; do not steal its run')
        failed={k:v for k,v in old.get('failed',{}).items() if v.get('state')!='paused'}
        by_resource={c['resource']:c for c in cases}
        for key,path in old.get('resource_profiles',{}).items():
            if key not in by_resource:raise ValueError('Unknown recovered resource profile')
            cfg=json.loads(Path(by_resource[key]['config']).read_text())
            if profile_verified(path,cfg,q.get('study_kind')=='grouped_linear'):profiles[key]=path

    # These full receipts are generated before execution, not inferred from directory names.
    for name in ['dependencies_acceptance.json','code_acceptance.json']:
        if not (root/name).exists():raise ValueError('Missing preparation receipt '+name)
    dependencies=json.loads((root/'dependencies_acceptance.json').read_text())
    if dependencies.get('passed') is not True or dependencies.get('test_used') is not False or dependencies['queue_sha256']!=sha(root/'queue.json'):
        raise ValueError('Dependencies or registered queue changed')
    for section in ('config_files','references','data_files'):
        for path,digest in dependencies[section].items():
            if sha(path)!=digest:raise ValueError('Dependency changed: '+path)
    code=json.loads((root/'code_acceptance.json').read_text())
    if q.get('study_kind')=='grouped_linear' and (code.get('source_commit')!=q['source_commit'] or code.get('framework_commit')!=q['framework_commit'] or code.get('passed_cpu') is not True or code.get('test_used') is not False):raise ValueError('Grouped code acceptance mismatch')
    for path,digest in code['files'].items():
        if sha(path)!=digest:raise ValueError('Code changed after acceptance: '+path)
    if q.get('study_kind')=='grouped_linear':
        from radon_bridge.studies.cohort_grouped import verify_deployment_receipts
        verify_deployment_receipts(root,q)
    (root/'profiles').mkdir(exist_ok=True);(root/'trials').mkdir(exist_ok=True)
    started=time.time()
    while True:
        for gpu,(proc,c,mode,output,log) in list(active.items()):
            if proc.poll() is None:continue
            log.close();del active[gpu]
            receipt=output/('accepted.json')
            if proc.returncode==0 and receipt.exists():
                if mode=='profile':
                    cfg=json.loads(Path(c['config']).read_text())
                    if profile_verified(receipt,cfg,q.get('study_kind')=='grouped_linear'):profiles[c['resource']]=str(receipt)
                elif verified(receipt,json.loads(Path(c['config']).read_text())):complete.add(c['name'])
            else:
                failed[c['name']]=dict(exit_code=proc.returncode,mode=mode,path=str(output),state='paused' if proc.returncode==75 else 'needs_review')
                if mode=='profile':
                    for other in cases:
                        if other['resource']==c['resource']:failed.setdefault(other['name'],dict(state='resource_class_quarantined',cause=c['name']))
        inflight={c['name'] for _,c,_,_,_ in active.values()}
        for gpu in args.devices:
            if gpu in active:continue
            if shutil.disk_usage(root).free < 102*1024**3:continue
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>1024:continue
            c=next((x for x in cases if x['name'] not in inflight|complete|failed.keys()),None)
            if c is None:continue
            cfg=json.loads(Path(c['config']).read_text());dest=root/'trials'/c['name']
            if verified(dest/'accepted.json',cfg):complete.add(c['name']);continue
            mode='train' if c['resource'] in profiles else 'profile'
            if mode=='profile' and any(x['resource']==c['resource'] and m=='profile' for _,x,m,_,_ in active.values()):continue
            output=dest if mode=='train' else root/'profiles'/(c['name']+'_'+str(time.time_ns()))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(cfg['seed']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            logs=root/'logs';logs.mkdir(exist_ok=True);log=(logs/(c['name']+'_'+mode+'_'+str(time.time_ns())+'.log')).open('w')
            cmd=[sys.executable,'-m','radon_bridge.studies.cohort_case',mode,'--config',c['config'],'--data',q['data'],'--output',str(output)]
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(proc,c,mode,output,log);inflight.add(c['name'])
        state='running' if active else 'needs_review' if failed else 'complete' if len(complete)==len(cases) else 'waiting_resources'
        write_json(root/'status.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),started_at=started,
            planned=len(cases),accepted=len(complete),failed=failed,resource_profiles=profiles,
            active={str(k):dict(pid=p.pid,name=c['name'],phase=m,output=str(o)) for k,(p,c,m,o,l) in active.items()},test_used=False))
        from radon_bridge.analysis.cohort_report import report
        try:
            report(root)
        except Exception as exc:
            write_json(root/'publication_failure.json',dict(updated_at=time.time(),error=repr(exc),state='needs_review',training_preserved=True))
        else:
            (root/'publication_failure.json').unlink(missing_ok=True)
        if not active and len(complete)+len(failed)==len(cases):break
        time.sleep(10)
    if len(complete)==len(cases):
        from radon_bridge.analysis.cohort_report import report
        report(root)


if __name__=='__main__':main()
