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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--devices',nargs='+',type=int,required=True)
    args=parser.parse_args();root=Path(args.root)
    lock=(root/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=json.loads((root/'queue.json').read_text());cases=q['cases'];active={};failed={};complete=set();profiles={}
    if (root/'status.json').exists():
        old=json.loads((root/'status.json').read_text())
        for worker in old.get('active',{}).values():
            try:os.kill(worker['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Previous worker still alive; do not steal its run')
        failed={k:v for k,v in old.get('failed',{}).items() if v.get('state')!='paused'}
        for key,path in old.get('resource_profiles',{}).items():
            if Path(path).exists() and json.loads(Path(path).read_text()).get('passed'):profiles[key]=path

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
    for path,digest in code['files'].items():
        if sha(path)!=digest:raise ValueError('Code changed after acceptance: '+path)
    (root/'profiles').mkdir(exist_ok=True);(root/'trials').mkdir(exist_ok=True)
    started=time.time()
    while True:
        for gpu,(proc,c,mode,output,log) in list(active.items()):
            if proc.poll() is None:continue
            log.close();del active[gpu]
            receipt=output/('accepted.json')
            if proc.returncode==0 and receipt.exists():
                if mode=='profile':
                    a=json.loads(receipt.read_text())
                    if not a.get('passed') or a.get('formal_updates')!=0 or a.get('configuration')!=json.loads(Path(c['config']).read_text()):raise ValueError('Invalid profile')
                    profiles[c['resource']]=str(receipt)
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
