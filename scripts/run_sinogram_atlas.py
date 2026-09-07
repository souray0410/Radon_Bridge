"""Wait for the existing locked test report, then run a separately registered atlas."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from radonbridge.artifacts import SOURCE,sha256
from scripts.geometry_evidence import read,write
from scripts.gpu_allocation import DEFAULT_PATH
from scripts.run_unified_test import queue


def run(a):
    a.output.mkdir(parents=True,exist_ok=True,mode=0o700);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with (a.output/'controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Immutable committed source required')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();os.environ['RB_EVALUATION_COMMIT']=commit
        registration=read(a.lock/'candidate_lock.json');parent=Path(registration['after_run'])
        def status(state,**kw):write(a.output/'status.json',dict(state=state,controller_pid=os.getpid(),source_commit=commit,
            lock_directory=str(a.lock),after_run=str(parent),updated_at=time.time(),**kw))
        while True:
            if stop[0]:raise InterruptedError('Stopped while waiting')
            previous=read(parent/'status.json') if (parent/'status.json').exists() else {}
            if previous.get('state')=='complete':break
            status('waiting_for_test_and_statistics',predecessor_state=previous.get('state'),predecessor_failure=(parent/'failure.json').exists())
            time.sleep(30)
        with (SOURCE/'.active.lock').open('a') as project:
            while True:
                if stop[0]:raise InterruptedError('Stopped before project lock')
                try:fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:status('waiting_for_project_lock');time.sleep(15)
            for name,h in registration['files'].items():
                if sha256(a.lock/name)!=h:raise ValueError('Atlas lock changed')
            if sha256(Path(registration['source_test_lock'])/'candidate_lock.json')!=registration['source_test_lock_sha256']:
                raise ValueError('Parent test lock changed')
            jobs=read(a.lock/'jobs.json');a.worker_module='radonbridge.sinogram_atlas'
            preflight=[]
            for arm,r,stages in [('svd_radon',64,[3]),('qr_radon',16,[3]),('svd_resample',16,[3]),('svd_multidepth',16,[2,3,4])]:
                preflight.append(next(j for j in jobs if j['display']==dict(arm=arm,seed=3416,r=r,stages=stages)))
            status('train_preflight');queue(preflight,'preflight',a,commit,stop)
            if stop[0]:raise InterruptedError('Stopped after preflight')
            status('read_only_atlas');queue(jobs,'all',a,commit,stop)
            status('figures_and_report')
            subprocess.run([sys.executable,'scripts/report_sinogram_atlas.py','--lock',str(a.lock),'--run',str(a.output)],check=True)
            status('complete',visual_review_required=True,training_tasks=0,accepted_checkpoints=len(jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allocation',type=Path,default=DEFAULT_PATH);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException:
        write(a.output/'failure.json',dict(state='needs_attention',error=traceback.format_exc(),updated_at=time.time()));raise
