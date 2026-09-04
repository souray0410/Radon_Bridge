"""Bounded two-GPU staged experiment controller with persistent accounting."""
import argparse
import atexit
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from radonbridge.experiment import write_json, bridge_configs


def read_json(path): return json.loads(Path(path).read_text())


def devices():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(row.split(',')[0]):{'uuid':row.split(',')[1].strip(),'free':int(row.split(',')[2])} for row in raw.splitlines()}


class Controller:
    def __init__(self,args):
        self.args=args; self.root=Path(args.output); self.root.mkdir(parents=True,exist_ok=True)
        self.protocol=read_json(args.protocol)
        if self.protocol['review_status']!='approved' or self.protocol['schema']!='integer_eem_direct_bp_v1':
            raise ValueError('Only the explicitly approved integer protocol may launch')
        self.limit=min(float(self.protocol['max_gpu_minutes']),240-float(self.protocol['prior_gpu_minutes']))
        self.ledger=read_json(self.root/'ledger.json') if (self.root/'ledger.json').exists() else {'jobs':[]}
        self.active={}; self.peak={}; self.phase=args.phase; self.stop=False; self.stop_reason=None
        self.commit=None
        if args.phase=='run':
            if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Source must be committed before training')
            self.commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            if (self.root/'run_source.json').exists(): raise RuntimeError('Run already started; inspect before any continuation')
            weights=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/resnet18-f37072fd.pth'
            audit=read_json(Path(args.data)/'audit.json')
            write_json(self.root/'run_source.json',{'commit':self.commit,'protocol_sha256':hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
                'weight_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'data_audit':audit,
                'validation_role':'development previously used for task selection','test_used':False})
        def handle(*_): self.stop=True; self.stop_reason='controller_signal'
        signal.signal(signal.SIGTERM,handle); signal.signal(signal.SIGINT,handle)
        atexit.register(self.shutdown)
        self.status('running')

    def shutdown(self):
        for v in self.active.values():
            if v['process'].poll() is None: v['process'].terminate()
        deadline=time.monotonic()+20
        for key,v in list(self.active.items()):
            try: v['process'].wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired: v['process'].kill();v['process'].wait()
            v['log'].close()
            self.ledger['jobs'].append({'id':key,'gpu':v['gpu'],'gpu_seconds':time.monotonic()-v['start'],
                'exit_code':v['process'].returncode,'sampled_peak_process_mib':self.peak.get(key,0),'controller_cleanup':True})
            del self.active[key]
        write_json(self.root/'ledger.json',self.ledger)

    def used(self):
        return sum(x['gpu_seconds'] for x in self.ledger['jobs'])/60+sum(time.monotonic()-x['start'] for x in self.active.values())/60

    def status(self,state,**kw):
        write_json(self.root/'status.json',{'state':state,'phase':self.phase,'controller_pid':os.getpid(),
            'source_commit':self.commit,'new_gpu_minutes':self.used(),'prior_gpu_minutes':self.protocol['prior_gpu_minutes'],
            'cumulative_gpu_minutes':self.protocol['prior_gpu_minutes']+self.used(),'budget_gpu_minutes':self.limit,
            'active':[{'id':k,'pid':v['process'].pid,'gpu':v['gpu']} for k,v in self.active.items()],
            'finished_jobs':len(self.ledger['jobs']),'updated_at':time.time(),**kw})

    def start(self,job,gpu):
        path=self.root/job['id']
        if path.exists():raise RuntimeError(f'Refuse to overwrite {path}')
        path.mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if job.get('geometry'):
            cmd=[sys.executable,'tests/check_integer_bridge.py','--device','cuda','--output',str(path/'summary.json')]
        else:
            write_json(path/'configuration.json',job['config']|{'source_commit':self.commit})
            cmd=[sys.executable,'-m','radonbridge.experiment','--config',str(path/'configuration.json'),
                 '--output',str(path),'--data',self.args.data]
        log=(path/'worker.log').open('w')
        tick=time.monotonic(); process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        self.active[job['id']]={'process':process,'gpu':gpu,'start':tick,'log':log,'stop_at':None}
        self.status('running')

    def run_jobs(self,jobs,allow_oom=False):
        queue=list(jobs); results={}; failed=False
        while queue or self.active:
            if not self.stop and self.used()+(20*len(self.active)+2)/60>=self.limit:
                self.stop=True; self.stop_reason='budget'
            if self.stop:
                queue.clear()
                for active in self.active.values():
                    if active['stop_at'] is None:
                        active['process'].terminate();active['stop_at']=time.monotonic()
                    elif time.monotonic()-active['stop_at']>20:active['process'].kill()
            elif not failed:
                info=devices(); occupied={v['gpu'] for v in self.active.values()}
                for gpu in [0,1]:
                    if queue and gpu in info and gpu not in occupied and info[gpu]['free']>=10240:
                        self.start(queue.pop(0),gpu)
            # Own workers are the only CUDA processes started by this locked controller.
            usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in usage.splitlines() if r.strip()}
            per_gpu={}
            for key,v in self.active.items():
                used=memory.get(v['process'].pid,0);self.peak[key]=max(self.peak.get(key,0),used)
                per_gpu[v['gpu']]=per_gpu.get(v['gpu'],0)+used
            if any(v>10240 for v in per_gpu.values()):
                self.stop=True; self.stop_reason='memory_limit'
                self.status('stopping',reason='Project GPU memory exceeds 10 GiB')
            for key,v in list(self.active.items()):
                code=v['process'].poll()
                if code is None:continue
                elapsed=time.monotonic()-v['start'];v['log'].close()
                self.ledger['jobs'].append({'id':key,'gpu':v['gpu'],'gpu_seconds':elapsed,'exit_code':code,
                                            'sampled_peak_process_mib':self.peak.get(key,0)})
                del self.active[key];write_json(self.root/'ledger.json',self.ledger)
                directory=self.root/key
                if code==0:
                    results[key]=read_json(directory/'summary.json')
                else:
                    error=read_json(directory/'failure.json') if (directory/'failure.json').exists() else {'state':'failed'}
                    if allow_oom and error['state']=='oom':results[key]=error
                    elif self.stop_reason=='budget':pass
                    else:failed=True;queue.clear();self.stop=True
            self.status('running' if not self.stop else 'stopping',queued=[j['id'] for j in queue])
            if queue or self.active:time.sleep(1)
        if failed:raise RuntimeError('Worker failed; see failure.json and worker.log')
        if self.stop:
            state='budget_complete' if self.stop_reason=='budget' else 'interrupted'
            self.status(state,reason=self.stop_reason)
            raise InterruptedError(self.stop_reason)
        if self.args.phase=='run':
            subprocess.run([sys.executable,'scripts/summarize_integer_experiment.py','--root',str(self.root)],check=True)
        return results

    def group_fits(self,jobs,seconds_per_epoch):
        estimate=sum(20+j['config']['epochs']*seconds_per_epoch for j in jobs)*1.2/60
        if self.used()+estimate>self.limit:
            self.status('budget_complete',reason='Insufficient budget for the next complete comparison group',estimated_group_gpu_minutes=estimate)
            return False
        return True


def config(seed,lr,stages=(),mode='radon',epochs=6,batch=16):
    return {'seed':seed,'backbone_lr':lr,'bridges':bridge_configs(stages,mode),'epochs':epochs,'microbatch':batch,'effective_batch':16}


def job(identifier,cfg):return {'id':identifier,'config':cfg}


def main(args):
    c=Controller(args)
    if args.phase=='preflight':
        c.phase='geometry_gpu';c.run_jobs([{'id':'gpu_geometry','geometry':True}])
        c.phase='microbatch_profile'
        for batch in [16,8,4,2]:
            cfg=config(3410,3e-5,(2,3),batch=batch);cfg['profile']=True
            result=c.run_jobs([job(f'profile_b{batch}',cfg)],allow_oom=True)[f'profile_b{batch}']
            if result.get('passed'):
                result['preflight_gpu_minutes']=c.used()
                write_json(c.root/'preflight.json',result);c.status('preflight_complete',microbatch=batch);return
        raise RuntimeError('No microbatch fits; do not freeze or change M/S/H')
    preflight=read_json(c.root/'preflight.json');assert preflight['passed']
    batch=preflight['microbatch'];report={'trials':{},'selections':{},'test_used':False,'source_commit':c.commit}
    per_epoch=max(preflight['step_seconds'][1:])*math_ceil(1264/batch)*1.6+5
    def collect(jobs):
        nonlocal per_epoch
        if not c.group_fits(jobs,per_epoch):return False
        completed=c.run_jobs(jobs);report['trials'].update(completed)
        observed=max(r['seconds']/max(r['configuration']['epochs'],1) for r in completed.values())
        per_epoch=observed if c.phase=='position_screen' else max(per_epoch,observed)
        write_json(c.root/'partial_summary.json',report);return True
    c.phase='learning_rate_calibration'
    calibration=[job('calibration_lr_low',config(3410,3e-6,epochs=4,batch=batch)),job('calibration_lr_high',config(3410,3e-5,epochs=4,batch=batch))]
    if not collect(calibration):return
    selected=max(calibration,key=lambda j:(report['trials'][j['id']]['fixed_last']['mean_task_macro_f1'],-j['config']['backbone_lr']))
    lr=selected['config']['backbone_lr'];report['selections']['backbone_lr']=lr
    write_json(c.root/'partial_summary.json',report)
    c.phase='position_screen'
    positions={'independent':(), 'stage2':(2,), 'stage3':(3,), 'stage23':(2,3)}
    screen=[job('screen_'+name,config(3411,lr,stages,batch=batch)) for name,stages in positions.items()]
    if not collect(screen):return
    selected=max(['stage2','stage3','stage23'],key=lambda name:(report['trials']['screen_'+name]['fixed_last']['mean_task_macro_f1'],-len(positions[name]),name=='stage3'))
    stages=positions[selected];report['selections']['bridge_positions']=list(stages)
    write_json(c.root/'partial_summary.json',report)
    c.phase='mechanism_screen'
    controls=[job('screen_'+mode,config(3411,lr,stages,mode,batch=batch)) for mode in ['self','pooled']]
    if not collect(controls):return
    write_json(c.root/'partial_summary.json',report)
    for seed in [3412,3413,3414]:
        c.phase=f'confirmation_{seed}'
        arms={'independent':((),'radon'),'radon':(stages,'radon'),'self':(stages,'self'),'pooled':(stages,'pooled')}
        group=[job(f'confirm_{seed}_{name}',config(seed,lr,pos,mode,epochs=8,batch=batch)) for name,(pos,mode) in arms.items()]
        if not collect(group):return
        hashes={report['trials'][j['id']]['initial_native_sha256'] for j in group}
        assert len(hashes)==1,'Paired native initializations differ'
    report['new_gpu_minutes']=c.used();report['cumulative_gpu_minutes']=c.protocol['prior_gpu_minutes']+c.used()
    write_json(c.root/'summary.json',report);c.status('complete',selections=report['selections'])


def math_ceil(x):return int(-(-x//1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','run'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--protocol',required=True);p.add_argument('--data',required=True)
    a=p.parse_args();lock_path=Path(a.output).parent.parent/'.active.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(a)
        except InterruptedError:
            subprocess.run([sys.executable,'scripts/summarize_integer_experiment.py','--root',a.output],check=False)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            path=Path(a.output)/'controller_failure.json';write_json(path,{'error':error,'time':time.time()})
            status_path=Path(a.output)/'status.json'
            if status_path.exists():
                state=read_json(status_path);state['state']='failed';state['error']=error;write_json(status_path,state)
            sys.exit(1)
