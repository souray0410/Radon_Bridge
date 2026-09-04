"""Bounded two-GPU staged experiment controller with persistent accounting."""
import argparse
import atexit
import fcntl
import hashlib
import json
import math
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
        if self.protocol['review_status']!='approved' or self.protocol['schema'] not in {'two_stage_ratio_convergence_v3','two_stage_ablation_v1'}:
            raise ValueError('Only the explicitly approved ratio and convergence protocol may launch')
        if self.protocol.get('gpu_time_policy') == 'unlimited_until_convergence':
            if self.protocol['max_gpu_minutes'] is not None:
                raise ValueError('Unlimited GPU time must use a null minute limit')
            self.limit=float('inf')
        else:
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
            'cumulative_gpu_minutes':self.protocol['prior_gpu_minutes']+self.used(),'budget_gpu_minutes':self.limit if math.isfinite(self.limit) else None,
            'active':[{'id':k,'pid':v['process'].pid,'gpu':v['gpu']} for k,v in self.active.items()],
            'finished_jobs':len(self.ledger['jobs']),'updated_at':time.time(),**kw})

    def start(self,job,gpu):
        path=self.root/job['id']
        if path.exists():raise RuntimeError(f'Refuse to overwrite {path}')
        path.mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if job.get('basis_fit'):
            write_json(path/'configuration.json',job['config'])
            cmd=[sys.executable,'-m','radonbridge.svd_basis','--config',str(path/'configuration.json'),
                 '--output',str(path),'--data',self.args.data]
        elif job.get('audit'):
            cmd=[sys.executable,'tests/check_two_stage.py','--device','cuda','--output',str(path/'summary.json')]
        elif job.get('geometry'):
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
                    elif self.stop_reason in ('budget','controller_signal'):pass
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
        estimate=sum(20+j['config'].get('budget_estimate_epochs',j['config']['convergence']['max_epochs'])*seconds_per_epoch for j in jobs)*1.2/60
        if self.used()+estimate>self.limit:
            self.status('budget_complete',reason='Insufficient budget for the next complete comparison group',estimated_group_gpu_minutes=estimate)
            return False
        return True


DEFAULT_POLICY={'min_epochs':8,'max_epochs':60,'patience':6,'min_delta':.001,'lr_patience':3,'lr_factor':.3}

def config(seed,lr,stages=(),mode='radon',epochs=60,batch=16,protocol=None):
    p=protocol or {}
    policy=dict(p.get('convergence',DEFAULT_POLICY))
    if protocol is None:policy['max_epochs']=epochs;policy['min_epochs']=min(policy['min_epochs'],epochs)
    return {'seed':seed,'backbone_lr':lr,
            'bridges':bridge_configs(stages,mode,M=p.get('M',16),S=p.get('S',64),rho=p.get('rho',.125)),
            'convergence':policy,'microbatch':batch,'effective_batch':16}


def job(identifier,cfg):return {'id':identifier,'config':cfg}


def source_hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['radonbridge','scripts','tests']
            for p in sorted(Path(directory).glob('*.py'))}


def main(args):
    c=Controller(args)
    if args.phase=='preflight':
        c.phase='geometry_checkpoint_and_ratio_audit'
        attempt=1+sum(1 for p in c.root.glob('two_stage_audit*') if p.is_dir())
        c.run_jobs([{'id':f'two_stage_audit_{attempt}','audit':True},{'id':f'gpu_geometry_{attempt}','geometry':True}])
        for batch in [16,8,4,2]:
            cfg=config(3415,3e-5,(2,3),batch=batch,protocol=c.protocol);cfg['profile']=True
            identifier=f'memory_profile_{batch}_{attempt}'
            result=c.run_jobs([job(identifier,cfg)],allow_oom=True)[identifier]
            if result.get('passed'):
                c.status('preflight_complete',microbatch=batch)
                write_json(c.root/'preflight.json',{'passed':True,'microbatch':batch,'profile':result,
                    'source_hashes':source_hashes(),'protocol_sha256':hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()})
                return
        raise RuntimeError('Even microbatch 2 did not fit; do not freeze or shrink bridge')
    preflight=read_json(c.root/'preflight.json')
    assert preflight['passed'] and preflight['source_hashes']==source_hashes(), 'Source changed since acceptance'
    assert preflight['protocol_sha256']==hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
    batch=preflight['microbatch']
    report={'trials':{},'training_protocol':'independent modality validation plateau then matched joint validation plateau',
            'source_commit':c.commit,'test_used':False}
    completed_modes=set()
    continuation=c.protocol.get('continuation_from')
    if continuation:
        source=Path(continuation['partial_summary'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=continuation['partial_summary_sha256']:
            raise ValueError('Inherited continuation summary hash mismatch')
        inherited=read_json(source)
        report['trials'].update(inherited['trials'])
        completed_modes={name.rsplit('_',1)[-1] for name in inherited['trials'] if name.startswith('confirm_')}
        report['continuation_reference']=continuation
        for identifier,item in c.protocol.get('inherited_trials',{}).items():
            directory=Path(item['directory'])
            if hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest()!=item['summary_sha256']:
                raise ValueError('Inherited trial summary hash mismatch')
            target=c.root/identifier
            if not target.exists():target.symlink_to(directory, target_is_directory=True)
    # The GPU profile is only a pre-training fallback.  Once a complete real
    # stage is available, its full epoch times include validation and shared
    # machine contention and are the safer basis for the next pair.
    seconds_per_epoch=max(preflight['profile']['step_seconds'][1:])*math_ceil(1264/batch)*1.4+5
    for seed in c.protocol['seeds']:
        c.phase=f'independent_pretraining_{seed}'
        inherited=c.protocol.get('pretrained_from',{}).get(str(seed))
        if inherited:
            source=Path(inherited['summary'])
            if hashlib.sha256(source.read_bytes()).hexdigest()!=inherited['summary_sha256']:
                raise ValueError('Inherited stage-one summary hash mismatch')
            result=read_json(source)
            if result['configuration']['seed']!=seed or result['configuration']['training_stage']!='independent':
                raise ValueError('Inherited checkpoint is not the requested independent stage')
            for branch in ('cfp','oct'):
                if result['modality_checkpoints'][branch]['sha256']!=inherited[f'{branch}_sha256']:
                    raise ValueError('Inherited modality checkpoint hash differs from the approved protocol')
            report['pretraining_reference']={str(seed):inherited}
        else:
            warm=config(seed,c.protocol['backbone_lr'],batch=batch,protocol=c.protocol)
            warm['training_stage']='independent';warm['budget_estimate_epochs']=warm['convergence']['min_epochs'];warm_job=job(f'pretrain_{seed}',warm)
            if not c.group_fits([warm_job],seconds_per_epoch):return
            completed=c.run_jobs([warm_job]);result=completed[warm_job['id']]
            report['trials'].update(completed);write_json(c.root/'partial_summary.json',report)
        if not result['converged_by_policy']:
            c.status('needs_attention',reason='Stage one hit epoch cap without validation plateau; no stage two launched');return
        parents=result['modality_checkpoints'];assert set(parents)=={'cfp','oct'}
        # Actual epoch duration can exceed a short profile under shared-machine load.
        if completed_modes:
            previous=[v for k,v in report['trials'].items() if k.startswith(f'confirm_{seed}_')]
            seconds_per_epoch=max(max(v['epoch_seconds']) for v in previous)
            expected_epochs=min(c.protocol['convergence']['max_epochs'],
                                math_ceil(max(v['epochs_ran'] for v in previous)*1.25))
        else:
            seconds_per_epoch=max(result['epoch_seconds'])*1.2
            expected_epochs=min(c.protocol['convergence']['max_epochs'],math_ceil(max(result['epochs_ran'],c.protocol['convergence']['min_epochs'])*1.5))
        hashes=set()
        for modes in c.protocol['arm_groups']:
            if set(modes)<=completed_modes:continue
            if set(modes)&completed_modes:raise ValueError('Only complete comparison pairs may be inherited')
            c.phase=f"continuations_{seed}_{'_'.join(modes)}"
            jobs=[]
            for mode in modes:
                cfg=config(seed,c.protocol['backbone_lr'],() if mode=='independent' else c.protocol['positions'],
                           'radon' if mode=='independent' else mode,batch=batch,protocol=c.protocol)
                cfg.update(training_stage='communication',parent_checkpoints=parents,
                           budget_estimate_epochs=expected_epochs)
                jobs.append(job(f'confirm_{seed}_{mode}',cfg))
            # Estimate from the observed stage-one plateau with 50% epoch headroom plus the group time margin.
            # Convergence is not predictable: the hard budget guard can still interrupt an incomplete pair.
            if not c.group_fits(jobs,seconds_per_epoch):return
            completed=c.run_jobs(jobs);report['trials'].update(completed)
            hashes.update(r['initial_native_sha256'] for r in completed.values())
            assert len(hashes)==1,'Continuation native initializations differ'
            assert all(r['parent_checkpoints']==parents for r in completed.values())
            write_json(c.root/'partial_summary.json',report)
            if not all(r['converged_by_policy'] for r in completed.values()):
                c.status('needs_attention',reason='Comparison contains a non-converged epoch-cap run; not a completed comparison');return
            seconds_per_epoch=max(seconds_per_epoch,max(t for r in completed.values() for t in r['epoch_seconds']))
            completed_modes.update(modes)
            expected_epochs=min(c.protocol['convergence']['max_epochs'],
                                math_ceil(max(r['epochs_ran'] for r in completed.values())*1.25))
    report['new_gpu_minutes']=c.used();report['cumulative_gpu_minutes']=c.protocol['prior_gpu_minutes']+c.used()
    write_json(c.root/'summary.json',report);c.status('complete')


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
