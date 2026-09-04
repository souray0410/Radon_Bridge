"""One-shot append-only SVD study; hold the project lock through acceptance and run."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def clean(repo):return not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
def head(repo):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
def sources(repo):
    return {str(p.relative_to(repo)):sha(p) for folder in ['radonbridge','scripts','tests'] for p in sorted((repo/folder).glob('*.py'))}
def completed(root, count):
    state=read(root/'status.json');assert state['state']=='complete' and not state['active'], f'Incomplete predecessor: {root}'
    p=read(root/'protocol.json');jobs=[j for g in p['groups'] for j in g['jobs']];assert len(jobs)==count
    for j in jobs:
        d=read(root/j['id']/'summary.json')
        assert d['converged_by_policy'] and d['state']=='complete' and d['stop_reason']=='validation_plateau' and d['test_used'] is False
    return p,state


def accept_references(root):
    p,state=completed(root,16);references={};groups=[];native={};acceptance=[]
    assert len(p['groups'])==4
    assert {(j['config']['seed'],j['config']['backbone_lr']) for g in p['groups'] for j in g['jobs']}=={(s,lr) for s in [3416,3417] for lr in [3e-5,6e-5]}
    policy={'min_epochs':8,'max_epochs':60,'patience':6,'min_delta':.001,'lr_patience':3,'lr_factor':.3}
    for g in p['groups']:
        jobs=g['jobs'];assert len(jobs)==4
        baseline=next(j for j in jobs if not j['config']['bridges'])
        standards=[j for j in jobs if j['config']['bridges']]
        assert sorted(j['config']['bridges'][0]['rho'] for j in standards)==[.0625,.125,.25]
        for j in jobs:
            cfg=j['config'];d=read(root/j['id']/'summary.json');actual=dict(d['configuration']);actual.pop('source_commit',None)
            expected=dict(cfg);expected.pop('source_commit',None);assert actual==expected,'Reference configuration mismatch'
            assert cfg['head_lr']==cfg['bridge_lr']==1e-4 and cfg['microbatch']==cfg['effective_batch']==16
            assert cfg['convergence']==policy and cfg['training_stage']=='communication'
            if cfg['bridges']:
                assert cfg['bridges']==[{'nodes':['cfp_stage3','oct_stage3'],'M':32,'S':64,'rho':cfg['bridges'][0]['rho'],'mode':'radon'}]
            assert d['parent_checkpoints']==cfg['parent_checkpoints']
            for parent in cfg['parent_checkpoints'].values():assert sha(parent['path'])==parent['sha256']
            native.setdefault(cfg['seed'],d['initial_native_sha256']);assert native[cfg['seed']]==d['initial_native_sha256']
            ref={'directory':str(root/j['id']),'summary_sha256':sha(root/j['id']/'summary.json'),
                 'configuration_sha256':sha(root/j['id']/'configuration.json'),'predictions_sha256':sha(root/j['id']/'selected_predictions.npz')}
            references[j['id']]=ref
            acceptance.append(dict(trial=j['id'],**ref,configuration=cfg,initial_native_sha256=d['initial_native_sha256'],converged_by_policy=True))
        groups.append({'name':g['name']+'_svd','baseline':baseline,'standards':standards})
    return p,references,groups,native,acceptance


def run(args):
    root=Path(args.root).resolve();candidate=Path(args.candidate).resolve();repo=Path(args.repo).resolve();old=Path(args.references).resolve();previous=Path(args.predecessor).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own, (root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'run_source.json').exists() and not (root/'preflight').exists(),'Queue already started; inspect before any continuation'
        assert clean(candidate) and head(candidate)==args.commit
        accepted=sources(candidate)
        write(root/'queue_status.json',{'state':'waiting_for_previous_queue','pid':os.getpid(),'predecessor':str(previous),'planned_trials':12,'source_commit':args.commit})
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        pp,state=completed(previous,16)
        original,refs,matched,native,acceptance=accept_references(old)
        assert clean(repo) and head(repo)==pp['source_commit'] and sources(repo)==pp['accepted_source_hashes'],'Active source changed unexpectedly'
        assert clean(candidate) and head(candidate)==args.commit and sources(candidate)==accepted
        write(root/'reference_acceptance.json',{'passed':True,'learned_count':12,'no_bridge_count':4,'references':acceptance,'test_used':False})
        write(root/'queue_status.json',{'state':'fitting_training_bases','pid':os.getpid(),'planned_trials':12,'source_commit':args.commit})
        # GPU work starts only here, after both old queues have ended and the project lock is held.
        os.chdir(candidate);sys.path.insert(0,str(candidate))
        from scripts.run_integer_experiment import Controller
        prior=state['prior_gpu_minutes']+sum(j['gpu_seconds'] for j in read(previous/'ledger.json')['jobs'])/60
        pre=root/'preflight';pre.mkdir()
        common={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence'}
        write(pre/'protocol.json',dict(common,source_commit=args.commit,accepted_source_hashes=accepted))
        c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=original['data']));c.commit=args.commit
        try:
            fit_jobs=[]
            for seed in [3416,3417]:
                cfg=next(g['baseline']['config'] for g in matched if g['baseline']['config']['seed']==seed)
                fit_jobs.append({'id':f'basis_seed{seed}','basis_fit':True,'config':{'seed':seed,'parent_checkpoints':cfg['parent_checkpoints'],'nodes':['cfp_stage3','oct_stage3'],'microbatch':16,'source_commit':args.commit}})
            fits=c.run_jobs(fit_jobs);basis={}
            for name,d in fits.items():
                assert d['passed'] and d['state']=='complete' and d['test_used'] is False
                assert d['provenance']['initial_native_sha256']==native[d['seed']]
                assert set(d['bases'])=={'cfp_stage3','oct_stage3'}
                assert d['peak_reserved_mib']<=10240
                basis[d['seed']]=d['bases']
            assert set(basis)=={3416,3417}
            groups=[];pairs={};comparisons=[]
            for g in matched:
                jobs=[]
                for ref in g['standards']:
                    cfg=copy.deepcopy(ref['config']);cfg['bridges'][0].update(compression='fixed_svd_channel',basis_files=basis[cfg['seed']])
                    name=ref['id']+'_svd';jobs.append({'id':name,'config':cfg})
                    pairs[name+'_minus_learned']=[ref['id'],name];pairs[name+'_minus_no_bridge']=[g['baseline']['id'],name]
                    comparisons.append({'fixed':name,'learned':ref['id'],'baseline':g['baseline']['id']})
                groups.append({'name':g['name'],'jobs':jobs,'estimate_seconds_per_epoch':80})
            cfg=copy.deepcopy(next(j['config'] for g in groups for j in g['jobs'] if j['config']['seed']==3416 and j['config']['backbone_lr']==6e-5 and j['config']['bridges'][0]['rho']==.25));cfg['profile']=True
            write(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'planned_trials':12,'rho':.25,'microbatch':16})
            profile=c.run_jobs([{'id':'profile_rho1_4_batch16','config':cfg}])['profile_rho1_4_batch16']
            assert profile['passed'] and profile['peak_reserved_mib']<=10240 and all(profile['modules_changed'].values())
            assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
            c.status('complete');write(root/'gpu_acceptance.json',{'passed':True,'profile':profile,'basis_fits':fits,'ledger':c.ledger,'source_commit':args.commit,'test_used':False})
        except BaseException:
            c.status('needs_attention',reason=traceback.format_exc());raise
        finally:c.shutdown()
        assert sources(candidate)==accepted and clean(candidate)
        # Deploy the exact validated commit only after preflight; no in-flight source mutation.
        subprocess.run(['git','fetch',str(candidate),'main'],cwd=repo,check=True)
        subprocess.run(['git','merge','--ff-only',args.commit],cwd=repo,check=True)
        assert head(repo)==args.commit and clean(repo) and sources(repo)==accepted
        p=dict(common,source_commit=args.commit,accepted_source_hashes=accepted,driver_sha256=sha(repo/'scripts/run_configured_experiment.py'),
               prior_gpu_minutes=prior+c.used(),references=refs,groups=groups,bootstrap_pairs=pairs,paired_comparisons=comparisons,
               predecessor=str(previous),learned_reference_root=str(old),data=original['data'],test_used=False,
               basis_fit_root=str(pre),basis_files={str(k):v for k,v in basis.items()},gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'),
               authorization='User approved 12 fixed SVD channel trials after existing32; ranks fixed by rho, no energy threshold, full convergence, batch16, unlimited GPU time, 10 GiB per GPU, no test data.')
        write(root/'protocol.json',p)
        os.chdir(repo)
        from scripts.run_configured_experiment import verify,run as run_study
        verify(p)
        env=dict(os.environ,PYTHONPATH='.:third_party/MHD_Project')
        log=(root/'report_worker.log').open('w')
        report=subprocess.Popen([sys.executable,'scripts/report_fixed_svd_study.py','--root',str(root),'--wait'],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        write(root/'queue_status.json',{'state':'running_trials','pid':os.getpid(),'report_pid':report.pid,'planned_trials':12,'source_commit':args.commit,'gpu_preflight_passed':True})
        run_study(root,p)  # same process retains the exclusive project lock without a handoff gap
        status=read(root/'status.json')
        write(root/'queue_status.json',{'state':status['state'],'pid':os.getpid(),'planned_trials':12,'source_commit':args.commit,'report_pid':report.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--candidate',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--repo',default='/home/mengh/RadonBridge');p.add_argument('--references',default='/data/mengh/RadonBridge/runs/2026_09_04_23_54_10');p.add_argument('--predecessor',default='/data/mengh/RadonBridge/runs/2026_09_05_00_05_58');a=p.parse_args()
    try:run(a)
    except BlockingIOError:
        raise SystemExit('Queue already owns its lock; no duplicate launched')
    except BaseException:
        root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
        write(root/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
