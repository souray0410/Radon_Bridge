"""Deferred centered-fit supplement: same linear runtime, 36 paired trainings."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace
from radonbridge.experiment import write_json
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.run_five_seed_study import canonical,verify,diagnostic_job,DATA
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.queue_fixed_svd_study import read,sha,head,clean

PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_09_29_16')
SVD_ARMS=['svd_radon','svd_self','svd_scrambled','svd_resample']
FILES=['summary.json','configuration.json','selected_predictions.npz','selected.pt']

def matrix():return [(s,a,r) for s in SEEDS for r in RHOS for a in SVD_ARMS]
def configuration(reference,bases):
    cfg=copy.deepcopy(reference['configuration']);cfg.pop('source_commit',None)
    assert len(cfg['bridges'])==1 and cfg['bridges'][0]['compression']=='fixed_svd_channel'
    cfg['bridges'][0].update(compression='fixed_centered_svd_channel',basis_files=bases)
    return cfg


def accept_previous(root):
    status=read(PREVIOUS/'queue_status.json');assert status['state']=='complete'
    old=read(PREVIOUS/'manifest.json');report=read(PREVIOUS/'report/results.json');rs=read(PREVIOUS/'report/report_status.json')
    assert rs['complete_trials']==rs['complete_diagnostics']==57 and old['seeds']==SEEDS and old['rhos']==RHOS
    records={};parents={};bases=old['svd_bases']
    for r in old['rows']:
        path=Path(r['directory']);d=read(path/'summary.json');cfg=read(path/'configuration.json')
        assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and not d['test_used']
        assert canonical(cfg)==canonical(r['configuration'])==canonical(d['configuration'])
        assert cfg['seed']==r['seed'] and cfg['backbone_lr']==6e-5 and cfg['microbatch']==cfg['effective_batch']==16
        for b in cfg['bridges']:assert b['rho']==r['rho'] and b['M']==32 and b['S']==64
        hashes={n:sha(path/n) for n in FILES}
        reported=next(x for x in report['rows'] if x['id']==r['id']);assert hashes==reported['result_hashes']
        record=dict(r,reused=True,accepted_hashes=hashes,reference_cost=reported['cost'])
        records[key(r['seed'],r['arm'],r['rho'])]=record;parents[r['seed']]=cfg['parent_checkpoints']
    assert len(records)==57
    for s in SEEDS:
        initial={read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in records.values() if r['seed']==s};assert len(initial)==1
    write_json(root/'reference_acceptance.json',{'passed':True,'references':list(records.values()),'previous_report_sha256':sha(PREVIOUS/'report/results.json'),'previous_manifest_sha256':sha(PREVIOUS/'manifest.json'),'test_used':False})
    return records,parents,{int(s):refs for s,refs in bases.items()},read(PREVIOUS/'study_summary.json')['cumulative_gpu_minutes']


def manifest(root,records,uncentered,centered):
    write_json(root/'manifest.json',{'seeds':SEEDS,'rhos':RHOS,'paired_arms':SVD_ARMS,'new_training_jobs':36,'total_results':93,
        'rows':list(records.values()),'uncentered_bases':{str(k):v for k,v in uncentered.items()},'centered_bases':{str(k):v for k,v in centered.items()},
        'fit_centered':True,'runtime_centering':False,'test_used':False})


def preflight(root,p,records,parents,uncentered):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    try:
        jobs=[{'id':f'centered_basis_seed{s}','basis_fit':True,'config':{'seed':s,'parent_checkpoints':parents[s],
            'nodes':['cfp_stage3','oct_stage3'],'microbatch':16,'fit_centered':True,'uncentered_basis_files':uncentered[s],'source_commit':p['source_commit']}} for s in SEEDS]
        fits=c.run_jobs(jobs);assert len(fits)==3 and all(v['passed'] for v in fits.values())
        centered={s:fits[f'centered_basis_seed{s}']['bases'] for s in SEEDS};jobs=[]
        for arm in SVD_ARMS:
            cfg=configuration(records[key(3416,arm,.25)],centered[3416]);cfg.update(profile=True,save_profile_checkpoint=True)
            jobs.append({'id':'profile_centered_'+arm,'config':cfg})
        profiles=c.run_jobs(jobs);assert len(profiles)==4 and all(v['passed'] for v in profiles.values())
        jobs=[diagnostic_job({'id':name,'directory':str(pre/name)},uncentered[3416],True,'profile_selected.pt') for name in profiles]
        jobs.append(diagnostic_job(dict(records[key(3416,'svd_radon',.25)],id='profile_uncentered_svd'),uncentered[3416],True))
        for job in jobs:job['config'].update(probe_count=128,energy_limit=None)
        ds=c.run_jobs(jobs);assert len(ds)==5 and all(v['passed'] for v in ds.values())
        assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
        c.status('complete');write_json(root/'gpu_acceptance.json',{'passed':True,'basis_fits':fits,'profiles':profiles,'diagnostics':ds,'ledger':c.ledger,
            'gpu_minutes':c.used(),'source_commit':p['source_commit'],'tested_source_hashes':p['accepted_source_hashes'],'test_used':False})
        return centered,c.used()
    except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
    finally:c.shutdown()


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Already started'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_previous_queue','pid':os.getpid(),'source_commit':commit,'predecessor':str(PREVIOUS),'new_training_jobs':36})
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        records,parents,uncentered,prior=accept_previous(root)
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
            'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','data':DATA,'seeds':SEEDS,'rhos':RHOS,
            'paired_arms':SVD_ARMS,'new_stage_two_jobs':36,'new_training_jobs':36,'new_pretraining_jobs':0,'basis_fit_jobs':3,'diagnostic_jobs':72,
            'reused_results':57,'total_stage_two_results':93,'predecessor':str(PREVIOUS),'fit_centered':True,'runtime_centering':False,
            'comparison':'centered covariance basis versus uncentered second-moment basis; identical Q^T X runtime',
            'convergence':dict(DEFAULT_POLICY),'test_used':False}
        verify(p,records);write_json(root/'plan_protocol.json',p);manifest(root,records,uncentered,{})
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit,'new_training_jobs':36})
        centered,cost=preflight(root,p,records,parents,uncentered)
        for s,a,rho in matrix():
            reference=records[key(s,a,rho)];arm='centered_'+a;name=identifier(s,arm,rho)
            records[key(s,arm,rho)]={'id':name,'seed':s,'arm':arm,'rho':rho,'paired_arm':a,'paired_id':reference['id'],'paired_directory':reference['directory'],
                'directory':str(root/name),'reused':False,'configuration':configuration(reference,centered[s])}
        p=dict(p,prior_gpu_minutes=prior+cost,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'));verify(p,records)
        if args.deploy_repo:
            repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(Path.cwd()),'main'],cwd=repo,check=True);subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p);manifest(root,records,uncentered,centered)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'planned_new_training_jobs':36,'gpu_preflight_passed':True})
            for seed in SEEDS:
                for rho in RHOS:
                    verify(p,records);rs=[records[key(seed,'centered_'+a,rho)] for a in SVD_ARMS]
                    c.phase=f'seed{seed}_rho{rho}_centered_training'
                    results=c.run_jobs([{'id':r['id'],'config':r['configuration']} for r in rs]);assert len(results)==4 and all(d['converged_by_policy'] for d in results.values()),'Epoch cap; needs attention'
                    for r in rs:
                        a=read(Path(r['directory'])/'summary.json');b=read(Path(r['paired_directory'])/'summary.json');assert a['initial_native_sha256']==b['initial_native_sha256'] and a['parent_checkpoints']==b['parent_checkpoints']
                    c.phase=f'seed{seed}_rho{rho}_paired_diagnostics'
                    pair_rs=rs+[records[key(seed,a,rho)] for a in SVD_ARMS]
                    ds=c.run_jobs([diagnostic_job(r,uncentered[seed]) for r in pair_rs]);assert len(ds)==8 and all(d['passed'] for d in ds.values())
                    write_json(root/'partial_summary.json',{'completed_centered_trials':sum((Path(r['directory'])/'summary.json').exists() for r in records.values() if not r['reused']),'test_used':False})
            verify(p,records);c.status('complete')
        except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
        from scripts.report_centered_study import build
        build(root)
        assert read(root/'report/report_status.json')['state']=='complete'
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'report_complete':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--deploy-repo');args=parser.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:
        write_json(Path(args.root)/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
