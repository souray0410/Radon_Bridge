"""Three seeds, all six bridge methods at three widths; accepted-result reuse."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace
from radonbridge.experiment import write_json
from radonbridge.svd_basis import save_random_basis
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.run_five_seed_study import base,configuration,canonical,verify,diagnostic_job,OLD,SVD,DATA
from scripts.queue_fixed_svd_study import accept_references,clean,head,sha,read

SEEDS=[3416,3417,3418]
RHOS=[.0625,.125,.25]
METHODS=['learned','svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']
PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_09_03_55')


def key(seed,arm,rho):return (seed,arm,rho or 0.)
def identifier(seed,arm,rho):return f'seed{seed}_{arm}'+(f'_rho1_{round(1/rho)}' if rho else '')
def matrix():return [(s,a,r) for s in SEEDS for a,r in [('no_bridge',None)]+[(a,r) for r in RHOS for a in METHODS]]
def config(seed,parents,svd,qr,arm,rho):
    cfg=configuration(seed,parents,svd,qr,'learned_equal' if arm=='learned' else arm)
    if cfg['bridges']:cfg['bridges'][0]['rho']=rho
    return cfg


def accept(path,seed,arm,rho,expected):
    d=read(path/'summary.json')
    assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and d['test_used'] is False
    assert canonical(d['configuration'])==canonical(expected)
    assert canonical(read(path/'configuration.json'))==canonical(expected)
    assert d['parent_checkpoints']==expected['parent_checkpoints']
    cost=next(j for j in read(path.parent/'ledger.json')['jobs'] if j['id']==path.name)
    return {'id':identifier(seed,arm,rho),'seed':seed,'arm':arm,'rho':rho,'directory':str(path),'reused':True,'configuration':expected,
            'accepted_hashes':{n:sha(path/n) for n in ['summary.json','configuration.json','selected_predictions.npz','selected.pt']},'reference_cost':cost}


def references(root):
    _,_,matched,native,_=accept_references(OLD)
    old=read(PREVIOUS/'manifest.json');status=read(PREVIOUS/'status.json')
    assert not status['active'] and read(PREVIOUS/'queue_status.json')['state'] in ['needs_attention','complete']
    parents={};bases={};qr={int(k):v for k,v in old['qr_bases'].items() if int(k) in SEEDS};records={}
    for seed in [3416,3417]:
        g=next(g for g in matched if g['baseline']['config']['seed']==seed and g['baseline']['config']['backbone_lr']==6e-5)
        parents[seed]=g['baseline']['config']['parent_checkpoints'];bases[seed]=read(SVD/'protocol.json')['basis_files'][str(seed)]
        candidates=[('no_bridge',None,OLD/g['baseline']['id'])]
        for rho in RHOS:
            candidates += [('learned',rho,OLD/next(j['id'] for j in g['standards'] if j['config']['bridges'][0]['rho']==rho)),
                           ('svd_radon',rho,SVD/f'seed{seed}_bb6e-05_rho1_{round(1/rho)}_svd')]
        for arm,rho,path in candidates:
            record=accept(path,seed,arm,rho,config(seed,parents[seed],bases[seed],qr.get(seed),arm,rho))
            assert read(path/'summary.json')['initial_native_sha256']==native[seed]
            records[key(seed,arm,rho)]=record
    historical=len(records);assert historical==14
    # Reuse only completed work from the superseded five-seed queue; failures are retained there.
    for old_record in old['rows']:
        seed=old_record['seed'];arm=old_record['arm']
        if seed not in SEEDS or arm not in ['svd_self','svd_scrambled','svd_resample','qr_radon']:continue
        path=Path(old_record['directory'])
        if not (path/'summary.json').exists():continue
        record=accept(path,seed,arm,.125,config(seed,parents[seed],bases[seed],qr.get(seed),arm,.125))
        assert read(path/'summary.json')['initial_native_sha256']==native[seed]
        records[key(seed,arm,.125)]=record
    write_json(root/'reference_acceptance.json',{'passed':True,'historical_results':historical,'superseded_completed_results':len(records)-historical,'references':list(records.values()),'test_used':False})
    verification=Path('/tmp/rb_three_diag_baseline/summary.json')
    extra=read(verification)
    assert extra['passed'] and extra['phases']['selected']['probe_participants']==128
    write_json(root/'diagnostic_fix_acceptance.json',{'summary':extra,'sha256':sha(verification),'directory':str(verification.parent),'gpu_minutes':extra['seconds']/60})
    return records,parents,bases,qr,read(PREVIOUS/'study_summary.json')['cumulative_gpu_minutes']+extra['seconds']/60


def manifest(root,records,bases,qr):
    write_json(root/'manifest.json',{'seeds':SEEDS,'rhos':RHOS,'arms':['no_bridge']+METHODS,'rows':[records[key(*x)] for x in matrix()],
        'svd_bases':{str(k):v for k,v in bases.items()},'qr_bases':{str(k):v for k,v in qr.items()},'test_used':False})


def preflight(root,p,parents,bases,qr):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    try:
        jobs=[]
        for arm in ['svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']:
            cfg=config(3416,parents[3416],bases[3416],qr[3416],arm,.25)
            cfg.update(profile=True,save_profile_checkpoint=True);jobs.append({'id':'profile_'+arm,'config':cfg})
        profiles=c.run_jobs(jobs);assert len(profiles)==5 and all(d['passed'] for d in profiles.values())
        jobs=[]
        for name in profiles:
            jobs.append(diagnostic_job({'id':name,'directory':str(pre/name)},bases[3416],True,'profile_selected.pt'))
        for arm,name in [('no_bridge','independent'),('learned','rho1_4')]:
            jobs.append(diagnostic_job({'id':'reference_'+arm,'directory':str(OLD/f'seed3416_bb6e-05_{name}')},bases[3416],True))
        for job in jobs:job['config'].update(probe_count=128,energy_limit=None)
        ds=c.run_jobs(jobs);assert len(ds)==7 and all(d['passed'] for d in ds.values())
        assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
        c.status('complete');write_json(root/'gpu_acceptance.json',{'passed':True,'profiles':profiles,'diagnostics':ds,'ledger':c.ledger,
            'gpu_minutes':c.used(),'source_commit':p['source_commit'],'tested_source_hashes':p['accepted_source_hashes'],'test_used':False})
        return c.used()
    except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
    finally:c.shutdown()


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Already started; inspect before continuation'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_project_lock','pid':os.getpid(),'source_commit':commit,'seeds':SEEDS})
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        records,parents,bases,qr,prior=references(root)
        reuse=len(records)
        for seed,arm,rho in matrix():
            name=identifier(seed,arm,rho)
            records.setdefault(key(seed,arm,rho),{'id':name,'seed':seed,'arm':arm,'rho':rho,'directory':str(root/name),'reused':False,'configuration':None})
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
           'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','data':DATA,'seeds':SEEDS,'rhos':RHOS,
           'arms':['no_bridge']+METHODS,'total_stage_two_results':57,'new_stage_two_jobs':57-reuse,'new_training_jobs':58-reuse,'reused_results':reuse,
           'supersedes':str(PREVIOUS),'superseded_reason':'User reduced to 3 seeds and expanded all mechanisms to 3 rho values; diagnostic memory fix',
           'convergence':dict(DEFAULT_POLICY),'test_used':False}
        verify(p,records);write_json(root/'plan_protocol.json',p);manifest(root,records,bases,qr)
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit,'seeds':SEEDS})
        pre_minutes=preflight(root,p,parents,bases,qr);p=dict(p,prior_gpu_minutes=prior+pre_minutes,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'))
        verify(p,records)
        if args.deploy_repo:
            repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(Path.cwd()),'main'],cwd=repo,check=True)
            subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'seeds':SEEDS,'planned_new_training_jobs':p['new_training_jobs'],'gpu_preflight_passed':True})
            for seed in SEEDS:
                verify(p,records)
                if seed not in parents:
                    c.phase=f'pretrain_{seed}'
                    result=c.run_jobs([{'id':f'pretrain_{seed}','config':base(seed)}])[f'pretrain_{seed}']
                    assert result['converged_by_policy'],'Pretraining epoch cap; needs attention'
                    parents[seed]=result['modality_checkpoints'];c.phase=f'basis_fit_{seed}'
                    fit=c.run_jobs([{'id':f'basis_seed{seed}','basis_fit':True,'config':{'seed':seed,'parent_checkpoints':parents[seed],'nodes':['cfp_stage3','oct_stage3'],'microbatch':16,'source_commit':commit}}])[f'basis_seed{seed}']
                    assert fit['passed'];bases[seed]=fit['bases']
                if seed not in qr:
                    tick=time.monotonic();qr[seed]={k:save_random_basis(v,root/'random_bases') for k,v in bases[seed].items()}
                    timings=read(root/'qr_generation.json') if (root/'qr_generation.json').exists() else {}
                    timings[str(seed)]={'cpu_wall_seconds':time.monotonic()-tick,'bases':qr[seed]};write_json(root/'qr_generation.json',timings)
                for s,arm,rho in [x for x in matrix() if x[0]==seed]:
                    r=records[key(s,arm,rho)]
                    if not r['reused']:r['configuration']=config(seed,parents[seed],bases[seed],qr[seed],arm,rho)
                manifest(root,records,bases,qr);verify(p,records)
                for rho in [None]+RHOS:
                    rs=[records[key(*x)] for x in matrix() if x[0]==seed and x[2]==rho]
                    jobs=[{'id':r['id'],'config':r['configuration']} for r in rs if not r['reused']]
                    c.phase=f'seed{seed}_rho{rho}_training';results=c.run_jobs(jobs)
                    assert all(d['converged_by_policy'] for d in results.values()),'Second-stage cap; needs attention'
                    native={read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in rs}
                    baseline=read(Path(records[key(seed,'no_bridge',None)]['directory'])/'summary.json')
                    assert native=={baseline['initial_native_sha256']}
                    c.phase=f'seed{seed}_rho{rho}_diagnostics'
                    ds=c.run_jobs([diagnostic_job(r,bases[seed]) for r in rs]);assert all(d['passed'] for d in ds.values())
                    write_json(root/'partial_summary.json',{'completed_groups':[j['id'] for j in c.ledger['jobs']],'test_used':False})
            verify(p,records);c.status('complete')
        except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();manifest(root,records,bases,qr)
            write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
        from scripts.report_three_seed_study import build
        build(root)
        assert read(root/'report/report_status.json')['state']=='complete'
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'report_complete':True,'seeds':SEEDS})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--deploy-repo');args=parser.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:
        write_json(Path(args.root)/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
