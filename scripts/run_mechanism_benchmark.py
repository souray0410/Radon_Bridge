"""Approved 57-training supplement; immutable 129 references, 186 final results."""
import argparse,copy,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.communication_analysis import derangements
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.run_five_seed_study import canonical,verify,diagnostic_job,DATA
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.queue_fixed_svd_study import read,sha,head,clean
from scripts.run_centered_study import FILES

PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_13_32_31')
MECHANISMS=['qr_resample','svd_oct_to_cfp','svd_cfp_to_oct','qr_oct_to_cfp','qr_cfp_to_oct']
BASELINES=['mmtm_r4','mmtm_r8','attention_d128','attention_d256']

def matrix():return [(s,a,r) for s in SEEDS for r in RHOS for a in MECHANISMS]+[(s,a,None) for s in SEEDS for a in BASELINES]

def configuration(records,seed,arm,rho):
    ref=records[key(seed,'qr_radon' if arm.startswith('qr_') else 'svd_radon',rho or .125)]
    cfg=copy.deepcopy(canonical(ref['configuration']));b=cfg['bridges'][0]
    if arm=='qr_resample':b['mode']='linear_resample'
    elif arm.endswith('_oct_to_cfp'):b['cross_edges']=[['oct_stage3','cfp_stage3']]
    elif arm.endswith('_cfp_to_oct'):b['cross_edges']=[['cfp_stage3','oct_stage3']]
    elif arm.startswith('mmtm_'):cfg['bridges']=[{'nodes':b['nodes'],'family':'mmtm','reduction_ratio':int(arm[6:])}]
    elif arm.startswith('attention_'):cfg['bridges']=[{'nodes':b['nodes'],'family':'cross_attention','attention_dimension':int(arm[11:]),'heads':4}]
    else:raise ValueError(arm)
    return cfg

def accept_previous(root):
    assert read(PREVIOUS/'queue_status.json')['state']=='complete'
    m=read(PREVIOUS/'manifest.json');report=read(PREVIOUS/'report/results.json');status=read(PREVIOUS/'report/report_status.json')
    assert status['complete_trials']==status['total_trials']==129 and m['seeds']==SEEDS and m['rhos']==RHOS
    old={r['id']:r for r in report['rows']};records={};identity=None
    for r in m['rows']:
        path=Path(r['directory']);s=read(path/'summary.json');cfg=read(path/'configuration.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and not s['test_used']
        assert canonical(cfg)==canonical(r['configuration'])==canonical(s['configuration'])
        assert cfg['backbone_lr']==6e-5 and cfg['head_lr']==cfg['bridge_lr']==1e-4 and cfg['microbatch']==cfg['effective_batch']==16
        assert cfg['convergence']==DEFAULT_POLICY
        for b in cfg['bridges']:assert b['rho']==r['rho'] and b['M']==32 and b['S']==64
        hashes={n:sha(path/n) for n in FILES};assert hashes==old[r['id']]['result_hashes']
        with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
            current=(z['ids'].copy(),z['y'].copy());assert len(current[0])==296 and np.bincount(current[1]).tolist()==[148,148]
            if identity is None:identity=current
            else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
        records[key(r['seed'],r['arm'],r['rho'])]=dict(r,reused=True,accepted_hashes=hashes,reference_cost=old[r['id']]['cost'])
    assert len(records)==129
    for seed in SEEDS:
        rs=[r for r in records.values() if r['seed']==seed]
        assert len({read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in rs})==1
        assert all(r['configuration']['parent_checkpoints']==rs[0]['configuration']['parent_checkpoints'] for r in rs)
    write_json(root/'reference_acceptance.json',{'passed':True,'references':list(records.values()),'previous_report_sha256':sha(PREVIOUS/'report/results.json'),'previous_manifest_sha256':sha(PREVIOUS/'manifest.json'),'participant_alignment':True,'test_used':False})
    np.savez_compressed(root/'private_permutations.npz',ids=identity[0],permutations=derangements(296))
    write_json(root/'permutation_protocol.json',{'algorithm':'independent NumPy default_rng(202609051); rejection-sampled uniform permutations with no fixed points','n':296,'count':20,'unit':'participant; both eyes move together; inverse permutation for reverse direction','labels_used':False,'sha256':sha(root/'private_permutations.npz'),'test_used':False})
    return records,{int(s):b for s,b in m['diagnostic_reference_bases'].items()},read(PREVIOUS/'study_summary.json')['cumulative_gpu_minutes']

def manifest(root,records,bases):
    write_json(root/'manifest.json',{'seeds':SEEDS,'rhos':RHOS,'new_training_jobs':57,'total_results':186,'rows':list(records.values()),'diagnostic_reference_bases':{str(s):b for s,b in bases.items()},'test_used':False})

def analysis_job(record,kind,root):
    path=Path(record['directory']);cfg={'trial_directory':str(path),'selected_sha256':sha(path/'selected.pt'),'analysis':kind}
    if kind=='pairing':cfg['permutation_file']=str(root/'private_permutations.npz')
    return {'id':kind+'_'+record['id'],'analysis':True,'config':cfg,**({'gpu':1} if kind=='latency' else {})}

def preflight(root,p,records,bases):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    try:
        jobs=[]
        for arm in MECHANISMS+BASELINES:
            cfg=configuration(records,3416,arm,None if arm in BASELINES else .25);cfg.update(profile=True,save_profile_checkpoint=True)
            jobs.append({'id':'profile_'+arm,'config':cfg})
        profiles=c.run_jobs(jobs);assert len(profiles)==9 and all(v['passed'] for v in profiles.values())
        jobs=[diagnostic_job({'id':n,'directory':str(pre/n)},bases[3416],True,'profile_selected.pt') for n in profiles]
        for j in jobs:j['config'].update(probe_count=128,energy_limit=None)
        ds=c.run_jobs(jobs);assert len(ds)==9 and all(v['passed'] for v in ds.values())
        # Full 296-participant cached/MHD identity check before deploying analysis.
        check=c.run_jobs([analysis_job(records[key(3416,'svd_radon',.25)],'pairing',root)])
        assert all(v['passed'] for v in check.values())
        assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
        c.status('complete');write_json(root/'gpu_acceptance.json',{'passed':True,'profiles':profiles,'diagnostics':ds,'paired_analysis':check,'ledger':c.ledger,'gpu_minutes':c.used(),'source_commit':p['source_commit'],'tested_source_hashes':p['accepted_source_hashes'],'test_used':False})
        return c.used()
    except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
    finally:c.shutdown()

def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Already started; inspect, do not duplicate'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_previous_queue','pid':os.getpid(),'source_commit':commit,'predecessor':str(PREVIOUS),'new_training_jobs':57})
        while True:
            previous=read(PREVIOUS/'queue_status.json')
            if previous['state']=='needs_attention':raise RuntimeError('Predecessor needs attention')
            if previous['state']=='complete':
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            time.sleep(10)
        records,bases,prior=accept_previous(root)
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
            'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','data':DATA,'seeds':SEEDS,'rhos':RHOS,
            'new_stage_two_jobs':57,'new_training_jobs':57,'new_pretraining_jobs':0,'basis_fit_jobs':0,'diagnostic_jobs':57,'pairing_checkpoints':54,'latency_checkpoints':33,'reused_results':129,
            'total_stage_two_results':186,'predecessor':str(PREVIOUS),'convergence':dict(DEFAULT_POLICY),
            'bootstrap_resamples':10000,'primary_contrasts':31,'simultaneous_interval':'centered bootstrap max absolute t, bootstrap SD scaling',
            'practical_margin_pp':1.,'development_previously_selected':True,'test_used':False}
        verify(p,records);write_json(root/'plan_protocol.json',p)
        for seed,arm,rho in matrix():
            name=identifier(seed,arm,rho);records[key(seed,arm,rho)]={'id':name,'seed':seed,'arm':arm,'rho':rho,'directory':str(root/name),'reused':False,'configuration':configuration(records,seed,arm,rho)}
        assert len(records)==186;manifest(root,records,bases);verify(p,records)
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit,'new_training_jobs':57})
        cost=preflight(root,p,records,bases);p=dict(p,prior_gpu_minutes=prior+cost,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'));verify(p,records)
        if args.deploy_repo:
            repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(Path.cwd()),'main'],cwd=repo,check=True);subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p);manifest(root,records,bases)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'planned_new_training_jobs':57,'gpu_preflight_passed':True})
            for seed in SEEDS:
                groups=[[records[key(seed,a,r)] for a in MECHANISMS] for r in RHOS]+[[records[key(seed,a,None)] for a in BASELINES]]
                for group_index,rs in enumerate(groups):
                    verify(p,records);c.phase=f'seed{seed}_group{group_index}_training'
                    results=c.run_jobs([{'id':r['id'],'config':r['configuration']} for r in rs]);assert len(results)==len(rs) and all(v['converged_by_policy'] for v in results.values()),'Epoch cap; needs attention'
                    native=read(Path(records[key(seed,'no_bridge',None)]['directory'])/'summary.json')
                    for r in rs:
                        d=read(Path(r['directory'])/'summary.json');assert d['initial_native_sha256']==native['initial_native_sha256'] and d['parent_checkpoints']==native['parent_checkpoints']
                    c.phase=f'seed{seed}_group{group_index}_diagnostics';ds=c.run_jobs([diagnostic_job(r,bases[seed]) for r in rs]);assert len(ds)==len(rs) and all(v['passed'] for v in ds.values())
                    write_json(root/'partial_summary.json',{'completed_new_trials':sum((Path(r['directory'])/'summary.json').exists() for r in records.values() if not r['reused']),'test_used':False})
            c.phase='paired_information_diagnostics'
            rs=[records[key(s,a,r)] for s in SEEDS for r in RHOS for a in ['svd_radon','qr_radon','svd_oct_to_cfp','svd_cfp_to_oct','qr_oct_to_cfp','qr_cfp_to_oct']]
            ds=c.run_jobs([analysis_job(r,'pairing',root) for r in rs]);assert len(ds)==54 and all(v['passed'] for v in ds.values())
            c.phase='controlled_forward_latency'
            rs=[records[key(s,a,r)] for s in SEEDS for a,r in [('no_bridge',None)]+[(a,None) for a in BASELINES]+[(a,r) for r in RHOS for a in ['svd_radon','qr_radon']]]
            # Sequential on GPU 1 (when available) avoids self-contention between timings.
            for r in rs:
                result=c.run_jobs([analysis_job(r,'latency',root)]);assert all(v['passed'] for v in result.values())
            verify(p,records);c.status('complete')
        except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
        from scripts.report_mechanism_benchmark import build
        build(root);assert read(root/'report/report_status.json')['state']=='complete'
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'report_complete':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deploy-repo');a=p.parse_args()
    try:run(a)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:
        write_json(Path(a.root)/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
