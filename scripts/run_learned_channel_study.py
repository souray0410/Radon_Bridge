"""Deferred 36-trial learned channel encoder/decoder study; immutable 93 references."""
import argparse,copy,fcntl,os,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
from radonbridge.experiment import write_json
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.run_five_seed_study import canonical,verify,diagnostic_job,DATA
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.queue_fixed_svd_study import read,sha,head,clean
from scripts.run_centered_study import SVD_ARMS,FILES

PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_10_09_52')
def matrix():return [(s,a,r) for s in SEEDS for r in RHOS for a in SVD_ARMS]
def configuration(reference):
    cfg=copy.deepcopy(reference['configuration']);cfg.pop('source_commit',None)
    b=cfg['bridges'][0];assert b['compression']=='fixed_svd_channel'
    b['compression']='learned_channel';b.pop('basis_files');return cfg

def accept_previous(root):
    assert read(PREVIOUS/'queue_status.json')['state']=='complete'
    m=read(PREVIOUS/'manifest.json');report=read(PREVIOUS/'report/results.json');status=read(PREVIOUS/'report/report_status.json')
    assert status['complete_trials']==status['total_trials']==93 and status['paired_diagnostics']==72
    assert m['seeds']==SEEDS and m['rhos']==RHOS
    byid={r['id']:r for r in report['rows']};records={}
    for r in m['rows']:
        path=Path(r['directory']);s=read(path/'summary.json');cfg=read(path/'configuration.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and not s['test_used']
        assert canonical(cfg)==canonical(r['configuration'])==canonical(s['configuration'])
        assert cfg['backbone_lr']==6e-5 and cfg['head_lr']==cfg['bridge_lr']==1e-4 and cfg['microbatch']==cfg['effective_batch']==16
        for b in cfg['bridges']:assert b['rho']==r['rho'] and b['M']==32 and b['S']==64
        hashes={n:sha(path/n) for n in FILES};assert hashes==byid[r['id']]['result_hashes']
        records[key(r['seed'],r['arm'],r['rho'])]=dict(r,reused=True,accepted_hashes=hashes,reference_cost=byid[r['id']]['cost'])
    assert len(records)==93
    for seed in SEEDS:
        assert len({read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in records.values() if r['seed']==seed})==1
    write_json(root/'reference_acceptance.json',{'passed':True,'references':list(records.values()),'previous_report_sha256':sha(PREVIOUS/'report/results.json'),'previous_manifest_sha256':sha(PREVIOUS/'manifest.json'),'test_used':False})
    return records,{int(s):b for s,b in m['uncentered_bases'].items()},read(PREVIOUS/'study_summary.json')['cumulative_gpu_minutes']

def manifest(root,records,bases):
    write_json(root/'manifest.json',{'seeds':SEEDS,'rhos':RHOS,'paired_arms':SVD_ARMS,'new_training_jobs':36,'total_results':129,
        'rows':list(records.values()),'diagnostic_reference_bases':{str(s):b for s,b in bases.items()},'compression':'learned_channel',
        'initialization':'same per-seed per-source QR as fixed random controls; encoder and decoder then train independently','test_used':False})

def preflight(root,p,records,bases):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    try:
        jobs=[]
        for arm in SVD_ARMS:
            cfg=configuration(records[key(3416,arm,.25)]);cfg.update(profile=True,save_profile_checkpoint=True)
            jobs.append({'id':'profile_channel_'+arm,'config':cfg})
        profiles=c.run_jobs(jobs);assert len(profiles)==4 and all(v['passed'] for v in profiles.values())
        jobs=[diagnostic_job({'id':n,'directory':str(pre/n)},bases[3416],True,'profile_selected.pt') for n in profiles]
        for j in jobs:j['config'].update(probe_count=128,energy_limit=None)
        ds=c.run_jobs(jobs);assert len(ds)==4 and all(v['passed'] for v in ds.values())
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
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Already started'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_previous_queue','pid':os.getpid(),'source_commit':commit,'predecessor':str(PREVIOUS),'new_training_jobs':36})
        while True:
            previous=read(PREVIOUS/'queue_status.json')
            if previous['state']=='needs_attention':raise RuntimeError('Predecessor needs attention; do not bypass it')
            if previous['state']=='complete':
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            time.sleep(10)
        records,bases,prior=accept_previous(root)
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
            'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','data':DATA,'seeds':SEEDS,'rhos':RHOS,
            'new_stage_two_jobs':36,'new_training_jobs':36,'new_pretraining_jobs':0,'basis_fit_jobs':0,'diagnostic_jobs':36,'reused_results':93,
            'total_stage_two_results':129,'predecessor':str(PREVIOUS),'compression':'learned_channel','encoder_decoder_tied':False,
            'initialization':'same QR control; independent trainable encoder Q^T and decoder Q; zero mixer',
            'convergence':dict(DEFAULT_POLICY),'test_used':False}
        verify(p,records);write_json(root/'plan_protocol.json',p);manifest(root,records,bases)
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit,'new_training_jobs':36})
        cost=preflight(root,p,records,bases)
        for seed,arm,rho in matrix():
            reference=records[key(seed,arm,rho)];a='channel_'+arm;name=identifier(seed,a,rho)
            records[key(seed,a,rho)]={'id':name,'seed':seed,'arm':a,'rho':rho,'paired_arm':arm,'paired_id':reference['id'],
                'paired_directory':reference['directory'],'directory':str(root/name),'reused':False,'configuration':configuration(reference)}
        p=dict(p,prior_gpu_minutes=prior+cost,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'));verify(p,records)
        if args.deploy_repo:
            repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(Path.cwd()),'main'],cwd=repo,check=True);subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p);manifest(root,records,bases)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'planned_new_training_jobs':36,'gpu_preflight_passed':True})
            for seed in SEEDS:
                for rho in RHOS:
                    verify(p,records);rs=[records[key(seed,'channel_'+a,rho)] for a in SVD_ARMS];c.phase=f'seed{seed}_rho{rho}_channel_training'
                    results=c.run_jobs([{'id':r['id'],'config':r['configuration']} for r in rs]);assert len(results)==4 and all(d['converged_by_policy'] for d in results.values()),'Epoch cap; needs attention'
                    for r in rs:
                        a=read(Path(r['directory'])/'summary.json');b=read(Path(r['paired_directory'])/'summary.json')
                        assert a['initial_native_sha256']==b['initial_native_sha256'] and a['parent_checkpoints']==b['parent_checkpoints']
                    c.phase=f'seed{seed}_rho{rho}_channel_diagnostics'
                    ds=c.run_jobs([diagnostic_job(r,bases[seed]) for r in rs]);assert len(ds)==4 and all(v['passed'] for v in ds.values())
                    write_json(root/'partial_summary.json',{'completed_channel_trials':sum((Path(r['directory'])/'summary.json').exists() for r in records.values() if not r['reused']),'test_used':False})
            verify(p,records);c.status('complete')
        except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
        from scripts.report_learned_channel_study import build
        build(root);assert read(root/'report/report_status.json')['state']=='complete'
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'report_complete':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deploy-repo');args=p.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:
        write_json(Path(args.root)/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
