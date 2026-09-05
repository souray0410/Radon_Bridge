"""Two authorized studies: 18 QR controls and 9 shared-width trainings, kept separate."""
import argparse,copy,fcntl,json,os,subprocess,time,traceback,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from radonbridge.experiment import write_json
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.run_five_seed_study import canonical,verify,diagnostic_job,DATA
from scripts.queue_fixed_svd_study import read,sha,head,clean
from scripts.run_centered_study import FILES

PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_18_12_57')
QR_ARMS=['qr_self','qr_scrambled']
NESTED_ARMS=['nested_learned_channel','nested_svd','nested_qr']
REF_ARMS={'nested_learned_channel':'channel_svd_radon','nested_svd':'svd_radon','nested_qr':'qr_radon'}


def matrix():
    return [(s,a,r) for s in SEEDS for r in RHOS for a in QR_ARMS]+[(s,a,.25) for s in SEEDS for a in NESTED_ARMS]


def configuration(records,seed,arm,rho):
    reference='qr_radon' if arm in QR_ARMS else REF_ARMS[arm]
    cfg=copy.deepcopy(canonical(records[key(seed,reference,rho)]['configuration']));b=cfg['bridges'][0]
    if arm in QR_ARMS:b['mode']='self' if arm=='qr_self' else 'scrambled'
    else:
        b['nested_rhos']=list(RHOS)
        assert b['compression']=={'nested_learned_channel':'learned_channel','nested_svd':'fixed_svd_channel','nested_qr':'fixed_random_orthogonal_channel'}[arm]
    return cfg


def accept_previous(root,previous):
    assert read(previous/'queue_status.json')['state']=='complete'
    manifest=read(previous/'manifest.json');report=read(previous/'report/results.json');reported={r['id']:r for r in report['rows']}
    assert len(manifest['rows'])==len(reported)==186
    records={};identity=None
    for r in manifest['rows']:
        path=Path(r['directory']);s=read(path/'summary.json');cfg=read(path/'configuration.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and not s['test_used']
        assert canonical(cfg)==canonical(s['configuration'])==canonical(r['configuration'])
        assert cfg['microbatch']==cfg['effective_batch']==16 and cfg['convergence']==DEFAULT_POLICY
        assert cfg['backbone_lr']==6e-5 and cfg['head_lr']==cfg['bridge_lr']==1e-4
        hashes={n:sha(path/n) for n in FILES};assert hashes==reported[r['id']]['result_hashes']
        with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
            current=(z['ids'].copy(),z['y'].copy());assert len(current[0])==296
            if identity is None:identity=current
            else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
        records[key(r['seed'],r['arm'],r['rho'])]=dict(r,reused=True,accepted_hashes=hashes,reference_cost=reported[r['id']]['cost'])
    for seed in SEEDS:
        rs=[r for r in records.values() if r['seed']==seed]
        assert len({read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in rs})==1
        assert all(r['configuration']['parent_checkpoints']==rs[0]['configuration']['parent_checkpoints'] for r in rs)
    # Verify existing SVD permutations against the exact constructor rule before adding QR controls.
    permutations=[]
    for seed in SEEDS:
        for rho in RHOS:
            ref=records[key(seed,'svd_scrambled',rho)];state=torch.load(Path(ref['directory'])/'selected.pt',map_location='cpu',weights_only=False)['model']['bridge_0_exchange']
            for i,(node,shape) in enumerate([('cfp_stage3',(14,14)),('oct_stage3',(8,6,6))]):
                expected=np.random.default_rng(seed+1009*i+len(shape)).permutation(np.prod(shape))
                assert np.array_equal(state[f'projectors.{i}.permutation'].numpy(),expected)
                assert np.array_equal(state[f'projectors.{i}.inverse_permutation'].numpy(),np.argsort(expected))
                permutations.append({'seed':seed,'rho':rho,'source':node,'permutation_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'reference_checkpoint_sha256':ref['accepted_hashes']['selected.pt']})
            del state
    write_json(root/'reference_acceptance.json',{'passed':True,'count':186,'references':list(records.values()),'previous_report_sha256':sha(previous/'report/results.json'),'permutation_pairing':permutations,'test_used':False})
    return records,{int(k):v for k,v in manifest['diagnostic_reference_bases'].items()},read(previous/'study_summary.json')['cumulative_gpu_minutes']


def view_records(record):
    return [dict(id=record['id']+'_'+str(round(1/v['rho'])),directory=v['directory'],rho=v['rho']) for v in read(Path(record['directory'])/'width_exports.json')]


def diagnostics_for(records,bases,preflight=False):
    jobs=[]
    for r in records:
        nested=r['arm'] in NESTED_ARMS
        targets=view_records(r) if nested else [r]
        for view in targets:
            j=diagnostic_job(view,bases[r['seed']],preflight,'selected.pt' if nested or not preflight else 'profile_selected.pt')
            j['config'].update(probe_count=128,energy_limit=None)
            if nested and preflight:j['config']['profile_export_identity']=True
            jobs.append(j)
    return jobs


def preflight(root,p,records,bases):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    try:
        rs=[]
        for arm in QR_ARMS+NESTED_ARMS:
            cfg=configuration(records,3416,arm,.25);cfg.update(profile=True,save_profile_checkpoint=True)
            rs.append({'id':'profile_'+arm,'arm':arm,'seed':3416,'directory':str(pre/('profile_'+arm)),'configuration':cfg})
        profiles=c.run_jobs([{'id':r['id'],'config':r['configuration']} for r in rs]);assert len(profiles)==5 and all(d['passed'] for d in profiles.values())
        ds=c.run_jobs(diagnostics_for(rs,bases,True));assert len(ds)==11 and all(d['passed'] for d in ds.values())
        assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
        c.status('complete');write_json(root/'gpu_acceptance.json',{'passed':True,'profiles':profiles,'diagnostics':ds,'gpu_minutes':c.used(),'ledger':c.ledger,'source_commit':p['source_commit'],'test_used':False})
        return c.used()
    except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
    finally:c.shutdown()


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True);previous=Path(args.predecessor).resolve()
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Already started; no blind relaunch'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_predecessor_and_project_lock','pid':os.getpid(),'source_commit':commit,'predecessor':str(previous),'new_training_jobs':27})
        while True:
            state=read(previous/'queue_status.json')['state']
            if state=='needs_attention':raise RuntimeError('Predecessor requires recovery; do not bypass its acceptance')
            if state=='complete':
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            time.sleep(10)
        records,bases,prior=accept_previous(root,previous)
        if args.prior_attempt:
            failed=Path(args.prior_attempt).resolve()
            assert read(failed/'queue_status.json')['state']=='needs_attention' and not (failed/'run_source.json').exists()
            old_ledger=read(failed/'preflight/ledger.json');carried=sum(j['gpu_seconds'] for j in old_ledger['jobs'])/60
            prior+=carried
            write_json(root/'prior_attempt_acceptance.json',{'directory':str(failed),'ledger_sha256':sha(failed/'preflight/ledger.json'),'carried_gpu_minutes':carried,'no_formal_training_started':True,'reason':'Align fixed nested channel contractions with legacy einsum kernels; preserve failed export-identity preflight and all costs, no tolerance relaxation.'})
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
           'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','gpu_indices':[1,0],'min_free_gpu_mib':12288,'data':DATA,
           'predecessor':str(previous),'seeds':SEEDS,'rhos':RHOS,'new_training_jobs':27,'qr_control_trainings':18,'joint_width_trainings':9,'joint_width_views':27,
           'historical_trainings':186,'total_distinct_training_configurations':213,'total_evaluation_views':231,'single_width_mechanism_results':204,
           'convergence':dict(DEFAULT_POLICY),'bootstrap_resamples':10000,'comparison_families':{'QR_mechanism':3,'joint_width_training':3},
           'nested_loss':'equal average of three widths and sum of two task CE; sequential forward/backward, one optimizer step per batch16',
           'nested_BN':'each width uses training batch statistics from same starting buffers; average three resulting running updates, increment counter once; not pooled mixture variance',
           'nested_selection':'one common checkpoint, equal mean of six branch-width macro-F1 values; unchanged minimum8/patience6/delta.001/LRpatience3/factor.3/cap60',
           'report_scope':'separate mechanism and training-method studies; no new worst-branch or bandwidth-cost specialty analyses',
           'development_previously_selected':True,'test_used':False}
        verify(p,records)
        for seed,arm,rho in matrix():
            name=identifier(seed,arm,rho);records[key(seed,arm,rho)]={'id':name,'seed':seed,'arm':arm,'rho':rho,'directory':str(root/name),'reused':False,'configuration':configuration(records,seed,arm,rho)}
        assert len(records)==213
        write_json(root/'manifest.json',{'rows':list(records.values()),'seeds':SEEDS,'rhos':RHOS,'new_training_jobs':27,'diagnostic_reference_bases':bases,'test_used':False})
        write_json(root/'plan_protocol.json',p);verify(p,records)
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit,'new_training_jobs':27})
        cost=preflight(root,p,records,bases);p=dict(p,prior_gpu_minutes=prior+cost,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'));verify(p,records)
        if args.deploy_repo:
            repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(Path.cwd()),'HEAD'],cwd=repo,check=True);subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'new_training_jobs':27})
            for group,arms in [('QR_mechanism',QR_ARMS),('joint_width_training',NESTED_ARMS)]:
                for seed in SEEDS:
                    rs=[r for r in records.values() if not r['reused'] and r['arm'] in arms and r['seed']==seed]
                    verify(p,records);c.phase=group+f'_seed{seed}_training'
                    results=c.run_jobs([{'id':r['id'],'config':r['configuration']} for r in rs]);assert len(results)==len(rs) and all(d['converged_by_policy'] for d in results.values()),'Cap without platform needs attention'
                    native=read(Path(records[key(seed,'no_bridge',None)]['directory'])/'summary.json')
                    for r in rs:
                        d=read(Path(r['directory'])/'summary.json');assert d['initial_native_sha256']==native['initial_native_sha256'] and d['parent_checkpoints']==native['parent_checkpoints']
                    c.phase=group+f'_seed{seed}_diagnostics';ds=c.run_jobs(diagnostics_for(rs,bases));assert all(d['passed'] for d in ds.values())
                    write_json(root/'partial_summary.json',{'completed_new_trainings':sum((Path(r['directory'])/'summary.json').exists() for r in records.values() if not r['reused']),'test_used':False})
            verify(p,records);c.status('complete')
        except BaseException:c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
        from scripts.report_qr_nested_supplement import build
        build(root)
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'separate_reports_complete':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--predecessor',default=str(PREVIOUS));parser.add_argument('--deploy-repo');parser.add_argument('--prior-attempt');args=parser.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:write_json(Path(args.root)/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
