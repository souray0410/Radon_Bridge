"""Accepted five-seed study: one lock, immutable references, preflight then training."""
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
from radonbridge.experiment import write_json
from radonbridge.svd_basis import save_random_basis
from scripts.run_integer_experiment import Controller,DEFAULT_POLICY,source_hashes
from scripts.queue_fixed_svd_study import accept_references,clean,head,sha,read

ARMS=['no_bridge','learned_equal','learned_wide','svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']
OLD=Path('/data/mengh/RadonBridge/runs/2026_09_04_23_54_10')
SVD=Path('/data/mengh/RadonBridge/runs/2026_09_05_01_29_50')
DATA='/data/mengh/RadonBridge/cache/full1264_296'


def base(seed,parents=None):
    return {'seed':seed,'backbone_lr':6e-5 if parents else 3e-5,'head_lr':1e-4,'bridge_lr':1e-4,
            'bridges':[],'microbatch':16,'effective_batch':16,'convergence':dict(DEFAULT_POLICY),
            'training_stage':'communication' if parents else 'independent',**({'parent_checkpoints':parents} if parents else {})}


def configuration(seed,parents,svd,qr,arm):
    cfg=base(seed,parents)
    if arm=='no_bridge':return cfg
    bridge={'nodes':['cfp_stage3','oct_stage3'],'M':32,'S':64,'rho':.25 if arm=='learned_wide' else .125,'mode':'radon'}
    if arm.startswith('svd_'):
        bridge.update(compression='fixed_svd_channel',basis_files=svd,mode={'svd_self':'self','svd_scrambled':'scrambled','svd_resample':'linear_resample'}.get(arm,'radon'))
    elif arm=='qr_radon':bridge.update(compression='fixed_random_orthogonal_channel',basis_files=qr)
    elif arm not in ['learned_equal','learned_wide']:raise ValueError(arm)
    cfg['bridges']=[bridge];return cfg


def canonical(cfg):return {k:v for k,v in cfg.items() if k not in ['source_commit','budget_estimate_epochs']}


def references(root):
    _,_,matched,native,_=accept_references(OLD);svdp=read(SVD/'protocol.json');svdstatus=read(SVD/'status.json');assert svdstatus['state']=='complete' and not svdstatus['active']
    records={};parents={};bases={}
    for seed in [3416,3417]:
        g=next(g for g in matched if g['baseline']['config']['seed']==seed and g['baseline']['config']['backbone_lr']==6e-5)
        parents[seed]=g['baseline']['config']['parent_checkpoints'];bases[seed]=svdp['basis_files'][str(seed)]
        choices={'no_bridge':(OLD,g['baseline']['id']),'learned_equal':(OLD,next(j['id'] for j in g['standards'] if j['config']['bridges'][0]['rho']==.125)),
                 'learned_wide':(OLD,next(j['id'] for j in g['standards'] if j['config']['bridges'][0]['rho']==.25)),
                 'svd_radon':(SVD,f'seed{seed}_bb6e-05_rho1_8_svd')}
        for arm,(directory,name) in choices.items():
            path=directory/name;d=read(path/'summary.json');assert d['converged_by_policy'] and d['state']=='complete' and d['test_used'] is False
            assert canonical(d['configuration'])==canonical(configuration(seed,parents[seed],bases[seed],None,arm))
            assert d['initial_native_sha256']==native[seed] and d['parent_checkpoints']==parents[seed]
            record={'id':f'seed{seed}_{arm}','seed':seed,'arm':arm,'directory':str(path),'reused':True,'configuration':d['configuration'],
                    'accepted_hashes':{name:sha(path/name) for name in ['summary.json','configuration.json','selected_predictions.npz','selected.pt']},
                    'reference_cost':next(j for j in read(directory/'ledger.json')['jobs'] if j['id']==name)}
            records[(seed,arm)]=record
    assert len(records)==8
    write_json(root/'reference_acceptance.json',{'passed':True,'references':list(records.values()),'test_used':False})
    return records,parents,bases,svdstatus['cumulative_gpu_minutes']


def verify(p,records):
    assert head(Path.cwd())==p['source_commit'] and clean(Path.cwd()) and source_hashes()==p['accepted_source_hashes']
    for r in records.values():
        if r['reused']:
            for name,digest in r['accepted_hashes'].items():assert sha(Path(r['directory'])/name)==digest
        cfg=r.get('configuration')
        if cfg:
            for parent in cfg.get('parent_checkpoints',{}).values():assert sha(parent['path'])==parent['sha256']
            for b in cfg['bridges']:
                for key,ref in b.get('basis_files',{}).items():
                    from radonbridge.svd_basis import _load_basis,BASIS_VERSION,QR_VERSION
                    assert sha(ref['path'])==ref['sha256'];_,_,meta=_load_basis(ref['path'],ref['sha256'])
                    assert meta['seed']==r['seed'] and meta['source_key']==key and meta['provenance']['parent_checkpoints']==cfg['parent_checkpoints']
                    assert meta['version']==(QR_VERSION if b['compression']=='fixed_random_orthogonal_channel' else BASIS_VERSION)


def manifest(root,records,bases,qr):
    write_json(root/'manifest.json',{'arms':ARMS,'seeds':list(range(3416,3421)),'rows':[records[k] for k in sorted(records)],
                                  'svd_bases':{str(k):v for k,v in bases.items()},'qr_bases':{str(k):v for k,v in qr.items()},'test_used':False})


def diagnostic_job(record,bases,preflight=False,selected_file='selected.pt'):
    path=Path(record['directory']);return {'id':'diagnostic_'+record['id'],'diagnostic':True,'config':{'trial_directory':str(path),'basis_files':bases,
          'selected_sha256':sha(path/selected_file),'selected_file':selected_file,'preflight':preflight}}


def preflight(root,p,parents,bases):
    pre=root/'preflight';pre.mkdir();write_json(pre/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(pre),protocol=str(pre/'protocol.json'),phase='preflight',data=DATA));c.commit=p['source_commit']
    tick=time.monotonic();qr={key:save_random_basis(ref,root/'random_bases') for key,ref in bases[3416].items()}
    write_json(root/'qr_generation.json',{'3416':{'cpu_wall_seconds':time.monotonic()-tick,'bases':qr}})
    try:
        jobs=[]
        for arm in ['svd_self','svd_scrambled','svd_resample','qr_radon']:
            cfg=configuration(3416,parents[3416],bases[3416],qr,arm);cfg.update(profile=True,save_profile_checkpoint=True)
            jobs.append({'id':'profile_'+arm,'config':cfg})
        profiles=c.run_jobs(jobs);assert all(d['passed'] for d in profiles.values())
        jobs=[diagnostic_job({'id':name,'directory':str(pre/name)},bases[3416],True,'profile_selected.pt') for name in profiles]
        for arm,path in [('learned_wide',OLD/'seed3416_bb6e-05_rho1_4'),('svd_radon',SVD/'seed3416_bb6e-05_rho1_8_svd')]:
            jobs.append(diagnostic_job({'id':'reference_'+arm,'directory':str(path)},bases[3416],True))
        diagnostics=c.run_jobs(jobs);assert all(d['passed'] for d in diagnostics.values())
        assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
        c.status('complete');acceptance={'passed':True,'profiles':profiles,'diagnostics':diagnostics,'source_commit':p['source_commit'],
                'tested_source_hashes':p['accepted_source_hashes'],'ledger':c.ledger,'gpu_minutes':c.used(),'test_used':False}
        write_json(root/'gpu_acceptance.json',acceptance);return qr,c.used()
    except BaseException:
        c.status('needs_attention',reason=traceback.format_exc());raise
    finally:c.shutdown()


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'preflight').exists() and not (root/'run_source.json').exists(),'Run already started; inspect before continuation'
        commit=head(Path.cwd());assert clean(Path.cwd())
        write_json(root/'queue_status.json',{'state':'waiting_for_project_lock','pid':os.getpid(),'source_commit':commit})
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        records,parents,bases,prior=references(root)
        for seed in range(3416,3421):
            for arm in ARMS:records.setdefault((seed,arm),{'id':f'seed{seed}_{arm}','seed':seed,'arm':arm,'directory':str(root/f'seed{seed}_{arm}'),'reused':False,'configuration':None})
        p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
           'prior_gpu_minutes':prior,'max_gpu_minutes':None,'gpu_time_policy':'unlimited_until_convergence','data':DATA,'seeds':list(range(3416,3421)),
           'arms':ARMS,'new_training_jobs':35,'new_stage_two_jobs':32,'reused_results':8,'convergence':dict(DEFAULT_POLICY),'test_used':False}
        verify(p,records);write_json(root/'plan_protocol.json',p);manifest(root,records,bases,{})
        write_json(root/'queue_status.json',{'state':'gpu_preflight','pid':os.getpid(),'source_commit':commit})
        first_qr,pre_minutes=preflight(root,p,parents,bases);qr={3416:first_qr};p=dict(p,prior_gpu_minutes=prior+pre_minutes,gpu_acceptance_sha256=sha(root/'gpu_acceptance.json'))
        verify(p,records)
        if args.deploy_repo:
            candidate=Path.cwd();repo=Path(args.deploy_repo);assert clean(repo)
            subprocess.run(['git','fetch',str(candidate),'main'],cwd=repo,check=True);subprocess.run(['git','merge','--ff-only',commit],cwd=repo,check=True)
            assert head(repo)==commit and clean(repo);os.chdir(repo);verify(p,records)
        write_json(root/'protocol.json',p)
        c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=DATA));c.args.phase='study'  # custom summary includes references and excludes diagnostics
        try:
            write_json(root/'queue_status.json',{'state':'running','pid':os.getpid(),'source_commit':commit,'planned_new_training_jobs':35,'gpu_preflight_passed':True})
            for seed in range(3416,3421):
                verify(p,records)
                if seed not in parents:
                    c.phase=f'pretrain_{seed}';cfg=base(seed)
                    result=c.run_jobs([{'id':f'pretrain_{seed}','config':cfg}])[f'pretrain_{seed}']
                    if not result['converged_by_policy']:raise RuntimeError(f'Pretraining seed{seed} reached epoch cap')
                    parents[seed]=result['modality_checkpoints'];assert set(parents[seed])=={'cfp','oct'}
                    c.phase=f'basis_fit_{seed}'
                    fit=c.run_jobs([{'id':f'basis_seed{seed}','basis_fit':True,'config':{'seed':seed,'parent_checkpoints':parents[seed],'nodes':['cfp_stage3','oct_stage3'],'microbatch':16,'source_commit':commit}}])[f'basis_seed{seed}']
                    assert fit['passed'];bases[seed]=fit['bases']
                if seed not in qr:
                    tick=time.monotonic();qr[seed]={key:save_random_basis(ref,root/'random_bases') for key,ref in bases[seed].items()}
                    timings=read(root/'qr_generation.json');timings[str(seed)]={'cpu_wall_seconds':time.monotonic()-tick,'bases':qr[seed]};write_json(root/'qr_generation.json',timings)
                jobs=[]
                for arm in ARMS:
                    r=records[(seed,arm)]
                    if not r['reused']:
                        r['configuration']=configuration(seed,parents[seed],bases[seed],qr[seed],arm);jobs.append({'id':r['id'],'config':r['configuration']})
                manifest(root,records,bases,qr);verify(p,records);c.phase=f'seed{seed}_matched_training'
                results=c.run_jobs(jobs)
                if not all(d['converged_by_policy'] for d in results.values()):raise RuntimeError('Second-stage epoch cap; comparison incomplete')
                hashes={read(Path(records[(seed,a)]['directory'])/'summary.json')['initial_native_sha256'] for a in ARMS};assert len(hashes)==1
                c.phase=f'seed{seed}_read_only_diagnostics'
                ds=c.run_jobs([diagnostic_job(records[(seed,a)],bases[seed]) for a in ARMS]);assert all(d['passed'] for d in ds.values())
                write_json(root/'partial_summary.json',{'completed_seeds':list(range(3416,seed+1)),'resolved_result_count':len([r for r in records.values() if (Path(r['directory'])/'summary.json').exists()]),'test_used':False})
            verify(p,records);c.status('complete')
        except BaseException:
            c.status('needs_attention',reason=traceback.format_exc());raise
        finally:
            c.shutdown();manifest(root,records,bases,qr)
            write_json(root/'study_summary.json',{'state':read(root/'status.json')['state'],'new_gpu_minutes':c.used(),'cumulative_gpu_minutes':p['prior_gpu_minutes']+c.used(),'test_used':False})
            from scripts.report_five_seed_study import build
            try:build(root)
            except BaseException:write_json(root/'report_failure.json',{'error':traceback.format_exc()});raise
        write_json(root/'queue_status.json',{'state':'complete','pid':os.getpid(),'source_commit':commit,'report_complete':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deploy-repo');a=p.parse_args()
    try:run(a)
    except BlockingIOError:raise SystemExit('Duplicate queue rejected')
    except BaseException:
        root=Path(a.root);root.mkdir(parents=True,exist_ok=True);write_json(root/'queue_status.json',{'state':'needs_attention','error':traceback.format_exc(),'time':time.time()});raise
