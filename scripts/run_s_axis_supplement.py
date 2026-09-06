"""Queue exactly 18 S-axis controls after the 54-task fusion benchmark."""
import argparse,copy,fcntl,os,subprocess,time,traceback,sys,shutil
from pathlib import Path
from radonbridge.experiment import write_json
from radonbridge.s_axis import fixed_permutation
from scripts.run_task_fusion_benchmark import live_workers,controller
from scripts.run_integer_experiment import DEFAULT_POLICY,source_hashes
from scripts.queue_fixed_svd_study import read,sha,head,clean
from scripts.run_five_seed_study import canonical,diagnostic_job

PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_22_42_53')
PREDECESSOR=Path('/data/mengh/RadonBridge/runs/2026_09_06_08_37_46')
SEEDS=[3416,3417,3418];RHOS=[1/16,1/8,1/4];BASES=['svd','qr']


def accept_references():
    assert read(PREVIOUS/'queue_status.json')['state']=='complete'
    manifest=read(PREVIOUS/'manifest.json')
    rows=[r for r in manifest['rows'] if r['arm'] in ('no_bridge','svd_radon','qr_radon')]
    assert len(rows)==21 and {(r['seed'],r['arm'],r['rho']) for r in rows if r['arm']!='no_bridge'}=={(s,b+'_radon',rho) for s in SEEDS for b in BASES for rho in RHOS}
    for r in rows:
        path=Path(r['directory']);s=read(path/'summary.json');cfg=s['configuration']
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
        assert canonical(cfg)==canonical(r['configuration']) and cfg['convergence']==DEFAULT_POLICY
        assert cfg['microbatch']==cfg['effective_batch']==16 and cfg['backbone_lr']==6e-5
        assert not cfg.get('task_fusion') and cfg['head_lr']==cfg['bridge_lr']==1e-4
        assert all(sha(path/n)==h for n,h in r['accepted_hashes'].items())
        for value in s['parent_checkpoints'].values():assert sha(Path(value['path']))==value['sha256']
        for bridge in cfg['bridges']:
            assert bridge['M']==32 and bridge['S']==64 and bridge['mode']=='radon'
            for value in bridge['basis_files'].values():assert sha(Path(value['path']))==value['sha256']
    for seed in SEEDS:
        group=[r for r in rows if r['seed']==seed]
        assert len({read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in group})==1
        assert all(r['configuration']['parent_checkpoints']==group[0]['configuration']['parent_checkpoints'] for r in group)
    return rows,{int(k):v for k,v in manifest['diagnostic_reference_bases'].items()}


def new_rows(references):
    rows=[];P=fixed_permutation(64)['permutation']
    for seed in SEEDS:
        for basis in BASES:
            for rho in RHOS:
                ref=next(r for r in references if (r['seed'],r['arm'],r['rho'])==(seed,basis+'_radon',rho))
                cfg=copy.deepcopy(canonical(ref['configuration']));cfg['bridges'][0]['s_axis_permutation']=P
                rows.append(dict(id=f'seed{seed}_{basis}_rho1_{round(1/rho)}_s_permuted',seed=seed,basis=basis,arm=basis+'_s_permuted',rho=rho,
                                 configuration=cfg,reference_id=ref['id'],directory=None,attempts=[],diagnostic=None))
    return rows


def accept_fusion():
    state=read(PREDECESSOR/'queue_status.json');assert state['state']=='complete' and state['report_complete']
    rows=read(PREDECESSOR/'manifest.json')['rows'];assert len(rows)==54
    for r in rows:
        path=Path(r['directory']);s=read(path/'summary.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
        assert all(sha(path/n)==h for n,h in r['accepted_hashes'].items())
    return dict(passed=True,directory=str(PREDECESSOR),manifest_sha256=sha(PREDECESSOR/'manifest.json'),results=54,
                cumulative_gpu_minutes=read(PREDECESSOR/'study_summary.json')['cumulative_gpu_minutes'])


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commit=head(Path.cwd());assert clean(Path.cwd()),'Committed source required'
        signature=dict(source_commit=commit,source_hashes=source_hashes(),protocol_document_sha256=sha(Path('experiments/s_axis_supplement/PROTOCOL.zh-CN.md')))
        if (root/'source_acceptance.json').exists():assert signature==read(root/'source_acceptance.json')
        else:write_json(root/'source_acceptance.json',signature)
        if (root/'queue_status.json').exists() and read(root/'queue_status.json')['state']=='complete':return
        references,bases=accept_references();write_json(root/'reference_acceptance.json',dict(passed=True,rows=references,test_used=False))
        expected=new_rows(references)
        rows=read(root/'manifest.json')['rows'] if (root/'manifest.json').exists() else expected
        assert len(rows)==18
        for actual,e in zip(rows,expected):
            for k in ('id','seed','basis','rho','configuration','reference_id'):assert actual[k]==e[k]
        def manifest():write_json(root/'manifest.json',dict(rows=rows,references=references,new_training_jobs=18,seeds=SEEDS,rhos=RHOS,test_used=False))
        manifest();write_json(root/'s_axis_permutation.json',fixed_permutation(64))
        # Reuse the already executed CPU probe; no GPU, training, or data access.
        acceptance=Path('experiments/s_axis_supplement/acceptance')
        assert read(acceptance/'cpu_acceptance.json')['passed']
        (root/'report').mkdir(exist_ok=True)
        shutil.copy2(acceptance/'synthetic_operator_probes.json',root/'report/synthetic_operator_probes.json')
        shutil.copy2(acceptance/'cpu_acceptance.json',root/'cpu_acceptance.json')
        shutil.copy2('experiments/s_axis_supplement/PROTOCOL.zh-CN.md',root/'PROTOCOL.zh-CN.md')
        write_json(root/'queue_status.json',dict(state='waiting_for_54_and_report',pid=os.getpid(),source_commit=commit,predecessor=str(PREDECESSOR),total=18,completed=sum(bool(r.get('accepted_hashes')) for r in rows)))
        while read(PREDECESSOR/'queue_status.json')['state']!='complete':time.sleep(30)
        predecessor=accept_fusion();write_json(root/'predecessor_acceptance.json',predecessor)
        with (root.parent.parent/'.active.lock').open('a') as project:
            write_json(root/'queue_status.json',dict(state='waiting_for_project_lock',pid=os.getpid(),source_commit=commit,total=18))
            while True:
                try:fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(30)
            assert not live_workers(root),'Surviving own workers require inspection'
            p=dict(schema='two_stage_ratio_convergence_v3',review_status='approved',source_commit=commit,accepted_source_hashes=source_hashes(),
                   gpu_time_policy='unlimited_until_convergence',max_gpu_minutes=None,prior_gpu_minutes=predecessor['cumulative_gpu_minutes'],gpu_indices=[1,0],
                   min_free_gpu_mib=12288,project_memory_limit_mib=10240,convergence=DEFAULT_POLICY,microbatch=16,effective_batch=16,seeds=SEEDS,rhos=RHOS,
                   backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,new_training_jobs=18,selection_metric='mean_branch_macro_f1',loss='CE_CFP + CE_OCT',
                   fixed_hypothesis_count=3,s_axis_permutation=fixed_permutation(64),test_used=False,protocol_document_sha256=signature['protocol_document_sha256'])
            if (root/'protocol.json').exists():assert p==read(root/'protocol.json')
            else:write_json(root/'protocol.json',p)
            if not (root/'gpu_acceptance.json').exists():
                pre=root/f'preflight_{1+len(list(root.glob("preflight_*"))):03d}';c=controller(pre,p,'preflight')
                try:
                    largest=[r for r in rows if r['seed']==3416 and r['rho']==.25]
                    jobs=[dict(id='profile_'+r['basis'],config=dict(r['configuration'],profile=True,save_profile_checkpoint=True)) for r in largest]
                    profiles=c.run_jobs(jobs);assert len(profiles)==2 and all(v['passed'] and v['peak_reserved_mib']<=10240 for v in profiles.values())
                    ds=c.run_jobs([diagnostic_job(dict(id='profile_'+r['basis'],directory=str(pre/('profile_'+r['basis']))),bases[3416],True,'profile_selected.pt') for r in largest])
                    assert len(ds)==2 and all(v['passed'] for v in ds.values()) and all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
                    write_json(root/'gpu_acceptance.json',dict(passed=True,profiles=profiles,diagnostics=ds,source_hashes=source_hashes(),ledger=c.ledger));c.status('complete')
                finally:c.shutdown()
            assert read(root/'gpu_acceptance.json')['source_hashes']==source_hashes()
            production=Path('/home/mengh/RadonBridge');assert clean(production)
            subprocess.run(['git','fetch',str(Path.cwd()),commit],cwd=production,check=True)
            subprocess.run(['git','merge','--ff-only',commit],cwd=production,check=True)
            pending=[]
            for r in rows:
                path=Path(r['directory']) if r['directory'] else None
                if path and (path/'summary.json').exists():
                    s=read(path/'summary.json');assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau','Nonplateau needs attention'
                    assert canonical(s['configuration'])==canonical(r['configuration']) and s['test_used'] is False
                    hashes={n:sha(path/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']}
                    if r.get('accepted_hashes'):assert hashes==r['accepted_hashes']
                    r['accepted_hashes']=hashes
                else:pending.append(r)
            manifest()
            if pending:
                segment=root/f'attempt_{1+len(list(root.glob("attempt_*"))):03d}';c=controller(segment,p,'run')
                try:
                    for r in pending:r['directory']=str(segment/r['id']);r['attempts'].append(r['directory'])
                    manifest()
                    for i in range(0,len(pending),2):
                        write_json(root/'queue_status.json',dict(state='running',pid=os.getpid(),source_commit=commit,total=18,completed=sum(bool(r.get('accepted_hashes')) for r in rows)))
                        batch=pending[i:i+2];results=c.run_jobs([dict(id=r['id'],config=r['configuration']) for r in batch])
                        for r in batch:
                            s=results[r['id']];assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau','Plateau required'
                            r['accepted_hashes']={n:sha(Path(r['directory'])/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']}
                        manifest()
                    c.status('complete')
                finally:c.shutdown()
            missing=[]
            for r in rows:
                if r.get('diagnostic') and (Path(r['diagnostic'])/'summary.json').exists():
                    d=read(Path(r['diagnostic'])/'summary.json');assert d['passed'] and d['selected_sha256']==r['accepted_hashes']['selected.pt']
                else:missing.append(r)
            if missing:
                segment=root/f'diagnostics_{1+len(list(root.glob("diagnostics_*"))):03d}';c=controller(segment,p,'diagnostics')
                try:
                    for r in missing:r['diagnostic']=str(segment/('diagnostic_'+r['id']))
                    manifest();write_json(root/'queue_status.json',dict(state='diagnostics',pid=os.getpid(),total=18,completed=18))
                    for i in range(0,len(missing),2):
                        ds=c.run_jobs([diagnostic_job(r,bases[r['seed']]) for r in missing[i:i+2]])
                        assert all(d['passed'] for d in ds.values())
                    c.status('complete')
                finally:c.shutdown()
        # Release GPU project lock before CPU-only synthesis, statistics and report.
        cost=sum(sum(j['gpu_seconds'] for j in read(path)['jobs'])/60 for path in root.glob('*/ledger.json'))
        write_json(root/'study_summary.json',dict(new_gpu_minutes=cost,cumulative_gpu_minutes=p['prior_gpu_minutes']+cost,training_jobs=18,test_used=False))
        write_json(root/'queue_status.json',dict(state='reporting',pid=os.getpid(),total=18,completed=18))
        from scripts.validate_s_axis_low_cost import build as synthetic
        if not (root/'report/synthetic_operator_probes.json').exists():synthetic(root/'report')
        from scripts.report_s_axis_supplement import build
        build(root)
        write_json(root/'queue_status.json',dict(state='complete',pid=os.getpid(),source_commit=commit,total=18,completed=18,report_complete=True,claim_inventory_locked=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Duplicate queue owner blocked')
    except BaseException:
        root=Path(args.root)
        if root.exists():write_json(root/'queue_status.json',dict(state='needs_attention',pid=os.getpid(),error=traceback.format_exc()))
        raise
