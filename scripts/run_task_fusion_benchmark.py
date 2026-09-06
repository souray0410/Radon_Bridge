"""Locked, restartable task-level comparison; no historical training overwrite."""
import argparse, copy, fcntl, hashlib, os, subprocess, time, traceback
from pathlib import Path
from types import SimpleNamespace
from radonbridge.experiment import write_json
from scripts.run_integer_experiment import Controller, DEFAULT_POLICY, source_hashes
from scripts.queue_fixed_svd_study import read, sha, head, clean
from scripts.run_five_seed_study import canonical, DATA

SEEDS=[3416,3417,3418]
LRS=[3e-5,6e-5]
ARMS=['concat_mlp','gated_mil','svd_rho1_16','svd_rho1_8','svd_rho1_4','mmtm_r4','mmtm_r8','attention_d128','attention_d256']
PREVIOUS=Path('/data/mengh/RadonBridge/runs/2026_09_05_22_42_53')


def matrix():
    return [(s,lr,a) for s in SEEDS for lr in LRS for a in ARMS]


def configuration(seed, lr, arm, references):
    ref=next(r for r in references if r['seed']==seed and r['arm']=='svd_radon' and r['rho']==.125)
    cfg=copy.deepcopy(canonical(ref['configuration']))
    cfg.update(backbone_lr=lr,task_fusion=dict(pooling='gated_mil' if arm=='gated_mil' else 'mean',hidden_dimension=256,attention_dimension=128),selection_metric='fusion_macro_f1')
    nodes=['cfp_stage3','oct_stage3']
    if arm in ('concat_mlp','gated_mil'):cfg['bridges']=[]
    elif arm.startswith('svd_'):cfg['bridges'][0]['rho']=1/int(arm.rsplit('_',1)[1])
    elif arm.startswith('mmtm_'):cfg['bridges']=[dict(nodes=nodes,family='mmtm',reduction_ratio=int(arm.split('r')[-1]))]
    elif arm.startswith('attention_'):cfg['bridges']=[dict(nodes=nodes,family='cross_attention',attention_dimension=int(arm.split('d')[-1]),heads=4)]
    else:raise ValueError(arm)
    return cfg


def accept_references(previous):
    assert read(previous/'queue_status.json')['state']=='complete'
    old=read(previous/'manifest.json');rows=[r for r in old['rows'] if r['arm'] in ('no_bridge','svd_radon')]
    assert len(rows)==12
    for r in rows:
        path=Path(r['directory']);s=read(path/'summary.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
        assert canonical(s['configuration'])==canonical(r['configuration'])
        assert all(sha(path/n)==h for n,h in r['accepted_hashes'].items())
        assert s['configuration']['convergence']==DEFAULT_POLICY
        assert s['configuration']['microbatch']==16 and s['configuration']['effective_batch']==16
        for value in s['parent_checkpoints'].values():assert sha(Path(value['path']))==value['sha256']
        for b in s['configuration']['bridges']:
            for value in b.get('basis_files',{}).values():assert sha(Path(value['path']))==value['sha256']
    for seed in SEEDS:
        subset=[r for r in rows if r['seed']==seed]
        assert len({read(Path(r['directory'])/'summary.json')['initial_native_sha256'] for r in subset})==1
        assert all(r['configuration']['parent_checkpoints']==subset[0]['configuration']['parent_checkpoints'] for r in subset)
    return rows


def live_workers(root):
    # Do not race surviving workers after a controller crash. No unrelated kill.
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeDecodeError):continue
        if ' -m radonbridge.' in cmd and str(root) in cmd:found.append(dict(pid=int(p.parent.name),command=cmd))
    return found


def controller(root, p, phase):
    root.mkdir(exist_ok=True);write_json(root/'protocol.json',p)
    c=Controller(SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase=phase,data=DATA))
    c.commit=p['source_commit'];return c


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(root.parent.parent/'.active.lock').open('a') as project:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commit=head(Path.cwd());assert clean(Path.cwd()),'Committed source required'
        if (root/'queue_status.json').exists() and read(root/'queue_status.json')['state']=='complete':return
        if (root/'protocol.json').exists():
            p=read(root/'protocol.json');assert p['source_commit']==commit and p['accepted_source_hashes']==source_hashes()
        else:
            p={'schema':'two_stage_ratio_convergence_v3','review_status':'approved','source_commit':commit,'accepted_source_hashes':source_hashes(),
               'gpu_time_policy':'unlimited_until_convergence','max_gpu_minutes':None,'prior_gpu_minutes':read(PREVIOUS/'study_summary.json')['cumulative_gpu_minutes'],
               'gpu_indices':[1,0],'min_free_gpu_mib':12288,'project_memory_limit_mib':10240,'convergence':DEFAULT_POLICY,'microbatch':16,'effective_batch':16,
               'seeds':SEEDS,'backbone_lrs':LRS,'head_lr':1e-4,'bridge_lr':1e-4,'arms':ARMS,'new_training_jobs':54,
               'selection_metric':'fusion_macro_f1','loss':'CE_CFP + CE_OCT + CE_fusion','fixed_hypothesis_count':36,'test_used':False,
               'prior_run':str(PREVIOUS),'prior_manifest_sha256':sha(PREVIOUS/'manifest.json'),'protocol_document_sha256':sha(Path('experiments/task_fusion_benchmark/PROTOCOL.zh-CN.md'))}
            write_json(root/'protocol.json',p)
        write_json(root/'queue_status.json',dict(state='waiting_for_project_lock',pid=os.getpid(),source_commit=commit,total=54))
        while True:
            try:fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        assert not live_workers(root),'Surviving own workers require inspection before restart'
        references=accept_references(PREVIOUS)
        write_json(root/'reference_acceptance.json',dict(passed=True,references=references,historical_training_reused=False,parents_and_SVD_bases_reused=True,test_used=False))
        if (root/'manifest.json').exists():rows=read(root/'manifest.json')['rows']
        else:
            rows=[dict(id=f'seed{s}_bb{lr:.0e}_{a}',seed=s,backbone_lr=lr,arm=a,configuration=configuration(s,lr,a,references),directory=None,attempts=[]) for s,lr,a in matrix()]
        assert len(rows)==54 and [(r['seed'],r['backbone_lr'],r['arm']) for r in rows]==matrix()
        for r in rows:assert r['configuration']==configuration(r['seed'],r['backbone_lr'],r['arm'],references)
        def manifest():write_json(root/'manifest.json',dict(rows=rows,total_training_jobs=54,seeds=SEEDS,backbone_lrs=LRS,arms=ARMS,test_used=False))
        manifest()
        if not (root/'gpu_acceptance.json').exists():
            pre=root/f'preflight_{1+len(list(root.glob("preflight_*"))):03d}';c=controller(pre,p,'preflight')
            try:
                jobs=[dict(id='profile_'+a,config=dict(configuration(3416,6e-5,a,references),profile=True)) for a in ARMS]
                results=c.run_jobs(jobs)
                assert len(results)==9 and all(x['passed'] and x['peak_reserved_mib']<=10240 for x in results.values())
                assert all(j['sampled_peak_process_mib']<=10240 for j in c.ledger['jobs'])
                write_json(root/'gpu_acceptance.json',dict(passed=True,profiles=results,source_commit=commit,source_hashes=source_hashes(),ledger=c.ledger))
                c.status('complete')
            finally:c.shutdown()
        assert read(root/'gpu_acceptance.json')['source_hashes']==source_hashes()
        # Deploy only the tested commit while holding the project lock.
        production=Path('/home/mengh/RadonBridge')
        assert clean(production)
        subprocess.run(['git','fetch',str(Path.cwd()),commit],cwd=production,check=True)
        subprocess.run(['git','merge','--ff-only',commit],cwd=production,check=True)
        pending=[]
        for r in rows:
            path=Path(r['directory']) if r['directory'] else None
            if path and (path/'summary.json').exists():
                s=read(path/'summary.json')
                assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau', 'Nonplateau result needs attention'
                assert canonical(s['configuration'])==canonical(r['configuration']) and s['test_used'] is False
                hashes={n:sha(path/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']}
                if r.get('accepted_hashes'):assert hashes==r['accepted_hashes']
                r['accepted_hashes']=hashes
            else:pending.append(r)
        manifest()
        if pending:
            segment=root/f'attempt_{1+len(list(root.glob("attempt_*"))):03d}';c=controller(segment,p,'run')
            try:
                for r in pending:
                    r['directory']=str(segment/r['id']);r['attempts'].append(r['directory'])
                manifest();write_json(root/'queue_status.json',dict(state='running',pid=os.getpid(),source_commit=commit,total=54,completed=54-len(pending),active_segment=str(segment)))
                # Small batches retain acceptance and progress after each pair.
                for i in range(0,len(pending),2):
                    batch=pending[i:i+2];results=c.run_jobs([dict(id=r['id'],config=r['configuration']) for r in batch])
                    for r in batch:
                        s=results[r['id']]
                        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau','Plateau required'
                        path=Path(r['directory']);r['accepted_hashes']={n:sha(path/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']}
                    manifest();write_json(root/'queue_status.json',dict(state='running',pid=os.getpid(),source_commit=commit,total=54,completed=sum(bool(r.get('accepted_hashes')) for r in rows),active_segment=str(segment)))
                c.status('complete')
            finally:c.shutdown()
        cost=sum(sum(x['gpu_seconds'] for x in read(path)['jobs'])/60 for path in root.glob('*/ledger.json'))
        write_json(root/'study_summary.json',dict(new_gpu_minutes=cost,cumulative_gpu_minutes=p['prior_gpu_minutes']+cost,training_jobs=54,test_used=False))
        write_json(root/'queue_status.json',dict(state='reporting',pid=os.getpid(),source_commit=commit,total=54,completed=54))
        from scripts.report_task_fusion_benchmark import build
        build(root)
        write_json(root/'queue_status.json',dict(state='complete',pid=os.getpid(),source_commit=commit,total=54,completed=54,report_complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    try:run(args)
    except BlockingIOError:raise SystemExit('Project queue already has an owner; duplicate blocked')
    except BaseException:
        root=Path(args.root)
        if root.exists():write_json(root/'queue_status.json',dict(state='needs_attention',pid=os.getpid(),error=traceback.format_exc()))
        raise
