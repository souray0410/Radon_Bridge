"""Register structural atlas choices without consulting performance values."""
import argparse
import json
from pathlib import Path
import shutil
from radonbridge.artifacts import sha256, resolve
from scripts.geometry_evidence import read, write


def select(refs,models):
    chosen=[]
    for seed in (3416,3417,3418):
        for arm,ranks in [('svd_radon',(16,32,64)),('qr_radon',(16,)),('svd_resample',(16,))]:
            for r in ranks:
                matches=[x for x in refs if x['source_group']=='history_final213' and x['seed']==seed and x['arm']==arm and x['rho']==r/256]
                if len(matches)!=1:raise ValueError(('Ambiguous structural reference',seed,arm,r,len(matches)))
                chosen.append((matches[0],dict(arm=arm,seed=seed,r=r,stages=[3])))
        for stages in ([2,3],[2,3,4]):
            matches=[x for x in refs if x['source_group']=='multidepth96' and x['seed']==seed
                and (x.get('structure') or {}).get('stages')==stages and x['structure']['mode']=='radon'
                and x['structure']['r']==16 and models[x['model_view_id']]['configuration']['backbone_lr']==6e-5]
            if len(matches)!=1:raise ValueError(('Ambiguous multidepth reference',seed,stages,len(matches)))
            chosen.append((matches[0],dict(arm='svd_multidepth',seed=seed,r=16,stages=stages)))
    if len(chosen)!=21 or len({x[0]['model_view_id'] for x in chosen})!=21:raise ValueError('Expected 21 unique checkpoints')
    return chosen


def build(source_lock,after,out):
    if out.exists():raise ValueError('Use a new unique lock directory')
    refs=read(source_lock/'model_reference_mapping.json');oldjobs=read(source_lock/'jobs.json')
    models={j['model']['model_view_id']:j['model'] for j in oldjobs if j['kind']=='model'}
    views={v['view_id']:v for v in read(source_lock/'model_views.json')}
    parents={m['seed']:m for m in models.values() if m.get('factory')=='PilotGraph_independent_parent_pair'}
    chosen=select(refs,models);jobs=[]
    def predictions(mid):
        v=views[mid];entry=read(after/'test/jobs'/v['job_id']/'acceptance.json');summary=Path(entry['summary_path'])
        if sha256(summary)!=entry['summary_sha256']:raise ValueError('Accepted test summary changed')
        s=read(summary)
        if s['state']!='accepted' or s['split']!='test':raise ValueError('Test model not accepted')
        result=dict(validation=str(resolve(v['development_prediction'])),test=str(summary.parent/v['file']))
        if sha256(result['test'])!=s['prediction_files'][v['file']]:raise ValueError('Accepted test predictions changed')
        return result
    for ref,display in chosen:
        m=models[ref['model_view_id']];c=m['configuration']
        if c['backbone_lr']!=6e-5 or c.get('native_frozen') or c.get('freeze_native'):raise ValueError('Wrong training regime')
        parent=parents[display['seed']]
        if {k:v['sha256'] for k,v in c['parent_checkpoints'].items()}!={k:v['sha256'] for k,v in parent['parent_checkpoints'].items()}:
            raise ValueError('Different parent lineage')
        for b in c['bridges']:
            if b['M']!=32 or b['S']!=64 or b.get('kernel_size',3)!=3:raise ValueError('Unexpected geometry')
            for basis in b['basis_files'].values():
                if sha256(basis['path'])!=basis['sha256']:raise ValueError('Basis changed')
        expected=dict(constructed_initial=predictions(parent['model_view_id']),selected=predictions(m['model_view_id']))
        jobs.append(dict(job_id='sinogram_'+m['model_view_id'],kind='sinogram_atlas',model=m,display=display,
            source_reference=ref['reference_id'],expected_predictions=expected,
            expected_prediction_sha256={phase:{s:sha256(p) for s,p in paths.items()} for phase,paths in expected.items()}))
    out.mkdir(parents=True,mode=0o700)
    write(out/'jobs.json',jobs)
    shutil.copyfile('experiments/geometry_mechanism/SINOGRAM_ATLAS.zh-CN.md',out/'protocol.zh-CN.md')
    source=read(source_lock/'candidate_lock.json')
    data=dict(train_development=source['development_data_directory'],test=source['data_directory'])
    write(out/'candidate_lock.json',dict(version='sinogram_atlas_v1',after_run=str(after),source_test_lock=str(source_lock),
        source_test_lock_sha256=sha256(source_lock/'candidate_lock.json'),files={f:sha256(out/f) for f in ('jobs.json','protocol.zh-CN.md')},
        data_directories=data,data_selected_sha256={k:sha256(Path(p)/'selected.csv') for k,p in data.items()},
        checkpoints=21,training_tasks=0,phases=['constructed_initial','selected'],cohorts=dict(train=1264,validation=296,test=290),
        examples='first16 sorted cache order; first4 figures; rank channels0,1,2; both eyes',
        selection_uses_performance=False,post_hoc_descriptive=True,primary_hypotheses_unchanged=True))
    print(json.dumps(dict(lock=str(out),checkpoints=len(jobs),training_tasks=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-lock','after','output'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();build(a.source_lock,a.after,a.output)
