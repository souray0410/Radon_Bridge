"""Post-test descriptive audit; no fitting, selection, or change to saved results.

Run only in the authorized data environment. Output contains aggregates only.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, json, statistics, time
import numpy as np
import pandas as pd


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distribution(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75))) if len(a)>1 else dict(n=len(a))
def counts(s):
    c=collections.Counter(str(x) if str(x) else 'missing' for x in s)
    return {**{k:v for k,v in c.items() if v>=10},'categories_below_10_combined':sum(v for v in c.values() if v<10)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);out=a.output
    lock=a.root/'unified_test_lock_v6';c=read(lock/'candidate_lock.json')
    data=Path(c['development_data_directory']);testdata=Path(c['data_directory'])
    frames=[pd.read_csv(data/'selected.csv',dtype=str).fillna(''),pd.read_csv(testdata/'selected.csv',dtype=str).fillna('')]
    # The new test cache only stores test rows; never silently deduplicate overlaps.
    f=pd.concat(frames,ignore_index=True)
    assert len(f)==1850 and not f.participant_id.duplicated().any() and not f.order.duplicated().any()
    assert f.groupby('split').size().to_dict()=={'train':1264,'validation':296,'test':290}
    metadata={}
    for split,g in f.groupby('split'):
        report={'participants':len(g),'labels':g.groupby('label_id').size().to_dict(),
                'age':distribution(pd.to_numeric(g.assessment_age,errors='coerce')),
                'missing_age':int(pd.to_numeric(g.assessment_age,errors='coerce').isna().sum())}
        for k in ('sex','assessment_centre','instance','candidate_status','reference_standard_type','reference_source','phenotype_profile','task_profile'):
            report[k]=counts(g[k])
        report['assessment_year']=counts(pd.to_datetime(g.assessment_date,errors='coerce').dt.year)
        report['case_age']=distribution(pd.to_numeric(g[g.label_id=='1'].assessment_age,errors='coerce'))
        report['control_age']=distribution(pd.to_numeric(g[g.label_id=='0'].assessment_age,errors='coerce'))
        cases=g[g.label_id=='1']; report['case_evidence_tokens']=counts(t for v in cases.evidence_sources for t in v.split(';') if t)
        report['incident_targets_nonempty']=int((g.incident_targets!='').sum())
        report['undated_targets_nonempty']=int((g.undated_targets!='').sum())
        metadata[split]=report
    jobs=read(lock/'jobs.json');views={v['view_id']:v for v in read(lock/'model_views.json')}
    accepted=read(a.root/'unified_test_2026_09_07_20_32_56/test/accepted_jobs.json')
    rows=[];history_rows=[];prediction_groups=collections.defaultdict(lambda:collections.defaultdict(list))
    refs={}
    wanted=('no_communication','mmtm_hidden128','mmtm_hidden256','mmtm_hidden512','mmtm_hidden1024','attention_d256','attention_d1024','radon_M32_S64_k3_r16','radon_M32_S64_k3_r32','linear_resample_M32_S64_k3_r16','linear_resample_M32_S64_k3_r32')
    for j in jobs:
        if j['kind']!='model':continue
        m=j['model'];ref=next((r for r in m['reference_ids'] if r.startswith('current_direct_and_host:branch_') and '_augment_' not in r),None)
        if not ref:continue
        identifier=ref.split(':',1)[1];name=identifier.removeprefix('branch_').split('_lr')[0]
        if name not in wanted:continue
        p=Path(m['checkpoint_path']).parent;s=read(p/'summary.json');h=[json.loads(l) for l in (p/'history.jsonl').read_text().splitlines()]
        refs[identifier]={'summary_sha256':sha(p/'summary.json'),'history_sha256':sha(p/'history.jsonl'),'selected_sha256':m['checkpoint_sha256']}
        row=dict(id=identifier,method=name,seed=m['configuration']['seed'],backbone_lr=m['configuration']['backbone_lr'],epochs=s['epochs_ran'],selected_epoch=s['selection']['joint']['best_epoch'],converged=s['converged_by_policy'],initial_dev=s['initial']['mean_task_macro_f1'],selected_dev=s['selected']['mean_task_macro_f1'],stop_dev=s['stopping_metrics']['mean_task_macro_f1'],stop_train=s['train_stopping_metrics']['mean_task_macro_f1'],first_train_ce=h[0]['train_ce_sum'],last_train_ce=h[-1]['train_ce_sum'],max_trained_dev=max(x['validation']['mean_task_macro_f1'] for x in h),min_trained_dev=min(x['validation']['mean_task_macro_f1'] for x in h),first_dev=h[0]['validation']['mean_task_macro_f1'])
        for b in ('cfp','oct'):
            for stage,key in [('initial','initial'),('stop','stopping_metrics'),('train','train_stopping_metrics')]:
                for metric in ('macro_f1','log_loss','auroc'):row[f'{stage}_{b}_{metric}']=s[key]['tasks'][b][metric]
        for k,v in s['stopping_metrics'].get('delta_over_feature_l2',{}).items():row['stop_delta_'+k]=v
        rates=s.get('initial_learning_rates',h[0]['learning_rates'])
        row['lr_source']='initial_learning_rates' if 'initial_learning_rates' in s else 'epoch1_history'
        row['lr_verified']=all(abs(v-(row['backbone_lr'] if '_stage' in k else 1e-4))<1e-12 for k,v in rates.items())
        rows.append(row)
        for x in h:history_rows.append({'id':identifier,'epoch':x['epoch'],'train_ce':x['train_ce_sum'],'dev_mean':x['validation']['mean_task_macro_f1'],'bridge_lr':x['learning_rates'].get('bridge_0_exchange')})
        ae=accepted[j['job_id']];assert sha(ae['summary_path'])==ae['summary_sha256']
        su=read(ae['summary_path']);assert su['strict_model_load'] and su['parameters_BN_gradients_RNG_preserved']
        paths={'validation':Path(views[m['model_view_id']]['development_prediction']),'test':Path(ae['summary_path']).parent/'predictions.npz'}
        assert sha(paths['test'])==su['prediction_files']['predictions.npz']
        for split,path in paths.items():
            z=np.load(path,allow_pickle=False);fg=f[f.split==split].set_index('order').loc[z['ids'].astype(str)]
            assert np.array_equal(z['y'],fg.label_id.astype(int).to_numpy())
            prediction_groups[(name,split)]['cfp'].append(z['cfp']);prediction_groups[(name,split)]['oct'].append(z['oct'])
            prediction_groups[(name,split)]['ids']=z['ids'];prediction_groups[(name,split)]['y']=z['y']
    assert len(rows)==66 and sum(r['method'].startswith('mmtm') for r in rows)==24
    from radonbridge.metrics import classification_metrics
    metrics=[];subgroups=[];error_overlap=[]
    for (name,split),group in prediction_groups.items():
        fg=f[f.split==split].set_index('order').loc[group['ids'].astype(str)];y=group['y']
        for branch in ('cfp','oct'):
            ps=np.stack(group[branch]);error=(ps.argmax(-1)!=y[None,:]);p=np.clip(ps,1e-7,1-1e-7)
            rr=dict(method=name,split=split,branch=branch,models=len(ps),n=len(y))
            for key in ('macro_f1','accuracy','auroc','log_loss'):
                rr[key]=float(np.mean([classification_metrics(y,x)[key] for x in ps]))
            rr.update(mean_confidence=float(ps.max(-1).mean()),wrong_prediction_confidence=float(ps.max(-1)[error].mean()),brier_binary=float(((ps[:,:,1]-y)**2).mean()),all_six_wrong=int(error.all(0).sum()),at_least_one_wrong=int(error.any(0).sum()))
            metrics.append(rr)
            for field,values in [('sex',fg.sex.to_numpy()),('label',y.astype(str)),('age_band',np.where(pd.to_numeric(fg.assessment_age)>=60,'60plus','under60')),('centre',fg.assessment_centre.to_numpy())]:
                for val in sorted(set(values)):
                    mask=values==val
                    if mask.sum()<20:continue
                    subgroups.append(dict(method=name,split=split,branch=branch,field=field,value=str(val),n=int(mask.sum()),mean_error_rate=float(error[:,mask].mean())))
            if name!='no_communication':
                base=prediction_groups[('no_communication',split)];be=np.stack(base[branch]).argmax(-1)!=y[None,:]
                # Sets defined by all six models within a family; descriptive, not sixfold sample size.
                error_overlap.append(dict(method=name,split=split,branch=branch,all_six_wrong_shared_with_no_communication=int((be.all(0)&error.all(0)).sum()),all_six_wrong_no_communication=int(be.all(0).sum()),all_six_wrong_method=int(error.all(0).sum())))
    def writecsv(name,rs):
        keys=list(dict.fromkeys(k for r in rs for k in r))
        with (out/name).open('w') as stream:w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rs)
    writecsv('training_audit.csv',rows);writecsv('training_curves.csv',history_rows);writecsv('prediction_audit.csv',metrics);writecsv('subgroup_error_audit.csv',subgroups);writecsv('error_overlap.csv',error_overlap)
    (out/'cohort_metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'source_hashes.json').write_text(json.dumps({'labels':{str(p):sha(p) for p in [data/'selected.csv',testdata/'selected.csv']},'models':refs},indent=2))
    # Raw image summaries remain descriptive proxies, never certified image-quality scores.
    pixel=collections.defaultdict(lambda:collections.defaultdict(list));native=collections.defaultdict(collections.Counter)
    for ix,row in enumerate(f.to_dict('records')):
        root=testdata if row['split']=='test' else data
        with np.load(root/(row['order']+'.npz'),allow_pickle=False) as z:
            oct_=z['oct'];cfp=np.load(root/'cfp224'/(row['order']+'.npy'),allow_pickle=False)
            assert oct_.shape==(2,1,32,96,96) and cfp.shape==(2,3,224,224)
            assert oct_.dtype==np.uint8 and cfp.dtype==np.uint8
            for k,x in [('cfp',cfp),('oct',oct_)]:
                pixel[row['split']][k+'_mean'].append(float(x.mean()/255));pixel[row['split']][k+'_sd'].append(float(x.std()/255));pixel[row['split']][k+'_black_fraction'].append(float((x<=8).mean()))
            if 'native_sizes' in z:
                for shape in z['native_sizes']:native[row['split']]['x'.join(map(str,shape))]+=1
        if (ix+1)%250==0:print(json.dumps({'cache_audited':ix+1,'total':len(f)}),flush=True)
    image={'all_1850_shapes_dtypes_verified':True,'split_summaries':{s:{k:distribution(v) for k,v in d.items()} for s,d in pixel.items()},'native_oct_dimensions_eye_counts':dict(native),'interpretation':'Image intensity proxies only; cannot diagnose image quality or clinical shifts.'}
    (out/'image_cache_audit.json').write_text(json.dumps(image,indent=2))
    (out/'audit_status.json').write_text(json.dumps(dict(state='complete',audit_type='post_test_descriptive_no_selection',participants=len(f),disjoint_splits=True,models=66,mmtm=24,new_training=False,new_inference=False,test_predictions_read=True,private_identifiers_exported=False,completed_at=time.time()),indent=2))
    print(json.dumps({'state':'complete','models':len(rows),'output':str(out)}),flush=True)

if __name__=='__main__':main()
