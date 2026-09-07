"""Complete depth table and prespecified shared-participant bootstrap contrasts."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from radonbridge.artifacts import resolve,sha256
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import classify
from scripts.geometry_evidence import read,write
from scripts.report_five_seed_study import f1
from scripts.report_geometry_mechanism import csv_write

def build(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    refs=read(root/'references.json');rows=read(root/'manifest.json')['rows']+refs['single_stage3']+refs['no_communication']
    assert len(rows)==114 and len({r['id'] for r in rows})==114
    definitions=read(root/'comparisons.json');lookup={r['id']:i for i,r in enumerate(rows)}
    ids=y=None;predictions=[];tables=[]
    for r in rows:
        table={k:r.get(k) for k in ('id','seed','backbone_lr','state','reused','seconds','epochs','best_epoch','parameters','peak_reserved_mib')}
        s=r['structure'];table.update(structure=s['id'],stages=s.get('stages'),mode=s.get('mode'),r=s.get('r'),h=s.get('h'),
            per_stage_rho=[b['rho'] for b in r['configuration']['bridges']],total_mixer_parameters=s.get('total_mixer_parameters'),
            bridge_flops_per_participant=s.get('cost',{}).get('flops_per_participant'))
        if r['state']!='accepted':predictions.append(None);tables.append(table);continue
        p=resolve(r['directory'])/'selected_predictions.npz';assert sha256(p)==r['accepted_hashes']['selected_predictions.npz']
        with np.load(p,allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy();y=z['y'].copy()
            assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
            predictions.append(np.stack([z['cfp'],z['oct']]).argmax(-1))
            for key,p in [('cfp',z['cfp']),('oct',z['oct']),('fixed_probability_average',(z['cfp']+z['oct'])/2)]:
                metrics=classification_metrics(y,p);metrics['brier_binary']=float(np.mean((p[:,1]-y)**2))
                table.update({key+'_'+k:v for k,v in metrics.items()})
            table['branch_mean_macro_f1']=(table['cfp_macro_f1']+table['oct_macro_f1'])/2
        tables.append(table)
    assert len(ids)==296
    pred=np.stack([p if p is not None else np.zeros((2,296),dtype=int) for p in predictions])
    point=100*f1(y,pred).mean(1)
    ix=np.random.default_rng(202609071).integers(0,296,(10000,296))
    boot=np.empty((10000,len(rows)))
    for lo in range(0,len(ix),20):
        indices=ix[lo:lo+20];boot[lo:lo+len(indices)]=(100*f1(y[indices],pred[:,:,indices])).mean(1).T
    records=[];valid=[];weights=[]
    for d in definitions:
        missing=[key for key in d['weights'] if key not in lookup or rows[lookup[key]]['state']!='accepted']
        if missing:records.append({k:v for k,v in d.items() if k!='weights'}|dict(estimable=False,missing=missing,classification='not_estimable'));continue
        w=np.zeros(len(rows))
        for key,value in d['weights'].items():w[lookup[key]]=value
        valid.append(d);weights.append(w)
    if valid:
        w=np.stack(weights);observed=w@point;draws=boot@w.T;sd=draws.std(0,ddof=1)
        primary=np.array([d['primary'] for d in valid]);active=(sd>1e-10)&primary
        critical=float(np.quantile(np.max(np.abs((draws[:,active]-observed[None,active])/sd[None,active]),1),.95)) if active.any() else None
        for j,d in enumerate(valid):
            ci=[float(observed[j]-critical*sd[j]),float(observed[j]+critical*sd[j])] if d['primary'] and sd[j]>1e-10 and critical is not None else None
            records.append({k:v for k,v in d.items() if k!='weights'}|dict(estimable=True,difference_pp=float(observed[j]),
                ci95_pp=np.quantile(draws[:,j],[.025,.975]).tolist(),family_simultaneous_ci95_pp=ci,bootstrap_sd_pp=float(sd[j]),
                classification=classify(ci) if ci else 'undefined_zero_variance' if sd[j]<=1e-10 else 'secondary_descriptive'))
    aggregates=[]
    for sid in dict.fromkeys(t['structure'] for t in tables):
        for lr in (3e-5,6e-5):
            group=[t for t in tables if t['structure']==sid and t['backbone_lr']==lr and t['state']=='accepted']
            for metric in ('cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1'):
                v=np.array([t[metric] for t in group])*100
                aggregates.append(dict(structure=sid,backbone_lr=lr,metric=metric,seeds=len(v),complete_three_seeds=len(v)==3,
                    mean_percent=float(v.mean()) if len(v) else None,sample_sd_pp=float(v.std(ddof=1)) if len(v)>1 else None))
    csv_write(out/'all_results.csv',tables);csv_write(out/'contrasts.csv',records);csv_write(out/'seed_aggregates.csv',aggregates)
    write(out/'statistics.json',dict(participants=296,resamples=10000,shared_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        comparison_sha256=sha256(root/'comparisons.json'),primary_comparisons=22,secondary_comparisons=38,contrasts=records,test_used=False,
        global_test_correction='pending locked union with all earlier primary families; these are development family intervals only',
        interpretation='Conditional on fitted and development-selected models; missing cells are not reweighted; no independent test evidence'))
    (out/'README.zh-CN.md').write_text('R&B 多深度机制补充\n\n完整结果见 all_results.csv；各种子原始结果及按学习率的三种子均值/样本标准差分别保留。contrasts.csv 含预定22项主要及38项次要比较。\n\n当前仅为开发集证据，普通95%区间与22项主要比较的族内同时区间分开报告；最终统一test还须进行全部主要比较的全局校正。±1 pp为研究参考范围。单格排名、三桥开关依赖及F1尺度训练交互均不自动构成临床或因果证明。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
