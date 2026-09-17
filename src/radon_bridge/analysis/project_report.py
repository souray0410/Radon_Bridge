"""Automatically regenerated, accepted-only matched evidence and interpretation."""
import csv
import json
from pathlib import Path
import numpy as np
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.studies.research_matrix import comparisons


def f1(c):
    tn,fp,fn,tp=np.moveaxis(c,-1,0)
    a=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)>0)
    b=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    return (a+b)/2


def bootstrap(y,pred,weights,iterations=10000,seed=73621):
    y=np.asarray(y);pred=np.asarray(pred);weights=np.asarray(weights,dtype=float)
    if pred.ndim!=2 or pred.shape[1]!=len(y) or weights.ndim!=2 or weights.shape[1]!=len(pred) or iterations<2:
        raise ValueError('Invalid matched bootstrap shapes')
    patterns,counts=np.unique(np.column_stack([y,pred.T]),axis=0,return_counts=True)
    indicator=(2*patterns[:,0,None,None]+patterns[:,1:,None]==np.arange(4)).reshape(len(patterns),-1).astype(float)
    observed=f1((counts@indicator).reshape(len(pred),4));point=weights@observed
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,iterations,32):
        counts_sample=rng.multinomial(len(y),counts/counts.sum(),size=min(32,iterations-start))
        values=f1((counts_sample@indicator).reshape(len(counts_sample),len(pred),4));draws.append(values@weights.T)
    draws=np.concatenate(draws);sd=draws.std(0,ddof=1);valid=sd>0
    critical=float(np.quantile(np.max(abs((draws[:,valid]-draws[:,valid].mean(0))/sd[valid]),1),.95)) if valid.any() else None
    records=[]
    for i,value in enumerate(point):
        interval=[float(value-critical*sd[i]),float(value+critical*sd[i])] if valid[i] else None
        judgement='尚不能分辨'
        if interval:
            if interval[0]>.01:judgement='支持实质提升'
            elif interval[1]<-.01:judgement='支持实质下降'
            elif interval[0]>=-.01 and interval[1]<=.01:judgement='支持实际接近'
        records.append(dict(difference=float(value),ordinary_95=np.quantile(draws[:,i],[.025,.975]).tolist(),
            simultaneous_95=interval,bootstrap_sd=float(sd[i]),zero_variance=not bool(valid[i]),judgement=judgement))
    return dict(iterations=iterations,participants=len(y),seed=seed,comparisons=records,
        interval_scope='conditional_on_selected_development_models_not_independent_test',max_t_critical=critical)


def report_case(spec,root,*,arm_ids=None,output=None):
    root=Path(root);out=Path(output) if output is not None else root/'report';out.mkdir(parents=True,exist_ok=True)
    selected=spec['arms'] if arm_ids is None else [a for a in spec['arms'] if a['id'] in arm_ids]
    if not selected or (arm_ids is not None and {a['id'] for a in selected}!=set(arm_ids)):
        raise ValueError('Unknown or empty report scope')
    rows=[];pred={};reference=None
    for arm in selected:
        path=root/'arms'/arm['id'];r=json.loads((path/'accepted.json').read_text())
        z=np.load(path/'development_predictions.npz',allow_pickle=False)
        if reference is None:reference=z
        elif not np.array_equal(reference['participant_ids'],z['participant_ids']) or not np.array_equal(reference['labels'],z['labels']):raise ValueError('Unpaired models')
        for branch in ('cfp','oct'):pred[(arm['id'],branch)]=z[branch].argmax(1)
        rows.append(dict(arm=arm['id'],seed=spec['seed'],disease=spec['disease'],cfp_f1=r['metrics']['cfp']['macro_f1'],oct_f1=r['metrics']['oct']['macro_f1'],mean_f1=r['metrics']['mean_macro_f1'],best_epoch=r['best_epoch'],stop_epoch=r['stop_epoch'],seconds=r['seconds'],frozen=arm['frozen'],r=arm['r'] if arm['family']=='radon' else None,M=arm['M'] if arm['family']=='radon' else None,S=arm['S'] if arm['family']=='radon' else None,k=arm['k'] if arm['family']=='radon' else None,stages='+'.join(map(str,arm['stages']))))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    keys=list(pred);names={a['id'] for a in selected}
    definitions=[c for c in comparisons(spec['disease'],spec['model']['name']) if {c['left'],c['right']}<=names];weights=[]
    for comparison in definitions:
        w=np.zeros(len(keys))
        branches=('cfp','oct') if comparison['branch']=='mean' else (comparison['branch'],)
        for branch in branches:
            w[keys.index((comparison['left'],branch))]+=1/len(branches);w[keys.index((comparison['right'],branch))]-=1/len(branches)
        weights.append(w)
    stats=bootstrap(reference['labels'],np.stack(list(pred.values())),weights,spec['bootstrap_iterations'])
    stats.update(definitions=definitions,arm_ids=[a['id'] for a in selected],
        family='registered_package_development; whole_study_intervals_are_separate')
    atomic_write_json(stats,out/'paired_statistics.json')
    lines=['# Radon_Bridge 匹配开发评价','', '主要指标为两分支macro-F1平均值；越高越好。差值按定义left减right，单位pp。',
        '本表仅为选中模型的开发证据。种子3416参与了父配方提名；不能当独立确认性验证。','',
        '| 方法 | CFP F1 % | OCT F1 % | 平均 F1 % |','|---|---:|---:|---:|']
    lines += [f"| {r['arm']} | {r['cfp_f1']*100:.2f} | {r['oct_f1']*100:.2f} | {r['mean_f1']*100:.2f} |" for r in rows]
    lines += ['', '| 比较 | 差值pp | 本比较族同时区间pp | 判断 |','|---|---:|---|---|']
    for d,s in zip(definitions,stats['comparisons']):
        ci=s['simultaneous_95'];text='未定义' if ci is None else f'[{ci[0]*100:.2f}, {ci[1]*100:.2f}]'
        lines.append(f"| {d['id']} | {s['difference']*100:.2f} | {text} | {s['judgement']} |")
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return stats
