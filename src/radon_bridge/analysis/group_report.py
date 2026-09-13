"""Matched multi-task reporting with participant resamples shared across cases."""
import csv
import json
from pathlib import Path
import numpy as np

from radon_bridge.analysis.project_report import f1
from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash


def comparisons(arms):
    by_id = {a['id']:a for a in arms}; rows = []
    reference = 'svd_radon' if 'svd_radon' in by_id else 'svd_all_radon'
    if reference not in by_id: raise ValueError('Missing reference method')
    for other in ('continue','svd_self','mmtm256','attention256'):
        if other in by_id:
            rows.append(dict(id=reference+'_minus_'+other, family='direct', weights={reference:1.,other:-1.}))
    for a in arms:
        if a.get('host'):
            if a['addition'] != 'continue':
                rows.append(dict(id=a['id']+'_minus_continuation', family='host_augmentation',
                                 weights={a['id']:1.,a['host']+'_plus_continue':-1.}))
            continue
        if a['mode'] == 'linear_resample':
            key = {k:v for k,v in a.items() if k not in ('id','mode')}
            matches = [b for b in arms if b['mode']=='radon' and {k:v for k,v in b.items() if k not in ('id','mode')}==key]
            if len(matches) != 1: raise ValueError('Ordinary communication lacks a unique matched Radon')
            rows.append(dict(id='geometry_'+a['id'], family='geometry', weights={matches[0]['id']:1.,a['id']:-1.}))
        if (a['id'] != reference and a['family']=='radon' and a['mode']!='linear_resample' and
                not a.get('host')):
            rows.append(dict(id=a['id']+'_minus_reference', family='mechanism', weights={a['id']:1.,reference:-1.}))
    return rows


def participant_draws(labels, predictions, iterations=10000, seed=73621):
    """Columns may have different task labels; participant row ownership is shared."""
    labels=np.asarray(labels); predictions=np.asarray(predictions)
    if labels.shape!=predictions.shape or labels.ndim!=2 or not len(labels):
        raise ValueError('Expected participant by model/task matrices')
    if not np.isin(labels,[0,1]).all() or not np.isin(predictions,[0,1]).all():
        raise ValueError('Binary confusion coding required')
    n,m=labels.shape
    indicators=(2*labels[...,None]+predictions[...,None]==np.arange(4)).reshape(n,4*m).astype(float)
    point=f1(indicators.sum(0).reshape(m,4))
    rng=np.random.default_rng(seed); draws=[]
    # Fixed participant probabilities, independent of model labels/predictions.
    # The same n, order and seed give identical weights for every seed/case.
    for start in range(0,iterations,32):
        count=rng.multinomial(n,np.full(n,1/n),size=min(32,iterations-start))
        draws.append(f1((count@indicators).reshape(len(count),m,4)))
    return point,np.concatenate(draws)


def intervals(point, draws):
    sd=draws.std(0,ddof=1); valid=sd>0
    critical=float(np.quantile(np.max(np.abs((draws[:,valid]-draws[:,valid].mean(0))/sd[valid]),1),.95)) if valid.any() else None
    rows=[]
    for i,value in enumerate(point):
        ci=[float(value-critical*sd[i]),float(value+critical*sd[i])] if valid[i] else None
        judgement='尚不能分辨'
        if ci:
            if ci[0]>.01: judgement='支持实质提升'
            elif ci[1]<-.01: judgement='支持实质下降'
            elif ci[0]>=-.01 and ci[1]<=.01: judgement='支持实际接近'
        rows.append(dict(difference=float(value),ordinary_95=np.quantile(draws[:,i],[.025,.975]).tolist(),
                         simultaneous_95=ci,zero_variance=not bool(valid[i]),judgement=judgement))
    return rows


def report_case(spec,root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    keys=[s['key'] for s in spec['sources']];rows=[];predictions=[];labels=[];ids=None
    for arm in spec['arms']:
        path=root/'arms'/arm['id'];r=json.loads((path/'accepted.json').read_text())
        if r.get('state')!='accepted' or not r.get('plateau') or r.get('test_access') is not False:
            raise ValueError('Incomplete training cannot enter matched statistics')
        with np.load(path/'development_predictions.npz',allow_pickle=False) as z:
            if ids is None:ids=z['participant_ids'].copy()
            if not np.array_equal(ids,z['participant_ids']):raise ValueError('Participant ordering changed')
            for s in spec['sources']:
                key=s['key'];predictions.append(z[key].argmax(1));labels.append(z['labels__'+key])
                rows.append(dict(arm=arm['id'],source=key,disease=s['disease'],modality=s['modality'],
                    architecture=s['architecture'],seed=spec['seed'],macro_f1=r['metrics'][key]['macro_f1'],
                    mean_macro_f1=r['metrics']['mean_macro_f1'],best_epoch=r['best_epoch'],stop_epoch=r['stop_epoch'],
                    seconds=r['seconds'],parameters_total=r['parameters_total'],parameters_trainable=r['parameters_trainable'],
                    **{name:arm[name] if arm['family']=='radon' else None for name in ('r','M','S','k')}))
    definitions=comparisons(spec['arms']);weights=np.zeros((len(definitions),len(predictions)))
    arm_order=[a['id'] for a in spec['arms']]
    for i,d in enumerate(definitions):
        for arm,w in d['weights'].items():
            index=arm_order.index(arm)*len(keys);weights[i,index:index+len(keys)]=w/len(keys)
    point,draws=participant_draws(np.stack(labels,1),np.stack(predictions,1),spec['bootstrap_iterations'])
    effects=weights@point;effect_draws=draws@weights.T
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Keep numeric model draws for cross-case families; no participant identifiers here.
    np.savez_compressed(out/'bootstrap_draws.npz',point=point,draws=draws,weights=weights,
                        keys=np.asarray([a+'::'+k for a in arm_order for k in keys]))
    result=dict(definitions=definitions,comparisons=intervals(effects,effect_draws),iterations=spec['bootstrap_iterations'],
        participants=len(ids),participant_order_sha256=stable_hash(ids.tolist()),resample_seed=73621,
        interval_scope='one matched development case; not global study or independent test evidence',
        whole_study_intervals='pending all registered matched cases',test_access=False,
        draws_sha256=file_sha256(out/'bootstrap_draws.npz'))
    atomic_write_json(result,out/'paired_statistics.json')
    (out/'README.zh-CN.md').write_text('Radon_Bridge开发集匹配结果。macro-F1越大越好，差值按weights的正项减负项计算。各来源保留自身疾病标签，先算各模型F1再等权平均，不平均不同疾病概率。CSV同时保留每个来源及整体均值。区间仅适用于本匹配组；全研究同时区间必须等完整组齐备后另算。\n')
    return result
