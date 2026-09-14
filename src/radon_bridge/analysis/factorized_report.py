"""Prespecified small-cohort paired development comparisons, no test reader."""
import csv
import json
from pathlib import Path
import numpy as np
from radon_bridge.studies.factorized_ws import write_json


def bootstrap_f1(y,p,counts):
    pred=p.argmax(1);values=[]
    for label in [0,1]:
        tp=counts@((y==label)&(pred==label)).astype(float)
        fp=counts@((y!=label)&(pred==label)).astype(float)
        fn=counts@((y==label)&(pred!=label)).astype(float)
        values.append(np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0))
    return (values[0]+values[1])/2


def report(root):
    root=Path(root);cases=json.loads((root/'queue.json').read_text())['cases'];rows=[];predictions={};ids=labels=None
    for c in cases:
        p=root/'trials'/c['name'];a=json.loads((p/'accepted.json').read_text())
        z=np.load(p/'selected_predictions.npz',allow_pickle=False)
        if ids is None:ids=z['ids'].copy();labels=z['y'].copy()
        assert np.array_equal(ids,z['ids']) and np.array_equal(labels,z['y'])
        predictions[c['name']]={k:z[k].copy() for k in ['cfp','oct']};z.close()
        rows.append(dict(name=c['name'],seed=c['seed'],cfp_f1=a['selected_validation']['tasks']['cfp']['macro_f1'],
            oct_f1=a['selected_validation']['tasks']['oct']['macro_f1'],mean_f1=a['selected_validation']['mean_task_macro_f1'],
            best_epoch=a['best_epoch'],stop_epoch=a['epochs_ran']))
    with (root/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts=np.random.default_rng(20260914).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=10000).astype(float)
    all_counts=np.concatenate([np.ones((1,len(labels))),counts]);f1={}
    for name,p in predictions.items():
        f1[name]={k:bootstrap_f1(labels,v,all_counts) for k,v in p.items()}
        f1[name]['mean']=(f1[name]['cfp']+f1[name]['oct'])/2
    contrasts=[];values=[]
    for rank in [256,512,1024]:
        for metric in ['cfp','oct','mean']:
            d=sum(f1[f'factorized_R{rank}_radon_seed{s}'][metric]-f1[f'factorized_R{rank}_linear_resample_seed{s}'][metric] for s in [3416,3417,3418])/3
            contrasts.append(dict(rank=rank,metric=metric,difference=float(d[0]),ci95=np.quantile(d[1:],[.025,.975]).tolist()));values.append(d)
    v=np.stack(values);sd=v[:,1:].std(1,ddof=1);valid=sd>0
    if valid.any():
        centered=v[valid,1:]-v[valid,1:].mean(1,keepdims=True)
        critical=float(np.quantile(np.max(np.abs(centered/sd[valid,None]),axis=0),.95))
    else:critical=None
    for i,c in enumerate(contrasts):
        c['simultaneous95']=[float(v[i,0]-critical*sd[i]),float(v[i,0]+critical*sd[i])] if valid[i] else None
        c['zero_variance']=not bool(valid[i])
    write_json(root/'comparisons.json',dict(test_used=False,cohort='ws02_small_development_exploration',participants=len(labels),
        resamples=10000,shared_participant_indices=True,seed_repetitions=3,primary_family_size=9,
        direction='Radon minus same-rank linear resampling',contrasts=contrasts))
