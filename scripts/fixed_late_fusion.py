"""Fixed late fusion using existing fixed-epoch predictions; aggregate-only export."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score

p=argparse.ArgumentParser();p.add_argument('--parent-root',required=True)
p.add_argument('--output',required=True);p.add_argument('--lock',required=True)
args=p.parse_args()

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p)),
            'log_loss':float(log_loss(y,p)), 'brier':float(np.mean((y-p)**2)),
            'accuracy_at_0.5':float(accuracy_score(y,p>=.5))}

with open(args.lock,'a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Commit source first')
    root=Path(args.parent_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('Output must be new')
    specs={'cfp96':('cfp96_s3407','baseline'),'cfp224':('cfp224_s3407','baseline'),
           'oct':('oct_s3407','baseline'),'joint96':('both96_s3407','baseline'),
           'joint224':('both224_s3407','baseline'),'radon224':('both224_s3407','radon')}
    preds={};source_configs={};ids=None;y=None
    for name,(run,arm) in specs.items():
        with np.load(root/run/(arm+'_last_predictions.npz'),allow_pickle=False) as a:
            if ids is None:ids=a['ids'].copy();y=a['y'].copy()
            assert np.array_equal(ids,a['ids']) and np.array_equal(y,a['y'])
            preds[name]=a['p'].astype(np.float64)
        source_configs[run]=json.loads((root/run/'config.json').read_text())['source_commit']
    preds['mean96']=(preds['cfp96']+preds['oct'])/2
    preds['mean224']=(preds['cfp224']+preds['oct'])/2
    def logit(p):
        p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
    preds['logit224']=1/(1+np.exp(-(logit(preds['cfp224'])+logit(preds['oct']))/2))
    comparisons=[]
    pairs=[('mean224',c) for c in ('cfp224','oct','joint224','radon224')]+[('mean96','oct'),('logit224','mean224')]
    for a,b in pairs:
        rng=np.random.default_rng(710);deltas=[]
        for _ in range(2000):
            ix=rng.integers(0,len(y),len(y))
            if len(np.unique(y[ix]))<2:continue
            deltas.append(roc_auc_score(y[ix],preds[a][ix])-roc_auc_score(y[ix],preds[b][ix]))
        comparisons.append({'a':a,'b':b,'delta_auroc':float(roc_auc_score(y,preds[a])-roc_auc_score(y,preds[b])),
                            'paired_bootstrap_95pct':np.quantile(deltas,[.025,.975]).tolist()})
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            'parent_source_commits':source_configs,'seed':3407,'validation_participants':len(y),
            'test_used':False,'gpu_training_minutes':0,'metrics':{k:metrics(y,v) for k,v in preds.items()},
            'comparisons':comparisons,'limitation':'Exploratory follow-up using shared validation and one training seed; no multiplicity adjustment.'}
    np.savez(out/'predictions.npz',ids=ids,y=y,**preds)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
