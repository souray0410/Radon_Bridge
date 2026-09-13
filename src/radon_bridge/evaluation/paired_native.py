"""Paired branch outputs; participant identity is retained only in restricted artifacts."""
from pathlib import Path
import numpy as np
import torch
from radon_bridge.runtime.state import atomic_write_json,file_sha256


def move(batch,device):
    if isinstance(batch,torch.Tensor):return batch.to(device)
    if isinstance(batch,dict):return {k:move(v,device) for k,v in batch.items()}
    return batch


@torch.no_grad()
def evaluate(model,loader,device,output=None,should_pause=lambda:False):
    if hasattr(model,'source_keys'):
        from radon_bridge.evaluation.group_native import evaluate as group_evaluate
        return group_evaluate(model,loader,device,output,should_pause)
    from expanded.native import metrics
    mode=model.training;model.eval();p={'cfp':[],'oct':[]};ids=[];labels=[]
    try:
        for batch in loader:
            if should_pause():raise InterruptedError('Pause during evaluation; replay this read-only stage')
            z,_=model(move(batch,device))
            for k in p:p[k].append(z[k].softmax(1).cpu().numpy())
            ids.extend(batch['participant_id']);labels.extend(batch['label'].tolist())
    finally:model.train(mode)
    p={k:np.concatenate(v) for k,v in p.items()};y=np.asarray(labels)
    if len(set(ids))!=len(ids) or not len(ids):raise ValueError('Evaluation coverage invalid')
    values={k:metrics(y,v) for k,v in p.items()}
    values['mean_macro_f1']=sum(v['macro_f1'] for v in values.values())/2
    values['fixed_probability_fusion']=metrics(y,(p['cfp']+p['oct'])/2)
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez(f,participant_ids=np.asarray(ids,dtype=str),labels=y,**p)
        tmp.replace(path)
    return values


def replay_matches(path,other):
    with np.load(path,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Prediction format changed')
        for k in a.files:
            exact=k in ('participant_ids','labels') or k.startswith('labels__')
            if exact and not np.array_equal(a[k],b[k]):raise ValueError('Prediction identity/order changed')
            if not exact and (not np.allclose(a[k],b[k],atol=1e-5,rtol=1e-4) or not np.array_equal(a[k].argmax(1),b[k].argmax(1))):
                raise ValueError('Selected prediction numerical replay failed')
