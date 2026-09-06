"""CPU-only prespecified transfer probes. No UKB files or neural pretraining."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from radonbridge.projector import Projector,LinearResampleProjector
from radonbridge.s_axis import fixed_permutation
from radonbridge.experiment import write_json


def field_basis(shape):
    axes=[np.linspace(-1,1,n) for n in shape]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,len(shape))
    if len(shape)==2:xyz=np.c_[xyz,np.zeros(len(xyz))]
    x,y,z=xyz.T
    f=np.stack([np.sin(np.pi*x),np.sin(np.pi*y),np.cos(np.pi*x),np.cos(np.pi*y),
                np.sin(np.pi*x)*np.cos(np.pi*y),np.cos(np.pi*x)*np.sin(np.pi*y),
                np.cos(np.pi*z)*np.sin(np.pi*x),np.cos(np.pi*z)*np.sin(np.pi*y)],0)
    return f-f.mean(1,keepdims=True)


def design(x,kind):
    source_shape=(4,5,4);target_shape=(4,5);M,S=4,11
    cls=LinearResampleProjector if kind=='resample' else Projector
    a=cls(source_shape,M,S).double();b=cls(target_shape,M,S).double()
    z=a(torch.from_numpy(x).reshape(-1,1,*source_shape))
    p=torch.tensor(fixed_permutation(S)['permutation']);inv=torch.argsort(p)
    if kind=='s_permuted':z=z[...,p]
    columns=[]
    for dst in range(M):
        for src in range(M):
            for offset in range(3):
                w=z.new_zeros(M,M,3);w[dst,src,offset]=1
                mixed=torch.nn.functional.conv1d(z,w,padding=1)
                if kind=='s_permuted':mixed=mixed[...,inv]
                columns.append(b.backproject(mixed).flatten(1).numpy())
    return np.stack(columns,-1)


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tick=time.monotonic();torch.set_num_threads(3)
    rows=[];train_n,eval_n=96,256
    for seed in (7711,7712,7713):
        for task in ('spatial_arrangement','global_mean','irrelevant_source'):
            rng=np.random.default_rng(seed);total=train_n+eval_n
            c=rng.normal(size=(total,8));other=rng.normal(size=(total,8))
            if task=='global_mean':
                amp=c[:,:1];x=np.repeat(amp,80,axis=1);y=np.repeat(amp,20,axis=1)
            else:
                x=c@field_basis((4,5,4));y=(other if task=='irrelevant_source' else c)@field_basis((4,5))
            # Same fixed noise and splits for all operators. No generated Radon targets.
            x=x+.1*rng.normal(size=x.shape);y=y+.1*rng.normal(size=y.shape)
            for method in ('radon','s_permuted','resample','global_pool','dense_linear'):
                if method=='dense_linear':
                    # Unrestricted linear capacity reference, explicitly not capacity matched.
                    d=x;train=d[:train_n];target=y[:train_n]
                    g=train.T@train/len(train);lam=1e-3*np.trace(g)/len(g)
                    w=np.linalg.solve(g+lam*np.eye(len(g)),train.T@target/len(train));pred=d[train_n:]@w;params=w.size
                else:
                    d=np.repeat(x.mean(1)[:,None,None],20,axis=1) if method=='global_pool' else design(x,method)
                    train=d[:train_n].reshape(-1,d.shape[-1]);target=y[:train_n].reshape(-1)
                    g=train.T@train/len(train);lam=1e-3*np.trace(g)/len(g)
                    w=np.linalg.solve(g+lam*np.eye(len(g)),train.T@target/len(train));pred=d[train_n:]@w;params=len(w)
                mse=float(np.mean((pred-y[train_n:])**2));zero=float(np.mean(y[train_n:]**2))
                rows.append(dict(seed=seed,task=task,method=method,synthetic_train=train_n,synthetic_evaluation=eval_n,mse=mse,
                                 normalized_mse=mse/zero,zero_prediction_mse=zero,parameters=params,ridge_lambda=float(lam)))
    protocol=dict(seeds=[7711,7712,7713],train=96,evaluation=256,source_shape=[4,5,4],target_shape=[4,5],M=4,S=11,channels=1,
                  target='2D plane of analytic smooth fields, global constant, or independent field',
                  fit='closed-form ridge on cross-source transfer operator; lambda=.001*trace(design Gram)/parameter count',
                  limitations='Synthetic regression operator probes, no CNN, no channel compression, no UKB label or clinical evidence. Pool and dense baselines have different capacity. No hyperparameter selection.')
    result=dict(state='complete',rows=rows,protocol=protocol,cpu_wall_seconds=time.monotonic()-tick,ukb_files_read=False)
    write_json(out/'synthetic_operator_probes.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
