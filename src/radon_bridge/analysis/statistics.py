"""Prespecified participant bootstrap, 31 primary contrasts, and strata.

Models are fixed. Seeds and widths are averaging factors, not independent
participants. All contrasts and secondary fusion metrics share the same draws.
"""
import hashlib
import numpy as np
from radon_bridge.evaluation.metrics import batched_macro_f1 as f1
from radon_bridge.studies.benchmark import SEEDS,RHOS,key
from radon_bridge.studies.benchmark import BASELINES


def classify(ci,margin=1.):
    lo,hi=ci
    if lo>margin:return 'supports_substantive_improvement'
    if hi<-margin:return 'supports_substantive_decline'
    if lo>=-margin and hi<=margin:return 'supports_practical_similarity'
    return 'unresolved'


def definitions(rows):
    lookup={key(r['seed'],r['arm'],r['rho']):i for i,r in enumerate(rows)}
    definitions=[]
    def add(name,terms,branches,seeds,rhos,primary):
        w=np.zeros((len(rows),2))
        for arm,coefficient in terms:
            for seed in seeds:
                actual_rhos=[None] if arm in BASELINES else rhos
                for rho in actual_rhos:
                    idx=lookup[key(seed,arm,rho)]
                    for b in branches:w[idx,b]+=coefficient/(len(seeds)*len(actual_rhos)*len(branches))
        definitions.append({'id':name,'terms':terms,'branches':['cfp' if b==0 else 'oct' for b in branches],'seeds':seeds,'rhos':rhos,'primary':primary,'weights':w})
    mechanics=[]
    for basis in ['svd','qr']:mechanics.append((basis+'_geometry',[(basis+'_radon',1),(basis+'_resample',-1)],[0,1]))
    mechanics.append(('basis_geometry_interaction',[('svd_radon',1),('svd_resample',-1),('qr_radon',-1),('qr_resample',1)],[0,1]))
    for basis in ['svd','qr']:
        mechanics.extend([(basis+'_oct_to_cfp_contribution',[(basis+'_radon',1),(basis+'_cfp_to_oct',-1)],[0]),(basis+'_cfp_to_oct_contribution',[(basis+'_radon',1),(basis+'_oct_to_cfp',-1)],[1])])
    for name,terms,bs in mechanics:add(name,terms,bs,SEEDS,RHOS,True)
    for basis in ['svd','qr']:
        for rho in RHOS:
            for baseline in BASELINES:add(f'{basis}_rho1_{round(1/rho)}_minus_{baseline}',[(basis+'_radon',1),(baseline,-1)],[0,1],SEEDS,[rho],True)
    assert len(definitions)==31
    # Every branch, width and individual seed, with the same paired draws.
    for name,terms,bs in mechanics:
        for seeds in [SEEDS]+[[s] for s in SEEDS]:
            for rhos in [RHOS]+[[r] for r in RHOS]:
                for branch in [[0],[1],[0,1]]:
                    add(name+f'_seeds{seeds}_rhos{rhos}_branches{branch}',terms,branch,seeds,rhos,False)
    for basis in ['svd','qr']:
        for rho in RHOS:
            for baseline in BASELINES:
                for seeds in [SEEDS]+[[s] for s in SEEDS]:
                    for branch in [[0],[1],[0,1]]:add(f'{basis}_minus_{baseline}_rho{rho}_seeds{seeds}_branches{branch}',[(basis+'_radon',1),(baseline,-1)],branch,seeds,[rho],False)
    return definitions


def summarize_bootstrap(observed,samples,primary_count=31):
    sd=np.std(samples,axis=0,ddof=1);valid=sd>0
    active=valid[:primary_count]
    centered=samples[:,:primary_count]-observed[None,:primary_count]
    standardized=np.divide(centered,sd[None,:primary_count],out=np.zeros_like(centered),where=active[None,:])
    critical=float(np.quantile(np.abs(standardized[:,active]).max(axis=1),.95)) if active.any() else None
    result=[]
    for j,value in enumerate(observed):
        ordinary=np.quantile(samples[:,j],[.025,.975]).tolist();primary=j<primary_count
        simultaneous=[float(value-critical*sd[j]),float(value+critical*sd[j])] if primary and valid[j] and critical is not None else None
        result.append({'difference_pp':float(value),'ci95_pp':ordinary,'bootstrap_sd_pp':float(sd[j]),'simultaneous_ci95_pp':simultaneous,
            'zero_variance':not bool(valid[j]),'classification':classify(simultaneous) if simultaneous is not None else 'undefined_zero_variance' if primary else 'secondary_no_simultaneous_classification'})
    return result,critical


def bootstrap(rows,probabilities,labels,resamples=10000,seed=20260905):
    """probabilities: model x branch x participant x class, branch order CFP/OCT."""
    preds=probabilities.argmax(-1);defs=definitions(rows);weights=np.stack([d.pop('weights') for d in defs]).reshape(len(defs),-1)
    point=100*f1(labels,preds);observed=weights@point.reshape(-1)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(labels),size=(resamples,len(labels)))
    boot_models=np.empty((resamples,len(rows),2));fusion_preds=probabilities.mean(axis=1).argmax(-1);fusion_boot=np.empty((resamples,len(rows)))
    for start in range(0,resamples,50):
        ix=indices[start:start+50]
        boot_models[start:start+len(ix)]=(100*f1(labels[ix],preds[:,:,ix])).transpose(2,0,1)
        fusion_boot[start:start+len(ix)]=(100*f1(labels[ix],fusion_preds[:,ix])).T
    samples=boot_models.reshape(resamples,-1)@weights.T
    intervals,critical=summarize_bootstrap(observed,samples)
    results=[dict(d,**v) for d,v in zip(defs,intervals)]
    return {'resamples':resamples,'seed':seed,'participants':len(labels),'resampling_unit':'participant, shared across every model/seed/rho','indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
        'primary_count':31,'max_abs_t_critical_95':critical,'practical_margin_pp':1.,'contrasts':results,
        'interpretation':'Conditional on fitted models and development-selected checkpoints. Seed and width are averaged after calculating each F1. No independent test evidence.'},boot_models,fusion_boot,indices
