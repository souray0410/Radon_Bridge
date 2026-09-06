"""Synthetic-only checks for the fixed matrix, paired statistics and figure paths."""
import json
from pathlib import Path
import numpy as np
from scripts.run_task_fusion_benchmark import matrix,configuration,SEEDS,ARMS
from scripts.report_task_fusion_benchmark import statistics,figures
from scripts.run_integer_experiment import DEFAULT_POLICY

def check():
    refs=[dict(seed=s,arm='svd_radon',rho=.125,configuration=dict(seed=s,backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,training_stage='communication',microbatch=16,effective_batch=16,convergence=DEFAULT_POLICY,parent_checkpoints={'cfp':{'path':'fixture','sha256':'fixture'},'oct':{'path':'fixture','sha256':'fixture'}},bridges=[dict(nodes=['cfp_stage3','oct_stage3'],M=32,S=64,rho=.125,mode='radon',compression='fixed_svd_channel',basis_files={})])) for s in SEEDS]
    rows=[]
    assert len(matrix())==54 and len(set(matrix()))==54
    for seed,lr,a in matrix():
        c=configuration(seed,lr,a,refs)
        assert c['microbatch']==16 and c['convergence']==DEFAULT_POLICY and c['selection_metric']=='fusion_macro_f1'
        assert c['head_lr']==c['bridge_lr']==1e-4 and c['backbone_lr']==lr and c['seed']==seed
        assert c['task_fusion']['pooling']==('gated_mil' if a=='gated_mil' else 'mean')
        rows.append(dict(seed=seed,backbone_lr=lr,arm=a))
    labels=np.tile([0,1],20);rng=np.random.default_rng(40);p=rng.random((54,4,40,2));p/=p.sum(-1,keepdims=True)
    result=statistics(rows,p,labels,200)
    assert result['primary_count']==36 and len(result['contrasts'])==38
    lookup={(r['seed'],r['backbone_lr'],r['arm']):i for i,r in enumerate(rows)}
    from scripts.report_five_seed_study import f1
    v=100*f1(labels,p.argmax(-1))
    for d in result['contrasts']:
        expected=np.mean([v[lookup[s,d['backbone_lr'],d['positive']],2]-v[lookup[s,d['backbone_lr'],d['negative']],2] for s in SEEDS])
        assert abs(d['difference_pp']-expected)<1e-12
    identical=np.broadcast_to(p[:1],p.shape).copy();z=statistics(rows,identical,labels,100)
    assert all(d['zero_variance'] and d['simultaneous_ci95_pp'] is None for d in z['contrasts'])
    for i,r in enumerate(rows):
        r.update({b+'_f1_percent':v[i,j] for j,b in enumerate(['cfp','oct','fusion','native_probability_average'])});r['parameters']=1000000*(i%9+20)
    out=Path('/tmp/radonbridge_task_fusion_fixture');out.mkdir(exist_ok=True)
    fs=figures(out,rows,result);assert len(fs)==16 and all((out/p).exists() for p in fs)
    print(json.dumps(dict(passed=True,synthetic_only=True,training_jobs=54,primary_contrasts=36,shared_participant_bootstrap=True,zero_variance_undefined=True,figures=len(fs))))
if __name__=='__main__':check()
