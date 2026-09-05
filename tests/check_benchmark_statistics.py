"""Synthetic fixtures validate the protocol and statistics; never real results."""
import copy,json
import numpy as np
import torch
from radonbridge.benchmark_statistics import definitions,bootstrap,summarize_bootstrap,classify
from radonbridge.communication_analysis import derangements,block_messages,cached_forward
from radonbridge.bridge import LinearMixer
from scripts.run_mechanism_benchmark import matrix,BASELINES,configuration
from scripts.run_three_seed_study import SEEDS,RHOS,key
from scripts.report_five_seed_study import f1

def check():
    old=['learned','svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']+['centered_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]+['channel_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]
    rows=[dict(seed=s,arm=a,rho=r) for s in SEEDS for a,r in [('no_bridge',None)]+[(a,r) for r in RHOS for a in old]]
    assert len(rows)==129;rows += [dict(seed=s,arm=a,rho=r) for s,a,r in matrix()];assert len(rows)==186 and len({key(r['seed'],r['arm'],r['rho']) for r in rows})==186
    defs=definitions(rows);assert sum(d['primary'] for d in defs)==31
    assert all(abs(d['weights'].sum())<1e-12 for d in defs)
    rng=np.random.default_rng(912);y=np.array([0,1,0,1,0,1,0,1]);p=rng.uniform(.01,.99,(186,2,8));p=np.stack([1-p,p],-1)
    result,boots,fused,indices=bootstrap(rows,p,y,resamples=101)
    w=np.stack([d['weights'] for d in defs]);ref=np.einsum('kmb,mb->k',w,100*f1(y,p.argmax(-1)))
    assert np.allclose(ref,[d['difference_pp'] for d in result['contrasts']])
    for i in [0,40,100]:
        assert np.array_equal(boots[i],100*f1(y[indices[i]],p.argmax(-1)[:,:,indices[i]]))
        assert np.array_equal(fused[i],100*f1(y[indices[i]],p.mean(1).argmax(-1)[:,indices[i]]))
    observed=np.array([2.,-3.,0.]);samples=np.tile(observed,(101,1));out,critical=summarize_bootstrap(observed,samples,3)
    assert critical is None and all(v['zero_variance'] and v['simultaneous_ci95_pp'] is None for v in out)
    assert classify([1.1,2])=='supports_substantive_improvement' and classify([-2,-1.1])=='supports_substantive_decline'
    assert classify([-.9,.9])=='supports_practical_similarity' and classify([0,2])=='unresolved'
    external=np.random.get_state();perms=derangements(296);assert np.array_equal(perms,derangements(296));assert all(np.all(p!=np.arange(296)) and np.array_equal(p[np.argsort(p)],np.arange(296)) for p in perms)
    assert all(np.array_equal(a,b) for a,b in zip(external,np.random.get_state()))
    torch.manual_seed(71);m=LinearMixer([3,5],3).double();torch.nn.init.normal_(m.conv.weight)
    x=[torch.randn(4,w,7,dtype=torch.float64) for w in [3,5]]
    assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(m(*x),block_messages(m,x,{})))
    z=block_messages(m,x,{(0,1):None,(1,0):None});changed=[x[0],x[1]+2]
    assert torch.equal(z[0],block_messages(m,changed,{(0,1):None})[0])
    print(json.dumps({'passed':True,'training_matrix':57,'total_results':186,'primary_contrasts':31,'shared_participant_bootstrap':True,'zero_variance_handling':True,'label_independent_derangements':True,'cross_only_message_overrides':True,'fixture_only':True}))
if __name__=='__main__':check()
