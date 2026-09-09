"""S adjacency acceptance: independent matrix reference and MHD integration."""
import copy,json,tempfile,random
from pathlib import Path
import numpy as np
import torch
from radon_bridge.bridge import LinearMixer,BridgeExchange,FeatureSpec
from radon_bridge.projector import Projector
from radon_bridge.s_axis import fixed_permutation
from check_integer_bridge import topology_check
from check_fixed_channel_basis import artifact,checkpoint_optimizer_checks
from radon_bridge.svd_basis import save_random_basis

def run():
    torch.set_num_threads(3);torch.manual_seed(761)
    before=torch.random.get_rng_state().clone();nb=np.random.get_state();rb=random.getstate()
    spec=fixed_permutation(11);p=spec['permutation'];assert spec==fixed_permutation(11)
    assert torch.equal(before,torch.random.get_rng_state()) and random.getstate()==rb
    assert all(np.array_equal(a,b) for a,b in zip(nb,np.random.get_state()))
    results={}
    for k in (1,3):
        a=LinearMixer([2,3],k).double();torch.nn.init.normal_(a.conv.weight,std=.1)
        b=LinearMixer([2,3],k,s_axis_permutation=p).double();b.conv.load_state_dict(a.conv.state_dict())
        x=torch.randn(2,5,11,dtype=torch.float64,requires_grad=True)
        ordered=torch.cat(a(*x.split([2,3],1)),1);permuted=torch.cat(b(*x.split([2,3],1)),1)
        if k==1:assert torch.equal(ordered,permuted)
        else:assert not torch.allclose(ordered,permuted)
        restored=copy.deepcopy(b);restored.load_state_dict(b.state_dict(),strict=True)
        assert torch.equal(torch.cat(restored(*x.split([2,3],1)),1),permuted)
        bad=copy.deepcopy(b.state_dict());bad['s_axis_permutation']=bad['s_axis_permutation'].roll(1)
        try:restored.load_state_dict(bad)
        except RuntimeError:pass
        else:raise AssertionError('Mismatched permutation checkpoint accepted')
        assert set(a.state_dict())=={'mask','conv.weight'}
        assert sum(v.numel() for v in a.parameters())==sum(v.numel() for v in b.parameters())
    # Independently materialize W with row=output and column=input; assemble B P^-1 W P A.
    for shape in ((3,4),(3,3,4)):
        m,S=3,11;pr=Projector(shape,m,S).double();n=int(np.prod(shape))
        mix=LinearMixer([m],3,s_axis_permutation=p).double();torch.nn.init.normal_(mix.conv.weight,std=.1)
        W=torch.zeros(m*S,m*S,dtype=torch.float64)
        for dst in range(m):
            for src in range(m):
                for t in range(S):
                    for offset in (-1,0,1):
                        if 0<=t+offset<S:W[dst*S+t,src*S+t+offset]=mix.conv.weight[dst,src,offset+1]
        P=torch.eye(S,dtype=torch.float64)[p];P=torch.kron(torch.eye(m,dtype=torch.float64),P)
        B=pr.backproject(torch.eye(m*S,dtype=torch.float64).reshape(m*S,m,S)).reshape(m*S,n).T
        K=B@P.T@W@P@pr.matrix
        x=torch.randn(2,1,*shape,dtype=torch.float64,requires_grad=True)
        actual=pr.backproject(mix(pr(x))[0]).flatten(1);expected=x.flatten(1)@K.T
        assert torch.allclose(actual,expected,atol=1e-11,rtol=1e-11)
        ga=torch.autograd.grad(actual.square().sum(),(x,mix.conv.weight),retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),(x,mix.conv.weight))
        assert all(torch.allclose(u,v,atol=1e-10,rtol=1e-10) for u,v in zip(ga,gb))
        results[str(shape)]={'matrix_error':float((actual-expected).abs().max().detach()),'gradient_error':max(float((u-v).abs().max()) for u,v in zip(ga,gb))}
    with tempfile.TemporaryDirectory() as temp:
        refs={key:artifact(Path(temp),c,key)[0] for key,c in [('network0_stage1',2),('network1_stage2',3)]}
        qr={key:save_random_basis(ref,Path(temp)/'qr') for key,ref in refs.items()}
        top=[topology_check([2,3],compression=comp,basis_files=files,s_axis_permutation=p) for comp,files in [('fixed_svd_channel',refs),('fixed_random_orthogonal_channel',qr)]]
        # At bridge level linear source blocks sum exactly, including reverse/own blocks.
        specs=[FeatureSpec('network0_stage1',2,(3,4)),FeatureSpec('network1_stage2',3,(3,4,3))]
        kw=dict(M=4,S=11,rho=.5,compression='fixed_svd_channel',basis_files=refs)
        legacy=BridgeExchange(specs,**kw).double();new=BridgeExchange(specs,**kw,s_axis_permutation=p).double()
        assert list(dict(new.named_parameters()))==['mixer.conv.weight']
        xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64) for s in specs]
        assert torch.equal(legacy(*xs),new(*xs))
        torch.nn.init.normal_(new.mixer.conv.weight,std=.1)
        all_=new(*xs);parts=new(xs[0],torch.zeros_like(xs[1]))+new(torch.zeros_like(xs[0]),xs[1])
        assert torch.allclose(all_,parts,atol=1e-11,rtol=1e-11)
        for overrides in [{'nested_rhos':[.5]},{'mode':'scrambled'},{'cross_edges':[]},{'s_axis_permutation':[0]*11}]:
            try:BridgeExchange(specs,**(kw|{'s_axis_permutation':p}|overrides))
            except ValueError:pass
            else:raise AssertionError(overrides)
        optimizer=checkpoint_optimizer_checks(temp,bridge_overrides={'s_axis_permutation':p})
    return dict(passed=True,optimizer=optimizer,explicit_operator=results,mhd=top,kernel1_cancels=True,kernel3_changes=True,source_blocks_add=True,default_state_keys_unchanged=True,rng_preserved=True,permutation=spec)

if __name__=='__main__':print(json.dumps(run()))
