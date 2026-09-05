"""Centered-fit basis semantics, exact linear runtime and MHD compatibility."""
import json,tempfile
from pathlib import Path
import torch
from radonbridge.svd_basis import save_centered_basis,save_basis,FixedChannelBasis,CENTERED_VERSION
from radonbridge.bridge import FeatureSpec,BridgeExchange
from radonbridge.projector import Projector
from check_integer_bridge import topology_check
from check_fixed_channel_basis import checkpoint_optimizer_checks
from scripts.run_centered_study import matrix


def centered_artifact(root,c,key,seed=3416):
    gen=torch.Generator().manual_seed(seed);f=torch.randn(c,100,generator=gen,dtype=torch.float64)*torch.linspace(2.,.3,c)[:,None]+torch.arange(c,dtype=torch.float64)[:,None]*3
    return save_centered_basis(f@f.T,f.sum(1),f.shape[1],root,key,seed,{'synthetic_test':True}),f


def check():
    torch.set_num_threads(3)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);before=torch.random.get_rng_state().clone();ref,f=centered_artifact(root,8,'a');assert torch.equal(before,torch.random.get_rng_state())
        q=FixedChannelBasis(8,4,'a',ref,CENTERED_VERSION);x=f-f.mean(1,keepdim=True);u,s,_=torch.linalg.svd(x,full_matrices=False)
        assert torch.allclose(q.q@q.q.T,u[:,:4]@u[:,:4].T,atol=1e-10,rtol=1e-10)
        assert abs(q.metadata['retained_variance_ratio']-float(s[:4].square().sum()/s.square().sum()))<1e-10
        assert abs(q.metadata['retained_energy_ratio']-float((q.q.T@f).square().sum()/f.square().sum()))<1e-10
        assert q.metadata['runtime_centering'] is False and not list(q.parameters())
        shifted=f+torch.arange(8,dtype=torch.float64)[:,None]*5
        translated=save_centered_basis(shifted@shifted.T,shifted.sum(1),100,root/'shift','a',3416,{})
        tq=FixedChannelBasis(8,4,'a',translated,CENTERED_VERSION)
        assert torch.allclose(q.q@q.q.T,tq.q@tq.q.T,atol=1e-10,rtol=1e-10)
        narrow=FixedChannelBasis(8,2,'a',ref,CENTERED_VERSION);assert torch.equal(narrow.q,q.q[:,:2])
        raw=save_basis(f@f.T,100,root/'raw','a',3416,{})
        rawq=FixedChannelBasis(8,4,'a',raw)
        assert not torch.allclose(rawq.q@rawq.q.T,q.q@q.q.T,atol=1e-5)
        try:FixedChannelBasis(8,4,'a',ref)
        except ValueError:pass
        else:raise AssertionError('Accepted centered artifact as legacy uncentered basis')
        for shape in [(4,5),(3,4,5)]:
            x=torch.randn(2,8,*shape,dtype=torch.float64);p=Projector(shape,4,11)
            assert torch.allclose(p(q.encode(x)),q.encode(p(x).reshape(2,8,4,11)).reshape(2,16,11),atol=1e-12,rtol=1e-12)
            z=torch.randn(2,16,11,dtype=torch.float64)
            assert torch.allclose(q.decode(p.backproject(z)),p.backproject(q.decode(z.reshape(2,4,4,11)).reshape(2,32,11)),atol=1e-12,rtol=1e-12)
        refs={k:centered_artifact(root,c,k)[0] for k,c in [('network0_stage1',2),('network1_stage2',3)]}
        results=[topology_check([2,3],compression='fixed_centered_svd_channel',basis_files=refs,mode=mode) for mode in ['radon','self','scrambled','linear_resample']]
        specs=[FeatureSpec('a',8,(4,5)),FeatureSpec('b',6,(3,4,5))];refs={'a':ref,'b':centered_artifact(root,6,'b')[0]}
        kw=dict(M=4,S=11,rho=.5,compression='fixed_centered_svd_channel',basis_files=refs)
        m=BridgeExchange(specs,**kw).double();assert list(dict(m.named_parameters()))==['mixer.conv.weight']
        xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64) for s in specs];assert torch.equal(m(*xs),torch.cat([x.flatten(1) for x in xs],1))
        torch.nn.init.normal_(m.mixer.conv.weight,std=.01);copy=BridgeExchange(specs,**kw).double();copy.load_state_dict(m.state_dict());assert torch.equal(m(*xs),copy(*xs))
        xs0=[torch.zeros_like(x) for x in xs];assert m(*xs0).abs().sum()==0,'Runtime introduced an affine shift'
        assert len(matrix())==len(set(matrix()))==36 and {s for s,_,_ in matrix()}=={3416,3417,3418}
        optimizer=checkpoint_optimizer_checks(root/'network',compression='fixed_centered_svd_channel',basis_factory=centered_artifact)
        print(json.dumps({'passed':True,'checkpoint_optimizer':optimizer,'centered_matches_explicit_svd':True,'translation_invariant_fit':True,'raw_and_variance_metrics_separate':True,'rng_preserved':True,'runtime_no_mean_shift':True,'nested_ranks':True,'only_conv_trainable':True,'checkpoint_reload_exact':True,'paired_matrix':36,'topology':results}))

if __name__=='__main__':check()
