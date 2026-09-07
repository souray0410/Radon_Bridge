"""Independent algebra, direction display and participant weighting acceptance."""
import numpy as np
import torch
from radonbridge.bridge import LinearMixer
from radonbridge.sinogram_atlas import block_decompose,angle_display,energy_terms,Aggregator,check_linear_sum


def run():
    torch.set_num_threads(3);torch.manual_seed(73)
    for self_only in (False,True):
        m=LinearMixer([3,5],3,self_only=self_only).double();torch.nn.init.normal_(m.conv.weight,std=.1)
        x=[torch.randn(4,c,7,dtype=torch.float64,requires_grad=True) for c in (3,5)]
        actual=m(*x);parts=block_decompose(m,x)
        for dst,(a,b) in enumerate(parts):
            torch.testing.assert_close(a+b,actual[dst],rtol=1e-12,atol=1e-12)
            grad=torch.autograd.grad((a+b).square().sum(),x[1-dst],retain_graph=True)[0]
            assert bool(grad.abs().sum()>0)==(not self_only)
            if self_only:assert torch.count_nonzero(b)==0
        check_linear_sum(*parts[0],actual[0])
    # Reversing a normal reverses the projection distance, not merely the row order.
    n=np.array([[1.,0.],[-1.,0.],[0.,1.]])
    z=np.array([[1,2,3],[3,2,1],[4,5,6]])
    values,theta,meta=angle_display(z,n)
    np.testing.assert_array_equal(values[:2],[[1,2,3],[1,2,3]])
    assert meta['s_reversed']==[False,True,False]
    x=torch.ones(4,1,2,2);a=x.clone();b=-a
    e=energy_terms(x,a,b);assert torch.equal(e['total_relative'],torch.zeros(2,dtype=torch.float64))
    assert torch.equal(e['self_cross_cosine'],-torch.ones(2,dtype=torch.float64))
    assert e['energy_identity_relative_error'].max()==0
    z=energy_terms(x,torch.zeros_like(x),torch.zeros_like(x));assert torch.isnan(z['self_cross_cosine']).all()
    # Two equal-weight people, irrespective of labels, eyes or the final batch size.
    agg=Aggregator();meta=dict(M=1,S=4,shape=[2,2])
    t=torch.arange(16,dtype=torch.float64).reshape(4,1,4)
    agg.update('x',meta,{'projection':t},{'input':t.reshape(4,1,2,2)},z,np.array([0,1]))
    out=agg.export();v=out['x/all'];assert v['participants']==2
    np.testing.assert_allclose(v['arrays']['projection_ms'],t.square().mean(0).numpy())
    np.testing.assert_allclose(np.sum(v['arrays']['projection_spectrum']),t.square().sum(-1).mean().numpy())
    assert v['scalars']['self_cross_cosine']['mean'] is None
    assert v['scalars']['self_cross_cosine']['undefined_participants']==2
    assert out['x/label0']['participants']==1
    collector_checks()
    print('PASS: masked block decomposition, gradients, signed angle display, destructive interference, undefined zero cosine, participant weighting, Parseval spectrum')


def collector_checks():
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from types import SimpleNamespace
    from check_fixed_channel_basis import artifact
    from radonbridge.bridge import BridgeExchange,FeatureSpec
    from radonbridge.sinogram_atlas import attach_collectors
    specs=[FeatureSpec('a',6,(3,4)),FeatureSpec('b',8,(3,4,3))]
    with TemporaryDirectory() as tmp:
        refs={s.key:artifact(tmp,s.channels,s.key)[0] for s in specs}
        for mode in ('radon','linear_resample'):
            ex=BridgeExchange(specs,M=4,S=11,rho=.5,compression='fixed_svd_channel',basis_files=refs,mode=mode).double()
            torch.nn.init.normal_(ex.mixer.conv.weight,std=.01)
            fake=SimpleNamespace(communication_groups=[dict(exchange_edge_name='exchange')],modules_by_name=lambda:{'exchange':ex})
            xs=[torch.randn(4,s.channels,*s.shape,dtype=torch.float64) for s in specs]
            agg=Aggregator();context=dict(labels=np.array([0,1]),ids=['first','second'],sample_ids={'first'})
            out=Path(tmp)/mode;out.mkdir()
            expected=ex(*xs).detach().clone();handles=attach_collectors(fake,agg,context,out)
            state={k:v.clone() for k,v in ex.state_dict().items()};rng=torch.random.get_rng_state().clone()
            with torch.no_grad():actual=ex(*xs)
            for h in handles:h.remove()
            assert torch.equal(expected,actual) and torch.equal(rng,torch.random.get_rng_state())
            assert all(torch.equal(v,ex.state_dict()[k]) for k,v in state.items())
            assert len(agg.export())==6 and len(list(out.glob('*.npz')))==2
            for source in ('a','b'):
                with np.load(out/('first__bridge0_'+source+'.npz'),allow_pickle=False) as z:
                    assert z['projection'].shape==(2,3,4,11)
            assert max(e['packet']['max_absolute'] for e in agg.errors)<1e-12


if __name__=='__main__':run()
