import copy
import pytest
import torch
from radon_bridge.methods.factorized import FactorizedMixer
from radon_bridge.methods.operator import BridgeExchange, FeatureSpec


@pytest.mark.parametrize('kernel',[1,3,5])
def test_dense_equivalence_outputs_inputs_and_factor_gradients(kernel):
    torch.manual_seed(37)
    m=FactorizedMixer([4,6],3,kernel).double()
    torch.nn.init.normal_(m.conv.weight,std=.1)
    x=torch.randn(2,10,7,dtype=torch.double,requires_grad=True)
    target=torch.randn_like(x)
    a=torch.cat(m(*x.split([4,6],1)),1)
    params=[x,*m.parameters()]
    ga=torch.autograd.grad((a*target).sum(),params)
    b=torch.nn.functional.conv1d(x,m.dense_weight(),padding=kernel//2)
    gb=torch.autograd.grad((b*target).sum(),params)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for p,q in zip(ga,gb):torch.testing.assert_close(p,q,rtol=1e-11,atol=1e-11)
    assert sum(p.numel() for p in m.parameters())==2*10*3+kernel*3*3


def test_zero_then_updates_reach_all_factors_and_sources_and_resume():
    torch.manual_seed(4);m=FactorizedMixer([4,6],3)
    opt=torch.optim.AdamW(m.parameters(),lr=.01)
    x=torch.randn(2,10,7);target=torch.randn(2,10,7)
    assert torch.count_nonzero(torch.cat(m(*x.split([4,6],1)),1))==0
    def step(model,optimizer):
        optimizer.zero_grad();y=torch.cat(model(*x.split([4,6],1)),1)
        (y-target).square().mean().backward();optimizer.step()
    step(m,opt)
    assert m.conv.weight.grad.abs().sum()>0
    assert m.encoder.weight.grad.abs().sum()==0
    step(m,opt)
    assert all(p.grad.abs().sum()>0 for p in m.parameters())
    z=x.clone().requires_grad_(True)
    loss=m(*z.split([4,6],1))[0].square().sum()
    assert torch.autograd.grad(loss,z)[0][:,4:].abs().sum()>0
    clone=copy.deepcopy(m);co=torch.optim.AdamW(clone.parameters(),lr=.01)
    co.load_state_dict(copy.deepcopy(opt.state_dict()))
    step(m,opt);step(clone,co)
    for p,q in zip(m.parameters(),clone.parameters()):assert torch.equal(p,q)


@pytest.mark.parametrize('mode',['radon','linear_resample'])
def test_exchange_linearity_full_inputs_and_original_shapes(mode):
    torch.manual_seed(7)
    specs=[FeatureSpec('cfp',3,(4,4)),FeatureSpec('oct',2,(4,4,4))]
    m=BridgeExchange(specs,M=2,S=8,rho=1,mode=mode,
                     compression='factorized_projected',bottleneck_rank=3).double()
    x=[torch.randn(2,s.channels,*s.shape,dtype=torch.double) for s in specs]
    assert torch.equal(m(*x),torch.cat([z.flatten(1) for z in x],1))
    assert m.export_fixed_bases('unused')==[]
    torch.nn.init.normal_(m.mixer.conv.weight,std=.01)
    y=[torch.randn_like(z) for z in x]
    torch.testing.assert_close(m(*[a+2*b for a,b in zip(x,y)]),m(*x)+2*m(*y),atol=1e-10,rtol=1e-10)
    assert not any(isinstance(v,(torch.nn.ReLU,torch.nn.BatchNorm1d)) for v in m.modules())
    clone=copy.deepcopy(m);clone.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(m(*x),clone(*x))


def test_invalid_controls_fail_explicitly():
    specs=[FeatureSpec('a',2,(4,4)),FeatureSpec('b',2,(4,4))]
    for kwargs in [dict(mode='self'),dict(rho=.5),dict(cross_edges=[['a','b']]),dict(bottleneck_rank=100)]:
        cfg=dict(M=2,S=8,rho=1,compression='factorized_projected',bottleneck_rank=3)
        cfg.update(kwargs)
        with pytest.raises(ValueError):BridgeExchange(specs,**cfg)
