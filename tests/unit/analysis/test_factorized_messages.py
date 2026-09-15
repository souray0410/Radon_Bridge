import copy
import pytest
import torch
from torch.nn import functional as F
from radon_bridge.methods.factorized import FactorizedMixer
from radon_bridge.analysis.factorized_messages import source_messages


@pytest.mark.parametrize('widths',[(3,5),(2,3,2,4,3,2)])
@pytest.mark.parametrize('kernel',[1,3,5])
def test_reconstruction_and_gradient_match_dense_formula_without_export(widths,kernel,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(82)
    m=FactorizedMixer(widths,4,kernel).double()
    with torch.no_grad():m.conv.weight.normal_()
    xs=[torch.randn(2,w,7,dtype=torch.float64,requires_grad=True) for w in widths]
    dense=m.dense_weight();before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    monkeypatch.setattr(m,'dense_weight',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No dense export')))
    actual=source_messages(m,xs)
    expected=F.conv1d(torch.cat(xs,1),dense,padding=kernel//2).split(widths,1)
    for a,b,c in zip(actual,expected,m(*xs)):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-10)
    targets=xs+list(m.parameters())
    ga=torch.autograd.grad(sum(v.square().sum() for v in actual),targets,retain_graph=True)
    gb=torch.autograd.grad(sum(v.square().sum() for v in expected),targets)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-9)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())


def test_source_edge_deletion_and_donor_preserve_receiver_and_other_destination():
    torch.manual_seed(11);m=FactorizedMixer([3,5],4,3).double()
    with torch.no_grad():m.conv.weight.normal_()
    x=[torch.randn(2,w,9,dtype=torch.float64,requires_grad=True) for w in m.widths]
    original=m(*x);deleted=source_messages(m,x,{(0,1):None})
    torch.testing.assert_close(deleted[1],original[1])
    grad=torch.autograd.grad(deleted[0].square().sum(),x,allow_unused=True,retain_graph=True)
    assert grad[0].abs().sum()>0 and grad[1] is None
    donor=x[1].detach().flip(0)
    perturbed=source_messages(m,x,{(0,1):donor})
    w=m.dense_weight()
    expected=F.conv1d(x[0],w[:3,:3],padding=1)+F.conv1d(donor,w[:3,3:],padding=1)
    torch.testing.assert_close(perturbed[0],expected)
    torch.testing.assert_close(perturbed[1],original[1])
    assert not torch.allclose(perturbed[0],original[0])


@pytest.mark.parametrize('overrides', [{(0,0):None},{(3,0):None},{(True,0):None},{(0,1):torch.zeros(1,4,7)}])
def test_reject_silent_source_identity_changes(overrides):
    m=FactorizedMixer([3,5],4)
    with pytest.raises(ValueError):source_messages(m,[torch.zeros(1,3,7),torch.zeros(1,5,7)],overrides)


def test_reject_latent_mask_and_keep_zero_initialization():
    m=FactorizedMixer([3,5],4);x=[torch.randn(1,w,7) for w in m.widths]
    assert all(torch.count_nonzero(v)==0 for v in source_messages(m,x))
    m.mask[0]=0
    with pytest.raises(ValueError,match='latent mask'):source_messages(m,x)
