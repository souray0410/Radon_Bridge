import torch
import pytest

from radon_bridge.methods.basis import save_basis
from radon_bridge.methods.operator import BridgeExchange, FeatureSpec, LinearMixer


def _identity_grouped_weights(mixer):
    with torch.no_grad():
        mixer.conv.weight.zero_()
        channels_per_group=mixer.conv.in_channels//mixer.conv.groups
        for out_channel in range(mixer.conv.out_channels):
            mixer.conv.weight[out_channel,out_channel%channels_per_group,mixer.conv.kernel_size[0]//2]=1


def test_group_one_preserves_legacy_state_output_and_gradient():
    torch.manual_seed(17)
    legacy=LinearMixer([4,4],3)
    explicit=LinearMixer([4,4],3,group_count=1,source_ranks=[2,2],directions=[2,2])
    assert set(legacy.state_dict())==set(explicit.state_dict())=={'conv.weight','mask'}
    explicit.load_state_dict(legacy.state_dict(),strict=True)
    a=torch.randn(2,4,7,requires_grad=True)
    b=torch.randn(2,4,7,requires_grad=True)
    aa=a.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_()
    expected=legacy(a,b)
    actual=explicit(aa,bb)
    assert all(torch.equal(x,y) for x,y in zip(expected,actual))
    sum(x.sum() for x in expected).backward()
    sum(x.sum() for x in actual).backward()
    assert torch.equal(a.grad,aa.grad) and torch.equal(b.grad,bb.grad)


def test_grouped_order_round_trips_and_cross_source_gradient_stays_within_group():
    mixer=LinearMixer([2,2],1,group_count=2,source_ranks=[2,2],directions=[1,1])
    assert mixer.group_permutation.tolist()==[0,2,1,3]
    assert mixer.group_inverse.tolist()==[0,2,1,3]
    _identity_grouped_weights(mixer)
    a=torch.tensor([[[1.],[2.]]])
    b=torch.tensor([[[3.],[4.]]])
    out_a,out_b=mixer(a,b)
    assert torch.equal(out_a,a) and torch.equal(out_b,b)

    asymmetric=LinearMixer([8,18],1,group_count=2,source_ranks=[4,6],directions=[2,3])
    assert asymmetric.group_permutation.tolist()==[0,1,2,3,8,9,10,11,12,13,14,15,16,4,5,6,7,17,18,19,20,21,22,23,24,25]
    assert not torch.equal(asymmetric.group_permutation,asymmetric.group_inverse)
    _identity_grouped_weights(asymmetric)
    aa=torch.arange(8,dtype=torch.float32).reshape(1,8,1)
    bb=torch.arange(100,118,dtype=torch.float32).reshape(1,18,1)
    round_a,round_b=asymmetric(aa,bb)
    assert torch.equal(round_a,aa) and torch.equal(round_b,bb)

    with torch.no_grad():
        mixer.conv.weight.zero_()
        # Group zero is [source0/channel0, source1/channel0].
        mixer.conv.weight[0,1,0]=1
    a=torch.zeros(1,2,1,requires_grad=True)
    b=torch.ones(1,2,1,requires_grad=True)
    out_a,_=mixer(a,b)
    out_a[:,0].sum().backward()
    assert b.grad[0,0,0]!=0
    assert b.grad[0,1,0]==0


def test_grouped_structure_rejects_rank_splitting_and_other_controls():
    with pytest.raises(ValueError,match='divisible'):
        LinearMixer([6,6],1,group_count=4,source_ranks=[3,3],directions=[2,2])
    with pytest.raises(ValueError,match='cannot combine'):
        LinearMixer([4,4],1,self_only=True,group_count=2,source_ranks=[2,2],directions=[2,2])


def test_grouped_bridge_zero_initialization_and_checkpoint_structure(tmp_path):
    specs=[FeatureSpec('cfp_stage3',4,(2,2)),FeatureSpec('oct_stage3',4,(2,2,2))]
    refs={}
    for spec in specs:
        moment=torch.diag(torch.tensor([4.,3.,2.,1.],dtype=torch.float64))
        refs[spec.key]=save_basis(moment,8,tmp_path/spec.key,spec.key,3416,{'synthetic_grouped_test':True})
    bridge=BridgeExchange(specs,M=2,S=5,rho=.5,mode='radon',compression='fixed_svd_channel',basis_files=refs,r=2,h=4,group_count=2).double()
    assert bridge.mixer.conv.groups==2
    assert bridge.metadata['group_count']==2
    assert bridge.metadata['grouped_connection_fraction']==.5
    assert bridge.metadata['grouped_mixer_parameters']*2==bridge.metadata['dense_equivalent_mixer_parameters']
    xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64) for s in specs]
    assert torch.equal(bridge(*xs),torch.cat([x.flatten(1) for x in xs],1))
    state={k:v.clone() for k,v in bridge.state_dict().items()}
    first=state['mixer.group_permutation'][0].clone()
    state['mixer.group_permutation'][0]=state['mixer.group_permutation'][1]
    state['mixer.group_permutation'][1]=first
    clone=BridgeExchange(specs,M=2,S=5,rho=.5,mode='radon',compression='fixed_svd_channel',basis_files=refs,r=2,h=4,group_count=2).double()
    with pytest.raises(RuntimeError,match='structural permutation'):
        clone.load_state_dict(state,strict=True)
