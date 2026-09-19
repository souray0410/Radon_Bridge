import torch
import torch.nn.functional as F
import pytest

from radon_bridge.methods.baselines import CMXRectifyExchange
from radon_bridge.methods.operator import FeatureSpec
from radon_bridge.studies.project_build import bridges_for


def _unpack(packet, x1, x2):
    n1=x1[0].numel()
    return packet[:,:n1].reshape_as(x1),packet[:,n1:].reshape_as(x2)


def test_cmx_frm_equal_grid_matches_author_formula():
    torch.manual_seed(31)
    specs=[FeatureSpec("cfp_stage3",4,(2,3)),FeatureSpec("oct_stage3",4,(2,3))]
    module=CMXRectifyExchange(specs,alignment_tokens=6)
    x1=torch.randn(2,4,2,3);x2=torch.randn(2,4,2,3)
    out1,out2=_unpack(module(x1,x2),x1,x2)

    joined=torch.cat((x1,x2),1)
    avg=F.adaptive_avg_pool2d(joined,1).flatten(1)
    maximum=F.adaptive_max_pool2d(joined,1).flatten(1)
    y=torch.cat((avg,maximum),1)
    y=F.linear(y,module.channel_mlp[0].weight,module.channel_mlp[0].bias)
    y=F.relu(y)
    y=torch.sigmoid(F.linear(y,module.channel_mlp[2].weight,module.channel_mlp[2].bias))
    channel=y.reshape(2,2,4,1,1)

    w0=module.spatial_mlp[0].weight.unsqueeze(-1)
    z=F.conv2d(joined,w0,module.spatial_mlp[0].bias)
    z=F.relu(z)
    w1=module.spatial_mlp[2].weight.unsqueeze(-1)
    spatial=torch.sigmoid(F.conv2d(z,w1,module.spatial_mlp[2].bias))

    expected1=x1+.5*channel[:,1]*x2+.5*spatial[:,1:2]*x2
    expected2=x2+.5*channel[:,0]*x1+.5*spatial[:,0:1]*x1
    torch.testing.assert_close(out1,expected1,rtol=0,atol=1e-6)
    torch.testing.assert_close(out2,expected2,rtol=0,atol=1e-6)


def test_cmx_frm_cross_dimensional_adapter_shapes_and_gradients():
    torch.manual_seed(37)
    specs=[FeatureSpec("cfp_stage3",8,(3,4)),FeatureSpec("oct_stage3",8,(2,3,4))]
    module=CMXRectifyExchange(specs,alignment_tokens=16)
    x1=torch.randn(2,8,3,4,requires_grad=True)
    x2=torch.randn(2,8,2,3,4,requires_grad=True)
    packet=module(x1,x2)
    assert packet.shape==(2,x1[0].numel()+x2[0].numel())
    y1,y2=_unpack(packet,x1,x2)
    assert y1.shape==x1.shape and y2.shape==x2.shape
    assert torch.isfinite(packet).all()
    packet.square().mean().backward()
    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert module.metadata["author_commit"]=="e251d860aebc2f583a6c4919877e6bebe7f1aff3"
    assert module.metadata["author_license"]=="MIT"
    assert "not full CMX" in module.metadata["author_component"]


def test_cmx_frm_requires_equal_channels():
    specs=[FeatureSpec("cfp_stage3",8,(2,2)),FeatureSpec("oct_stage3",16,(2,2,2))]
    with pytest.raises(ValueError,match="equal channel"):
        CMXRectifyExchange(specs,alignment_tokens=8)


def test_project_builder_emits_explicit_cmx_frm_config():
    class Graph:
        feature_channels={"stage3":256}
    class Parent:
        graph=Graph()
    parents={"cfp":Parent(),"oct":Parent()}
    arm={"family":"cmx_frm","stages":[3],"alignment_tokens":64}
    cfg=bridges_for(parents,{},arm,{},3416)
    assert cfg==[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_frm","alignment_tokens":64}]
