import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import types

import torch

from radon_bridge.methods.baselines import CMXFullExchange, CMXRectifyExchange
from radon_bridge.methods.operator import FeatureSpec, attach_group
from radon_bridge.studies.project_build import bridges_for


AUTHOR_NET_UTILS_SHA256="ada5e36e14d83c35d9230618c4eb84352a76ec2275a086a624b280e7638d8473"


def _author_module():
    path=Path(__file__).resolve().parents[2]/"fixtures/cmx_author_e251d860/net_utils.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==AUTHOR_NET_UTILS_SHA256
    # The pinned author file imports only trunc_normal_ from timm.  The test
    # supplies the equivalent torch initializer so it executes the exact
    # vendored author class bodies without making timm a production dependency.
    saved={name:sys.modules.get(name) for name in ("timm","timm.models","timm.models.layers")}
    timm=types.ModuleType("timm");timm.__path__=[]
    models=types.ModuleType("timm.models");models.__path__=[]
    layers=types.ModuleType("timm.models.layers");layers.trunc_normal_=torch.nn.init.trunc_normal_
    sys.modules["timm"]=timm;sys.modules["timm.models"]=models;sys.modules["timm.models.layers"]=layers
    try:
        spec=importlib.util.spec_from_file_location("cmx_author_e251d860_net_utils",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return module


def _author_core(dim=8,heads=2):
    source=_author_module()
    frm=source.FeatureRectifyModule(dim=dim,reduction=1,lambda_c=.5,lambda_s=.5).eval()
    ffm=source.FeatureFusionModule(dim=dim,reduction=1,num_heads=heads).eval()
    return frm,ffm


def _load_author_weights(project,author_frm,author_ffm):
    assert project.frm.state_dict().keys()==author_frm.state_dict().keys()
    assert project.ffm.state_dict().keys()==author_ffm.state_dict().keys()
    project.frm.load_state_dict(author_frm.state_dict(),strict=True)
    project.ffm.load_state_dict(author_ffm.state_dict(),strict=True)


def _assert_parameter_gradients_equal(project,author):
    p=dict(project.named_parameters());a=dict(author.named_parameters())
    assert p.keys()==a.keys()
    for name in p:
        assert p[name].grad is not None and a[name].grad is not None
        torch.testing.assert_close(p[name].grad,a[name].grad,rtol=1e-6,atol=1e-7)


def _load_author_frm_into_historical(historical,author):
    with torch.no_grad():
        historical.channel_mlp[0].weight.copy_(author.channel_weights.mlp[0].weight)
        historical.channel_mlp[0].bias.copy_(author.channel_weights.mlp[0].bias)
        historical.channel_mlp[2].weight.copy_(author.channel_weights.mlp[2].weight)
        historical.channel_mlp[2].bias.copy_(author.channel_weights.mlp[2].bias)
        historical.spatial_mlp[0].weight.copy_(author.spatial_weights.mlp[0].weight.squeeze(-1))
        historical.spatial_mlp[0].bias.copy_(author.spatial_weights.mlp[0].bias)
        historical.spatial_mlp[2].weight.copy_(author.spatial_weights.mlp[2].weight.squeeze(-1))
        historical.spatial_mlp[2].bias.copy_(author.spatial_weights.mlp[2].bias)
        historical.residual_gate.fill_(1.)


def _unpack(packet,*features):
    values=[];start=0
    for x in features:
        length=x[0].numel()
        values.append(packet[:,start:start+length].reshape_as(x));start+=length
    return values


def test_cmx_full_same_grid_matches_pinned_author_source_forward_inputs_and_parameter_gradients():
    torch.manual_seed(733)
    specs=[FeatureSpec("cfp_stage3",8,(4,4)),FeatureSpec("oct_stage3",8,(4,4))]
    project=CMXFullExchange(specs,alignment_tokens=16,heads=2).eval()
    author_frm,author_ffm=_author_core();_load_author_weights(project,author_frm,author_ffm)
    px1=torch.randn(2,8,4,4,requires_grad=True);px2=torch.randn(2,8,4,4,requires_grad=True)
    ax1=px1.detach().clone().requires_grad_(True);ax2=px2.detach().clone().requires_grad_(True)
    got=project.core_forward_shared(px1,px2)
    expected=author_ffm(*author_frm(ax1,ax2))
    torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-7)
    probe=torch.randn_like(got)
    (got*probe).sum().backward();(expected*probe).sum().backward()
    torch.testing.assert_close(px1.grad,ax1.grad,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(px2.grad,ax2.grad,rtol=1e-6,atol=1e-7)
    _assert_parameter_gradients_equal(project.frm,author_frm)
    _assert_parameter_gradients_equal(project.ffm,author_ffm)


def test_left_0058_native_to_shared_align_first_matches_pinned_author_source_and_not_native_first():
    torch.manual_seed(741)
    specs=[FeatureSpec("cfp_stage3",8,(3,4)),FeatureSpec("oct_stage3",8,(2,3,4))]
    project=CMXFullExchange(specs,alignment_tokens=16,heads=2).eval()
    author_frm,author_ffm=_author_core();_load_author_weights(project,author_frm,author_ffm)
    with torch.no_grad():project.return_gate.fill_(1.)
    px1=torch.randn(2,8,3,4,requires_grad=True);px2=torch.randn(2,8,2,3,4,requires_grad=True)
    ax1=px1.detach().clone().requires_grad_(True);ax2=px2.detach().clone().requires_grad_(True)
    got1,got2=_unpack(project(px1,px2),px1,px2)
    shared1=project._align_to_shared(ax1);shared2=project._align_to_shared(ax2)
    fused=author_ffm(*author_frm(shared1,shared2))
    expected1=ax1+CMXRectifyExchange._resize(fused.flatten(2),ax1.flatten(2).shape[-1]).reshape_as(ax1)
    expected2=ax2+CMXRectifyExchange._resize(fused.flatten(2),ax2.flatten(2).shape[-1]).reshape_as(ax2)
    torch.testing.assert_close(got1,expected1,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(got2,expected2,rtol=1e-6,atol=1e-7)

    historical=CMXRectifyExchange(specs,alignment_tokens=16).eval()
    _load_author_frm_into_historical(historical,author_frm)
    old1,old2=_unpack(historical(px1.detach(),px2.detach()),px1,px2)
    old_shared1=project._align_to_shared(old1);old_shared2=project._align_to_shared(old2)
    old_fused=author_ffm(old_shared1,old_shared2)
    assert float((old_fused-fused.detach()).abs().max().detach())>.05

    probe1=torch.randn_like(got1);probe2=torch.randn_like(got2)
    ((got1*probe1).sum()+(got2*probe2).sum()).backward()
    ((expected1*probe1).sum()+(expected2*probe2).sum()).backward()
    torch.testing.assert_close(px1.grad,ax1.grad,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(px2.grad,ax2.grad,rtol=1e-6,atol=1e-7)
    _assert_parameter_gradients_equal(project.frm,author_frm)
    _assert_parameter_gradients_equal(project.ffm,author_ffm)


def test_cmx_full_cross_dimensional_identity_and_gradients():
    torch.manual_seed(223)
    specs=[FeatureSpec("cfp_stage3",8,(3,4)),FeatureSpec("oct_stage3",8,(2,3,4))]
    module=CMXFullExchange(specs,alignment_tokens=16,heads=2)
    x1=torch.randn(2,8,3,4,requires_grad=True)
    x2=torch.randn(2,8,2,3,4,requires_grad=True)
    y1,y2=_unpack(module(x1,x2),x1,x2)
    torch.testing.assert_close(y1,x1,rtol=0,atol=0)
    torch.testing.assert_close(y2,x2,rtol=0,atol=0)
    (y1.square().mean()+y2.square().mean()).backward()
    assert module.return_gate.grad is not None
    assert torch.isfinite(module.return_gate.grad).all()
    assert torch.count_nonzero(module.return_gate.grad)>0
    assert module.metadata["author_commit"]=="e251d860aebc2f583a6c4919877e6bebe7f1aff3"
    assert "FeatureRectifyModule + FeatureFusionModule" in module.metadata["author_component"]

    module.zero_grad(set_to_none=True)
    with torch.no_grad():module.return_gate.fill_(1.)
    a=torch.randn(2,8,3,4,requires_grad=True)
    b=torch.randn(2,8,2,3,4,requires_grad=True)
    packet=module(a,b)
    packet.square().mean().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    core_grads=[p.grad for p in list(module.frm.parameters())+list(module.ffm.parameters()) if p.requires_grad]
    assert core_grads and all(g is not None and torch.isfinite(g).all() for g in core_grads)
    assert any(torch.count_nonzero(g)>0 for g in core_grads)


def test_cmx_full_requires_square_lattice_and_equal_channels():
    specs=[FeatureSpec("a",8,(2,2)),FeatureSpec("b",8,(2,2,2))]
    try:
        CMXFullExchange(specs,alignment_tokens=15,heads=2)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("nonsquare lattice accepted")
    bad=[FeatureSpec("a",8,(2,2)),FeatureSpec("b",16,(2,2,2))]
    try:
        CMXFullExchange(bad,alignment_tokens=16,heads=2)
    except ValueError as exc:
        assert "equal channel" in str(exc)
    else:
        raise AssertionError("unequal channels accepted")
    three=[FeatureSpec("a",8,(2,2)),FeatureSpec("b",8,(2,2,2)),FeatureSpec("c",8,(4,))]
    try:
        CMXFullExchange(three,alignment_tokens=16,heads=2)
    except ValueError as exc:
        assert "two distinct sources" in str(exc)
    else:
        raise AssertionError("three-modality CMX core accepted")


def test_project_builder_emits_cmx_full_core_config():
    class Graph:
        feature_channels={"stage3":256}
    class Parent:
        graph=Graph()
    parents={"cfp":Parent(),"oct":Parent()}
    arm={"family":"cmx_full","stages":[3],"alignment_tokens":64,"heads":4}
    cfg=bridges_for(parents,{},arm,{},3416)
    assert cfg==[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_full","alignment_tokens":64,"heads":4}]


def test_operator_routes_explicit_cmx_full_family_without_radon_fields():
    specs=[FeatureSpec("cfp_stage3",8,(2,2)),FeatureSpec("oct_stage3",8,(2,2,2))]
    inputs={"cfp_stage3":11,"oct_stage3":12};created=[];edges=[]
    def node(name):
        created.append(name);return 100+len(created)
    def edge(name,module,heads,tails,**kwargs):
        edges.append((name,module,tuple(heads),tuple(tails),kwargs))
    returned,meta=attach_group(
        node,edge,specs,inputs,"cmx_",family="cmx_full",alignment_tokens=16,heads=2)
    assert returned==inputs
    assert meta["family"]=="cmx_full"
    assert meta["author_commit"]=="e251d860aebc2f583a6c4919877e6bebe7f1aff3"
    assert [row[0] for row in edges]==["cmx_exchange","cmx_cfp_stage3_return","cmx_oct_stage3_return"]


def test_cmx_full_optimizer_resume_matches_uninterrupted_next_update():
    torch.manual_seed(227)
    specs=[FeatureSpec("cfp_stage3",8,(3,4)),FeatureSpec("oct_stage3",8,(2,3,4))]
    module=CMXFullExchange(specs,alignment_tokens=16,heads=2).train()
    optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    x1=torch.randn(3,8,3,4);x2=torch.randn(3,8,2,3,4)

    def update(model,opt):
        opt.zero_grad(set_to_none=True)
        packet=model(x1,x2)
        loss=packet.square().mean()
        loss.backward();opt.step()
        return float(loss.detach())

    first=update(module,optimizer)
    checkpoint_model=copy.deepcopy(module.state_dict())
    checkpoint_optimizer=copy.deepcopy(optimizer.state_dict())
    uninterrupted_loss=update(module,optimizer)
    uninterrupted=copy.deepcopy(module.state_dict())

    torch.manual_seed(991)
    resumed=CMXFullExchange(specs,alignment_tokens=16,heads=2).train()
    resumed_optimizer=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    resumed.load_state_dict(checkpoint_model,strict=True)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    resumed_loss=update(resumed,resumed_optimizer)

    assert math.isfinite(first) and math.isfinite(uninterrupted_loss)
    assert resumed_loss==uninterrupted_loss
    for key,value in uninterrupted.items():
        torch.testing.assert_close(resumed.state_dict()[key],value,rtol=0,atol=0)
