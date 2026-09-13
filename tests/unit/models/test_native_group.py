from types import SimpleNamespace
import copy
import pytest
import torch
from torch import nn
from mhd_framework.models.graph import NamedModelGraph
from radon_bridge.models.observed_participant import ObservedParticipantModel
from radon_bridge.models.native_group import NativeGroup
from radon_bridge.studies.complete_matrix import groups


class Spatial(nn.Module):
    def __init__(self, dims, last):
        super().__init__(); self.conv=(nn.Conv2d if dims==2 else nn.Conv3d)(1,4,1); self.last=last
    def forward(self,x):
        x=self.conv(x); return x.movedim(1,-1) if self.last else x


class Pool(nn.Module):
    def __init__(self,last): super().__init__();self.last=last
    def forward(self,x):
        if self.last:x=x.movedim(-1,1)
        return x.flatten(2).mean(-1)


def fixture(device='cpu'):
    torch.manual_seed(414)
    sources=copy.deepcopy(next(g for g in groups() if len(g['sources'])==6)['sources'])
    # One NHWC source exercises Swin layout without substituting for real acceptance.
    sources[0]['model']='swin_b'
    parents={};shapes={};inputs={};labels={}
    for i,s in enumerate(sources):
        key=s['key'];dims=s['spatial_dims'];last=s['model']=='swin_b'
        ops=[('stage2',Spatial(dims,last)),('stage3',nn.Identity()),('stage4',nn.Identity()),('features',Pool(last)),('logits',nn.Linear(4,2))]
        config=SimpleNamespace(name=s['model'],views=1,spatial_dims=dims)
        graph=NamedModelGraph(config,nn.Identity(),ops,dict(stage2=4,stage3=4,stage4=4,features=4))
        parents[key]=ObservedParticipantModel(graph).eval();shapes[key]=(1,*([4]*dims))
        inputs[key]=torch.randn(3,*shapes[key],device=device);labels[key]=torch.tensor([(i//2)%2,(i//2+1)%2],device=device)
    return parents,shapes,sources,dict(inputs=inputs,labels=labels,counts=[1,2],participant_id=['a','b'])


@pytest.mark.parametrize('family',['none','mmtm','cross_attention'])
def test_six_native_outputs_gradients_and_state(family):
    torch.set_num_threads(2)
    parents,shapes,sources,batch=fixture();names=[s['key']+'_stage3' for s in sources]
    cfg=[] if family=='none' else [dict(nodes=names,family=family,**(dict(hidden_dimension=8) if family=='mmtm' else dict(attention_dimension=8,heads=2)))]
    model=NativeGroup(parents,shapes,sources,cfg);model.eval()
    z,loss=model(batch)
    for key,parent in parents.items():assert torch.equal(z[key],parent(batch['inputs'][key],batch['counts']))
    expected=sum(torch.nn.functional.cross_entropy(z[k],batch['labels'][k]) for k in z)
    assert torch.equal(loss,expected)
    optimizer=torch.optim.AdamW(model.groups(1e-3,1e-3,1e-3))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);_,loss=model(batch);model.backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True);_,direct=model.native_forward(batch);direct.backward()
        assert torch.allclose(loss,direct)
        for n,p in model.named_parameters():
            if n in grads:assert torch.allclose(p.grad,grads[n],atol=1e-6,rtol=1e-4),n
        model.clip(5);optimizer.step()
    if family!='none':
        optimizer.zero_grad(set_to_none=True)
        b=copy.deepcopy(batch)
        for v in b['inputs'].values():v.requires_grad_(True)
        out,_=model.native_forward(b)
        torch.nn.functional.cross_entropy(out[sources[0]['key']],b['labels'][sources[0]['key']]).backward()
        assert all(v.grad is not None and v.grad.abs().sum()>0 for v in b['inputs'].values())
    state=copy.deepcopy(model.state_dict());clone=NativeGroup(parents,shapes,sources,cfg).eval();clone.load_state_dict(state,strict=True)
    with torch.no_grad():
        a,_=model(batch);b,_=clone(batch)
        assert all(torch.equal(a[k],b[k]) for k in a)
