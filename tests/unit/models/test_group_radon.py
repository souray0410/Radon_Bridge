import copy
import pytest
import torch
from tests.unit.models.test_native_group import fixture
from radon_bridge.methods.basis import save_basis
from radon_bridge.models.native_group import NativeGroup


@pytest.mark.parametrize('mode',['radon','linear_resample','self'])
def test_six_fixed_channel_mhd_and_cross_source_mask(tmp_path,mode):
    torch.set_num_threads(2)
    parents,shapes,sources,batch=fixture();names=[s['key']+'_stage3' for s in sources]
    refs={key:save_basis(torch.eye(4,dtype=torch.float64),1,tmp_path/key,key,3416,{'fixture':True}) for key in names}
    config=dict(nodes=names,M=4,S=8,rho=.5,r=2,h=8,mode=mode,compression='fixed_svd_channel',basis_files=refs)
    model=NativeGroup(parents,shapes,sources,[config],frozen=True).eval()
    z,_=model(batch)
    for key,parent in parents.items():assert torch.equal(z[key],parent(batch['inputs'][key],batch['counts']))
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    assert len(trainable)==1 and trainable[0].endswith('mixer.conv.weight')
    exchange=model.task.modules_by_name()['bridge_0_exchange'];inner=getattr(exchange,'exchange',exchange)
    with torch.no_grad():inner.mixer.conv.weight.normal_(0,.01);inner.mixer.conv.weight.mul_(inner.mixer.mask)
    test=copy.deepcopy(batch)
    for x in test['inputs'].values():x.requires_grad_(True)
    logits,_=model.native_forward(test)
    key=sources[0]['key'];torch.nn.functional.cross_entropy(logits[key],test['labels'][key]).backward()
    other=test['inputs'][sources[1]['key']].grad
    if mode=='self':assert other is None or torch.count_nonzero(other)==0
    else:assert other is not None and torch.count_nonzero(other)>0
    model.zero_grad(set_to_none=True);_,loss=model(batch);model.backward()
    gradient=inner.mixer.conv.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True);_,native=model.native_forward(batch);native.backward()
    assert torch.allclose(gradient,inner.mixer.conv.weight.grad,atol=1e-6,rtol=1e-5)
    assert torch.equal(inner.mixer.conv.weight.grad*(1-inner.mixer.mask),torch.zeros_like(gradient))
