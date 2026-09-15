from types import SimpleNamespace
import copy
import torch
import pytest
from tests.unit.models.test_native_group import fixture
from tests.unit.models.test_group_lifecycle import SmallData
from radon_bridge.models.native_group import NativeGroup
from radon_bridge.methods.basis import save_basis
from radon_bridge.analysis.group_pairing import diagnose


@pytest.mark.parametrize('mode',['radon','linear_resample'])
@pytest.mark.parametrize('compression',['fixed_svd_channel','factorized_projected'])
def test_two_source_full_mhd_pairing_views_preserve_receiver_and_labels(tmp_path,compression,mode):
    torch.set_num_threads(2)
    parents,shapes,sources,batch=fixture();sources=sources[:2];keys=[s['key'] for s in sources]
    parents={k:parents[k] for k in keys};shapes={k:shapes[k] for k in keys}
    batch=dict(batch,inputs={k:batch['inputs'][k] for k in keys},labels={k:batch['labels'][k] for k in keys})
    names=[k+'_stage3' for k in keys]
    refs={k:save_basis(torch.eye(4,dtype=torch.float64),1,tmp_path/k,k,3416,{'fixture':True}) for k in names}
    cfg=dict(nodes=names,M=4,S=8,rho=.5,r=2,h=8,mode=mode,compression=compression,basis_files=refs)
    if compression=='factorized_projected':cfg=dict(nodes=names,M=4,S=8,rho=1,mode=mode,compression=compression,bottleneck_rank=4)
    model=NativeGroup(parents,shapes,sources,[cfg]).train()
    outer=model.task.modules_by_name()['bridge_0_exchange'];exchange=getattr(outer,'exchange',outer)
    with torch.no_grad():exchange.mixer.conv.weight.normal_(0,.01)
    dataset=SmallData(batch,'development',4);dataset.sources=sources
    dataset.datasets={keys[0]:SimpleNamespace(rows=[dict(eyes=['left']) for _ in range(4)])}
    dataset.indices={keys[0]:{pid:i for i,pid in enumerate(dataset.participant_ids)}}
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    for p in model.parameters():p.grad=torch.ones_like(p)
    gradients=[p.grad.clone() for p in model.parameters()]
    result=diagnose(model,dataset,tmp_path/'pairing',torch.device('cpu'),lambda:False)
    assert result['conditions']==64 and not result['test_access']
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),gradients))
    assert result['compression']==compression
    assert len(result['rows'])==64
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(v<1e-5 for v in result['rows'][0]['mean_absolute_probability_change'].values())

    # Interruption must also restore caller state; this scratch attempt is not accepted.
    flags=[m.training for m in model.modules()]
    calls=[0]
    def pause_after_cache_entry():
        calls[0]+=1
        return calls[0]>1
    with pytest.raises(InterruptedError):
        diagnose(model,dataset,tmp_path/'interrupted',torch.device('cpu'),pause_after_cache_entry)
    assert [m.training for m in model.modules()]==flags
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),gradients))
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
