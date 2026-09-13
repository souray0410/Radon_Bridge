from types import SimpleNamespace
import copy
import torch
from tests.unit.models.test_native_group import fixture
from tests.unit.models.test_group_lifecycle import SmallData
from radon_bridge.models.native_group import NativeGroup
from radon_bridge.methods.basis import save_basis
from radon_bridge.analysis.group_pairing import diagnose


def test_two_source_full_mhd_pairing_views_preserve_receiver_and_labels(tmp_path):
    torch.set_num_threads(2)
    parents,shapes,sources,batch=fixture();sources=sources[:2];keys=[s['key'] for s in sources]
    parents={k:parents[k] for k in keys};shapes={k:shapes[k] for k in keys}
    batch=dict(batch,inputs={k:batch['inputs'][k] for k in keys},labels={k:batch['labels'][k] for k in keys})
    names=[k+'_stage3' for k in keys]
    refs={k:save_basis(torch.eye(4,dtype=torch.float64),1,tmp_path/k,k,3416,{'fixture':True}) for k in names}
    model=NativeGroup(parents,shapes,sources,[dict(nodes=names,M=4,S=8,rho=.5,r=2,h=8,mode='radon',compression='fixed_svd_channel',basis_files=refs)]).eval()
    outer=model.task.modules_by_name()['bridge_0_exchange'];exchange=getattr(outer,'exchange',outer)
    with torch.no_grad():exchange.mixer.conv.weight.normal_(0,.01)
    dataset=SmallData(batch,'development',4);dataset.sources=sources
    dataset.datasets={keys[0]:SimpleNamespace(rows=[dict(eyes=['left']) for _ in range(4)])}
    dataset.indices={keys[0]:{pid:i for i,pid in enumerate(dataset.participant_ids)}}
    before=copy.deepcopy(model.state_dict())
    result=diagnose(model,dataset,tmp_path/'pairing',torch.device('cpu'),lambda:False)
    assert result['conditions']==64 and not result['test_access']
    assert len(result['rows'])==64
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(v<1e-5 for v in result['rows'][0]['mean_absolute_probability_change'].values())
