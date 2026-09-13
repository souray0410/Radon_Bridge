import copy
import pytest
import torch

from radon_bridge.models.native_group import NativeGroup
from radon_bridge.models.placement import place
from radon_bridge.runtime.host_checkpoint import capture_rng, restore_rng, save, load
from radon_bridge.training.paired_native import Schedule, DEFAULTS
from tests.unit.models.test_native_group import fixture


@pytest.mark.parametrize('distributed', [False, True])
def test_source_placement_update_and_resume(tmp_path, distributed):
    if distributed and torch.cuda.device_count() < 2:
        pytest.skip('Two CUDA devices required for actual source-parallel validation')
    torch.set_num_threads(2)
    parents, shapes, sources, batch = fixture()
    cfg = [dict(nodes=[s['key']+'_stage3' for s in sources], family='mmtm', hidden_dimension=8)]
    a = NativeGroup(parents, shapes, sources, cfg).eval()
    device = 'cuda:0' if distributed else 'cpu'
    b = NativeGroup(parents, shapes, sources, cfg, device=device).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    placement = {s['key']: f'cuda:{i%2}' if distributed else 'cpu' for i,s in enumerate(sources)}
    place(a, placement, device)
    assert a.node_identity() == b.node_identity()
    oa = torch.optim.AdamW(a.groups(1e-3,1e-3,1e-3))
    ob = torch.optim.AdamW(b.groups(1e-3,1e-3,1e-3))
    sa = Schedule(oa, dict(DEFAULTS))
    def update(model,opt):
        from radon_bridge.evaluation.paired_native import move
        opt.zero_grad(set_to_none=True)
        _, loss = model(move(batch,device)); model.backward(); model.clip(5); opt.step(); opt.zero_grad(set_to_none=True)
        return float(loss.detach())
    for _ in range(3):
        assert abs(update(a,oa)-update(b,ob)) < 1e-5
        for k,v in a.state_dict().items():
            assert torch.allclose(v.cpu(), b.state_dict()[k].cpu(), atol=1e-6,rtol=1e-5), k
    save(tmp_path/'resume.pt', model=a, optimizer=oa, scheduler=sa, identity='fixture', progress={'step':3},node_ids=a.node_identity())
    update(a,oa); expected=copy.deepcopy(a.state_dict())
    load(tmp_path/'resume.pt',model=a,optimizer=oa,scheduler=sa,identity='fixture',node_ids=a.node_identity())
    update(a,oa)
    assert all(torch.equal(v,expected[k]) for k,v in a.state_dict().items())

