import copy
import pytest
import torch
from radon_bridge.runtime.pilot_checkpoint import read, save, validate


def test_full_resume_preserves_selected_and_next_update(tmp_path):
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.randn(4, 3)
    def update(m, o):
        o.zero_grad(set_to_none=True)
        m(x).square().mean().backward()
        o.step()
        o.zero_grad(set_to_none=True)
    update(model, optimizer)
    cfg = dict(seed=31, bridges=[])
    selected = save(dict(model={'native':model.state_dict()},configuration=cfg,epoch=0),
                    tmp_path/'best.pt', kind='selected')
    save(dict(model={'native':model.state_dict()},optimizer=optimizer.state_dict(),
              identity='run',configuration=cfg,epoch=1,monitor={'best_epoch':0},history=[{'epoch':1}],
              rng={'torch':torch.get_rng_state()},selected=selected), tmp_path/'resume.pt', kind='resume')
    update(model, optimizer)
    state = read(tmp_path/'resume.pt',kind='resume',configuration=cfg)
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(state['model']['native'],strict=True)
    resumed = torch.optim.AdamW(restored.parameters(),lr=.01)
    resumed.load_state_dict(state['optimizer'])
    update(restored,resumed)
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for pid,value in optimizer.state_dict()['state'].items():
        for name,tensor in value.items():
            torch.testing.assert_close(tensor,resumed.state_dict()['state'][pid][name],rtol=0,atol=0)
    assert state['selected']['epoch']==0
    with pytest.raises(ValueError,match='checkpoint required'):
        read(tmp_path/'resume.pt',kind='selected')
    bad=copy.deepcopy(state);del bad['rng']
    with pytest.raises(ValueError,match='Incomplete'):
        validate(bad,kind='resume')
    bad=copy.deepcopy(state);del bad['selected']['framework_api']
    with pytest.raises(ValueError,match='explicitly migrate'):
        validate(bad,kind='resume')


@pytest.mark.parametrize('header',[{}, {'framework_api':'V4'}, {'framework_api':'V5'}])
def test_unconverted_or_ambiguous_states_rejected(tmp_path,header):
    path=tmp_path/'old.pt'
    torch.save(dict(model={'native':{'weight':torch.ones(1)}},**header),path)
    with pytest.raises(ValueError,match='explicitly migrate'):
        read(path,kind='selected')


def test_native_parent_role_and_configuration_are_checked(tmp_path):
    state=dict(model={'cfp':{'weight':torch.ones(1)}},branch='cfp',seed=31)
    save(state,tmp_path/'parent.pt',kind='native_parent')
    assert read(tmp_path/'parent.pt',kind='native_parent')['branch']=='cfp'
    with pytest.raises(ValueError,match='configuration changed'):
        read(tmp_path/'parent.pt',kind='native_parent',configuration={'seed':32})
