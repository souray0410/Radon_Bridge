import numpy as np
import torch
from radon_bridge.analysis.group_report import participant_draws, intervals, comparisons
from radon_bridge.analysis.group_diagnostics import disabled_exchange
from radon_bridge.analysis.communication import block_messages
from radon_bridge.methods.operator import LinearMixer
from radon_bridge.studies.complete_matrix import groups, group_arms


def test_distinct_task_labels_shared_resampling_and_zero_variance():
    labels=np.array([[0,1],[0,1],[1,0],[1,0]])
    predictions=labels.copy()
    point,draws=participant_draws(labels,predictions,128)
    assert np.array_equal(point,[1,1])
    assert np.array_equal(draws[:,0],draws[:,1])
    assert intervals(np.array([0.]),(draws[:,0]-draws[:,1])[:,None])[0]['simultaneous_95'] is None
    _,one=participant_draws(labels[:,:1],predictions[:,:1],128)
    assert np.array_equal(one[:,0],draws[:,0])


def test_comparison_references_are_complete():
    seen=set()
    for group in groups():
        if group['package'] in seen:continue
        seen.add(group['package']);arms=group_arms(group);names={a['id'] for a in arms}
        for row in comparisons(arms):
            assert set(row['weights'])<=names
            assert sum(row['weights'].values())==0


def test_disabling_delta_path_does_not_add_an_extra_identity():
    from types import SimpleNamespace
    x=(torch.ones(2,3,4),torch.full((2,3,4),2.))
    output=torch.randn(2,24)
    assert torch.equal(disabled_exchange(SimpleNamespace(delta_only=True),x,output),torch.zeros_like(output))
    assert torch.equal(disabled_exchange(SimpleNamespace(delta_only=False),x,output),torch.cat([v.flatten(1) for v in x],1))


def test_block_decomposition_preserves_s_axis_operator():
    torch.manual_seed(71)
    mixer=LinearMixer([3,4],keys=['a','b'],s_axis_permutation=[2,0,3,1])
    with torch.no_grad():mixer.conv.weight.normal_()
    inputs=(torch.randn(2,3,4),torch.randn(2,4,4))
    expected=mixer(*inputs);got=block_messages(mixer,inputs,{})
    assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(expected,got))
