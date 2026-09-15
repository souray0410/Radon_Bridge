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


def factor_arms():
    from radon_bridge.studies.research_matrix import changed
    result=[changed('svd_radon'),changed('svd_ordinary',mode='linear_resample')]
    for rank in (256,512,1024):
        for mode in ('radon','linear_resample'):
            a=changed(f'factor_{rank}_{mode}',compression='factorized_projected',mode=mode,bottleneck_rank=rank)
            a.pop('r');result.append(a)
    return result


def test_factorization_metadata_never_fabricates_channel_rank():
    import pytest
    from radon_bridge.analysis.group_report import geometry_metadata
    arms=factor_arms();a=arms[2];m=geometry_metadata(a)
    assert m['r'] is None and m['bottleneck_rank']==256 and m['rank_scope']=='global_concatenated_projection'
    assert geometry_metadata(arms[0])['r']==32
    assert geometry_metadata(dict(a,family='none'))['bottleneck_rank'] is None
    with pytest.raises(ValueError):geometry_metadata(dict(a,r=32))


def test_factorization_interaction_direction_and_exact_matching():
    import pytest
    from radon_bridge.analysis.group_report import factorization_interactions
    arms=factor_arms();rows=factorization_interactions(arms,['svd_radon','svd_ordinary'])
    assert len(rows)==3
    # Factorized +.03 vs SVD +.01 is a +.02 interaction, not a direct gain.
    values={a['id']:.6 for a in arms};values.update(svd_radon=.61,factor_256_radon=.63)
    assert sum(w*values[k] for k,w in rows[0]['weights'].items())==pytest.approx(.02)
    assert all(sum(r['weights'].values())==0 for r in rows)
    with pytest.raises(ValueError):factorization_interactions(arms[:-1],['svd_radon','svd_ordinary'])
    changed=[dict(a,S=128) if a['id'].startswith('factor_256') else a for a in arms]
    with pytest.raises(ValueError):factorization_interactions(changed,['svd_radon','svd_ordinary'])
    with pytest.raises(ValueError):factorization_interactions(arms+[arms[2]],['svd_radon','svd_ordinary'])


def test_full_factorization_report_writes_global_rank(tmp_path):
    import csv,json
    from radon_bridge.analysis.group_report import report_case
    arms=factor_arms();ids=np.arange(8);labels=np.array([0,1]*4)
    metrics={'a':{'macro_f1':1.},'mean_macro_f1':1.}
    for arm in arms:
        p=tmp_path/'arms'/arm['id'];p.mkdir(parents=True)
        (p/'accepted.json').write_text(json.dumps(dict(state='accepted',plateau=True,test_access=False,
            metrics=metrics,best_epoch=1,stop_epoch=8,seconds=1.,parameters_total=1,parameters_trainable=1)))
        np.savez(p/'development_predictions.npz',participant_ids=ids,a=np.eye(2)[labels],labels__a=labels)
    spec=dict(sources=[dict(key='a',disease='synthetic',modality='cfp',architecture='fixture')],
              seed=3416,arms=arms,bootstrap_iterations=32)
    result=report_case(spec,tmp_path)
    rows=list(csv.DictReader((tmp_path/'report/metrics.csv').open()))
    assert rows[2]['r']=='' and rows[2]['bottleneck_rank']=='256'
    assert rows[0]['r']=='32' and rows[0]['bottleneck_rank']==''
    # The old comparison family is not silently expanded with supplementary interactions.
    assert not any(r['family']=='parameterization_geometry' for r in result['definitions'])
