from types import SimpleNamespace

from radon_bridge.studies.grouped_linear import GROUP_COUNTS, G1_REUSE, grouped_arms, new_grouped_arms, grouped_comparisons, manifest
from radon_bridge.studies.project_build import bridges_for
from radon_bridge.studies.research_matrix import arms


def _parents():
    graph=SimpleNamespace(feature_channels={'stage3':64})
    return {'cfp':SimpleNamespace(graph=graph),'oct':SimpleNamespace(graph=graph)}


def _bases():
    return {'fixed_svd_channel':{'cfp_stage3':'cfp_basis.npz','oct_stage3':'oct_basis.npz'}}


def test_grouped_supplement_is_finite_and_separate():
    supplement_arms=grouped_arms()
    assert GROUP_COUNTS==(1,2,4,8,16)
    assert len(supplement_arms)==10 and len({a['id'] for a in supplement_arms})==10
    assert len(grouped_comparisons())==5
    assert len(new_grouped_arms())==8 and all(a['group_count']>1 for a in new_grouped_arms())
    assert G1_REUSE=={'grouped_g1_radon':'svd_radon','grouped_g1_linear_resample':'svd_resample'}
    data=manifest()
    assert data['fixed']=={'compression':'fixed_svd_channel','r':32,'M':32,'S':64,'k':3,'stages':[3]}
    assert data['uncompressed_grouped_budget_matched'].startswith('pending_')
    assert data['reuse']==G1_REUSE and len(data['new_execution_arms'])==8
    assert not data['test_access']
    grouped=grouped_arms()
    base=arms('glaucoma')
    assert bridges_for(_parents(),{},grouped[0],_bases(),3416)==bridges_for(_parents(),{},base[1],_bases(),3416)
    assert bridges_for(_parents(),{},grouped[1],_bases(),3416)==bridges_for(_parents(),{},base[2],_bases(),3416)


def test_grouped_radon_and_linear_resample_configs_are_matched_except_geometry():
    pairs={}
    for arm in grouped_arms():
        group=arm['group_count']
        if group==1:
            continue
        config=bridges_for(_parents(),{},arm,_bases(),3416)[0]
        pairs.setdefault(group,{})[arm['mode']]=config
    for group,pair in pairs.items():
        assert pair['radon']['group_count']==pair['linear_resample']['group_count']==group
        left=dict(pair['radon']);right=dict(pair['linear_resample'])
        assert left.pop('mode')=='radon'
        assert right.pop('mode')=='linear_resample'
        assert left==right
