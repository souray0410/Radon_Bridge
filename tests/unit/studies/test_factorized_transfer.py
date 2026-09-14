from types import SimpleNamespace
import pytest
from radon_bridge.studies.factorized_transfer import configuration, cost
from radon_bridge.studies.project_build import bridges_for
from radon_bridge.methods.factorized import FactorizedMixer


def arm(mode='radon'):
    return dict(id='global', family='radon', compression='factorized_projected',
                mode=mode, stages=[3], r=None, M=32, S=64, k=3, bottleneck_rank=256, frozen=False)


@pytest.mark.parametrize('mode', ['radon', 'linear_resample'])
def test_native_pair_preserves_full_unequal_channels_and_global_rank(mode):
    parents={k:SimpleNamespace(graph=SimpleNamespace(feature_channels={'stage3':c}))
             for k,c in [('cfp',1024),('oct',512)]}
    result=bridges_for(parents,{},arm(mode),{},3416)
    assert result == [configuration(['cfp_stage3','oct_stage3'],
                                    {'cfp_stage3':1024,'oct_stage3':512},arm(mode))]
    assert result[0]['rho']==1 and result[0]['bottleneck_rank']==256
    assert not {'basis_files','r','h'} & result[0].keys()


def test_cost_matches_the_executed_operator():
    report=cost([3,5],2,4,3)
    mixer=FactorizedMixer([6,10],4,3)
    assert report['parameters']==sum(p.numel() for p in mixer.parameters())
    assert report['unfactorized_parameters']==3*16**2


@pytest.mark.parametrize('count',[2,6])
def test_named_group_adapter_uses_the_same_contract(monkeypatch,count):
    import torch
    from radon_bridge.studies import group_build
    sources=[{'key':f'disease{i}__modality{i}'} for i in range(count)]
    features={s['key']+'_stage3':torch.zeros(1,8+i,2,2)
              for i,s in enumerate(sources)}
    native=SimpleNamespace(builder=SimpleNamespace(edges=[],native_forward=lambda _:features),
                           probe_inputs={},metadata={'channel_axes':{n:1 for n in features}})
    monkeypatch.setattr(group_build,'definition',lambda *args:(native,None))
    result=group_build.configurations({}, {}, sources, arm(), {}, 3416)
    expected=configuration(list(features),{n:t.shape[1] for n,t in features.items()},arm())
    assert result==[expected]


@pytest.mark.parametrize('change',[{'r':32},{'direction':['oct','cfp']},
    {'topology':'same_modality'},{'mode':'self'},{'bottleneck_rank':999999},
    {'M':True},{'k':2}])
def test_no_silent_change_of_scientific_meaning(change):
    a=arm();a.update(change)
    with pytest.raises(ValueError):configuration(['a','b'],{'a':32,'b':64},a)
