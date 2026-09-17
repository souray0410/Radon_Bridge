import pytest
from radon_bridge.studies.parent_routes import catalog_routes, group_keys


def test_family_routes_use_actual_two_and_three_dimensional_names():
    names = ['resnet50', 'densenet121', 'monai_densenet121_3d',
             'swin_b', 'swin_unetr_encoder_3d']
    routes = catalog_routes([{'model': name} for name in names])
    assert list(routes) == ['densenet121', 'resnet50', 'swin_b']
    assert group_keys('glaucoma', routes['densenet121']) == (
        'glaucoma/densenet121/cfp_2d', 'glaucoma/monai_densenet121_3d/oct_volume_3d')
    assert group_keys('cataract', routes['swin_b']) == (
        'cataract/swin_b/cfp_2d', 'cataract/swin_unetr_encoder_3d/oct_volume_3d')


def test_partial_catalog_does_not_invent_an_accepted_partner():
    routes = catalog_routes([{'model': 'monai_densenet121_3d'}])
    groups = {'glaucoma/monai_densenet121_3d/oct_volume_3d': {'selected': True}}
    assert not all(groups.get(key, {}).get('selected')
                   for key in group_keys('glaucoma', routes['densenet121']))


def test_unknown_architecture_is_not_silently_paired_by_name():
    with pytest.raises(ValueError, match='Unregistered'):
        catalog_routes([{'model': 'future_network'}])
