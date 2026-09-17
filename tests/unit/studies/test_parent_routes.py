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


def test_native_readiness_honors_multifamily_registered_catalog(tmp_path):
    import json
    from radon_bridge.runtime.state import file_sha256
    from radon_bridge.studies.native_prerequisites import audit
    rows = []
    for name, track, dims in [('resnet50', 'cfp_2d', 2),
                              ('monai_densenet121_3d', 'oct_volume_3d', 3),
                              ('swin_unetr_encoder_3d', 'oct_volume_3d', 3)]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(dict(model=dict(name=name, spatial_dims=dims),
            track=track, disease='glaucoma', training=dict(seed=3416), test_used=False)))
        rows.append(dict(model=name, track=track, disease='glaucoma',
            spec=str(path), spec_sha256=file_sha256(path), run_dir=str(tmp_path/name)))
    keys = [f"glaucoma/{r['model']}/{r['track']}" for r in rows]
    catalog = dict(schema='radon_bridge_native_screen_v1', test_access=False,
                   candidates=rows, registered_groups=keys)
    result = audit(catalog, lambda *_: pytest.fail('No completed model exists'))
    assert set(result['groups']) == set(keys)
    assert all(x['state'] == 'waiting' and x['candidate_count'] == 1
               for x in result['groups'].values())
    catalog['registered_groups'] = keys[:1]
    with pytest.raises(ValueError, match='outside'):
        audit(catalog, lambda *_: None)
    catalog['registered_groups'] = ['glaucoma/monai_densenet121_3d/cfp_2d']
    with pytest.raises(ValueError, match='Unregistered'):
        audit(catalog, lambda *_: None)


def test_complete_controller_tick_accepts_dense_and_swin_routes(tmp_path):
    import json
    from radon_bridge.runtime.state import file_sha256
    from radon_bridge.studies.autoresearch import Controller
    from radon_bridge.studies.native_prerequisites import collect
    tasks = []
    for name, track, dims in [('densenet121','cfp_2d',2),
                              ('monai_densenet121_3d','oct_volume_3d',3),
                              ('swin_b','cfp_2d',2),
                              ('swin_unetr_encoder_3d','oct_volume_3d',3)]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(dict(model=dict(name=name, spatial_dims=dims),
            track=track,disease='glaucoma',training=dict(seed=3416),test_used=False,
            train_manifest_sha256='train',development_manifest_sha256='dev',
            cache_receipt_sha256='cache',aggregation='valid_eye_feature_mean_v1')))
        tasks.append(dict(spec=str(path),spec_sha256=file_sha256(path),run_dir=str(tmp_path/name)))
    queue=tmp_path/'queues'/'one'/'queue.json';queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps(dict(tasks=tasks)))
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(collect([queue])))
    protocol=tmp_path/'protocol';protocol.write_text('Frozen fixture')
    c=Controller(dict(schema='radon_bridge_autoresearch_v1',test_access=False,
        catalog=dict(path=str(catalog),sha256=file_sha256(catalog)),
        protocol=dict(path=str(protocol),sha256=file_sha256(protocol)),source_pins=[],
        output=str(tmp_path/'output'),queues_root=str(tmp_path/'queues'),models_root=str(tmp_path/'models')),
        lambda *_:pytest.fail('No accepted model'),lambda *_:pytest.fail('No selected recipe'))
    result=c.tick()
    assert result['current_round_candidates']==4
    assert result['accepted_native']==0
    assert result['groups']['glaucoma/monai_densenet121_3d/oct_volume_3d']['total']==1
    assert not result['formal_project_dispatch_ready']
