import itertools
from radon_bridge.studies.complete_matrix import groups, group_arms, manifest, candidates, candidate_grid, cross_edges


def test_exact_approved_matrix():
    m = manifest(); gs = m['groups']
    assert len(gs) == 864 and len({g['id'] for g in gs}) == 864
    assert sum(g['tuning_reference'] for g in gs) == 12
    assert m['upper_bound'] == 92526 and m['fixed_training_positions'] == 65070
    assert m['counts']['same_family_mechanisms'] == 1485
    assert m['counts']['other_pair_core'] == 4536
    assert m['counts']['six_network_core'] == 59049
    for g in gs:
        aa = group_arms(g)
        assert len({a['id'] for a in aa}) == len(aa)
        assert all(a['host'] in {b['id'] for b in aa} for a in aa if a.get('host'))


def test_topologies_partition_all_directed_cross_edges():
    sources = next(g for g in groups() if len(g['sources']) == 6)['sources']
    masks = [set(map(tuple, cross_edges(sources, t))) for t in ('same_disease','same_modality','different_disease_and_modality')]
    assert list(map(len,masks)) == [6,12,12]
    assert not any(a & b for a,b in itertools.combinations(masks,2))
    assert set.union(*masks) == set(map(tuple,cross_edges(sources,'all')))


def test_search_coverage_and_matched_geometry():
    assert candidates('radon') == candidates('linear_resample')
    for family in ('radon','mmtm','cross_attention'):
        rows = candidates(family); levels,_ = candidate_grid(family)
        assert len(rows) == 32 and len({tuple(sorted(r.items())) for r in rows}) == 32
        assert {(k,v) for r in rows for k,v in r.items()} == {(k,v) for k,vs in levels.items() for v in vs}
