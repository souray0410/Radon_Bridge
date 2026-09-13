from radon_bridge.studies.research_matrix import arms,comparisons


def test_matched_modes_and_finite_positions():
    all_arms=arms('glaucoma');assert len(all_arms)==55
    for stages in ('23','234'):
        for mode in ('radon','linear_resample','self'):
            a=next(a for a in all_arms if a['id']=='depth'+stages+'_'+mode)
            assert a['mode']==mode
    assert len(arms('cataract'))==6
    assert len(arms('glaucoma','resnet101'))==6
    for disease in ('cataract','glaucoma','macular_degeneration'):
        names={a['id'] for a in arms(disease)}
        assert len(names)==len(arms(disease))
        assert all(d['left'] in names and d['right'] in names for d in comparisons(disease))


def test_reference_questions_and_other_models_do_not_expand_mechanism_search():
    assert {d['question'] for d in comparisons('glaucoma')} >= {'direct','matched_geometry','coadaptation','direction','bridge_depth','host_augmentation'}
    assert len(comparisons('glaucoma','resnet34'))==5
