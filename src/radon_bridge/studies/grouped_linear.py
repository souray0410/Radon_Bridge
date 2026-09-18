"""Approved grouped-linear supplement; separate from the 49-arm reference mechanism matrix."""
from radon_bridge.studies.research_matrix import changed

GROUP_COUNTS=(1,2,4,8,16)
G1_REUSE={'grouped_g1_radon':'svd_radon','grouped_g1_linear_resample':'svd_resample'}


def grouped_arms():
    """Return the prespecified fixed-SVD grouped Radon/control pairs."""
    result=[]
    for groups in GROUP_COUNTS:
        result.append(changed(f'grouped_g{groups}_radon',group_count=groups))
        result.append(changed(f'grouped_g{groups}_linear_resample',group_count=groups,mode='linear_resample'))
    return result


def new_grouped_arms():
    """Only G>1 requires new execution; G=1 is exact accepted-core reuse when identity matches."""
    return [a for a in grouped_arms() if a['group_count']>1]


def grouped_comparisons():
    return [
        dict(
            id=f'grouped_g{groups}_geometry',
            left=f'grouped_g{groups}_radon',
            right=f'grouped_g{groups}_linear_resample',
            question='matched_geometry_under_grouped_connectivity',
            branch='mean',
        )
        for groups in GROUP_COUNTS
    ]


def manifest():
    return {
        'schema':'radon_grouped_linear_supplement_v1',
        'scope':'approved_independent_supplement_not_reference_49_arm_matrix',
        'groups':list(GROUP_COUNTS),
        'arms':grouped_arms(),
        'new_execution_arms':new_grouped_arms(),
        'reuse':dict(G1_REUSE),
        'comparisons':grouped_comparisons(),
        'fixed':{'compression':'fixed_svd_channel','r':32,'M':32,'S':64,'k':3,'stages':[3]},
        'uncompressed_grouped_budget_matched':'pending_feasibility_review_no_matching_rule_selected',
        'test_access':False,
    }
