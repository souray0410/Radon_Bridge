"""Prespecified WS02 centered-SVD finite supplement."""
COMPARISONS=(
    ('centered_radon','uncentered_radon'),
    ('centered_linear_resample','uncentered_linear_resample'),
    ('centered_radon','centered_linear_resample'),
)


def manifest():
    return {
        'schema':'radon_centered_svd_supplement_v1',
        'scope':'ws02_single_seed_centered_basis_direction_ablation',
        'seed':3416,
        'new_execution_arms':['centered_radon','centered_linear_resample'],
        'strict_reuse':{'uncentered_radon':'svd','uncentered_linear_resample':'linear'},
        'comparisons':[list(v) for v in COMPARISONS],
        'fixed':{'stage':3,'r':32,'M':32,'S':64,'k':3,'rho':.125,'group_count':1},
        'basis_definition':{
            'uncentered':'FF^T/N',
            'centered':'FF^T/N - mu mu^T',
            'runtime_centering':False,
        },
        'test_access':False,
    }
