"""Explicit modality-specific architectures for the approved expert routes.

A route names a research family, not two identical native implementations.
This mapping never waives parent acceptance or actual runtime integration.
"""

RESNETS = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
ROUTES = {
    **{name: {'cfp': name, 'oct': name} for name in RESNETS},
    'densenet121': {'cfp': 'densenet121', 'oct': 'monai_densenet121_3d'},
    'swin_b': {'cfp': 'swin_b', 'oct': 'swin_unetr_encoder_3d'},
}


def catalog_routes(candidates):
    names = {row['model'] for row in candidates}
    known = {name for pair in ROUTES.values() for name in pair.values()}
    if names - known:
        raise ValueError('Unregistered native architecture route: ' + ', '.join(sorted(names - known)))
    return {route: dict(pair) for route, pair in sorted(ROUTES.items())
            if names.intersection(pair.values())}


def group_keys(disease, pair):
    return (f"{disease}/{pair['cfp']}/cfp_2d",
            f"{disease}/{pair['oct']}/oct_volume_3d")
