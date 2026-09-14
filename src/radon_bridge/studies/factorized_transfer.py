"""Shared construction contract for the same operator across cohort adapters.

This constructs configurations, not accepted study cases or GPU work orders.
The caller still owns cohort/parent locks, the scientific protocol and admission.
"""


def configuration(names, channels, arm):
    if arm.get('compression') != 'factorized_projected':
        raise ValueError('Not a global factorization arm')
    if arm.get('mode') not in ('radon', 'linear_resample'):
        raise ValueError('Only the paired geometric comparison is supported')
    if any(arm.get(k) is not None for k in ('r', 'h', 'direction', 'cross_edges', 'host', 's_axis_permutation')):
        raise ValueError('Global rank is not per-source channel rank or a masked bridge')
    if arm.get('topology') not in (None, 'all') or arm.get('s_axis_scramble'):
        raise ValueError('Masked factorization requires a separate implementation')
    if len(names) != len(set(names)) or set(names) != set(channels):
        raise ValueError('Source channels must match distinct endpoints')
    M, S, k, rank = (arm[x] for x in ('M', 'S', 'k', 'bottleneck_rank'))
    if any(type(x) is not int or x <= 0 for x in (M, S, k, rank)) or S < 2 or k % 2 == 0:
        raise ValueError('Invalid geometry or global rank')
    if not channels or any(type(c) is not int or c <= 0 for c in channels.values()):
        raise ValueError('Invalid native channel count')
    if rank > sum(channels.values()) * M:
        raise ValueError('Global rank exceeds the concatenated projected width')
    return dict(nodes=list(names), M=M, S=S, rho=1, mode=arm['mode'],
                compression='factorized_projected', kernel_size=k, bottleneck_rank=rank)


def cost(channels, M, rank, kernel_size):
    """Stored learnable mixer parameters; excludes geometry and backbone costs."""
    width = sum(channels) * M
    return dict(projected_width=width, global_rank=rank,
                parameters=2 * width * rank + kernel_size * rank * rank,
                unfactorized_parameters=kernel_size * width * width)
