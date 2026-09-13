"""Shape-verified multi-task bridge construction from complete parent models."""
import torch
import numpy as np
from radon_bridge.models.native_group import NativeGroup, definition
from radon_bridge.studies.complete_matrix import cross_edges
from radon_bridge.methods.projection import geometry_budget


def configurations(parents, shapes, sources, arm, bases, seed):
    if arm['family'] == 'none': return []
    if arm.get('host'):
        host = dict(arm, id=arm['host'], family='mmtm' if arm['host'].startswith('mmtm') else 'cross_attention')
        host.pop('host'); host.pop('addition')
        result = configurations(parents, shapes, sources, host, bases, seed)
        if arm['addition'] == 'continue': return result
        extra = dict(arm, family='radon', mode=arm['addition']); extra.pop('host'); extra.pop('addition')
        result += configurations(parents, shapes, sources, extra, bases, seed)
        result[1]['parallel_to'] = 0
        return result
    native, _ = definition(parents, shapes, sources)
    modules = [e.edge_operations[0].function for e in native.builder.edges]
    for m in modules: m.eval()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        features = native.builder.native_forward(native.probe_inputs)
    configs = []
    for stage in arm['stages']:
        names = [s['key']+f'_stage{stage}' for s in sources]
        channels = {n: features[n].shape[native.metadata['channel_axes'][n]] for n in names}
        if arm['family'] == 'mmtm':
            configs.append(dict(nodes=names, family='mmtm', hidden_dimension=arm.get('hidden_dimension', 256)))
            continue
        if arm['family'] == 'cross_attention':
            configs.append(dict(nodes=names, family='cross_attention', attention_dimension=arm.get('attention_dimension',256), heads=arm.get('heads',4)))
            continue
        r, M, S = arm['r'], arm['M'], arm['S']
        if any(r > c for c in channels.values()): raise ValueError('Channel rank exceeds available channels')
        config = dict(nodes=names, M=M, S=S, rho={n:r/c for n,c in channels.items()},
                      mode=arm['mode'], compression=arm['compression'], kernel_size=arm['k'])
        if arm['compression'].startswith('fixed_'):
            config['basis_files'] = {n: bases[arm['compression']][n] for n in names}
        if arm['compression'] in ('fixed_svd_channel', 'fixed_random_orthogonal_channel'):
            config.update(r=r, h=r*M)
        if arm.get('topology'):
            config['cross_edges'] = [[k+f'_stage{stage}' for k in pair] for pair in cross_edges(sources, arm['topology'])]
        if arm.get('direction'):
            by_modality = {s['modality']:s['key'] for s in sources}
            if len(by_modality) != len(sources): raise ValueError('Ambiguous legacy modality direction')
            config['cross_edges'] = [[by_modality[k]+f'_stage{stage}' for k in arm['direction']]]
        if arm.get('s_axis_scramble'):
            config['s_axis_permutation'] = np.random.default_rng(seed+907).permutation(S).tolist()
        configs.append(config)
    return configs


def build(parents, shapes, sources, arm, bases, seed, device='cpu'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with geometry_budget(128_000_000):
            return NativeGroup(parents, shapes, sources, configurations(parents,shapes,sources,arm,bases,seed),
                               device=device, frozen=arm['frozen'])
