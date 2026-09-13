"""Finite UKB protocol: logical positions are not accepted training executions."""
import argparse
import copy
import itertools
import json
from pathlib import Path

from radon_bridge.runtime.state import stable_hash, atomic_write_json, atomic_write_text
from radon_bridge.studies.research_matrix import arms as legacy_arms, changed

VERSION = 'ukb_complete_20260913_v1'
DISEASES = ('cataract', 'glaucoma', 'macular_degeneration')
ARCHITECTURES = ('resnet50', 'densenet121', 'swin_b')
SEEDS = (3416, 3417, 3418)
ROLES = tuple((d, m) for d in DISEASES for m in ('cfp', 'oct'))
TOPOLOGIES = ('same_disease', 'same_modality', 'different_disease_and_modality', 'all')
FAMILIES = ('radon', 'linear_resample', 'mmtm', 'cross_attention')
MODELS = {
    'resnet50': {'cfp': 'resnet50', 'oct': 'resnet50'},
    'densenet121': {'cfp': 'densenet121', 'oct': 'monai_densenet121_3d'},
    'swin_b': {'cfp': 'swin_b', 'oct': 'swin_unetr_encoder_3d'},
}


def source(role, architecture):
    disease, modality = role
    return dict(key=f'{disease}__{modality}__{architecture}', disease=disease,
                modality=modality, architecture=architecture, model=MODELS[architecture][modality],
                spatial_dims=2 if modality == 'cfp' else 3)


def groups():
    for roles in itertools.combinations(ROLES, 2):
        for architectures in itertools.product(ARCHITECTURES, repeat=2):
            sources = [source(r, a) for r, a in zip(roles, architectures)]
            deep = roles[0][0] == roles[1][0] and architectures[0] == architectures[1]
            yield dict(id='pair_' + stable_hash(sources)[:20], sources=sources,
                       package='same_family_mechanisms' if deep else 'other_pair_core',
                       tuning_reference=deep)
    for architectures in itertools.product(ARCHITECTURES, repeat=6):
        sources = [source(r, a) for r, a in zip(ROLES, architectures)]
        yield dict(id='six_' + stable_hash(sources)[:20], sources=sources,
                   package='six_network_core', tuning_reference=len(set(architectures)) == 1)


def cross_edges(sources, topology):
    if topology not in TOPOLOGIES:
        raise ValueError('Undeclared connection relation')
    edges = []
    for sender, receiver in itertools.permutations(sources, 2):
        same_d = sender['disease'] == receiver['disease']
        same_m = sender['modality'] == receiver['modality']
        if (topology == 'all' or topology == 'same_disease' and same_d or
                topology == 'same_modality' and same_m or
                topology == 'different_disease_and_modality' and not same_d and not same_m):
            edges.append([sender['key'], receiver['key']])
    return edges


def group_arms(group):
    full = legacy_arms('glaucoma', 'resnet50')
    if group['package'] == 'same_family_mechanisms':
        return copy.deepcopy(full)
    if group['package'] == 'other_pair_core':
        return copy.deepcopy(full[:6] + [a for a in full if a.get('host')])
    result = [changed('continue', family='none')]
    for compression, label in [('fixed_svd_channel', 'svd'), ('fixed_random_orthogonal_channel', 'qr')]:
        result.append(changed(label + '_self', compression=compression, mode='self'))
        for topology, mode in itertools.product(TOPOLOGIES, ('radon', 'linear_resample')):
            result.append(changed(f'{label}_{topology}_{mode}', compression=compression,
                                  topology=topology, mode=mode))
    result.extend(copy.deepcopy(full[4:6] + [a for a in full if a.get('host')]))
    return result


def position_id(group, arm, seed, phase='fixed'):
    return stable_hash(dict(protocol=VERSION, group=group, arm=arm, seed=seed, phase=phase))


def positions():
    for g in groups():
        for a, seed in itertools.product(group_arms(g), SEEDS):
            yield dict(id=position_id(g['id'], a['id'], seed), protocol=VERSION,
                       group=g['id'], arm=a['id'], seed=seed, phase='fixed', package=g['package'],
                       dependencies=([position_id(g['id'], a['host'], seed)] if a.get('host') else []))


def all_positions():
    """Logical references including unresolved selected configurations, never fake runs."""
    yield from positions()
    for g in groups():
        cardinality = len(g['sources'])
        if g['tuning_reference']:
            for family, candidate in itertools.product(FAMILIES, range(32)):
                arm = f'{family}_candidate_{candidate:02d}'
                yield dict(id=position_id(g['id'], arm, 3416, 'tuning'), protocol=VERSION,
                           group=g['id'], arm=arm, seed=3416, phase='tuning', family=family,
                           candidate=candidate, dependencies=[])
        for family, seed in itertools.product(FAMILIES, SEEDS):
            selection = f'public_config_{cardinality}_{family}'
            yield dict(id=position_id(g['id'], family, seed, 'selected'), protocol=VERSION,
                       group=g['id'], arm=family, seed=seed, phase='selected', family=family,
                       dependencies=[selection], unresolved_selection=selection)
        for family, addition, seed in itertools.product(('mmtm', 'cross_attention'),
                                                       ('continue', 'radon', 'linear_resample'), SEEDS):
            arm = f'{family}_{addition}'
            selections = [f'public_config_{cardinality}_{family}']
            # Both additions use the selected Radon geometry, not the ordinary method's optimum.
            if addition != 'continue': selections.append(f'public_config_{cardinality}_radon')
            yield dict(id=position_id(g['id'], arm, seed, 'selected_host'), protocol=VERSION,
                       group=g['id'], arm=arm, seed=seed, phase='selected_host', family=family,
                       addition=addition, dependencies=[position_id(g['id'], family, seed, 'selected'), *selections])


def candidate_grid(family):
    common = dict(backbone_lr=[3e-5, 6e-5], head_bridge_lr=[3e-5, 1e-4, 3e-4],
                  weight_decay=[1e-4, .01, .05])
    if family in ('radon', 'linear_resample'):
        levels = dict(r=[16, 32, 64], M=[16, 32, 64], S=[32, 64, 128], k=[1, 3, 5], **common)
    elif family == 'mmtm':
        levels = dict(hidden_dimension=[128, 256, 512, 1024], **common)
    elif family == 'cross_attention':
        levels = dict(attention_dimension=[128, 256, 512, 1024], heads=[4, 8], **common)
    else:
        raise ValueError('Unknown search family')
    return levels, [dict(zip(levels, values)) for values in itertools.product(*levels.values())]


def candidates(family):
    """Hash-stratified without process RNG; geometry twins have identical candidates."""
    levels, grid = candidate_grid(family)
    reference = dict(backbone_lr=6e-5, head_bridge_lr=1e-4, weight_decay=.01)
    reference.update(dict(r=32, M=32, S=64, k=3) if family in ('radon', 'linear_resample') else
                     dict(hidden_dimension=256) if family == 'mmtm' else dict(attention_dimension=256, heads=4))
    namespace = 'geometry' if family in ('radon', 'linear_resample') else family
    order = lambda row: stable_hash(dict(protocol=VERSION, sampling_seed=3416, family=namespace, parameters=row))
    remaining = sorted((r for r in grid if r != reference), key=order)
    chosen = [reference]
    covered = set(reference.items())
    while len(chosen) < 32:
        # Cover marginal parameter levels before hash-ordered filling.
        best = min(remaining, key=lambda r: (-len(set(r.items()) - covered), order(r)))
        chosen.append(best); remaining.remove(best); covered.update(best.items())
    assert covered == {(k, v) for k, values in levels.items() for v in values}
    return chosen


def manifest():
    gs = list(groups())
    counts = {p: 0 for p in ('same_family_mechanisms', 'other_pair_core', 'six_network_core')}
    definitions = {}
    for g in gs:
        aa = group_arms(g)
        key = g['package']
        definitions.setdefault(key, aa)
        counts[key] += len(aa) * len(SEEDS)
    upper = dict(counts, tuning_and_selected_replay_upper=12*4*32+864*4*3,
                 tuned_host_augmentation_upper=864*2*3*3)
    return dict(schema=VERSION, test_access=False, seeds=list(SEEDS), groups=gs,
                arm_sets=definitions, candidates={f: candidates(f) for f in FAMILIES},
                counts=upper, upper_bound=sum(upper.values()), fixed_training_positions=sum(counts.values()),
                selected_transfer='one configuration per method and source-count; no per-arrangement retuning',
                selection_ties=['mean_dev_macro_f1_desc', 'mean_arithmetic_cost_asc', 'mean_parameters_asc', 'fingerprint_asc'],
                publication_gate='data_parent_resource_replay_training_diagnostics_statistics_test_audit',
                allocation=dict(minimum_hours=48, default_hours=48, native_min_gpus=4),
                status='protocol_locked_not_training_acceptance')


def write_protocol(output):
    import fcntl
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    m = manifest(); digest = stable_hash(m)
    with (out/'protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        artifacts = {'protocol.json': json.dumps(m, sort_keys=True, indent=2)+'\n',
                     'positions.jsonl': ''.join(json.dumps(p, sort_keys=True)+'\n' for p in all_positions())}
        for name, content in artifacts.items():
            path = out/name
            if path.exists() and path.read_text() != content:
                raise ValueError('Refuse to overwrite a changed protocol artifact: '+name)
        for name, content in artifacts.items():
            if not (out/name).exists(): atomic_write_text(content, out/name)
        # A controller owns mutable execution progress; registration never resets it.
        if not (out/'status.json').exists():
            atomic_write_json(dict(protocol_sha256=digest, **m['counts'], upper_bound=m['upper_bound'],
                                   test_access=False, executable_tasks=0,
                                   status='waiting_parent_cohort_and_execution_acceptance'), out/'status.json')
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); m = write_protocol(args.output)
    print(json.dumps(dict(counts=m['counts'], upper_bound=m['upper_bound']), indent=2))
