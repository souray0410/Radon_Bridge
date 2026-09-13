"""Audited common-cohort cases for complete independent MHD task networks."""
import json
from pathlib import Path

from radon_bridge.runtime.state import file_sha256, stable_hash
from radon_bridge.studies.complete_matrix import VERSION, groups, group_arms
from radon_bridge.training.paired_native import validate, DEFAULTS
from radon_bridge.models.native_materialization import verify_selected, load_selected
from radon_bridge.data.observed_group import ObservedGroup
from radon_bridge.evaluation.native_replay import replay_selected


def read(path):
    return json.loads(Path(path).read_text())


def cohort_manifest(reference, split):
    path = Path(reference['path'])
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Locked common cohort changed')
    m = read(path)
    if (m.get('schema') != 'radon_common_cohort_v1' or m.get('split') != split or
            m.get('test_access') is not False or m.get('state') != 'accepted'):
        raise ValueError('Cohort audit not accepted')
    expected = 58358 if split == 'train' else 12502
    if len(m['participants']) != expected or len({r['id'] for r in m['participants']}) != expected:
        raise ValueError('Locked common cohort count/identity differs')
    for field in ('source_manifests', 'label_definition_sha256', 'visit_audit_sha256', 'cross_task_split_audit_sha256'):
        if not m.get(field): raise ValueError('Incomplete cohort provenance: '+field)
    for name, field in (('visit','visit_audit_sha256'),('cross_task','cross_task_split_audit_sha256')):
        ref=m['audits'][name]
        if ref['sha256'] != m[field] or file_sha256(ref['path']) != m[field] or read(ref['path']).get('state') != 'accepted':
            raise ValueError('Cohort audit evidence differs: '+name)
    return m


def validate_spec(spec):
    if spec.get('schema') != 'radon_group_case_v1' or spec.get('protocol') != VERSION or spec.get('test_access') is not False:
        raise ValueError('Unknown or unsealed multi-source case')
    group = next((g for g in groups() if g['id'] == spec['group']), None)
    if group is None or spec['sources'] != group['sources'] or spec['arms'] != group_arms(group):
        raise ValueError('Fixed research matrix differs from locked definition')
    if spec['seed'] not in (3416,3417,3418): raise ValueError('Undeclared seed')
    validate(spec['training'])
    fixed=dict(DEFAULTS,microbatch=spec['training']['microbatch'],num_workers=spec['training']['num_workers'])
    if spec['training'] != fixed:
        raise ValueError('Fixed mechanism case cannot silently change the scientific training recipe')
    # Microbatch affects BN. It is locked separately by matched group, never chosen after results.
    if not spec.get('microbatch_lock_sha256'): raise ValueError('Missing matched microbatch acceptance')
    lock=spec['training_lock']
    if file_sha256(lock['path']) != spec['microbatch_lock_sha256'] or lock['sha256'] != spec['microbatch_lock_sha256']:
        raise ValueError('Matched training lock changed')
    accepted=read(lock['path'])
    if (accepted.get('schema')!='radon_matched_training_lock_v1' or accepted.get('group')!=spec['group'] or
            accepted.get('training')!=spec['training'] or accepted.get('test_access') is not False or
            accepted.get('locked_before_performance') is not True):
        raise ValueError('Matched training recipe is not locked')
    if set(spec['parents']) != {s['key'] for s in spec['sources']}:
        raise ValueError('Incomplete source parents')
    manifests = {split:cohort_manifest(spec['cohort'][split], split) for split in ('train','development')}
    if {r['id'] for r in manifests['train']['participants']} & {r['id'] for r in manifests['development']['participants']}:
        raise ValueError('Common cohort split overlap')
    for s in spec['sources']:
        ref = spec['parents'][s['key']]; root = Path(ref['path'])
        if file_sha256(root/'selected_artifact.json') != ref['manifest_sha256']:
            raise ValueError('Parent receipt changed')
        _, parent, _ = verify_selected(root)
        from radon_bridge.data.cohort_audit import check_parent_splits
        check_parent_splits(parent, manifests['train'])
        track = 'cfp_2d' if s['modality']=='cfp' else 'oct_volume_3d'
        if (parent['training']['seed'] != spec['seed'] or parent['track'] != track or
                parent['disease'] != s['disease'] or parent['model']['name'] != s['model']):
            raise ValueError('Parent role, architecture or seed mismatch')
    for ref in spec['source_pins']:
        if file_sha256(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Immutable source changed')


def prepare(spec, out, device, should_pause):
    from expanded.native import Inputs, collate
    from mhd_framework.models import create_model
    validate_spec(spec)
    parents = {}; parent_specs = {}
    for source in spec['sources']:
        key = source['key']; root = Path(spec['parents'][key]['path'])
        parent_specs[key] = read(root/'spec.json')
        model = load_selected(root, create_model, device='cpu', allow_inference_equivalence=True)
        replay = out/'parents'/key/'replay.json'
        if replay.exists():
            r = read(replay)
            if (r['best_sha256'] != file_sha256(root/'best.pt') or r.get('status') != 'accepted' or
                    r.get('inference_compatibility') != model.inference_compatibility):
                raise ValueError('Parent replay evidence changed')
        else:
            if should_pause(): raise InterruptedError('Pause before parent replay')
            model.to(device)
            try: replay_selected(model, root, Inputs, collate, device, 1, replay)
            finally: model.cpu()
        parents[key] = model
    def dataset(split, augment=False):
        manifest = cohort_manifest(spec['cohort'][split], split)
        datasets = {}
        for s in spec['sources']:
            key = s['key']; p = parent_specs[key]
            datasets[key] = Inputs(p[split+'_manifest'], p[split+'_manifest_sha256'], p['track'],
                                   seed=spec['seed'], augment=augment, recipe=p['training'].get('recipe'))
        ds = ObservedGroup(datasets, spec['sources'], split, [str(r['id']) for r in manifest['participants']], augment=augment)
        # Recheck actual labels/eyes against the common audit, not just its length.
        for row in manifest['participants']:
            for s in spec['sources']:
                actual = datasets[s['key']].rows[ds.indices[s['key']][str(row['id'])]]
                if actual['eyes'] != row['eyes'] or int(actual['label']) != row['labels'][s['disease']]:
                    raise ValueError('Audited labels or eye ownership no longer match')
        return ds
    train = dataset('train', True); fit = dataset('train'); dev = dataset('development')
    sample = fit[0]
    shapes = {key:tuple(x.shape[1:]) for key,x in sample['inputs'].items()}
    return parents, shapes, train, fit, dev
