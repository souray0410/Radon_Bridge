"""Audit recorded current-cohort training ancestry without evaluating test."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
from radonbridge.artifacts import SOURCE, ARCHIVE, STUDY, resolve, sha256
from scripts.geometry_evidence import read, write

LABEL_SHA = 'eae604091a1094ed70ff4edbcf6e59d00b57b5be124fb4ac5f76ece1beac0d8d'
SELECTED_SHA = '4285250208dd1496a6912ed6e072abf9d2aa87e05216e78be6ad405e9ea7d24c'
WEIGHT_SHA = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def dataset_splits(source):
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'PairedDataset':
            split = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == 'split'), None)
            if not isinstance(split, ast.Constant) or split.value not in ('train', 'validation'):
                raise ValueError('Training source has a nonliteral or disallowed dataset split')
            calls.append(dict(line=node.lineno, split=split.value))
    if {c['split'] for c in calls} != {'train', 'validation'}:
        raise ValueError('Could not establish train/development dataset construction')
    return calls


def main(args):
    if args.output.exists():
        raise ValueError('Use a new immutable audit output directory')
    args.output.mkdir(parents=True, mode=0o700)
    if sha256(args.labels) != LABEL_SHA or sha256(args.data/'selected.csv') != SELECTED_SHA:
        raise ValueError('Original labels or development cache manifest changed')
    label = pd.read_csv(args.labels, dtype={'participant_id': str})
    selected = pd.read_csv(args.data/'selected.csv', dtype={'participant_id': str})
    if label.participant_id.duplicated().any() or selected.participant_id.duplicated().any():
        raise ValueError('Duplicate participant')
    groups = {k:set(label.loc[label.split == k, 'participant_id']) for k in ('train', 'validation', 'test')}
    if {k:len(v) for k,v in groups.items()} != {'train':1264, 'validation':296, 'test':290}:
        raise ValueError('Unexpected split counts')
    if any(groups[a] & groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Current split overlap')
    if set(selected.participant_id) != groups['train'] | groups['validation']:
        raise ValueError('Cache participants differ from current train/development')
    merged = selected.merge(label[['participant_id','split','label_id']], on='participant_id', suffixes=('_cache','_original'), validate='one_to_one')
    if not (merged.split_cache == merged.split_original).all() or not (merged.label_id_cache == merged.label_id_original).all():
        raise ValueError('Cached split or label differs from original')
    inv = read(args.inventory/'status.json')
    if sha256(args.inventory/'candidate_models.json') != inv['models_sha256'] or sha256(args.inventory/'reference_mapping.json') != inv['reference_mapping_sha256']:
        raise ValueError('Inventory changed')
    models = read(args.inventory/'candidate_models.json')
    refs = read(args.inventory/'reference_mapping.json')
    model_by_id = {r['model_view_id']:r for r in models}
    source_records, code_records, parent_records, rows, issues = {}, {}, {}, [], []

    def find_source(summary):
        p = resolve(summary).parent
        candidates = [p]
        try:
            candidates.append(SOURCE/'runs'/STUDY/p.relative_to(ARCHIVE/'current'))
        except ValueError:
            pass
        try:
            candidates.append(SOURCE/p.relative_to(ARCHIVE/'history'))
        except ValueError:
            pass
        for candidate in candidates:
            for ancestor in [candidate, *candidate.parents]:
                f = ancestor/'run_source.json'
                if f.exists():
                    return f
        raise ValueError('No recorded run_source.json in original or archived ancestry')

    def check_source(summary, cfg):
        f = find_source(summary)
        if str(f) not in source_records:
            s = read(f); audit = s['data_audit']
            if audit['labels_sha256'] != LABEL_SHA or audit['selected_sha256'] != SELECTED_SHA or audit['test_used'] is not False:
                raise ValueError('Recorded run data differs from locked train/development')
            if audit['participants'] != 1560 or s['weight_sha256'] != WEIGHT_SHA:
                raise ValueError('Unexpected input cohort or pretrained initialization')
            source_records[str(f)] = dict(sha256=sha256(f), commit=s['commit'], data_audit=audit,
                                          pretrained_weight_sha256=s['weight_sha256'])
        entry = source_records[str(f)]
        commit = cfg['source_commit']
        if commit != entry['commit']:
            raise ValueError('Training configuration and run source commit differ')
        if commit not in code_records:
            source = subprocess.check_output(['git','-C',str(args.repository),'show',commit+':radonbridge/experiment.py'],text=True)
            code_records[commit] = dict(experiment_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                        dataset_calls=dataset_splits(source))
        return str(f)

    # Parent identities must be unique per training seed and branch.
    parent_refs = {}
    for model in models:
        cfg = model.get('configuration', model)
        for branch, ref in cfg.get('parent_checkpoints', {}).items():
            key = (cfg['seed'], branch)
            if key in parent_refs and parent_refs[key]['sha256'] != ref['sha256']:
                raise ValueError('Seed does not have a unique independent parent')
            parent_refs[key] = ref
    for (seed, branch), ref in sorted(parent_refs.items()):
        try:
            path = resolve(ref['path'])
            if sha256(path) != ref['sha256']:
                raise ValueError('Parent checkpoint SHA changed')
            summary_path = path.parent/'summary.json'
            s = read(summary_path); cfg = s['configuration']
            if cfg['seed'] != seed or cfg['training_stage'] != 'independent' or cfg['bridges'] or cfg.get('parent_checkpoints'):
                raise ValueError('Unexpected independent parent construction')
            if s['state'] != 'complete' or not s['converged_by_policy'] or s['test_used'] is not False:
                raise ValueError('Parent training did not pass')
            if s['modality_checkpoints'][branch]['sha256'] != ref['sha256']:
                raise ValueError('Parent hash not recorded by pretraining summary')
            provenance = check_source(summary_path, cfg)
            parent_records[f'{seed}_{branch}'] = dict(checkpoint_sha256=ref['sha256'], summary_sha256=sha256(summary_path), run_source=provenance)
        except Exception as exc:
            issues.append(dict(reference=f'parent_{seed}_{branch}', error=repr(exc)))
    for ref in refs:
        if 'summary_path' not in ref:
            continue
        try:
            path = resolve(ref['summary_path'])
            if sha256(path) != ref['summary_sha256']:
                raise ValueError('Summary SHA changed')
            s = read(path); cfg = s['configuration']
            if s['state'] != 'complete' or s['test_used'] is not False or not s['converged_by_policy']:
                raise ValueError('Training summary did not pass')
            source = check_source(path, cfg)
            for branch, p in cfg['parent_checkpoints'].items():
                if p['sha256'] != parent_refs[(cfg['seed'],branch)]['sha256']:
                    raise ValueError('Training used a different independent parent')
            if cfg.get('host_checkpoint'):
                h = cfg['host_checkpoint']
                if sha256(h['path']) != h['sha256']:
                    raise ValueError('Host checkpoint SHA changed')
                host_models = [m for m in models if m.get('checkpoint_sha256') == h['sha256']]
                if not host_models:
                    raise ValueError('Augmentation host is outside accepted model ancestry')
            rows.append(dict(reference_id=ref['reference_id'], run_source=source, source_commit=cfg['source_commit'],
                             independent_parent_pair_verified=True))
        except Exception as exc:
            issues.append(dict(reference=ref['reference_id'], error=repr(exc)))
    result = dict(state='recorded_current_lineage_verified' if not issues else 'needs_attention',
        scope='recorded current cohort and model ancestry; not all previous LOOK studies',
        model_views=len(models), training_references_checked=len(rows), independent_parent_branches=len(parent_records),
        run_source_records=len(source_records), training_code_versions=len(code_records), issues=issues,
        split_counts={k:len(v) for k,v in groups.items()}, train_development_test_overlap=0,
        labels_sha256=LABEL_SHA, cache_selected_sha256=SELECTED_SHA,
        current_training_cache_excludes_test=True, inventory_sha256=inv['models_sha256'],
        evidence_limits='Recorded execution provenance and committed dataset construction; not a proof of all historical research decisions or unrecorded external use.',
        audit_implementation_sha256=sha256(Path(__file__)),
        earlier_LOOK_overlap_retained_as_background=True, test_performance_read=False,
        test_inference_permitted=False, updated_at=time.time())
    write(args.output/'source_evidence.json',dict(run_sources=source_records, code_versions=code_records, parent_branches=parent_records, training_references=rows))
    write(args.output/'status.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inventory','output','labels','data','repository'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
