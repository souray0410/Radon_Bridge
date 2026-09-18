"""Explicit, read-only migration of factorized_ws_queue_v1 to safe report evidence.

Run on authorized storage. No training/inference/test reads; hash all accepted
artifacts, independently recalculate dev F1, preserve original scientific cutoff.
Only allowlisted aggregate fields leave the server. Historical raw schema remains
isolated here, not added to the current training or report runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def migrate(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    q = read('queue.json')
    dep = read('dependencies_acceptance.json')
    old_audit = read('independent_acceptance_20260915/accepted.json')
    state = read('status.json')
    if q['schema'] != 'factorized_ws_queue_v1' or len(q['cases']) != 24:
        raise ValueError('Unsupported or incomplete source matrix')
    if dep['queue_sha256'] != sha(root / 'queue.json') or not dep['passed'] or dep['test_used']:
        raise ValueError('Dependency acceptance changed')
    if old_audit['accepted'] != 24 or old_audit['test_used'] or state['accepted'] != 24 or state['state'] != 'complete':
        raise ValueError('Source not complete')
    old = {r['name']: r for r in old_audit['rows']}
    rows = []
    ids = labels = None
    seen = set()
    for c in q['cases']:
        name = c['name']
        if name in seen:
            raise ValueError('Duplicate scientific case')
        seen.add(name)
        trial = root / 'trials' / name
        cfg_path = Path(c['config'])
        cfg = json.loads(cfg_path.read_text())
        a = json.loads((trial / 'accepted.json').read_text())
        if sha(cfg_path) != dep['config_files'][str(cfg_path)] or cfg != a['configuration']:
            raise ValueError('Configuration changed')
        if sha(trial / 'accepted.json') != old[name]['receipt_sha256']:
            raise ValueError('Original independent receipt changed')
        if a['test_used'] or not a['converged_by_policy'] or a['state'] != 'complete':
            raise ValueError('Unaccepted execution')
        for file, digest in a['files'].items():
            if sha(trial / file) != digest:
                raise ValueError('Artifact changed: ' + name + '/' + file)
        for parent in cfg['parents'].values():
            if sha(parent['path']) != parent['sha256']:
                raise ValueError('Parent changed')
        with np.load(trial / 'selected_predictions.npz', allow_pickle=False) as z:
            if ids is None:
                ids, labels = z['ids'].copy(), z['y'].copy()
            if len(ids) != 296 or len(set(ids.tolist())) != 296 or not np.array_equal(ids, z['ids']) or not np.array_equal(labels, z['y']):
                raise ValueError('Participant order/cohort mismatch')
            scores = {k: float(f1_score(labels, z[k].argmax(1), average='macro')) for k in ('cfp', 'oct')}
        for k, value in scores.items():
            if abs(value - a['selected_validation']['tasks'][k]['macro_f1']) > 1e-12:
                raise ValueError('Metric replay mismatch')
        mean = sum(scores.values()) / 2
        if abs(mean - a['selected_validation']['mean_task_macro_f1']) > 1e-12:
            raise ValueError('Mean metric mismatch')
        bridges = json.loads(json.dumps(cfg['bridges']))
        for bridge in bridges:
            for node, basis in bridge.get('basis_files', {}).items():
                if sha(basis['path']) != basis['sha256']:
                    raise ValueError('Basis changed')
                basis['artifact_ref'] = f'bases/seed{cfg["seed"]}/{node}'
                del basis['path']
        rows.append(dict(id=name, seed=cfg['seed'], family=name.rsplit('_seed', 1)[0],
            configuration_sha256=sha(cfg_path), bridges=bridges,
            parents={k: {'artifact_ref': f'parents/seed{cfg["seed"]}/{k}.pt', 'sha256': p['sha256']} for k, p in cfg['parents'].items()},
            branch_f1=scores, mean_f1=mean, best_epoch=a['best_epoch'], stop_epoch=a['epochs_ran'],
            receipt_sha256=sha(trial/'accepted.json'),
            artifacts={f: {'artifact_ref': f'Radon_Bridge/{root.name}/{name}/{f}', 'sha256': d, 'retention': 'retained_hash_verified'} for f, d in a['files'].items()}))
    comp = read('comparisons.json')
    if comp['test_used'] or comp['participants'] != 296 or comp['seed_repetitions'] != 3 or len(comp['contrasts']) != 9:
        raise ValueError('Statistical scope mismatch')
    return dict(schema='radon_factorized_publication_v1', sequence_id=root.name,
        source_commit=read('code_acceptance.json')['source_commit'],
        evidence_cutoff_unix=state['updated_at'], original_independent_audit_unix=old_audit['checked_at'],
        test_used=False, train=1264, development=296, seeds=[3416,3417,3418], complete=True,
        verification='all receipt/config/artifact/parent SHA; ordered dev participants/labels and sklearn F1; no new inference, no raw-cache rescan, no new bootstrap',
        source_hashes={n: sha(root/n) for n in ['queue.json','results.csv','comparisons.json','code_acceptance.json','dependencies_acceptance.json','independent_acceptance_20260915/accepted.json']},
        configuration=dict(backbones='ResNet18 2D / inflated3D',cfp='two eyes RGB224x224',oct='two eyes 1x32x96x96',stage=3,M=32,S=64,kernel=3,batch=16,min_epochs=8,max_epochs=60,patience=6,min_delta=.001,optimizer='AdamW',backbone_lr=6e-5,head_bridge_lr=1e-4,weight_decay=.01,precision='FP32; cuDNN TF32 true',selection='dev mean branch macro-F1 incl epoch0'),
        results=rows, comparisons=comp,
        limitations=['historical exploratory development cohort','SVD versus factorized not parameter matched','bootstrap fixed selected models and three observed seeds; not general seed uncertainty'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); result = migrate(a.root)
    # Validate everything before touching the last valid aggregate export.
    out = Path(a.out); content = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if not out.exists() or out.read_text() != content:
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp'); temp.write_text(content); temp.replace(out)
    print(json.dumps({'accepted': len(result['results']), 'publication_sha256': sha(out)}))


if __name__ == '__main__':
    main()
