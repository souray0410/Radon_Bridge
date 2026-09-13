"""Metadata-only common cohort audit. Participant records stay in restricted storage."""
import argparse
import gzip
import json
import re
from pathlib import Path

from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash

DISEASES = ('cataract', 'glaucoma', 'macular_degeneration')
COUNTS = {'train': 58358, 'development': 12502}


def read(path):
    return json.loads(Path(path).read_text())


def immutable(value, path):
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing audit differs; use an independent audit directory')
    else:
        atomic_write_json(value, path)
    return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def audit(expansion, output):
    root, out = Path(expansion), Path(output)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = root/'cohort_attempt_001/participants.jsonl.gz'
    summary = read(root/'cohort_attempt_001/cohort_summary.json')
    if file_sha256(raw) != summary['participant_manifest_sha256']:
        raise ValueError('Original participant ledger changed')
    cache = root/'cache_224_attempt_001'
    identity = read(cache/'identity.json')
    if identity['cohort_sha256'] != summary['participant_manifest_sha256']:
        raise ValueError('Cache and original ledger do not match')
    ledger = {}; source = {}
    with gzip.open(raw, 'rt') as f:
        for line in f:
            row = json.loads(line); pid = str(row['id'])
            if pid in ledger: raise ValueError('Duplicate original participant')
            ledger[pid] = row['split']
            # Only split membership is retained for test; no test labels/images are used.
            if row['split'] in COUNTS: source[pid] = row
    split_ref = immutable(dict(schema='ukb_global_split_ledger_v1', participants=ledger,
        source_sha256=file_sha256(raw), usage='split_membership_audit_only',
        test_prediction_access=False), out/'split_ledger.json')
    visits = {}; manifests = {}; common = {}; all_source_refs = {}
    for split, expected in COUNTS.items():
        rows = {}; refs = {}
        for disease in DISEASES:
            path = cache/f'{disease}_{split}.json'; payload = read(path)
            index = {str(r['id']): r for r in payload['samples']}
            if len(index) != len(payload['samples']): raise ValueError('Duplicate cache participant')
            if any(ledger.get(pid) != split for pid in index): raise ValueError('Cross-task split conflict')
            rows[disease] = index; refs[disease] = dict(path=str(path), sha256=file_sha256(path))
        ids = sorted(set.intersection(*(set(r) for r in rows.values())))
        if len(ids) != expected: raise ValueError('Common participant count changed')
        participants = []
        for pid in ids:
            original = source[pid]; eyes = rows[DISEASES[0]][pid]['eyes']
            original_eyes = {eye['eye']: eye for eye in original['eyes']}
            if (not eyes or len(set(eyes)) != len(eyes) or
                    not set(eyes) <= set(original_eyes)):
                raise ValueError('Eye provenance differs')
            for eye in eyes:
                for modality in ('cfp', 'oct'):
                    item = original_eyes[eye][modality]
                    name = str(item['member'])
                    match = re.search(r'_(21015|21016|21017|21018)_(\d+)_(\d+)\.', name)
                    if match is None:
                        # OCT outer archives carry the participant/visit identity.
                        match = re.search(r'_(21015|21016|21017|21018)_(\d+)_(\d+)\.', str(item['archive']))
                    if match is None or (int(match[2]), int(match[3])) != (original['visit'], original['array']):
                        raise ValueError('Unverified image visit/array identity')
                    wanted = {'cfp': {'left':'21015','right':'21016'},
                              'oct': {'left':'21017','right':'21018'}}[modality][eye]
                    if match[1] != wanted: raise ValueError('Image field and eye do not match')
            first = rows[DISEASES[0]][pid]
            labels = {}
            for disease in DISEASES:
                row = rows[disease][pid]
                if (row['eyes'] != eyes or row['path'] != first['path'] or row['files'] != first['files'] or
                        row['label'] != original['labels'][disease] or row['label'] not in (0,1)):
                    raise ValueError('Labels, files or ordered eyes differ across tasks')
                labels[disease] = row['label']
            participants.append(dict(id=pid, eyes=eyes, labels=labels, visit=original['visit'], array=original['array']))
        common[split] = participants; all_source_refs[split] = refs
        visits[split] = dict(participants=len(ids), conflicting_visits=0, conflicting_eyes=0,
            ordered_eye_policy='exact cache order; source ledger membership checked',
            image_file_provenance='original source field, visit and array; cache identity and file digests')
    visit_ref = immutable(dict(schema='ukb_visit_audit_v1', state='accepted', splits=visits,
        original_sha256=file_sha256(raw), cache_identity_sha256=file_sha256(cache/'identity.json'),
        test_access=False), out/'visit_audit.json')
    split_audit = immutable(dict(schema='ukb_cross_task_split_audit_v1', state='accepted',
        source_manifests=all_source_refs, ledger=split_ref, conflicting_participants=0,
        scope='six current train/dev manifests; every future parent separately checked against ledger',
        test_prediction_access=False), out/'cross_task_split_audit.json')
    for split in COUNTS:
        manifests[split] = immutable(dict(schema='radon_common_cohort_v1', state='accepted', split=split,
            participants=common[split], source_manifests=all_source_refs[split],
            label_definition_sha256=summary['source_label_sha256'],
            visit_audit_sha256=visit_ref['sha256'], cross_task_split_audit_sha256=split_audit['sha256'],
            audits=dict(visit=visit_ref, cross_task=split_audit),
            label_description=summary['target'], control_description=summary['control'],
            test_access=False), out/f'{split}.json')
    result = dict(schema='radon_common_cohort_acceptance_v1', state='accepted', manifests=manifests,
        counts=COUNTS, data_role='train/development; global split metadata audited separately',
        parent_usage_audit='mandatory per parent at case validation', test_prediction_access=False)
    immutable(result,out/'accepted.json')
    return result


def check_parent_splits(parent, cohort):
    audit_ref = cohort['audits']['cross_task']; audit_path = Path(audit_ref['path'])
    if file_sha256(audit_path) != audit_ref['sha256']: raise ValueError('Split audit changed')
    record = read(audit_path); ref = record['ledger']
    if file_sha256(ref['path']) != ref['sha256']: raise ValueError('Global split ledger changed')
    ledger = read(ref['path'])['participants']
    for split in COUNTS:
        path = Path(parent[split+'_manifest'])
        if file_sha256(path) != parent[split+'_manifest_sha256']: raise ValueError('Parent data manifest changed')
        payload = read(path)
        if any(ledger.get(str(row['id'])) != split for row in payload['samples']):
            raise ValueError('Parent cross-task participant usage is incompatible with global split')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expansion',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.expansion,args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'manifests'},indent=2))
