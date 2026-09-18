"""Fail closed when a registered completed research package disappears from reports.

This gate checks declared coverage, not whether all science has been discovered or
whether a protocol is sound. Add new packages to coverage.json before dispatch.
"""
import argparse
import json
from pathlib import Path
from radon_bridge.studies.research_matrix import arms


def check(root):
    root = Path(root)
    registry = json.loads((root/'coverage.json').read_text())
    package_ids = [p['id'] for p in registry['packages']]
    if len(set(package_ids)) != len(package_ids):
        raise ValueError('Duplicate package identity')
    required = {'ws02_channel_compression','ws02_core','ws02_factorized','grouped_linear','ibex_core','ibex_mechanisms','ibex_factorized','existing_method_augmentation','later_transfer_multinetwork'}
    if not required.issubset(package_ids):
        raise ValueError('Approved package removed from registry')
    mechanisms = next(p for p in registry['packages'] if p['id']=='ibex_mechanisms')
    if mechanisms['expected_arms'] != arms('glaucoma')[6:]:
        raise ValueError('Approved mechanism matrix changed or omitted')
    publications = {}
    for p in registry['packages']:
        if not p.get('report') or not (root/p['report']).is_file():
            raise ValueError('Missing coverage/report entry: '+p['id'])
        if p['state'] != 'accepted':
            if not p.get('remaining'):
                raise ValueError('Unfinished scope has no next action')
            continue
        data = json.loads((root/p['data']).read_text())
        ids = [r['id'] for r in data['results']]
        if p['id']=='ws02_factorized':
            families=['none','svd']+[f'factorized_R{r}_{m}' for r in (256,512,1024) for m in ('radon','linear_resample')]
            expected={f'{f}_seed{s}' for f in families for s in (3416,3417,3418)}
            if set(p['expected_ids'])!=expected:
                raise ValueError('Locked factorized matrix changed')
        if p['id']=='ws02_core' and set(p['expected_ids'])!={'none','svd','linear','self','mmtm','attention'}:
            raise ValueError('Locked six-arm matrix changed')
        if len(ids) != len(set(ids)) or set(ids) != set(p['expected_ids']):
            raise ValueError('Missing, duplicate or unexpected results: '+p['id'])
        if not data['complete'] or data['test_used']:
            raise ValueError('Unaccepted publication scope: '+p['id'])
        if data['sequence_id'] != p['sequence_id']:
            raise ValueError('Scientific identity changed: '+p['id'])
        publications[p['id']] = {r['id']: r for r in data['results']}
    for reuse in registry['reuse']:
        a = publications[reuse['source_package']][reuse['source_id']]
        b = publications[reuse['target_package']][reuse['target_id']]
        source_prediction = a['artifacts']['selected_predictions.npz']['sha256'] if 'artifacts' in a else a['prediction_sha256']
        source_model = a['artifacts']['best.pt']['sha256'] if 'artifacts' in a else a['selected_model']['sha256']
        if source_prediction != b['prediction_sha256']:
            raise ValueError('Reused prediction identity changed')
        if source_model != b['selected_model']['sha256']:
            raise ValueError('Reused model identity changed')
        if abs(a['mean_f1']-b['mean_f1'])>1e-12:
            raise ValueError('Reused metric changed')
    return dict(accepted_packages=len(publications), indexed_packages=len(registry['packages']),
                unique_accepted_executions=sum(len(r) for r in publications.values())-len(registry['reuse']),
                unreviewed_history=registry['unreviewed_history'], full_project_accepted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    print(json.dumps(check(p.parse_args().root),ensure_ascii=False))


if __name__=='__main__':main()
