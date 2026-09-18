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
    required = {'ws02_channel_compression','ws02_core','ws02_factorized','grouped_linear','ws02_centered_basis',
                'ibex_core','ibex_mechanisms','ibex_factorized','existing_method_augmentation','later_transfer_multinetwork'}
    if not required.issubset(package_ids):
        raise ValueError('Approved package removed from registry')
    mechanisms = next(p for p in registry['packages'] if p['id']=='ibex_mechanisms')
    if mechanisms['expected_arms'] != arms('glaucoma')[6:]:
        raise ValueError('Approved mechanism matrix changed or omitted')
    centered = next(p for p in registry['packages'] if p['id']=='ws02_centered_basis')
    if (centered.get('expected_ids')!=['uncentered_radon','uncentered_linear_resample','centered_radon','centered_linear_resample']
            or centered.get('comparisons')!=[['centered_radon','uncentered_radon'],
                ['centered_linear_resample','uncentered_linear_resample'],['centered_radon','centered_linear_resample']]
            or centered.get('strict_reuse')!={'uncentered_radon':'ws02_core/svd','uncentered_linear_resample':'ws02_core/linear'}
            or centered.get('fixed')!={'stage':3,'r':32,'M':32,'S':64,'k':3,'rho':.125,'group_count':1,'seed':3416}):
        raise ValueError('Locked centered WS02 package changed')
    if centered.get('state')=='accepted':
        review=centered.get('acceptance_review',{})
        if (centered.get('accepted_scope')!='ws02_single_seed_centered_basis_direction'
                or review.get('kind')!='left_independent_review' or review.get('passed') is not True
                or review.get('audit_sha256')!='43a11f9c322e84c3e7501d487bb0c74d1c4c86542b5f248e3d91624afc58f5bd'
                or review.get('participants')!=296 or review.get('source_files_exact')!=91
                or review.get('asset_and_input_files_verified')!=3160 or review.get('bootstrap_resamples')!=10000
                or review.get('test_used') is not False):
            raise ValueError('Accepted centered WS02 review identity changed')
    centered_reuse=[
        {'source_package':'ws02_core','source_id':'svd','target_package':'ws02_centered_basis','target_id':'uncentered_radon'},
        {'source_package':'ws02_core','source_id':'linear','target_package':'ws02_centered_basis','target_id':'uncentered_linear_resample'},
    ]
    if centered.get('state')=='accepted' and any(row not in registry['reuse'] for row in centered_reuse):
        raise ValueError('Accepted centered WS02 package lacks exact global reuse registration')
    reuse_fields={'source_package','source_id','target_package','target_id'}
    if any(set(row)!=reuse_fields for row in registry['reuse']):
        raise ValueError('Invalid reuse registration fields')
    reuse_targets=[(row['target_package'],row['target_id']) for row in registry['reuse']]
    if len(reuse_targets)!=len(set(reuse_targets)):
        raise ValueError('Duplicate reuse target registration')
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
        if p['id']=='grouped_linear':
            expected={f'grouped_g{g}_{mode}' for g in (1,2,4,8,16) for mode in ('radon','linear_resample')}
            if set(p['expected_ids'])!=expected or p.get('groups')!=[1,2,4,8,16] or p.get('accepted_scope')!='ws02_single_seed_fixed_svd_grouped':
                raise ValueError('Locked grouped WS02 package changed')
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
