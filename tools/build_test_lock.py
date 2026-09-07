"""Compile outcome-independent comparisons and evaluation jobs before test access."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from radonbridge.artifacts import resolve, sha256, ARCHIVE
from scripts.geometry_evidence import read, write
from radonbridge.communication_analysis import derangements


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(root, inventory, replay, lineage, data, out):
    if out.exists():
        raise ValueError('Use a new lock directory; never overwrite a preregistration')
    out.mkdir(parents=True, mode=0o700)
    models = read(inventory/'candidate_models.json'); refs = read(inventory/'reference_mapping.json')
    inv = read(inventory/'status.json'); replay_status = read(replay/'status.json'); audit = read(lineage/'status.json')
    assert inv['models_sha256'] == sha256(inventory/'candidate_models.json')
    assert inv['reference_mapping_sha256'] == sha256(inventory/'reference_mapping.json')
    assert not inv['issues'] and not audit['issues'] and audit['state'] == 'recorded_current_lineage_verified'
    assert audit['inventory_sha256'] == inv['models_sha256']
    assert replay_status['accepted'] == len(models) == 777 and replay_status['failed'] == 0
    assert replay_status['state'] == 'development_replay_complete_test_still_sealed'
    accepted = read(replay/'accepted_models.json')
    assert set(accepted) == {m['model_view_id'] for m in models}
    for m in models:
        a = accepted[m['model_view_id']]; s = read(a['summary_path'])
        assert sha256(a['summary_path']) == a['summary_sha256'] and s['state'] == 'accepted'
        assert s['strict_model_load'] and s['development_forward_replay'] and s['parameters_BN_gradients_RNG_preserved']
        assert sha256(Path(a['summary_path']).parent/'predictions.npz') == s['replay_prediction_sha256']
    prep = read(data/'preparation_status.json')
    assert prep['state'] == 'verified_inputs_only_test_evaluation_still_sealed'
    assert prep['all_1560_train_development_reconstructions_pixel_identical'] and prep['test_participants'] == 290
    assert sha256(data/'selected.csv') == prep['selected_sha256']
    assert sha256(data/'cache_files.json') == prep['cache_manifest_sha256']
    # Re-hash all prepared caches without any model execution or outcome inspection.
    for item in read(data/'cache_files.json'):
        assert sha256(data/item['relative']) == item['sha256']
    selected = pd.read_csv(data/'selected.csv', dtype={'participant_id':str})
    tests = selected[selected.split == 'test']
    assert len(tests) == 290 and not tests.participant_id.duplicated().any()
    test_ids = tests['order'].to_numpy(dtype=str)
    permutations = derangements(290)
    np.savez(out/'test_permutations.npz', ids=test_ids, permutations=permutations)
    jobs = [dict(job_id=m['model_view_id'], kind='model', model=m) for m in models]
    views = {m['model_view_id']:dict(view_id=m['model_view_id'], job_id=m['model_view_id'], file='predictions.npz',
              development_prediction=str(Path(accepted[m['model_view_id']]['summary_path']).parent/'predictions.npz')) for m in models}
    by_id = collections.defaultdict(set)
    for r in refs: by_id[r.get('source_row_id', r['reference_id'])].add(r['model_view_id'])
    by_prediction = collections.defaultdict(set)
    for m in models:
        if m.get('development_prediction_sha256'): by_prediction[m['development_prediction_sha256']].add(m['model_view_id'])

    def row_view(r):
        candidates = by_id.get(r['id'], set())
        if len(candidates) == 1: return next(iter(candidates))
        pred = r.get('accepted_hashes', {}).get('selected_predictions.npz')
        if pred is None: pred = sha256(resolve(r['directory'])/'selected_predictions.npz')
        candidates = by_prediction[pred]
        checkpoint = r.get('accepted_hashes',{}).get('selected.pt') or sha256(resolve(r['directory'])/'selected.pt')
        candidates = {m['model_view_id'] for m in models if m['model_view_id'] in candidates and m.get('checkpoint_sha256')==checkpoint}
        if len(candidates) != 1: raise ValueError('Ambiguous model view: '+r['id'])
        return next(iter(candidates))

    definitions = []
    sources = {}
    def source(path):
        path = resolve(path); sources[str(path)] = sha256(path); return read(path)
    def add(identifier, family, terms, primary=True, metadata=None):
        # Keys contain actual model view and branch, not display aliases.
        w = collections.defaultdict(float)
        for view, branch, weight in terms:
            assert view in views and branch in ('cfp','oct')
            w[(view,branch)] += float(weight)
        terms = [dict(view_id=v, branch=b, weight=c) for (v,b),c in sorted(w.items()) if abs(c)>1e-14]
        assert terms and abs(sum(t['weight'] for t in terms)) < 1e-10
        definitions.append(dict(id=identifier, family=family, primary=primary, terms=terms, metadata=metadata or {}))
    def dense(defs, rows, family=None):
        ids = [r['_view'] if '_view' in r else row_view(r) for r in rows]
        for d in defs:
            w = np.asarray(d['weights'])
            if w.ndim == 1: w = np.repeat(w[:,None]/2, 2, axis=1)
            if not d.get('estimable',True):
                definitions.append(dict(id=d['id'],family=family or d['family'],primary=d.get('primary',True),
                    terms=[],estimable=False,metadata={k:v for k,v in d.items() if k!='weights'}))
                continue
            add(d['id'],family or d['family'],[(v,b,w[i,j]) for i,v in enumerate(ids) for j,b in enumerate(('cfp','oct')) if w[i,j]],
                d.get('primary',True), {k:v for k,v in d.items() if k not in ('weights','id','family','primary')})

    branch = source(root/'branch_only/manifest.json')
    diagnostics=[]
    for sub in ('branch_only','pretest_completion_v2','multidepth96'):
        for row in source(root/sub/'manifest.json')['rows']:
            d=row.get('diagnostic') or row.get('depth_diagnostic')
            if not d:continue  # Reused references and frozen-training arms follow their recorded scope.
            p=resolve(d['directory'])/'summary.json';digest=d.get('sha256',d.get('summary_sha256'))
            assert sha256(p)==digest
            summary=source(p);assert summary['passed']
            diagnostics.append(dict(reference=row['id'],path=str(p),sha256=digest))
    from scripts.report_geometry_mechanism import definitions as geometry_defs
    dense(geometry_defs(branch['rows'],branch['catalog']), branch['rows'])
    history = source(ARCHIVE/'history/runs/2026_09_05_22_42_53/manifest.json')['rows']
    # Expanded widths already have uniquely hashed inference configurations in the registry.
    hrows = [dict(id=r['source_row_id'],seed=r['seed'],arm=r['arm'],rho=r['rho'],_view=r['model_view_id'])
             for r in refs if r['source_group']=='history_final213']
    from radonbridge.benchmark_statistics import definitions as benchmark_defs
    dense(benchmark_defs(hrows),hrows,'mechanism31')
    from scripts.report_qr_nested_supplement import make_weights
    for group in ('A','B'): dense(make_weights(hrows,group),hrows,'qr_mechanism3' if group=='A' else 'joint_width3')
    # Existing exploratory compression comparisons retain secondary status.
    lookup = {(r['seed'],r['arm'],r['rho']):r['_view'] for r in hrows}
    pairs = []
    for mode in ('radon','self','scrambled','resample'):
        pairs += [(f'centered_svd_{mode}',f'svd_{mode}'),(f'channel_svd_{mode}',f'svd_{mode}'),
                  (f'channel_svd_{mode}',f'centered_svd_{mode}')]
    pairs += [('svd_radon',a) for a in ('no_bridge','learned','qr_radon','svd_self','svd_scrambled','svd_resample')]
    pairs += [('channel_svd_radon',a) for a in ('no_bridge','learned','qr_radon','channel_svd_self','channel_svd_scrambled','channel_svd_resample')]
    for a,b in pairs:
        for rho in (1/16,1/8,1/4):
            for seeds in ((3416,3417,3418),(3416,),(3417,),(3418,)):
                for bs in (('cfp',),('oct',),('cfp','oct')):
                    ts=[(lookup[(s,arm,None if arm=='no_bridge' else rho)],br,sgn/(len(seeds)*len(bs)))
                        for arm,sgn in ((a,1),(b,-1)) for s in seeds for br in bs]
                    add(f'{a}_minus_{b}_rho{rho}_seeds{seeds}_branches{bs}','compression_secondary',ts,False)
    s_axis = source(ARCHIVE/'history/runs/2026_09_06_10_12_59/manifest.json')
    srows = s_axis['rows']+s_axis['references']; sl={(r['seed'],r['arm'],r['rho']):row_view(r) for r in srows}
    for basis_terms, name in [([('svd',1)],'svd_ordered_minus_permuted'),([('qr',1)],'qr_ordered_minus_permuted'),
                              ([('svd',1),('qr',-1)],'basis_S_axis_interaction')]:
        ts=[(sl[(seed,b+'_radon' if sign==1 else b+'_s_permuted',rho)],branch,coef*sign/18)
            for b,coef in basis_terms for sign in (1,-1) for seed in (3416,3417,3418)
            for rho in (1/16,1/8,1/4) for branch in ('cfp','oct')]
        add(name,'s_axis3',ts)
    # Add aliases for reused rows before translating sparse, pre-existing definitions.
    aliases = {}
    for r in refs:
        if len(by_id[r.get('source_row_id', r['reference_id'])]) == 1:
            aliases[r.get('source_row_id',r['reference_id'])] = r['model_view_id']
    depth_ref = source(root/'multidepth96/references.json')
    for rows in depth_ref.values():
        for r in rows: aliases[r['id']] = row_view(r)
    for r in source(root/'pretest_completion_v2/references.json'):aliases[r['id']]=row_view(r)
    for path in (root/'pretest_completion_v2/new_contrasts.json',root/'multidepth96/comparisons.json'):
        for d in source(path):
            assert d['outcome']=='branch_mean_macro_f1'
            add(d['id'],d['family'],[(aliases[k],b,c/2) for k,c in d['weights'].items() for b in ('cfp','oct')],d.get('primary',True))
    frozen = source(root/'dependency_supplement/manifest.json')['rows']
    for control in ('linear_resample','parent'):
        add('frozen_radon_minus_'+control,'frozen2',[(aliases['frozen_radon_seed'+str(s)] if sign==1 else
            aliases[('frozen_linear_resample_seed' if control!='parent' else 'parent_seed')+str(s)],b,sign/6)
            for s in (3416,3417,3418) for sign in (1,-1) for b in ('cfp','oct')])
    components = source(root/'dependency_supplement/diagnostics.json')
    component_views = {}
    for key, meta in components.items():
        original = aliases[meta['source_id']]; model=next(m for m in models if m['model_view_id']==original)
        ds = source(resolve(meta['directory'])/'summary.json')
        assert sha256(resolve(meta['directory'])/'summary.json')==meta['summary_sha256'] and ds['passed']
        assert ds['parameters_buffers_gradients_rng_preserved'] and ds['original_node_ids_preserved']
        jobid='component_'+original
        jobs.append(dict(job_id=jobid,kind='component',model=model))
        for state in ('both_on','host_only','new_only','both_off'):
            pred=resolve(meta['directory'])/(state+'.npz')
            assert sha256(pred)==ds['prediction_files'][state]['sha256']
            view=original if state=='both_on' else original+'__'+state
            if state!='both_on': views[view]=dict(view_id=view,job_id=jobid,file=state+'.npz',development_prediction=str(pred))
            component_views[(key,state)]=view
    for host in ('mmtm_hidden256','attention_d256'):
        for mode in ('radon','linear_resample'):
            keys=[k for k,v in components.items() if v['host']==host and v['mode']==mode];assert len(keys)==6
            for state in ('host_only','new_only'):
                add(f'{host}_{mode}_both_on_minus_{state}','component8',[(component_views[(k,t)],b,sign/12)
                    for k in keys for t,sign in (('both_on',1),(state,-1)) for b in ('cfp','oct')])
    pairing = [r for r in hrows if r['arm'] in ('svd_radon','qr_radon','svd_oct_to_cfp','svd_cfp_to_oct','qr_oct_to_cfp','qr_cfp_to_oct')]
    assert len(pairing)==54 and len(components)==24
    for r in pairing:
        original=r['_view']; model=next(m for m in models if m['model_view_id']==original)
        jobs.append(dict(job_id='pairing_'+original,kind='pairing',model=model))
    # Equal signed weights are one hypothesis even if referenced by several studies.
    unique={}
    for d in definitions:
        signature=fingerprint([{**t,'weight':round(t['weight'],13)} for t in d['terms']] if d['terms'] else {'not_estimable':d['id']})
        if signature not in unique:
            unique[signature]=dict(contrast_id=signature,terms=d['terms'],primary=d['primary'],estimable=bool(d['terms']),aliases=[])
        entry=unique[signature];entry['primary']|=d['primary']
        entry['aliases'].append({k:v for k,v in d.items() if k!='terms'})
    write(out/'jobs.json',jobs); write(out/'model_views.json',list(views.values()));write(out/'comparisons.json',list(unique.values()))
    write(out/'model_reference_mapping.json',refs);write(out/'source_manifests.json',sources)
    write(out/'data_acceptance.json',prep)
    write(out/'diagnostic_acceptance.json',diagnostics)
    payload=dict(state='candidate_test_lock_awaiting_implementation_preflight',model_views=777,component_checkpoints=24,
        pairing_checkpoints=54,jobs=len(jobs),statistical_model_and_component_views=len(views),
        pairing_evaluation_states_per_checkpoint=64,paired_original_reused=True,
        unique_comparisons=len(unique),primary_comparisons=sum(x['primary'] for x in unique.values()),
        primary_aliases=sum(d['primary'] for d in definitions),
        primary_family_counts=dict(collections.Counter(d['family'] for d in definitions if d['primary'])),
        inventory_sha256=inv['models_sha256'],lineage_status=dict(path=str(lineage/'status.json'),sha256=sha256(lineage/'status.json')),
        replay_status=dict(path=str(replay/'status.json'),sha256=sha256(replay/'status.json')),
        replay_directory=str(replay),data_directory=str(data),development_data_directory=str(ARCHIVE/'history/cache/full1264_296'),
        files={name:sha256(out/name) for name in ('jobs.json','model_views.json','comparisons.json','model_reference_mapping.json',
                                               'source_manifests.json','data_acceptance.json','test_permutations.npz','diagnostic_acceptance.json')},
        statistics=dict(resamples=10000,seed=202609072,shared_participant_indices=True,primary='mean of branch macro-F1',
            max_abs_t='center bootstrap minus observed, scale by bootstrap sample SD; ordinary/family/global intervals',
            zero_sd_tolerance_pp=1e-10,practical_margin_pp=1.,no_test_selection=True),
        scientific_scope='Current unified UKB cohort; historical LOOK overlap retained; not all-history untouched test or external clinical validation',
        test_performance_read=False,test_inference_permitted=False,created_at=time.time())
    write(out/'candidate_lock.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('files','statistics','source_manifests')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','inventory','replay','lineage','data','output'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();build(a.root,a.inventory,a.replay,a.lineage,a.data,a.output)
