import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from radon_bridge.analysis.cohort_report import report
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.methods.basis import (
    BASIS_VERSION,
    CENTERED_VERSION,
    save_statistics_state,
    statistics_fingerprint,
    tensor_sha,
)
from radon_bridge.studies.cohort_case import sha
from radon_bridge.studies.cohort_centered import (
    COMPARISONS,
    FIXED,
    finalize_package,
    prepare_package,
    validate_basis_fit,
    validate_queue,
)
from radon_bridge.studies.centered_basis import manifest


def _write(path,data=b'x'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path


def _basis(path,key,parents,participant_sha,centered=False):
    q=np.eye(256,dtype=np.float64);values=np.linspace(256.,1.,256,dtype=np.float64);second=np.diag(values)
    provenance={'parent_checkpoints':parents,'initial_native_sha256':'native','participant_ids_sha256':participant_sha,
        'participants':1264,'source_commit':'old','data_audit_sha256':'data','batchnorm':'eval; unchanged',
        'fit_domain':'native stage3 channel features; all eyes and spatial positions equally weighted'}
    metadata={'version':CENTERED_VERSION if centered else BASIS_VERSION,'source_key':key,'seed':3416,'channels':256,
        'sampled_channel_vectors':495488 if key=='cfp_stage3' else 728064,'centered':centered,'fit_split':'train','test_used':False,
        'normalization':'FF^T/N - mean mean^T; eigenvalues are centered variances; energies are q^T(FF^T/N)q' if centered else 'FF^T/N',
        'ordering':'descending centered variance; maximum-absolute entry of each column nonnegative' if centered else 'descending energy; maximum-absolute entry of each column nonnegative',
        'full_master_sha256':tensor_sha(torch.from_numpy(q)),'torch_version':torch.__version__,'provenance':provenance}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if centered:
        mean=np.zeros(256,dtype=np.float64);metadata.update(runtime_centering=False,runtime_projection='Q^T X; decode Q delta; no mean subtraction or addition',
            centering_scope='training-set channel mean over all participants, eyes and spatial positions',
            mean_energy_fraction=0.,mean_sha256=tensor_sha(torch.from_numpy(mean)))
        np.savez(path,q=q,eigenvalues=values,energies=values,mean=mean,second_moment=second,covariance=second,metadata=json.dumps(metadata,sort_keys=True))
    else:
        metadata['normalization']='FF^T/N; energy values are squared singular values divided by N'
        np.savez(path,q=q,eigenvalues=values,second_moment=second,metadata=json.dumps(metadata,sort_keys=True))
    return {'path':str(path),'sha256':sha(path)}


def _accepted(trial,cfg,prob):
    trial=Path(trial);trial.mkdir(parents=True,exist_ok=True);y=np.arange(296)%2
    np.savez(trial/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=prob,oct=prob)
    _write(trial/'best.pt',(cfg['name']+' model').encode())
    files={n:sha(trial/n) for n in ('selected_predictions.npz','best.pt')}
    metrics={k:classification_metrics(y,prob) for k in ('cfp','oct')}
    value=dict(configuration=cfg,test_used=False,converged_by_policy=True,files=files,best_epoch=1,epochs_ran=8,
        selected_validation=dict(tasks=metrics))
    (trial/'accepted.json').write_text(json.dumps(value));return value


def _reference(tmp_path):
    root=tmp_path/'reference';(root/'configs').mkdir(parents=True);(root/'trials').mkdir()
    parents={'cfp':{'path':str(_write(tmp_path/'parents/cfp.pt')),'sha256':sha(tmp_path/'parents/cfp.pt')},
             'oct':{'path':str(_write(tmp_path/'parents/oct.pt')),'sha256':sha(tmp_path/'parents/oct.pt')}}
    train_ids=[f'train-{i}' for i in range(1264)]
    participant_sha=hashlib.sha256(json.dumps(train_ids,separators=(',',':')).encode()).hexdigest()
    bases={'cfp_stage3':_basis(tmp_path/'bases/cfp.npz','cfp_stage3',parents,participant_sha),
           'oct_stage3':_basis(tmp_path/'bases/oct.npz','oct_stage3',parents,participant_sha)}
    y=np.arange(296)%2;prob=np.stack([1-y,y],axis=1)*.8+.1
    cases=[];audit_rows=[]
    for key,mode in [('svd','radon'),('linear','linear_resample')]:
        name=key+'_seed3416';cfg=dict(schema='factorized_ws_v1',name=name,seed=3416,parents=parents,
            bridges=[dict(nodes=['cfp_stage3','oct_stage3'],M=32,S=64,rho=.125,mode=mode,compression='fixed_svd_channel',basis_files=bases,kernel_size=3)])
        path=root/'configs'/(name+'.json');path.write_text(json.dumps(cfg));receipt=_accepted(root/'trials'/name,cfg,prob)
        cases.append(dict(id=key,name=name,config=str(path),resource=key,seed=3416,provenance='accepted reference'))
        audit_rows.append(dict(id=key,prediction_sha256=receipt['files']['selected_predictions.npz']))
    data=_write(tmp_path/'data/audit.json',b'data');_write(tmp_path/'data/selected.csv',b'selected')
    queue=dict(schema='radon_small_cohort_core_v1',sequence_id='accepted_core',cases=cases,data=str(data.parent),test_used=False)
    (root/'queue.json').write_text(json.dumps(queue))
    (root/'dependencies_acceptance.json').write_text(json.dumps(dict(passed=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        config_files={c['config']:sha(c['config']) for c in cases},references={},data_files={str(data):sha(data)})))
    (root/'status.json').write_text(json.dumps(dict(state='complete',planned=6,accepted=6,failed={},active={},test_used=False)))
    (root/'independent_final_audit.json').write_text(json.dumps(dict(passed=True,participants=296,independent_sklearn_f1=True,
        ordered_ids_labels_matched=True,all_artifact_sha_verified=True,test_used=False,rows=audit_rows)))
    publication=root/'publication';publication.mkdir()
    contrast=dict(reference='linear',difference=.04,ordinary95=[-.01,.09],simultaneous95=[-.02,.10])
    (publication/'current.json').write_text(json.dumps(dict(complete=True,test_used=False,sequence_id='accepted_core',
        results=[{'id':'svd'},{'id':'linear'}],comparisons={'contrasts':[contrast]})))
    return root,parents,bases,participant_sha


def _install_centered_fit(root,parents,bases,participant_sha):
    fit=root/'basis_fit';(fit/'bases').mkdir(parents=True)
    cfg=json.loads((root/'basis_fit_config.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    fingerprint,payload=statistics_fingerprint(cfg,protocol['data'],'native')
    ids=[f'train-{i}' for i in range(1264)];moments={};sums={};counts={}
    for key,ref in bases.items():
        with np.load(ref['path'],allow_pickle=False) as z:second=torch.from_numpy(z['second_moment'].copy())
        count=495488 if key=='cfp_stage3' else 728064
        moments[key]=second*count;sums[key]=torch.zeros(256,dtype=torch.float64);counts[key]=count
    stats=fit/'sufficient_statistics.pt';save_statistics_state(stats,fingerprint,1264,moments,sums,counts,ids)
    stats_sha=sha(stats)
    centered={}
    for key in ('cfp_stage3','oct_stage3'):
        ref=_basis(fit/'bases'/(key+'_centered.npz'),key,parents,participant_sha,centered=True)
        with np.load(ref['path'],allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
        meta['provenance']['source_commit']='source123';meta['provenance']['uncentered_basis_files']=bases
        meta['provenance']['statistics_fingerprint']=fingerprint;meta['provenance']['statistics_fingerprint_payload']=payload
        meta['provenance']['sufficient_statistics_sha256']=stats_sha
        with np.load(ref['path'],allow_pickle=False) as z:
            arrays={name:z[name].copy() for name in z.files if name!='metadata'}
        np.savez(ref['path'],**arrays,metadata=json.dumps(meta,sort_keys=True));ref['sha256']=sha(ref['path']);centered[key]=ref
    summary=dict(state='complete',passed=True,seed=3416,bases=centered,
        provenance={'parent_checkpoints':parents,'initial_native_sha256':'native','participant_ids_sha256':participant_sha,
            'participants':1264,'source_commit':'source123','data_audit_sha256':'data','batchnorm':'eval; unchanged',
            'fit_domain':'native stage3 channel features; all eyes and spatial positions equally weighted',
            'uncentered_basis_files':bases,'statistics_fingerprint':fingerprint,'statistics_fingerprint_payload':payload,
            'sufficient_statistics_sha256':stats_sha},seconds=1.,peak_reserved_mib=10.,test_used=False)
    (fit/'summary.json').write_text(json.dumps(summary))
    value=validate_basis_fit(root);(root/'basis_acceptance.json').write_text(json.dumps(value));return value


def _profile(root,case):
    cfg=json.loads(Path(case['config']).read_text());path=root/'profiles'/case['id']/'accepted.json'
    resume=_write(path.parent/'resume.pt',b'resume '+case['id'].encode())
    value=dict(passed=True,test_used=False,formal_updates=0,configuration=cfg,checkpoint_update_exact=True,
        node_ids_preserved=True,autograd_equivalence=True,full_development_participants=296,
        resume_sha256=sha(resume),peak_reserved_gib=1.0,peak_process_sampled_gib=1.1,seconds=2.0)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path,value


def _full_audit(root,queue,status,current):
    rows=[]
    for item in queue['references']+queue['cases']:
        reused='trial' in item;trial=Path(item['trial']) if reused else root/'trials'/item['name']
        receipt=json.loads((trial/'accepted.json').read_text())
        row=next(r for r in current['results'] if r['id']==item['id'])
        rows.append(dict(id=item['id'],reused=reused,metrics=row['metrics'],best_epoch=receipt['best_epoch'],
            stop_epoch=receipt['epochs_ran'],prediction_sha256=receipt['files']['selected_predictions.npz'],
            model_sha256=receipt['files']['best.pt']))
    profiles={}
    for case in queue['cases']:
        path=Path(status['resource_profiles'][case['resource']]);value=json.loads(path.read_text())
        profiles[case['id']]=dict(path=str(path),sha256=sha(path),resume_sha256=value['resume_sha256'],
            peak_reserved_gib=value['peak_reserved_gib'],peak_process_sampled_gib=value['peak_process_sampled_gib'],seconds=value['seconds'])
    return dict(schema='radon_centered_final_audit_v1',passed=True,participants=296,ordered_ids_labels_matched=True,
        independent_metrics=True,all_artifact_sha_verified=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        status_sha256=sha(root/'status.json'),deployment={'dependencies_sha256':'d','code_sha256':'c','cpu_acceptance_sha256':'u','basis_acceptance_sha256':'b'},
        basis=json.loads((root/'basis_acceptance.json').read_text()),rows=rows,profiles=profiles,
        historical_reference=queue['historical_reference'])


def test_centered_manifest_locks_three_exploratory_contrasts():
    value=manifest()
    assert value['comparisons']==COMPARISONS and value['new_execution_arms']==['centered_radon','centered_linear_resample']
    assert value['strict_reuse']=={'uncentered_radon':'svd','uncentered_linear_resample':'linear'}
    assert value['basis_definition']['runtime_centering'] is False and value['test_access'] is False


def test_centered_prepare_basis_finalize_and_report(tmp_path,monkeypatch):
    reference,parents,bases,participant_sha=_reference(tmp_path);root=tmp_path/'centered'
    result=prepare_package(reference,root,'centered_seq','source123','framework456')
    assert result['comparisons']==COMPARISONS and result['new_cases']==2 and result['reused_cases']==2
    protocol=json.loads((root/'protocol.json').read_text());assert protocol['fixed']==FIXED and protocol['comparisons']==COMPARISONS
    fit=json.loads((root/'basis_fit_config.json').read_text())
    assert fit['fit_centered'] and fit['microbatch']==16 and fit['uncentered_basis_files']==bases
    _install_centered_fit(root,parents,bases,participant_sha)
    final=finalize_package(root);assert final['passed'] and validate_queue(root)['new_cases']==2
    queue=json.loads((root/'queue.json').read_text())
    assert [c['id'] for c in queue['cases']]==['centered_radon','centered_linear_resample']
    for c in queue['cases']:
        cfg=json.loads(Path(c['config']).read_text());assert cfg['bridges'][0]['compression']=='fixed_centered_svd_channel'
    y=np.arange(296)%2
    for i,c in enumerate(queue['cases']):
        p=np.stack([1-y,y],axis=1)*(.78-.02*i)+(.11+.01*i)
        _accepted(root/'trials'/c['name'],json.loads(Path(c['config']).read_text()),p)
    profile_paths={}
    for case in queue['cases']:
        path,_=_profile(root,case);profile_paths[case['resource']]=str(path)
    status=dict(state='complete',planned=2,accepted=2,failed={},active={},resource_profiles=profile_paths,updated_at=1)
    (root/'status.json').write_text(json.dumps(status))
    deployment={'dependencies_sha256':'d','code_sha256':'c','cpu_acceptance_sha256':'u','basis_acceptance_sha256':'b'}
    monkeypatch.setattr('radon_bridge.studies.cohort_centered.verify_deployment_receipts',lambda *_:deployment)
    current=report(root);assert current['matched_results_complete'] and not current['complete'] and len(current['results'])==4
    assert 'comparisons' not in current and current['historical_reference']['simultaneous_family']=='original_core_five_contrasts'
    assert '完整后才生成10,000次配对bootstrap' in (root/'publication/README.md').read_text()
    (root/'independent_final_audit.json').write_text(json.dumps(dict(passed=True,test_used=False,queue_sha256=sha(root/'queue.json'),status_sha256=sha(root/'status.json'))))
    current=report(root);assert not current['complete'] and 'comparisons' not in current
    full=_full_audit(root,queue,status,current);(root/'independent_final_audit.json').write_text(json.dumps(full))
    current=report(root);assert not current['complete'] and 'comparisons' not in current
    current=report(root,_allow_centered_audit_build=True);assert current['complete'] and len(current['comparisons']['contrasts'])==3
    full['publication_sha256']=sha(root/'publication/current.json');(root/'independent_final_audit.json').write_text(json.dumps(full))
    current=report(root);assert current['complete'] and current['study_kind']=='centered_basis'
    assert len(current['comparisons']['contrasts'])==3
    text=(root/'publication/README.md').read_text()
    assert '中心化SVD基比较' in text and '三项同时95%区间' in text and '概率质量与校准方向' in text
    assert '原未中心化几何比较（引用原比较族）' in text and '运行时没有输入去均值' in text
