"""Finite WS02 centered-SVD basis supplement; two new arms plus strict core reuse."""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from radon_bridge.methods.basis import BASIS_VERSION, CENTERED_VERSION, _load_basis, fit_training_bases, load_statistics_state, same_parent_identity
from radon_bridge.studies.cohort_case import sha, write_json
from radon_bridge.studies.cohort_grouped import (
    _accepted_reference_package,
    _pair_signature,
    _receipt,
    git_identity,
    read,
)

REFERENCE_IDS={'uncentered_radon':'svd','uncentered_linear_resample':'linear'}
NEW_IDS=('centered_radon','centered_linear_resample')
COMPARISONS=[
    ['centered_radon','uncentered_radon'],
    ['centered_linear_resample','uncentered_linear_resample'],
    ['centered_radon','centered_linear_resample'],
]
FIXED={'stage':3,'channel_rank':32,'M':32,'S':64,'kernel':3,'rho':.125,'group_count':1,'seed':3416}


def reference_pair(reference_root):
    root=Path(reference_root);queue=read(root/'queue.json')
    if queue.get('schema')!='radon_small_cohort_core_v1' or queue.get('test_used') is not False:
        raise ValueError('Unaccepted centered reference queue')
    package_evidence=_accepted_reference_package(root,queue);by_id={c['id']:c for c in queue['cases']};result={}
    for centered_id,source_id in REFERENCE_IDS.items():
        case=by_id[source_id];cfg=read(case['config']);trial=root/'trials'/case['name'];receipt=_receipt(trial/'accepted.json',cfg)
        result[centered_id]=dict(id=centered_id,name=case['name'],config=case['config'],trial=str(trial),
            config_sha256=sha(case['config']),receipt_sha256=sha(trial/'accepted.json'),seed=3416,
            provenance='strict accepted-core uncentered reuse; no new training',source_id=source_id,
            source_sequence_id=queue['sequence_id'],receipt=receipt)
    a,amode=_pair_signature(read(result['uncentered_radon']['config']))
    b,bmode=_pair_signature(read(result['uncentered_linear_resample']['config']))
    if amode!='radon' or bmode!='linear_resample' or a!=b:
        raise ValueError('Uncentered matched references changed')
    return queue,result,package_evidence


def _historical_geometry(reference_root):
    root=Path(reference_root);publication=read(root/'publication/current.json')
    if publication.get('complete') is not True or publication.get('test_used') is not False:
        raise ValueError('Core publication is not accepted')
    rows={r['id']:r for r in publication.get('results',[])}
    if 'svd' not in rows or 'linear' not in rows:
        raise ValueError('Core publication lacks uncentered geometry rows')
    contrast=next((c for c in publication.get('comparisons',{}).get('contrasts',[]) if c.get('reference')=='linear'),None)
    if contrast is None:
        raise ValueError('Core publication lacks original SVD-linear contrast')
    return dict(source_sequence_id=publication['sequence_id'],publication_sha256=sha(root/'publication/current.json'),
        left='uncentered_radon',right='uncentered_linear_resample',
        difference=contrast['difference'],ordinary95=contrast['ordinary95'],simultaneous95=contrast['simultaneous95'],
        simultaneous_family='original_core_five_contrasts')


def validate_protocol(root,protocol=None):
    root=Path(root);p=read(root/'protocol.json') if protocol is None else protocol
    if (p.get('schema')!='radon_small_cohort_centered_protocol_v1' or p.get('study_kind')!='centered_basis'
            or p.get('test_used') is not False or p.get('fixed')!=FIXED or p.get('comparisons')!=COMPARISONS):
        raise ValueError('Centered protocol changed')
    if [r['id'] for r in p.get('references',[])]!=list(REFERENCE_IDS):
        raise ValueError('Centered strict references changed')
    cfg=read(root/'basis_fit_config.json')
    if (cfg.get('seed')!=3416 or cfg.get('nodes')!=['cfp_stage3','oct_stage3'] or cfg.get('microbatch')!=16
            or cfg.get('fit_centered') is not True or cfg.get('source_commit')!=p.get('source_commit')):
        raise ValueError('Centered basis fit protocol changed')
    base=read(p['references'][0]['config'])
    bridge=base['bridges'][0]
    if cfg.get('parent_checkpoints')!=base['parents'] or cfg.get('uncentered_basis_files')!=bridge['basis_files']:
        raise ValueError('Centered basis fit no longer matches accepted uncentered parent/basis identity')
    for ref in list(cfg['parent_checkpoints'].values())+list(cfg['uncentered_basis_files'].values()):
        if sha(ref['path'])!=ref['sha256']:
            raise ValueError('Centered protocol dependency changed')
    return dict(passed=True,test_used=False,comparisons=COMPARISONS,new_cases=2,reused_cases=2)


def prepare_package(reference_root,output,sequence_id,source_commit,framework_commit):
    reference_root=Path(reference_root);output=Path(output)
    old_queue,refs,package_evidence=reference_pair(reference_root)
    old_dep=read(reference_root/'dependencies_acceptance.json')
    if old_dep.get('passed') is not True or old_dep.get('test_used') is not False or old_dep.get('queue_sha256')!=sha(reference_root/'queue.json'):
        raise ValueError('Centered reference dependency receipt changed')
    output.mkdir(parents=True,exist_ok=False);(output/'configs').mkdir();(output/'trials').mkdir()
    references=[]
    for key in REFERENCE_IDS:
        ref={k:v for k,v in refs[key].items() if k!='receipt'};references.append(ref)
    base=read(refs['uncentered_radon']['config']);bridge=base['bridges'][0]
    fit=dict(seed=3416,parent_checkpoints=base['parents'],nodes=['cfp_stage3','oct_stage3'],microbatch=16,
        source_commit=source_commit,fit_centered=True,uncentered_basis_files=bridge['basis_files'],
        fit_domain='native stage3 channel features; all eyes and spatial positions equally weighted')
    write_json(output/'basis_fit_config.json',fit)
    protocol=dict(schema='radon_small_cohort_centered_protocol_v1',study_kind='centered_basis',sequence_id=sequence_id,
        source_commit=source_commit,framework_commit=framework_commit,data=old_queue['data'],test_used=False,
        fixed=FIXED,comparisons=copy.deepcopy(COMPARISONS),references=references,
        historical_reference=_historical_geometry(reference_root),reference_package=package_evidence,
        reference_queue_sha256=sha(reference_root/'queue.json'),reference_dependencies_sha256=sha(reference_root/'dependencies_acceptance.json'))
    write_json(output/'protocol.json',protocol)
    preparation_files={str(output/'protocol.json'):sha(output/'protocol.json'),str(output/'basis_fit_config.json'):sha(output/'basis_fit_config.json')}
    for ref in references:
        preparation_files[ref['config']]=ref['config_sha256'];preparation_files[str(Path(ref['trial'])/'accepted.json')]=ref['receipt_sha256']
        receipt=refs[ref['id']]['receipt'];trial=Path(ref['trial'])
        for name,digest in receipt['files'].items():preparation_files[str(trial/name)]=digest
    for item in list(base['parents'].values())+list(bridge['basis_files'].values()):
        if sha(item['path'])!=item['sha256']:raise ValueError('Centered preparation dependency changed')
        preparation_files[item['path']]=item['sha256']
    for path,digest in old_dep['data_files'].items():
        if sha(path)!=digest:raise ValueError('Centered data dependency changed')
    write_json(output/'preparation_acceptance.json',dict(schema='radon_centered_preparation_v1',passed=True,test_used=False,
        protocol_sha256=sha(output/'protocol.json'),basis_fit_config_sha256=sha(output/'basis_fit_config.json'),
        files=preparation_files,data_files=old_dep['data_files']))
    return validate_protocol(output,protocol)


def _resource_lock():
    import psutil
    if psutil.virtual_memory().available < .15*psutil.virtual_memory().total:
        raise MemoryError('Host reserve below 15 percent')
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not device or ',' in device:raise ValueError('Centered basis fit requires one explicit CUDA device')
    used,total=map(int,subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True).strip().split(','))
    if total-used < 20*1024:raise MemoryError('Need 10GiB worker plus 10GiB reserve')
    uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    locks=Path(os.environ['RESEARCH_GPU_LOCK_ROOT']);locks.mkdir(parents=True,exist_ok=True)
    handle=(locks/(uuid+'.lock')).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle


def validate_basis_fit(root):
    root=Path(root);validate_protocol(root);cfg=read(root/'basis_fit_config.json')
    summary=read(root/'basis_fit/summary.json')
    if (summary.get('state')!='complete' or summary.get('passed') is not True or summary.get('test_used') is not False
            or summary.get('seed')!=3416 or summary.get('provenance',{}).get('participants')!=1264
            or summary.get('provenance',{}).get('parent_checkpoints')!=cfg['parent_checkpoints']
            or summary.get('provenance',{}).get('source_commit')!=cfg['source_commit']):
        raise ValueError('Centered basis fit summary changed')
    stats_path=root/'basis_fit/sufficient_statistics.pt';stats_sha=summary.get('provenance',{}).get('sufficient_statistics_sha256')
    fingerprint=summary.get('provenance',{}).get('statistics_fingerprint')
    if not stats_path.exists() or not fingerprint or stats_sha!=sha(stats_path):
        raise ValueError('Centered sufficient-statistics receipt changed')
    stats=load_statistics_state(stats_path,fingerprint,cfg['nodes'],True,1264)
    stats_ids_sha=__import__('hashlib').sha256(json.dumps(stats['ids'],separators=(',',':')).encode()).hexdigest()
    if stats_ids_sha!=summary['provenance'].get('participant_ids_sha256'):
        raise ValueError('Centered sufficient-statistics participant order changed')
    old_meta={};centered_meta={}
    if set(summary.get('bases',{}))!=set(cfg['nodes']):raise ValueError('Centered basis sources changed')
    first_old=None
    for key in cfg['nodes']:
        old_ref=cfg['uncentered_basis_files'][key];new_ref=summary['bases'][key]
        old_q,_,om=_load_basis(old_ref['path'],old_ref['sha256']);new_q,_,nm=_load_basis(new_ref['path'],new_ref['sha256'])
        old_parents=om.get('provenance',{}).get('parent_checkpoints',{});new_provenance=nm.get('provenance',{})
        if (set(old_parents)!=set(cfg['parent_checkpoints']) or any(old_parents[k].get('sha256')!=cfg['parent_checkpoints'][k].get('sha256') for k in old_parents)
                or om.get('version')!=BASIS_VERSION or om.get('centered') is not False or nm.get('version')!=CENTERED_VERSION
                or nm.get('centered') is not True or nm.get('runtime_centering') is not False
                or nm.get('source_key')!=key or nm.get('seed')!=3416 or nm.get('channels')!=256
                or nm.get('sampled_channel_vectors')!=om.get('sampled_channel_vectors')
                or new_provenance.get('uncentered_basis_files')!=cfg['uncentered_basis_files']
                or new_provenance.get('source_commit')!=cfg['source_commit']
                or new_provenance.get('participant_ids_sha256')!=om.get('provenance',{}).get('participant_ids_sha256')
                or new_provenance.get('statistics_fingerprint')!=fingerprint
                or new_provenance.get('sufficient_statistics_sha256')!=stats_sha
                or not same_parent_identity(new_provenance.get('parent_checkpoints',{}),cfg['parent_checkpoints'])):
            raise ValueError('Centered basis metadata mismatch')
        if old_q.shape!=new_q.shape or new_q.shape!=(256,256):raise ValueError('Centered basis shape changed')
        with np.load(old_ref['path'],allow_pickle=False) as z:old_second=z['second_moment'].copy()
        with np.load(new_ref['path'],allow_pickle=False) as z:
            second=z['second_moment'].copy();cov=z['covariance'].copy();mean=z['mean'].copy()
        if not np.allclose(second,old_second,atol=1e-10,rtol=1e-10):raise ValueError('Centered refit second moment differs from accepted uncentered statistics')
        if not (np.isfinite(cov).all() and np.isfinite(mean).all() and np.allclose(cov,cov.T,atol=1e-12,rtol=0)):
            raise ValueError('Centered sufficient statistics invalid')
        old_meta[key]=om;centered_meta[key]=nm
        first_old=first_old or om
    if summary['provenance'].get('participant_ids_sha256')!=first_old.get('provenance',{}).get('participant_ids_sha256'):
        raise ValueError('Centered basis participant order changed')
    value=dict(schema='radon_centered_basis_acceptance_v1',passed=True,test_used=False,
        basis_fit_config_sha256=sha(root/'basis_fit_config.json'),summary_sha256=sha(root/'basis_fit/summary.json'),
        participant_ids_sha256=summary['provenance']['participant_ids_sha256'],participants=1264,
        bases=summary['bases'],uncentered_basis_files=cfg['uncentered_basis_files'],
        centered_metadata={key:{k:meta[k] for k in ('version','source_key','seed','channels','sampled_channel_vectors','centered',
            'runtime_centering','runtime_projection','fit_split','normalization','ordering','mean_energy_fraction','full_master_sha256','mean_sha256')}
            for key,meta in centered_meta.items()})
    return value


def fit_basis(root):
    root=Path(root);validate_protocol(root)
    if (root/'basis_acceptance.json').exists():
        value=read(root/'basis_acceptance.json');current=validate_basis_fit(root)
        if value!=current:raise ValueError('Centered basis acceptance changed')
        return value
    if (root/'basis_fit/summary.json').exists():
        value=validate_basis_fit(root);write_json(root/'basis_acceptance.json',value);return value
    lock=_resource_lock()
    try:
        fit_training_bases(read(root/'basis_fit_config.json'),root/'basis_fit',read(root/'protocol.json')['data'])
    finally:
        lock.close()
    value=validate_basis_fit(root);write_json(root/'basis_acceptance.json',value);return value


def _centered_config(base,mode,name,bases):
    cfg=copy.deepcopy(base);cfg['name']=name;bridge=cfg['bridges'][0]
    bridge['mode']=mode;bridge['compression']='fixed_centered_svd_channel';bridge['basis_files']=copy.deepcopy(bases);bridge.pop('group_count',None)
    return cfg


def finalize_package(root):
    root=Path(root);protocol=read(root/'protocol.json');validate_protocol(root,protocol)
    if (root/'queue.json').exists():return validate_queue(root)
    basis=validate_basis_fit(root)
    if not (root/'basis_acceptance.json').exists() or read(root/'basis_acceptance.json')!=basis:
        raise ValueError('Centered basis has not been accepted')
    refs={r['id']:r for r in protocol['references']};base=read(refs['uncentered_radon']['config']);cases=[];config_files={}
    for key,mode in [('centered_radon','radon'),('centered_linear_resample','linear_resample')]:
        name=key+'_seed3416';cfg=_centered_config(base,mode,name,basis['bases']);path=root/'configs'/(name+'.json');write_json(path,cfg)
        config_files[str(path)]=sha(path);cases.append(dict(id=key,name=name,config=str(path),resource=key,seed=3416,
            provenance='new approved centered-SVD single-seed execution'))
    queue=dict(schema='radon_small_cohort_centered_v1',study_kind='centered_basis',sequence_id=protocol['sequence_id'],
        source_commit=protocol['source_commit'],framework_commit=protocol['framework_commit'],references=protocol['references'],
        cases=cases,data=protocol['data'],test_used=False,comparisons=copy.deepcopy(COMPARISONS),fixed=FIXED,
        centered_basis_acceptance_sha256=sha(root/'basis_acceptance.json'),historical_reference=protocol['historical_reference'])
    write_json(root/'queue.json',queue)
    prep=read(root/'preparation_acceptance.json');reference_files=dict(prep['files'])
    reference_files[str(root/'basis_fit/summary.json')]=sha(root/'basis_fit/summary.json')
    reference_files[str(root/'basis_acceptance.json')]=sha(root/'basis_acceptance.json')
    for ref in basis['bases'].values():reference_files[ref['path']]=ref['sha256']
    write_json(root/'dependencies_acceptance.json',dict(passed=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        config_files=config_files,references=reference_files,data_files=prep['data_files'],
        preparation_acceptance_sha256=sha(root/'preparation_acceptance.json'),basis_acceptance_sha256=sha(root/'basis_acceptance.json'),
        reference_package=protocol['reference_package']))
    return validate_queue(root,queue)


def validate_queue(root,queue=None):
    root=Path(root);q=read(root/'queue.json') if queue is None else queue
    if (q.get('schema')!='radon_small_cohort_centered_v1' or q.get('study_kind')!='centered_basis'
            or q.get('test_used') is not False or q.get('fixed')!=FIXED or q.get('comparisons')!=COMPARISONS):
        raise ValueError('Centered scientific queue changed')
    if [r['id'] for r in q.get('references',[])]!=list(REFERENCE_IDS) or [c['id'] for c in q.get('cases',[])]!=list(NEW_IDS):
        raise ValueError('Centered queue arm identity changed')
    if q.get('centered_basis_acceptance_sha256')!=sha(root/'basis_acceptance.json'):
        raise ValueError('Centered basis acceptance identity changed')
    basis=validate_basis_fit(root);refs={r['id']:r for r in q['references']}
    for ref in q['references']:
        if sha(ref['config'])!=ref['config_sha256'] or sha(Path(ref['trial'])/'accepted.json')!=ref['receipt_sha256']:
            raise ValueError('Centered reference identity changed')
        _receipt(Path(ref['trial'])/'accepted.json',read(ref['config']))
    a,amode=_pair_signature(read(refs['uncentered_radon']['config']));b,bmode=_pair_signature(read(refs['uncentered_linear_resample']['config']))
    if amode!='radon' or bmode!='linear_resample' or a!=b:raise ValueError('Centered uncentered references no longer matched')
    base=read(refs['uncentered_radon']['config']);deps=read(root/'dependencies_acceptance.json')
    for case,(key,mode) in zip(q['cases'],[('centered_radon','radon'),('centered_linear_resample','linear_resample')]):
        if case['id']!=key or case.get('seed')!=3416:raise ValueError('Centered case order/seed changed')
        cfg=read(case['config'])
        if sha(case['config'])!=deps['config_files'][case['config']] or cfg!=_centered_config(base,mode,case['name'],basis['bases']):
            raise ValueError('Centered case configuration changed')
    return dict(passed=True,new_cases=2,reused_cases=2,comparisons=copy.deepcopy(COMPARISONS),test_used=False)


def write_cpu_acceptance(output,source,framework,source_commit,framework_commit):
    output=Path(output);source=Path(source);framework=Path(framework)
    source_git=git_identity(source,source_commit);framework_git=git_identity(framework,framework_commit)
    env=dict(os.environ,PYTHONPATH=str(source/'src')+os.pathsep+str(framework/'src'))
    commands=[[sys.executable,'-m','pytest','-q','tests/unit'],[sys.executable,'scripts/manage.py','check']]
    results=[]
    for command in commands:
        completed=subprocess.run(command,cwd=source,env=env,text=True,capture_output=True)
        results.append(dict(argv=command[1:],returncode=completed.returncode,
            stdout_tail='\n'.join(completed.stdout.splitlines()[-20:]),stderr_tail='\n'.join(completed.stderr.splitlines()[-20:])))
        if completed.returncode!=0:raise RuntimeError('Centered target CPU acceptance failed')
    import torch as _torch
    value=dict(schema='radon_centered_cpu_acceptance_v1',passed=True,test_used=False,source_commit=source_commit,
        framework_commit=framework_commit,source_git=source_git,framework_git=framework_git,python=sys.version.split()[0],
        torch=str(_torch.__version__),commands=results)
    write_json(output,value);return value


def write_code_acceptance(root,source,framework,source_commit,framework_commit,cpu_receipt):
    root=Path(root);source=Path(source);framework=Path(framework);cpu_receipt=Path(cpu_receipt);cpu=read(cpu_receipt)
    if (cpu.get('schema')!='radon_centered_cpu_acceptance_v1' or cpu.get('passed') is not True or cpu.get('test_used') is not False
            or cpu.get('source_commit')!=source_commit or cpu.get('framework_commit')!=framework_commit):
        raise ValueError('Centered CPU acceptance mismatch')
    source_git=git_identity(source,source_commit);framework_git=git_identity(framework,framework_commit)
    if cpu.get('source_git')!=source_git or cpu.get('framework_git')!=framework_git:raise ValueError('Centered CPU Git identity changed')
    files=dict(source_git['tracked_files']);files.update(framework_git['tracked_files'])
    value=dict(schema='radon_centered_code_acceptance_v1',source_commit=source_commit,framework_commit=framework_commit,
        source_root=str(source.resolve()),framework_root=str(framework.resolve()),source_tree=source_git['tree'],framework_tree=framework_git['tree'],
        files=files,passed_cpu=True,test_used=False,cpu_acceptance=dict(path=str(cpu_receipt),sha256=sha(cpu_receipt)))
    write_json(root/'code_acceptance.json',value);return value


def verify_deployment_receipts(root,queue):
    root=Path(root);deps=read(root/'dependencies_acceptance.json');code=read(root/'code_acceptance.json')
    if deps.get('passed') is not True or deps.get('test_used') is not False or deps.get('queue_sha256')!=sha(root/'queue.json'):
        raise ValueError('Centered dependencies changed')
    for section in ('config_files','references','data_files'):
        for path,digest in deps[section].items():
            if sha(path)!=digest:raise ValueError('Centered dependency bytes changed')
    if deps.get('basis_acceptance_sha256')!=sha(root/'basis_acceptance.json'):raise ValueError('Centered basis acceptance changed after queue finalization')
    package=deps.get('reference_package',{});reference_root=Path(queue['references'][0]['trial']).parents[1]
    if package.get('status_sha256')!=sha(reference_root/'status.json') or package.get('audit_sha256')!=sha(reference_root/'independent_final_audit.json'):
        raise ValueError('Centered core source package acceptance changed')
    if (code.get('schema')!='radon_centered_code_acceptance_v1' or code.get('source_commit')!=queue['source_commit']
            or code.get('framework_commit')!=queue['framework_commit'] or code.get('passed_cpu') is not True or code.get('test_used') is not False):
        raise ValueError('Centered code acceptance mismatch')
    current_source=git_identity(code['source_root'],queue['source_commit']);current_framework=git_identity(code['framework_root'],queue['framework_commit'])
    if current_source['tree']!=code['source_tree'] or current_framework['tree']!=code['framework_tree']:raise ValueError('Centered deployed Git tree changed')
    cpu=code.get('cpu_acceptance',{})
    if not cpu or sha(cpu['path'])!=cpu['sha256']:raise ValueError('Centered target CPU acceptance changed')
    for path,digest in code['files'].items():
        if sha(path)!=digest:raise ValueError('Centered deployed source changed')
    return dict(dependencies_sha256=sha(root/'dependencies_acceptance.json'),code_sha256=sha(root/'code_acceptance.json'),
        cpu_acceptance_sha256=cpu['sha256'],basis_acceptance_sha256=sha(root/'basis_acceptance.json'))


def calibration_bias(y,prob):
    return float(np.asarray(prob)[:,1].mean()-np.asarray(y).mean())


def validate_final_audit(root,value=None,require_publication=True):
    from radon_bridge.studies.cohort_delivery import profile_verified
    root=Path(root);q=read(root/'queue.json');value=read(root/'independent_final_audit.json') if value is None else value
    status=read(root/'status.json')
    deployment=verify_deployment_receipts(root,q)
    required={'schema','passed','participants','ordered_ids_labels_matched','independent_metrics','all_artifact_sha_verified',
        'test_used','queue_sha256','status_sha256','deployment','basis','rows','profiles','historical_reference'}
    expected=required|({'publication_sha256'} if require_publication else set())
    if set(value)!=expected:
        raise ValueError('Centered final audit fields changed')
    if (value.get('schema')!='radon_centered_final_audit_v1' or value.get('passed') is not True or value.get('participants')!=296
            or value.get('ordered_ids_labels_matched') is not True or value.get('independent_metrics') is not True
            or value.get('all_artifact_sha_verified') is not True or value.get('test_used') is not False
            or value.get('queue_sha256')!=sha(root/'queue.json') or value.get('status_sha256')!=sha(root/'status.json')
            or value.get('deployment')!=deployment or value.get('basis')!=read(root/'basis_acceptance.json')
            or value.get('historical_reference')!=q['historical_reference']):
        raise ValueError('Centered final audit identity changed')
    if require_publication and (not (root/'publication/current.json').exists()
            or value.get('publication_sha256')!=sha(root/'publication/current.json')):
        raise ValueError('Centered final audit publication identity changed')
    items=q['references']+q['cases'];rows=value.get('rows',[])
    if [row.get('id') for row in rows]!=[item['id'] for item in items]:
        raise ValueError('Centered final audit rows changed')
    for item,row in zip(items,rows):
        reused='trial' in item;trial=Path(item['trial']) if reused else root/'trials'/item['name'];cfg=read(item['config'])
        receipt=_receipt(trial/'accepted.json',cfg)
        if (row.get('reused') is not reused or row.get('best_epoch')!=receipt['best_epoch'] or row.get('stop_epoch')!=receipt['epochs_ran']
                or row.get('prediction_sha256')!=receipt['files']['selected_predictions.npz'] or row.get('model_sha256')!=receipt['files']['best.pt']):
            raise ValueError('Centered final audit row artifact identity changed')
        metrics=row.get('metrics',{})
        if set(metrics)!= {'cfp','oct'}:raise ValueError('Centered final audit metric branches changed')
        for key in ('cfp','oct'):
            ref=receipt['selected_validation']['tasks'][key];current=metrics[key]
            for field in ('macro_f1','log_loss','auroc'):
                if field not in current or abs(current[field]-ref[field])>1e-12:
                    raise ValueError('Centered final audit metric identity changed')
            if not np.isfinite(current.get('calibration_in_the_large',np.nan)):
                raise ValueError('Centered final audit calibration direction missing')
    profiles=value.get('profiles',{})
    if set(profiles)!=set(c['id'] for c in q['cases']):raise ValueError('Centered final audit profiles changed')
    for case in q['cases']:
        path=Path(status['resource_profiles'][case['resource']]);profile=profiles[case['id']];cfg=read(case['config'])
        if (profile.get('path')!=str(path) or profile.get('sha256')!=sha(path) or profile.get('resume_sha256')!=read(path).get('resume_sha256')
                or not profile_verified(path,cfg,False)):
            raise ValueError('Centered final audit profile identity changed')
    return True


def audit(root):
    from radon_bridge.analysis.cohort_report import report
    from radon_bridge.evaluation.metrics import classification_metrics
    root=Path(root);q=read(root/'queue.json');validate_queue(root,q);deployment=verify_deployment_receipts(root,q);status=read(root/'status.json')
    if status.get('state')!='complete' or status.get('planned')!=2 or status.get('accepted')!=2 or status.get('failed') or status.get('active'):
        raise ValueError('Centered finite owner not complete')
    current=report(root)
    if not current.get('matched_results_complete') or len(current.get('results',[]))!=4:raise ValueError('Centered report inputs incomplete')
    rows=[];ids0=y0=None
    for item in q['references']+q['cases']:
        trial=Path(item['trial']) if 'trial' in item else root/'trials'/item['name'];cfg=read(item['config']);receipt=_receipt(trial/'accepted.json',cfg)
        with np.load(trial/'selected_predictions.npz',allow_pickle=False) as z:
            if ids0 is None:ids0=z['ids'].copy();y0=z['y'].copy()
            if len(z['ids'])!=296 or len(set(z['ids'].tolist()))!=296 or not np.array_equal(z['ids'],ids0) or not np.array_equal(z['y'],y0):
                raise ValueError('Centered participant order mismatch')
            metrics={k:classification_metrics(z['y'],z[k]) for k in ('cfp','oct')}
            for k,m in metrics.items():
                ref=receipt['selected_validation']['tasks'][k]
                for field in ('macro_f1','log_loss','auroc'):
                    if abs(m[field]-ref[field])>1e-12:raise ValueError('Centered independent metric mismatch')
                m['calibration_in_the_large']=calibration_bias(z['y'],z[k])
        rows.append(dict(id=item['id'],reused='trial' in item,metrics=metrics,best_epoch=receipt['best_epoch'],
            stop_epoch=receipt['epochs_ran'],prediction_sha256=receipt['files']['selected_predictions.npz'],model_sha256=receipt['files']['best.pt']))
    profiles={}
    for case in q['cases']:
        path=Path(status['resource_profiles'][case['resource']]);value=read(path)
        if (value.get('passed') is not True or value.get('test_used') is not False or value.get('configuration')!=read(case['config'])
                or value.get('formal_updates')!=0 or value.get('checkpoint_update_exact') is not True
                or value.get('node_ids_preserved') is not True or value.get('autograd_equivalence') is not True
                or value.get('full_development_participants')!=296 or sha(path.parent/'resume.pt')!=value.get('resume_sha256')):
            raise ValueError('Centered profile acceptance mismatch')
        profiles[case['id']]=dict(path=str(path),sha256=sha(path),resume_sha256=value['resume_sha256'],
            peak_reserved_gib=value['peak_reserved_gib'],peak_process_sampled_gib=value.get('peak_process_sampled_gib'),seconds=value['seconds'])
    receipt=dict(schema='radon_centered_final_audit_v1',passed=True,participants=296,ordered_ids_labels_matched=True,
        independent_metrics=True,all_artifact_sha_verified=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        status_sha256=sha(root/'status.json'),deployment=deployment,basis=read(root/'basis_acceptance.json'),
        rows=rows,profiles=profiles,historical_reference=q['historical_reference'])
    validate_final_audit(root,receipt,require_publication=False)
    write_json(root/'independent_final_audit.json',receipt);final=report(root,_allow_centered_audit_build=True)
    if not final.get('complete'):raise ValueError('Centered provisional publication did not accept final audit')
    receipt['publication_sha256']=sha(root/'publication/current.json');write_json(root/'independent_final_audit.json',receipt)
    final=report(root)
    if not final.get('complete') or receipt['publication_sha256']!=sha(root/'publication/current.json'):
        raise ValueError('Centered final publication/audit seal changed')
    return receipt


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference-root',required=True);p.add_argument('--output',required=True);p.add_argument('--sequence-id',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--framework-commit',required=True)
    f=sub.add_parser('fit-basis');f.add_argument('--root',required=True)
    z=sub.add_parser('finalize');z.add_argument('--root',required=True)
    u=sub.add_parser('cpu-acceptance');u.add_argument('--output',required=True);u.add_argument('--source',required=True);u.add_argument('--framework',required=True)
    u.add_argument('--source-commit',required=True);u.add_argument('--framework-commit',required=True)
    c=sub.add_parser('code-acceptance');c.add_argument('--root',required=True);c.add_argument('--source',required=True);c.add_argument('--framework',required=True)
    c.add_argument('--source-commit',required=True);c.add_argument('--framework-commit',required=True);c.add_argument('--cpu-receipt',required=True)
    a=sub.add_parser('audit');a.add_argument('--root',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare_package(args.reference_root,args.output,args.sequence_id,args.source_commit,args.framework_commit)
    elif args.command=='fit-basis':result=fit_basis(args.root)
    elif args.command=='finalize':result=finalize_package(args.root)
    elif args.command=='cpu-acceptance':result=write_cpu_acceptance(args.output,args.source,args.framework,args.source_commit,args.framework_commit)
    elif args.command=='code-acceptance':result=write_code_acceptance(args.root,args.source,args.framework,args.source_commit,args.framework_commit,args.cpu_receipt)
    else:result=audit(args.root)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
