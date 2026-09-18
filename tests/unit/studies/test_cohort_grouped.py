import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from radon_bridge.analysis.cohort_report import report
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.methods.operator import LinearMixer
from radon_bridge.studies.cohort_case import grouped_structure_probe, sha
from radon_bridge.studies.cohort_grouped import GROUPS, build_package, git_identity, validate_grouped_profile_structure, validate_queue, write_code_acceptance
from radon_bridge.studies.cohort_delivery import profile_verified


def _write(path,data=b'x'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path



def _git_repo(root,files):
    root=Path(root);root.mkdir(parents=True)
    subprocess.run(['git','init','-q',str(root)],check=True)
    for name,data in files.items():_write(root/name,data)
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-q','-m','fixture'],check=True)
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def _accepted(trial,cfg,prob=None):
    trial=Path(trial);trial.mkdir(parents=True,exist_ok=True)
    y=np.arange(296)%2
    if prob is None:prob=np.stack([1-y,y],axis=1)*.8+.1
    np.savez(trial/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=prob,oct=prob)
    _write(trial/'best.pt',b'model')
    files={name:sha(trial/name) for name in ('selected_predictions.npz','best.pt')}
    metrics={k:classification_metrics(y,prob) for k in ('cfp','oct')}
    value=dict(configuration=cfg,test_used=False,converged_by_policy=True,files=files,best_epoch=1,epochs_ran=8,
        selected_validation=dict(tasks=metrics))
    (trial/'accepted.json').write_text(json.dumps(value))
    return value


def _reference(tmp_path):
    root=tmp_path/'reference';(root/'configs').mkdir(parents=True);(root/'trials').mkdir()
    parent_cfp=_write(tmp_path/'parents/cfp.pt');parent_oct=_write(tmp_path/'parents/oct.pt')
    basis_cfp=_write(tmp_path/'bases/cfp.npz');basis_oct=_write(tmp_path/'bases/oct.npz')
    parents={'cfp':{'path':str(parent_cfp),'sha256':sha(parent_cfp)},'oct':{'path':str(parent_oct),'sha256':sha(parent_oct)}}
    bases={'cfp_stage3':{'path':str(basis_cfp),'sha256':sha(basis_cfp)},'oct_stage3':{'path':str(basis_oct),'sha256':sha(basis_oct)}}
    cases=[];audit_rows=[]
    for key,mode in [('svd','radon'),('linear','linear_resample')]:
        name=key+'_seed3416'
        cfg=dict(schema='factorized_ws_v1',name=name,seed=3416,parents=parents,
            bridges=[dict(nodes=['cfp_stage3','oct_stage3'],M=32,S=64,rho=.125,mode=mode,
                compression='fixed_svd_channel',basis_files=bases,kernel_size=3)])
        path=root/'configs'/(name+'.json');path.write_text(json.dumps(cfg))
        receipt=_accepted(root/'trials'/name,cfg)
        cases.append(dict(id=key,name=name,config=str(path),resource=key,seed=3416,provenance='accepted reference'))
        audit_rows.append(dict(id=key,prediction_sha256=receipt['files']['selected_predictions.npz']))
    data=_write(tmp_path/'data/fixture.bin',b'data')
    queue=dict(schema='radon_small_cohort_core_v1',sequence_id='accepted_core',cases=cases,data=str(data.parent),test_used=False)
    (root/'queue.json').write_text(json.dumps(queue))
    dep=dict(passed=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        config_files={c['config']:sha(c['config']) for c in cases},references={},data_files={str(data):sha(data)})
    (root/'dependencies_acceptance.json').write_text(json.dumps(dep))
    (root/'status.json').write_text(json.dumps(dict(state='complete',planned=6,accepted=6,failed={},active={},test_used=False)))
    (root/'independent_final_audit.json').write_text(json.dumps(dict(passed=True,participants=296,independent_sklearn_f1=True,
        ordered_ids_labels_matched=True,all_artifact_sha_verified=True,test_used=False,rows=audit_rows)))
    return root


def test_grouped_package_registers_two_references_and_eight_new_cases(tmp_path):
    reference=_reference(tmp_path);root=tmp_path/'grouped'
    result=build_package(reference,root,'grouped_seq','source123','framework456')
    assert result==dict(passed=True,new_cases=8,reused_cases=2,groups=list(GROUPS),test_used=False)
    queue=json.loads((root/'queue.json').read_text())
    assert len(queue['references'])==2 and len(queue['cases'])==8 and len(queue['comparisons'])==5
    assert all(json.loads(Path(c['config']).read_text())['bridges'][0]['group_count'] in (2,4,8,16) for c in queue['cases'])
    assert queue['references'][0]['source_sequence_id']=='accepted_core'
    assert validate_queue(root)['passed']
    case=queue['cases'][0];path=Path(case['config']);original=path.read_bytes();cfg=json.loads(original);cfg['bridges'][0]['group_count']=8;path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='config changed|configuration changed'):validate_queue(root)
    path.write_bytes(original)
    assert validate_queue(root)['passed']


def test_grouped_code_acceptance_records_deployed_source_and_framework(tmp_path):
    root=tmp_path/'run';root.mkdir();source=tmp_path/'source';framework=tmp_path/'framework'
    source_commit=_git_repo(source,{'src/radon_bridge/a.py':b'a=1\n','scripts/check.py':b'print(1)\n','tests/test_a.py':b'def test_a(): pass\n'})
    framework_commit=_git_repo(framework,{'src/mhd_framework/b.py':b'b=2\n','tests/test_b.py':b'def test_b(): pass\n'})
    source_git=git_identity(source,source_commit);framework_git=git_identity(framework,framework_commit)
    cpu=tmp_path/'cpu.json'
    cpu.write_text(json.dumps(dict(schema='radon_grouped_cpu_acceptance_v1',passed=True,test_used=False,
        source_commit=source_commit,framework_commit=framework_commit,source_git=source_git,framework_git=framework_git)))
    value=write_code_acceptance(root,source,framework,source_commit,framework_commit,cpu)
    assert value['passed_cpu'] and value['test_used'] is False and len(value['files'])==5
    assert value['cpu_acceptance']['sha256']==sha(cpu)
    assert json.loads((root/'code_acceptance.json').read_text())==value


def test_grouped_report_combines_strict_reuse_and_new_cases(tmp_path):
    reference=_reference(tmp_path);root=tmp_path/'grouped'
    build_package(reference,root,'grouped_seq','source123','framework456')
    queue=json.loads((root/'queue.json').read_text())
    for case in queue['cases']:_accepted(root/'trials'/case['name'],json.loads(Path(case['config']).read_text()))
    current=report(root)
    assert current['matched_results_complete'] and not current['complete'] and current['study_kind']=='grouped_linear' and current['reused_references']==2
    assert len(current['results'])==10 and len(current['comparisons']['contrasts'])==5
    reused=next(r for r in current['results'] if r['id']=='grouped_g1_radon')
    assert reused['reused'] and reused['selected_model']['artifact_ref']=='Radon_Bridge/accepted_core/svd/best.pt'
    text=(root/'publication/README.md').read_text()
    assert 'G=1/2/4/8/16' in text and 'G=16' in text and '跨组直接梯度为零不代表完整网络彼此独立' in text


def test_grouped_profile_probe_checks_direct_connectivity():
    mixer=LinearMixer([8,18],3,group_count=2,source_ranks=[4,6],directions=[2,3])
    exchange=SimpleNamespace(mixer=mixer)
    graph=SimpleNamespace(modules_by_name=lambda:{'bridge_0_exchange':exchange})
    value=grouped_structure_probe(graph,{'bridges':[{'group_count':2}]})
    assert value['group_count']==2 and value['conv_groups']==2
    assert value['source_widths']==[8,18]
    assert value['within_group_cross_source_gradient']>0
    assert value['cross_group_direct_gradient']==0
    assert value['group_source_counts']==[[4,9],[4,9]]
    assert validate_grouped_profile_structure(value,{'bridges':[{'group_count':2}]})
    bad=dict(value,group_source_counts=[])
    with pytest.raises(ValueError,match='source width/count'):validate_grouped_profile_structure(bad,{'bridges':[{'group_count':2}]})


def test_profile_recovery_binds_exact_config_structure_and_resume(tmp_path):
    cfg={'bridges':[{'group_count':2}]};profile=tmp_path/'profile/accepted.json';resume=_write(profile.parent/'resume.pt',b'resume')
    structure=dict(group_count=2,conv_groups=2,source_widths=[8,18],group_source_counts=[[4,9],[4,9]],
        within_group_cross_source_gradient=1.0,cross_group_direct_gradient=0.0,inverse_permutation_verified=True)
    value=dict(passed=True,test_used=False,formal_updates=0,configuration=cfg,checkpoint_update_exact=True,
        node_ids_preserved=True,autograd_equivalence=True,full_development_participants=296,resume_sha256=sha(resume),
        grouped_structure=structure)
    profile.parent.mkdir(parents=True,exist_ok=True);profile.write_text(json.dumps(value))
    assert profile_verified(profile,cfg,True)
    wrong={'bridges':[{'group_count':4}]}
    with pytest.raises(ValueError,match='Invalid resource profile|Invalid grouped'):profile_verified(profile,wrong,True)
    resume.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='resume changed'):profile_verified(profile,cfg,True)
