"""Finite WS02 grouped-linear supplement registration and independent acceptance."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from radon_bridge.studies.cohort_case import sha, write_json

GROUPS=(1,2,4,8,16)
REFERENCE_IDS={'grouped_g1_radon':'svd','grouped_g1_linear_resample':'linear'}


def read(path):
    return json.loads(Path(path).read_text())


def _receipt(path, config):
    path=Path(path);value=read(path)
    if value.get('configuration')!=config or value.get('test_used') is not False or not value.get('converged_by_policy'):
        raise ValueError('Unaccepted grouped reference')
    for name,digest in value['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Changed grouped reference artifact')
    return value


def _pair_signature(config):
    value=copy.deepcopy(config);value.pop('name',None)
    if value.get('seed')!=3416 or len(value.get('bridges',[]))!=1:raise ValueError('Grouped reference identity mismatch')
    bridge=value['bridges'][0];mode=bridge.pop('mode',None)
    if mode not in ('radon','linear_resample'):raise ValueError('Grouped reference geometry mismatch')
    expected={'nodes':['cfp_stage3','oct_stage3'],'M':32,'S':64,'rho':.125,'compression':'fixed_svd_channel','kernel_size':3}
    if {k:bridge.get(k) for k in expected}!=expected:raise ValueError('Grouped fixed protocol changed')
    if 'group_count' in bridge:raise ValueError('G1 reference must use legacy dense configuration')
    return value,mode


def _accepted_reference_package(root,queue):
    root=Path(root)
    status=read(root/'status.json');audit=read(root/'independent_final_audit.json')
    if (status.get('state')!='complete' or status.get('planned')!=6 or status.get('accepted')!=6
            or status.get('failed') or status.get('active') or status.get('test_used') is not False):
        raise ValueError('G1 source package is not a complete accepted core')
    if (audit.get('passed') is not True or audit.get('participants')!=296 or audit.get('independent_sklearn_f1') is not True
            or audit.get('ordered_ids_labels_matched') is not True or audit.get('all_artifact_sha_verified') is not True
            or audit.get('test_used') is not False):
        raise ValueError('G1 source independent audit is not accepted')
    rows={row['id']:row for row in audit.get('rows',[])}
    by_id={c['id']:c for c in queue['cases']}
    for key in ('svd','linear'):
        if key not in rows or key not in by_id:raise ValueError('G1 source audit missing matched reference')
        trial=root/'trials'/by_id[key]['name'];receipt=read(trial/'accepted.json')
        if rows[key].get('prediction_sha256')!=receipt.get('files',{}).get('selected_predictions.npz'):
            raise ValueError('G1 source audit prediction identity changed')
    return dict(status_sha256=sha(root/'status.json'),audit_sha256=sha(root/'independent_final_audit.json'))


def reference_pair(reference_root):
    root=Path(reference_root);queue=read(root/'queue.json')
    if queue.get('schema')!='radon_small_cohort_core_v1' or queue.get('test_used') is not False:raise ValueError('Unaccepted reference queue')
    package_evidence=_accepted_reference_package(root,queue);by_id={c['id']:c for c in queue['cases']};result={}
    for grouped_id,source_id in REFERENCE_IDS.items():
        case=by_id[source_id];cfg=read(case['config']);trial=root/'trials'/case['name']
        receipt=_receipt(trial/'accepted.json',cfg)
        result[grouped_id]=dict(id=grouped_id,name=case['name'],config=case['config'],trial=str(trial),
            config_sha256=sha(case['config']),receipt_sha256=sha(trial/'accepted.json'),seed=3416,
            provenance='G=1 strict accepted-core reuse; no new training',source_id=source_id,
            source_sequence_id=queue['sequence_id'],receipt=receipt)
    a,amode=_pair_signature(read(result['grouped_g1_radon']['config']))
    b,bmode=_pair_signature(read(result['grouped_g1_linear_resample']['config']))
    if amode!='radon' or bmode!='linear_resample' or a!=b:raise ValueError('G1 matched pair changed')
    return queue,result,package_evidence


def expected_case_ids():
    return [f'grouped_g{g}_{mode}' for g in GROUPS[1:] for mode in ('radon','linear_resample')]


def _new_config(base,group,mode,name):
    cfg=copy.deepcopy(base);cfg['name']=name;bridge=cfg['bridges'][0]
    bridge['mode']=mode;bridge['group_count']=group
    return cfg


def build_package(reference_root,output,sequence_id,source_commit,framework_commit):
    reference_root=Path(reference_root);output=Path(output)
    old_queue,refs,package_evidence=reference_pair(reference_root)
    old_dep=read(reference_root/'dependencies_acceptance.json')
    if old_dep.get('passed') is not True or old_dep.get('test_used') is not False or old_dep.get('queue_sha256')!=sha(reference_root/'queue.json'):
        raise ValueError('Reference dependency receipt changed')
    output.mkdir(parents=True,exist_ok=False);(output/'configs').mkdir();(output/'trials').mkdir()
    base=read(refs['grouped_g1_radon']['config']);cases=[];config_files={}
    for group in GROUPS[1:]:
        for mode in ('radon','linear_resample'):
            key=f'grouped_g{group}_{mode}';name=key+'_seed3416'
            cfg=_new_config(base,group,mode,name);path=output/'configs'/(name+'.json');write_json(path,cfg)
            config_files[str(path)]=sha(path)
            cases.append(dict(id=key,name=name,config=str(path),resource=key,seed=3416,
                provenance='new approved grouped-linear single-seed execution'))
    references=[]
    reference_files={}
    for key in REFERENCE_IDS:
        ref={k:v for k,v in refs[key].items() if k!='receipt'};references.append(ref)
        config_files[ref['config']]=ref['config_sha256']
        receipt=refs[key]['receipt'];trial=Path(ref['trial'])
        reference_files[str(trial/'accepted.json')]=ref['receipt_sha256']
        for name,digest in receipt['files'].items():reference_files[str(trial/name)]=digest
        cfg=read(ref['config'])
        for item in list(cfg['parents'].values())+list(cfg['bridges'][0]['basis_files'].values()):
            if sha(item['path'])!=item['sha256']:raise ValueError('Grouped dependency artifact changed')
            reference_files[item['path']]=item['sha256']
    comparisons=[[f'grouped_g{g}_radon',f'grouped_g{g}_linear_resample'] for g in GROUPS]
    queue=dict(schema='radon_small_cohort_grouped_v1',study_kind='grouped_linear',sequence_id=sequence_id,
        source_commit=source_commit,framework_commit=framework_commit,groups=list(GROUPS),references=references,cases=cases,
        data=old_queue['data'],test_used=False,comparisons=comparisons,
        fixed=dict(stage=3,channel_rank=32,M=32,S=64,kernel=3,seed=3416))
    write_json(output/'queue.json',queue)
    for path,digest in old_dep['data_files'].items():
        if sha(path)!=digest:raise ValueError('Grouped data dependency changed')
    write_json(output/'dependencies_acceptance.json',dict(passed=True,test_used=False,queue_sha256=sha(output/'queue.json'),
        config_files=config_files,references=reference_files,data_files=old_dep['data_files'],
        reused_reference_queue_sha256=sha(reference_root/'queue.json'),reference_package=package_evidence))
    return validate_queue(output,queue)


def git_identity(root,expected_commit):
    root=Path(root)
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=expected_commit:raise ValueError('Target Git HEAD does not match declared commit')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('Target Git checkout is not clean')
    tree=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD^{tree}'],text=True).strip()
    names=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).split(b'\0')
    tracked=[name.decode() for name in names if name and (root/name.decode()).is_file()]
    if not tracked:raise ValueError('Target Git checkout has no tracked files')
    files={str((root/name).resolve()):sha(root/name) for name in tracked}
    return dict(head=head,tree=tree,tracked_files=files)


def validate_grouped_profile_structure(structure,config):
    expected_group=config['bridges'][0].get('group_count',1)
    if expected_group<=1 or not structure or structure.get('group_count')!=expected_group or structure.get('conv_groups')!=expected_group:
        raise ValueError('Invalid grouped resource profile identity')
    widths=structure.get('source_widths')
    counts=structure.get('group_source_counts')
    if (not isinstance(widths,list) or len(widths)!=2 or any(type(v) is not int or v<=0 or v%expected_group for v in widths)
            or not isinstance(counts,list) or len(counts)!=expected_group):
        raise ValueError('Invalid grouped source width/count structure')
    expected_row=[width//expected_group for width in widths]
    if any(not isinstance(row,list) or len(row)!=2 or row!=expected_row for row in counts):
        raise ValueError('Grouped source counts do not match exact partition')
    if (structure.get('within_group_cross_source_gradient',0)<=0 or structure.get('cross_group_direct_gradient')!=0
            or structure.get('inverse_permutation_verified') is not True):
        raise ValueError('Invalid grouped direct-gradient structure')
    return True


def write_cpu_acceptance(output,source,framework,source_commit,framework_commit):
    output=Path(output);source=Path(source);framework=Path(framework)
    source_git=git_identity(source,source_commit);framework_git=git_identity(framework,framework_commit)
    env=dict(os.environ,PYTHONPATH=str(source/'src')+os.pathsep+str(framework/'src'))
    commands=[
        [sys.executable,'-m','pytest','-q','tests/unit'],
        [sys.executable,'scripts/manage.py','check'],
    ]
    results=[]
    for command in commands:
        completed=subprocess.run(command,cwd=source,env=env,text=True,capture_output=True)
        results.append(dict(argv=command[1:],returncode=completed.returncode,
            stdout_tail='\n'.join(completed.stdout.splitlines()[-20:]),stderr_tail='\n'.join(completed.stderr.splitlines()[-20:])))
        if completed.returncode!=0:raise RuntimeError('Target CPU acceptance failed: '+' '.join(command[1:]))
    import torch
    value=dict(schema='radon_grouped_cpu_acceptance_v1',passed=True,test_used=False,
        source_commit=source_commit,framework_commit=framework_commit,source_git=source_git,framework_git=framework_git,
        python=sys.version.split()[0],torch=str(torch.__version__),commands=results)
    write_json(output,value);return value


def write_code_acceptance(root,source,framework,source_commit,framework_commit,cpu_receipt):
    root=Path(root);source=Path(source);framework=Path(framework);cpu_receipt=Path(cpu_receipt)
    cpu=read(cpu_receipt)
    if (cpu.get('schema')!='radon_grouped_cpu_acceptance_v1' or cpu.get('passed') is not True or cpu.get('test_used') is not False
            or cpu.get('source_commit')!=source_commit or cpu.get('framework_commit')!=framework_commit):
        raise ValueError('Grouped CPU acceptance mismatch')
    source_git=git_identity(source,source_commit);framework_git=git_identity(framework,framework_commit)
    if cpu.get('source_git')!=source_git or cpu.get('framework_git')!=framework_git:
        raise ValueError('Grouped CPU Git identity changed')
    files=dict(source_git['tracked_files']);files.update(framework_git['tracked_files'])
    value=dict(schema='radon_grouped_code_acceptance_v1',source_commit=source_commit,framework_commit=framework_commit,
        source_root=str(source.resolve()),framework_root=str(framework.resolve()),source_tree=source_git['tree'],framework_tree=framework_git['tree'],
        files=files,passed_cpu=True,test_used=False,
        cpu_acceptance=dict(path=str(cpu_receipt),sha256=sha(cpu_receipt)))
    write_json(root/'code_acceptance.json',value);return value


def validate_queue(root,queue=None):
    root=Path(root);queue=read(root/'queue.json') if queue is None else queue
    if queue.get('schema')!='radon_small_cohort_grouped_v1' or queue.get('study_kind')!='grouped_linear' or queue.get('test_used') is not False:
        raise ValueError('Unknown grouped cohort contract')
    if tuple(queue.get('groups',()))!=GROUPS or queue.get('fixed')!=dict(stage=3,channel_rank=32,M=32,S=64,kernel=3,seed=3416):
        raise ValueError('Grouped scientific protocol changed')
    if [c['id'] for c in queue.get('cases',[])]!=expected_case_ids():raise ValueError('Grouped new case list changed')
    if [r['id'] for r in queue.get('references',[])]!=list(REFERENCE_IDS):raise ValueError('Grouped G1 references changed')
    expected_comparisons=[[f'grouped_g{g}_radon',f'grouped_g{g}_linear_resample'] for g in GROUPS]
    if queue.get('comparisons')!=expected_comparisons:raise ValueError('Grouped comparison family changed')
    reference_configs={}
    for ref in queue['references']:
        if sha(ref['config'])!=ref['config_sha256'] or sha(Path(ref['trial'])/'accepted.json')!=ref['receipt_sha256']:
            raise ValueError('Grouped reference identity changed')
        cfg=read(ref['config']);_receipt(Path(ref['trial'])/'accepted.json',cfg);reference_configs[ref['id']]=cfg
    baseline,_=_pair_signature(reference_configs['grouped_g1_radon'])
    other,_=_pair_signature(reference_configs['grouped_g1_linear_resample'])
    if baseline!=other:raise ValueError('Grouped G1 pair no longer matched')
    base=reference_configs['grouped_g1_radon']
    for case in queue['cases']:
        if case.get('seed')!=3416:raise ValueError('Grouped seed changed')
        cfg=read(case['config'])
        if sha(case['config'])!=read(root/'dependencies_acceptance.json')['config_files'][case['config']]:
            raise ValueError('Grouped case config changed')
        parts=case['id'].split('_');group=int(parts[1][1:]);mode='_'.join(parts[2:])
        expected_mode='radon' if mode=='radon' else 'linear_resample'
        if cfg!=_new_config(base,group,expected_mode,case['name']):raise ValueError('Grouped case configuration changed')
    return dict(passed=True,new_cases=8,reused_cases=2,groups=list(GROUPS),test_used=False)


def verify_deployment_receipts(root,queue):
    root=Path(root);dependencies=read(root/'dependencies_acceptance.json');code=read(root/'code_acceptance.json')
    if (dependencies.get('passed') is not True or dependencies.get('test_used') is not False
            or dependencies.get('queue_sha256')!=sha(root/'queue.json')):
        raise ValueError('Grouped dependency acceptance changed')
    for section in ('config_files','references','data_files'):
        for path,digest in dependencies[section].items():
            if sha(path)!=digest:raise ValueError('Grouped dependency bytes changed')
    package=dependencies.get('reference_package',{})
    reference_root=Path(queue['references'][0]['trial']).parents[1]
    if package.get('status_sha256')!=sha(reference_root/'status.json') or package.get('audit_sha256')!=sha(reference_root/'independent_final_audit.json'):
        raise ValueError('Grouped source package acceptance changed')
    if (code.get('schema')!='radon_grouped_code_acceptance_v1' or code.get('source_commit')!=queue['source_commit'] or code.get('framework_commit')!=queue['framework_commit']
            or code.get('passed_cpu') is not True or code.get('test_used') is not False):
        raise ValueError('Grouped code acceptance changed')
    cpu=code.get('cpu_acceptance',{})
    current_source=git_identity(code.get('source_root',''),queue['source_commit'])
    current_framework=git_identity(code.get('framework_root',''),queue['framework_commit'])
    if current_source['tree']!=code.get('source_tree') or current_framework['tree']!=code.get('framework_tree'):
        raise ValueError('Grouped deployed Git tree changed')
    if not cpu or sha(cpu['path'])!=cpu['sha256']:
        raise ValueError('Grouped target CPU acceptance changed')
    for path,digest in code['files'].items():
        if sha(path)!=digest:raise ValueError('Grouped deployed source changed')
    return dict(dependencies_sha256=sha(root/'dependencies_acceptance.json'),code_sha256=sha(root/'code_acceptance.json'),
        cpu_acceptance_sha256=cpu['sha256'])


def audit(root):
    import numpy as np
    from sklearn.metrics import f1_score
    from radon_bridge.analysis.cohort_report import report
    root=Path(root);queue=read(root/'queue.json');validate_queue(root,queue);deployment=verify_deployment_receipts(root,queue)
    status=read(root/'status.json')
    if status.get('state')!='complete' or status.get('planned')!=8 or status.get('accepted')!=8 or status.get('failed') or status.get('active'):
        raise ValueError('Grouped finite owner not complete')
    current=report(root)
    if not current.get('matched_results_complete') or len(current.get('results',[]))!=10:raise ValueError('Grouped report inputs incomplete')
    rows=[];ids0=y0=None
    for item in queue['references']+queue['cases']:
        trial=Path(item['trial']) if 'trial' in item else root/'trials'/item['name']
        cfg=read(item['config']);receipt=_receipt(trial/'accepted.json',cfg)
        with np.load(trial/'selected_predictions.npz',allow_pickle=False) as z:
            if ids0 is None:ids0=z['ids'].copy();y0=z['y'].copy()
            if len(z['ids'])!=296 or len(set(z['ids'].tolist()))!=296 or not np.array_equal(z['ids'],ids0) or not np.array_equal(z['y'],y0):
                raise ValueError('Grouped participant order mismatch')
            metrics={k:float(f1_score(z['y'],z[k].argmax(1),average='macro')) for k in ('cfp','oct')}
            for k,v in metrics.items():
                if abs(v-receipt['selected_validation']['tasks'][k]['macro_f1'])>1e-12:raise ValueError('Grouped independent F1 mismatch')
        rows.append(dict(id=item['id'],reused='trial' in item,metrics=metrics,best_epoch=receipt['best_epoch'],
            stop_epoch=receipt['epochs_ran'],prediction_sha256=receipt['files']['selected_predictions.npz']))
    profiles={}
    for case in queue['cases']:
        path=Path(status['resource_profiles'][case['resource']]);value=read(path)
        grouped=value.get('grouped_structure')
        if (value.get('passed') is not True or value.get('test_used') is not False or value.get('configuration')!=read(case['config'])
                or value.get('formal_updates')!=0 or value.get('checkpoint_update_exact') is not True
                or value.get('node_ids_preserved') is not True or value.get('autograd_equivalence') is not True
                or value.get('full_development_participants')!=296):
            raise ValueError('Grouped profile acceptance mismatch')
        validate_grouped_profile_structure(grouped,read(case['config']))
        if sha(path.parent/'resume.pt')!=value.get('resume_sha256'):raise ValueError('Grouped profile resume artifact changed')
        profiles[case['id']]=dict(path=str(path),sha256=sha(path),grouped_structure=grouped,
            peak_reserved_gib=value['peak_reserved_gib'],peak_process_sampled_gib=value.get('peak_process_sampled_gib'),seconds=value['seconds'])
    receipt=dict(schema='radon_grouped_final_audit_v1',passed=True,participants=296,independent_sklearn_f1=True,
        ordered_ids_labels_matched=True,all_artifact_sha_verified=True,test_used=False,queue_sha256=sha(root/'queue.json'),
        deployment=deployment,status_sha256=sha(root/'status.json'),rows=rows,profiles=profiles)
    write_json(root/'independent_final_audit.json',receipt)
    final=report(root)
    if not final.get('complete'):raise ValueError('Grouped publication did not accept final audit')
    receipt['publication_sha256']=sha(root/'publication/current.json');write_json(root/'independent_final_audit.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register');p.add_argument('--reference-root',required=True);p.add_argument('--output',required=True);p.add_argument('--sequence-id',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--framework-commit',required=True)
    u=sub.add_parser('cpu-acceptance');u.add_argument('--output',required=True);u.add_argument('--source',required=True);u.add_argument('--framework',required=True)
    u.add_argument('--source-commit',required=True);u.add_argument('--framework-commit',required=True)
    c=sub.add_parser('code-acceptance');c.add_argument('--root',required=True);c.add_argument('--source',required=True);c.add_argument('--framework',required=True)
    c.add_argument('--source-commit',required=True);c.add_argument('--framework-commit',required=True);c.add_argument('--cpu-receipt',required=True)
    a=sub.add_parser('audit');a.add_argument('--root',required=True)
    args=parser.parse_args()
    if args.command=='register':result=build_package(args.reference_root,args.output,args.sequence_id,args.source_commit,args.framework_commit)
    elif args.command=='cpu-acceptance':result=write_cpu_acceptance(args.output,args.source,args.framework,args.source_commit,args.framework_commit)
    elif args.command=='code-acceptance':result=write_code_acceptance(args.root,args.source,args.framework,args.source_commit,args.framework_commit,args.cpu_receipt)
    else:result=audit(args.root)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
