"""Independent work units referencing one unchanged scientific case.

This module compiles/validates work, not allocations or claims. Existing shared
owners claim the unit directories. Core delivery never certifies the full case.
"""
import json
from pathlib import Path
import time

from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash

CORE = ('continue', 'svd_radon', 'svd_resample', 'svd_self', 'mmtm256', 'attention256')


def read(path):
    return json.loads(Path(path).read_text())


def checked_files(root, files):
    root = Path(root).resolve()
    for name, digest in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError('Changed project-unit evidence')


def verify_prepared(spec, root):
    root = Path(root); receipt = read(root/'prepared.json')
    if receipt.get('identity') != stable_hash(spec) or receipt.get('state') != 'accepted' or receipt.get('test_access') is not False:
        raise ValueError('Unaccepted shared preparation')
    required = {'spec.json','bases/accepted.json','parent_predictions.npz',
                'parents/cfp/replay.json','parents/oct/replay.json'}
    if not required <= receipt['files'].keys():
        raise ValueError('Incomplete shared preparation')
    checked_files(root, receipt['files'])
    basis = read(root/'bases/accepted.json')
    if basis.get('identity') != stable_hash(spec) or basis.get('fit_split') != 'train' or basis.get('test_access') is not False:
        raise ValueError('Basis provenance mismatch')
    for entries in basis['bases'].values():
        for ref in entries.values():
            if file_sha256(ref['path']) != ref['sha256']:
                raise ValueError('Changed basis bytes')
    return receipt


def arm_identity(spec, root, arm):
    root = Path(root); bases = read(root/'bases/accepted.json')['bases']; host_sha = None
    if arm.get('host'):
        host_sha = file_sha256(root/'arms'/arm['host']/'best.pt')
        receipt = read(root/'host_bases'/arm['host']/'accepted.json')
        expected = stable_hash(dict(case=stable_hash(spec),host=arm['host'],best_sha256=host_sha))
        if receipt.get('identity') != expected:
            raise ValueError('Host basis belongs to another checkpoint')
        bases = receipt['bases']
    for entries in bases.values():
        for ref in entries.values():
            if file_sha256(ref['path']) != ref['sha256']:
                raise ValueError('Arm basis changed')
    return stable_hash(dict(case=stable_hash(spec),arm=arm,bases=bases,host_best_sha256=host_sha))


def verify_training(spec, root, name):
    from radon_bridge.studies.project_case import verify_arm
    arm = next(a for a in spec['arms'] if a['id'] == name)
    return verify_arm(Path(root)/'arms'/name, arm_identity(spec,root,arm))


def verify_core(spec, root):
    root=Path(root); out=root/'packages/core'; receipt=read(out/'accepted.json')
    if (receipt.get('identity') != stable_hash(spec) or receipt.get('state') != 'accepted'
            or receipt.get('test_access') is not False or receipt.get('arm_ids') != list(CORE)):
        raise ValueError('Core package identity changed')
    required={'report/metrics.csv','report/paired_statistics.json','report/README.zh-CN.md',
              'report/comparison.svg','diagnostics.json'}
    if not required <= receipt['files'].keys():raise ValueError('Core report incomplete')
    checked_files(out,receipt['files'])
    for name in CORE:verify_training(spec,root,name)
    diagnostics=read(out/'diagnostics.json')
    if diagnostics.get('identity')!=stable_hash(spec) or diagnostics.get('arm_ids')!=list(CORE) or diagnostics.get('state')!='accepted':
        raise ValueError('Core diagnostics incomplete')
    checked_files(root/'diagnostics',diagnostics['files'])
    return receipt


def finish_core(spec,root,parents,shapes,bases,train,dev,device,paused):
    from radon_bridge.analysis.native_diagnostics import diagnose
    from radon_bridge.analysis.project_report import report_case
    root=Path(root);out=root/'packages/core';out.mkdir(parents=True,exist_ok=True)
    if (out/'accepted.json').exists():return verify_core(spec,root)
    for name in CORE:verify_training(spec,root,name)
    diagnostic=diagnose(spec,root,parents,shapes,bases,train,dev,device,paused,arm_ids=CORE)
    atomic_write_json(diagnostic,out/'diagnostics.json')
    report_case(spec,root,arm_ids=CORE,output=out/'report')
    render_core(out/'report')
    files={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='accepted.json'}
    receipt=dict(state='accepted',identity=stable_hash(spec),arm_ids=list(CORE),files=files,
                 scope='six_arm_core_only; remaining_mechanisms_are_not_complete',test_access=False)
    atomic_write_json(receipt,out/'accepted.json');return verify_core(spec,root)


def render_core(out):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    with (out/'metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for field,label in [('cfp_f1','CFP'),('oct_f1','OCT'),('mean_f1','Branch mean')]:
        ax.plot(range(len(rows)),[float(r[field])*100 for r in rows],marker='o',label=label)
    ax.set_xticks(range(len(rows)),[r['arm'] for r in rows],rotation=25,ha='right')
    ax.set_ylabel('Development macro-F1 (%) — higher is better');ax.set_ylim(0,100);ax.legend()
    fig.savefig(out/'comparison.svg');plt.close(fig)
    with (out/'README.zh-CN.md').open('a') as stream:
        stream.write('\n![六臂匹配结果](comparison.svg)\n\n横轴为六种匹配方法，纵轴为dev macro-F1百分比，越高越好。三条线分别为CFP、OCT及其指标平均，连线仅方便阅读，不是连续变量趋势或概率融合。图不证明跨种子显著性；差值及区间见表。本核心包不代表机制补充全部完成。\n')


def load_unit(unit):
    if unit.get('schema')!='radon_project_unit_v1' or unit.get('test_access') is not False:
        raise ValueError('Unregistered project unit')
    if file_sha256(unit['case_spec'])!=unit['case_spec_sha256']:
        raise ValueError('Case specification changed')
    spec=read(unit['case_spec']);root=Path(unit['case_root'])
    allowed={'prepare','report_core','report_full'}|{'arm_'+a['id'] for a in spec['arms']}
    if unit['unit'] not in allowed:raise ValueError('Unknown project unit')
    return spec,root


def verify_unit(output,unit):
    spec,root=load_unit(unit);out=Path(output);receipt=read(out/'accepted.json')
    if out.resolve()!=(root/'units'/unit['unit']).resolve():raise ValueError('Unit output mismatch')
    if receipt.get('identity')!=stable_hash(unit) or receipt.get('state')!='accepted' or receipt.get('test_access') is not False:
        raise ValueError('Unit acceptance changed')
    verify_prepared(spec,root)
    if unit['unit'].startswith('arm_'):verify_training(spec,root,unit['unit'][4:])
    elif unit['unit']=='report_core':verify_core(spec,root)
    elif unit['unit']=='report_full':
        from radon_bridge.studies.project_case import verify_case
        verify_case(root,spec)
    return receipt


def prerequisites(spec,root,name):
    """Only real dependencies; peers never depend on one another's scores."""
    if name=='prepare':return
    verify_prepared(spec,root)
    if name.startswith('arm_'):
        arm=next(a for a in spec['arms'] if a['id']==name[4:])
        if arm.get('host'):verify_training(spec,root,arm['host'])
    else:
        for arm in (CORE if name=='report_core' else [a['id'] for a in spec['arms']]):
            verify_training(spec,root,arm)
        if name=='report_full':verify_core(spec,root)


def compile_units(case_spec,root):
    """Deterministic unit specs reuse the case and original arm identities."""
    spec=read(case_spec);root=Path(root);tasks=[]
    names=['prepare']+['arm_'+a['id'] for a in spec['arms']]+['report_core','report_full']
    for name in names:
        unit=dict(schema='radon_project_unit_v1',case_spec=str(case_spec),case_spec_sha256=file_sha256(case_spec),
                  case_root=str(root),unit=name,seed=spec['seed'],test_access=False)
        path=root/'unit_specs'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and read(path)!=unit:raise ValueError('Unit specification changed')
        if not path.exists():atomic_write_json(unit,path)
        priority=0 if name=='prepare' else 1 if name=='report_core' else 2 if name[4:] in CORE else 3
        tasks.append(dict(id=stable_hash(unit),spec=str(path),spec_sha256=file_sha256(path),
            run_dir=str(root/'units'/name),execution='radon_unit',priority=priority))
    return tasks


def unit_ready(task):
    unit=read(task['spec']);spec,root=load_unit(unit)
    try:prerequisites(spec,root,unit['unit'])
    except FileNotFoundError:return False
    return True


def execute(unit,output,device='cuda:0'):
    from radon_bridge.studies.project_case import execute as execute_case
    spec,root=load_unit(unit);out=Path(output)
    if out.resolve()!=(root/'units'/unit['unit']).resolve():raise ValueError('Unit output mismatch')
    if (out/'accepted.json').exists():return verify_unit(out,unit)
    prerequisites(spec,root,unit['unit'])
    result=execute_case(spec,root,device,unit=unit['unit'])
    if result.get('state')!='accepted':return result
    receipt=dict(identity=stable_hash(unit),state='accepted',test_access=False)
    atomic_write_json(receipt,out/'accepted.json');verify_unit(out,unit)
    atomic_write_json(dict(state='completed',updated_at=time.time(),test_access=False),out/'status.json')
    return receipt
