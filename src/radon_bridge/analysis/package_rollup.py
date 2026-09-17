"""Atomic cumulative core deliveries, separate from full-case completion."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from radon_bridge.runtime.state import stable_hash,file_sha256,atomic_write_json


def collect(tasks,verify):
    coverage=[];rows=[];seen=set()
    for task in tasks:
        root=Path(task['run_dir']).resolve()
        if str(root) in seen:raise ValueError('Duplicate case in package rollup')
        seen.add(str(root))
        if file_sha256(task['spec'])!=task['spec_sha256']:raise ValueError('Case specification changed')
        spec=json.loads(Path(task['spec']).read_text());package=root/'packages/core'
        row=dict(run=str(root),disease=spec['disease'],architecture=spec['model']['name'],seed=spec['seed'],
                 spec_sha256=task['spec_sha256'],state='awaiting_core_delivery')
        if (package/'accepted.json').exists():
            verify(spec,root)
            row.update(state='accepted_core',receipt_sha256=file_sha256(package/'accepted.json'))
            with (package/'report/metrics.csv').open() as stream:
                rows.extend(dict(r,run=str(root),architecture=spec['model']['name']) for r in csv.DictReader(stream))
        coverage.append(row)
    return dict(schema='radon_cumulative_core_v1',coverage=coverage,rows=rows,test_access=False,
                interpretation='core_packages_only; selected_development_evidence; not_full_mechanism_or_test')


def render(snapshot,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=snapshot['rows']
    if not rows:return
    # Pagination is stable by case order, never by outcome rank.
    accepted=[c for c in snapshot['coverage'] if c['state']=='accepted_core']
    for index,case in enumerate(accepted):
        selected=[r for r in rows if r['run']==case['run']]
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for key,label in [('cfp_f1','CFP'),('oct_f1','OCT'),('mean_f1','Branch mean')]:
            ax.plot(range(len(selected)),[100*float(r[key]) for r in selected],marker='o',label=label)
        ax.set_xticks(range(len(selected)),[r['arm'] for r in selected],rotation=25,ha='right')
        ax.set_ylim(0,100);ax.set_ylabel('Dev macro-F1 (%) — higher is better')
        ax.set_title(f"{case['disease']} / {case['architecture']} / {case['seed']}");ax.legend()
        fig.savefig(out/f'comparison_{index+1}.svg');plt.close(fig)


def publish(tasks,output,verify=None,renderer=render):
    if verify is None:
        from radon_bridge.studies.project_units import verify_core
        verify=verify_core
    snapshot=collect(tasks,verify);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    digest=stable_hash(snapshot);target=root/'releases'/digest
    with (root/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            from radon_bridge.studies.project_units import checked_files
            manifest=json.loads((target/'manifest.json').read_text())
            if manifest.get('snapshot')!=digest:raise ValueError('Publication identity changed')
            if not {'snapshot.json','results.csv','README.zh-CN.md'}<=manifest['files'].keys():raise ValueError('Incomplete publication')
            checked_files(target,manifest['files'])
        else:
            temp=Path(tempfile.mkdtemp(prefix='.publish-',dir=root))
            try:
                atomic_write_json(snapshot,temp/'snapshot.json')
                fields=list(dict.fromkeys(k for r in snapshot['rows'] for k in r)) or ['run','arm','seed','disease','architecture','cfp_f1','oct_f1','mean_f1']
                with (temp/'results.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(snapshot['rows'])
                renderer(snapshot,temp)
                lines=['# Radon_Bridge 累计核心包结果','','只纳入验收完整的六臂包；未完成包显示状态，不填零，不混作全机制完成。',
                       '图横轴为六种方法，纵轴为dev macro-F1百分比，越高越好；线为CFP、OCT及指标平均，不是概率融合。',
                       '不同疾病/架构/种子分图，按登记顺序展示；选中dev结果不构成独立确认性证据。区间和诊断见各原run/packages/core。',
                       '','','|疾病|架构|种子|核心包状态|','|---|---|---:|---|']
                lines += [f"|{c['disease']}|{c['architecture']}|{c['seed']}|{c['state']}|" for c in snapshot['coverage']]
                lines += [f'\n![累计匹配图]({p.name})' for p in sorted(temp.glob('comparison_*.svg'))]
                (temp/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
                manifest=dict(snapshot=digest,files={p.name:file_sha256(p) for p in temp.iterdir() if p.is_file()})
                atomic_write_json(manifest,temp/'manifest.json');target.parent.mkdir(exist_ok=True);os.replace(temp,target)
            finally:
                if temp.exists():shutil.rmtree(temp)
        current=root/'current';unchanged=current.is_symlink() and current.resolve()==target.resolve()
        if not unchanged:
            link=root/f'.current-{os.getpid()}'
            link.unlink(missing_ok=True);link.symlink_to(target.relative_to(root));os.replace(link,current)
    return dict(accepted=sum(c['state']=='accepted_core' for c in snapshot['coverage']),
                state='unchanged' if unchanged else 'published',snapshot=digest)
