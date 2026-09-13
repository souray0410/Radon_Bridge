"""Accepted-only progress, complete-seed comparisons and evidence coverage."""
import csv
import json
from pathlib import Path
import numpy as np
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.studies.research_matrix import comparisons
from radon_bridge.analysis.project_report import bootstrap


def summarize(tasks,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    progress=[];rows=[];groups={}
    for task in tasks:
        run=Path(task['run_dir']);spec=json.loads(Path(task['spec']).read_text())
        if file_sha256(Path(task['spec']))!=task['spec_sha256']:raise ValueError('Case spec changed')
        item=dict(task=task['id'],architecture=spec['model']['name'],disease=spec['disease'],seed=spec['seed'],run_id=run.name)
        if not (run/'accepted.json').exists():
            state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'pending'}
            progress.append(dict(item,state=state['state'],stage=state.get('stage')));continue
        from radon_bridge.studies.project_case import verify_case
        verify_case(run,spec)
        progress.append(dict(item,state='accepted',receipt_sha256=file_sha256(run/'accepted.json')))
        with (run/'report/metrics.csv').open() as f:
            rows.extend(dict(item,**{k:v for k,v in r.items() if k not in item}) for r in csv.DictReader(f))
        groups.setdefault((item['disease'],item['architecture']),[]).append((spec,run))
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['task','architecture','disease','seed','arm','cfp_f1','oct_f1','mean_f1']
    with (root/'development_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    findings=[]
    for (disease,architecture),cases in sorted(groups.items()):
        if sorted(s['seed'] for s,r in cases)!=[3416,3417,3418]:continue
        signature=stable_hash([file_sha256(r/'accepted.json') for s,r in sorted(cases,key=lambda x:x[0]['seed'])])
        path=root/f'{disease}_{architecture}_three_seed.json'
        if path.exists() and json.loads(path.read_text()).get('signature')==signature:
            findings.append(json.loads(path.read_text()));continue
        keys=[];pred=[];reference=None;raw=[]
        definitions=comparisons(disease,architecture)
        for spec,run in sorted(cases,key=lambda x:x[0]['seed']):
            for arm in spec['arms']:
                with np.load(run/'arms'/arm['id']/'development_predictions.npz',allow_pickle=False) as z:
                    identity=(z['participant_ids'].copy(),z['labels'].copy())
                    if reference is None:reference=identity
                    elif any(not np.array_equal(a,b) for a,b in zip(reference,identity)):raise ValueError('Cross-seed participants are not matched')
                    for branch in ('cfp','oct'):
                        keys.append((spec['seed'],arm['id'],branch));pred.append(z[branch].argmax(1))
        weights=[]
        for d in definitions:
            w=np.zeros(len(keys));branches=('cfp','oct') if d['branch']=='mean' else (d['branch'],)
            for seed in (3416,3417,3418):
                for b in branches:
                    w[keys.index((seed,d['left'],b))]+=1/(3*len(branches));w[keys.index((seed,d['right'],b))]-=1/(3*len(branches))
            weights.append(w)
        stats=bootstrap(reference[1],np.stack(pred),weights,10000)
        stats.update(signature=signature,disease=disease,architecture=architecture,definitions=definitions,
            family='prespecified_comparisons_within_disease_architecture_averaged_three_seeds',
            missing_whole_study_global_family=True,screening_seed=3416,replication_seeds=[3417,3418])
        atomic_write_json(stats,path);findings.append(stats)
    accepted=sum(p['state']=='accepted' for p in progress)
    result=dict(schema='radon_project_progress_v1',registered=len(tasks),accepted=accepted,
        complete=bool(tasks) and accepted==len(tasks),tasks=progress,rows=len(rows),complete_seed_groups=len(findings),
        test_access=False,scope='development; current registered cases, not undiscovered architectures')
    atomic_write_json(result,root/'progress.json')
    lines=['# Radon_Bridge 多模型开发研究','',f'已登记完整研究单元：{len(tasks)}；验收：{accepted}；三种子匹配组：{len(findings)}。',
        '每个研究单元包括多项训练及诊断；不能把一个研究单元称为一次训练。',
        '指标为CFP/OCT各自macro-F1及二者平均，越高越好。固定概率融合单列，不替代主指标。',
        '已完成但尚未凑齐种子的结果保留，不据部分结果排名。完整三种子组使用共同参与者重采样；不同疾病不混作一个诊断任务。','',
        '| 模型 | 疾病 | 比较 | 差值 pp | 同时区间 pp | 解释 |','|---|---|---|---:|---|---|']
    for g in findings:
        for d,v in zip(g['definitions'],g['comparisons']):
            ci=v['simultaneous_95'];ci='未定义' if ci is None else f'[{ci[0]*100:.2f}, {ci[1]*100:.2f}]'
            lines.append(f"| {g['architecture']} | {g['disease']} | {d['id']} | {v['difference']*100:.2f} | {ci} | {v['judgement']} |")
    lines+=['','差值left减right；正值有利于left。同时区间只覆盖标明的疾病/架构比较族，不能用于全研究优越性主张。',
        '完整数据来自同一UKB资源；跨架构重复仍不是跨数据集或外部中心验证。']
    (root/'README.zh-CN.md').write_text('\n'.join(lines)+'\n');return result
