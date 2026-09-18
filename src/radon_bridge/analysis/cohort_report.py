"""Allowlisted cumulative single-seed core comparison, separate from large cohort."""
import json
from pathlib import Path
import numpy as np
from radon_bridge.studies.cohort_case import sha,write_json
from radon_bridge.evaluation.metrics import classification_metrics


def report(root):
    root=Path(root);q=json.loads((root/'queue.json').read_text());rows=[];predictions={};ids=labels=None
    labels_name={'none':'无通信继续训练','svd':'SVD-Radon','linear':'匹配普通通信','self':'自身处理','mmtm':'MMTM适配','attention':'交叉注意力适配'}
    for c in q['cases']:
        p=root/'trials'/c['name'];receipt=p/'accepted.json'
        if not receipt.exists():continue
        a=json.loads(receipt.read_text());cfg=json.loads(Path(c['config']).read_text())
        if a['configuration']!=cfg or a['test_used'] is not False or not a['converged_by_policy']:raise ValueError('Unaccepted case')
        for name,digest in a['files'].items():
            if sha(p/name)!=digest:raise ValueError('Evidence changed')
        with np.load(p/'selected_predictions.npz',allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy();labels=z['y'].copy()
            if not np.array_equal(ids,z['ids']) or not np.array_equal(labels,z['y']):raise ValueError('Participants not matched')
            if len(ids)!=296 or len(set(ids.tolist()))!=296:raise ValueError('Cohort count/uniqueness')
            metrics={k:classification_metrics(labels,z[k]) for k in ('cfp','oct')}
            for k in metrics:
                if abs(metrics[k]['macro_f1']-a['selected_validation']['tasks'][k]['macro_f1'])>1e-12:raise ValueError('F1 replay differs')
            predictions[c['id']]={k:z[k].copy() for k in metrics}
        rows.append(dict(id=c['id'],name=labels_name[c['id']],seed=cfg['seed'],metrics=metrics,
            mean_f1=float(np.mean([metrics[k]['macro_f1'] for k in metrics])),best_epoch=a['best_epoch'],stop_epoch=a['epochs_ran'],
            provenance=c['provenance'],prediction_sha256=a['files']['selected_predictions.npz'],
            selected_model=dict(artifact_ref='Radon_Bridge/'+q['sequence_id']+'/'+c['id']+'/best.pt',sha256=a['files']['best.pt'])))
    out=root/'publication';out.mkdir(exist_ok=True)
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    status={k:state.get(k) for k in ('state','accepted','planned','updated_at')}
    status['active']=[dict(device=g,name=v['name'],phase=v['phase']) for g,v in state.get('active',{}).items()]
    status['failed']={name:{k:v[k] for k in ('state','exit_code','mode') if k in v} for name,v in state.get('failed',{}).items()}
    current=dict(schema='radon_small_cohort_publication_v1',sequence_id=q['sequence_id'],
        test_used=False,seed=3416,train=1264,development=296,
        input=dict(cfp='two eyes x RGB x224x224',oct='two eyes x1x32x96x96'),
        backbone='ResNet18 2D / inflated3D',selection='development mean of branch macro-F1',
        configuration=dict(stage=3,channel_rank=32,M=32,S=64,kernel=3,real_batch=16,
            minimum_epochs=8,maximum_epochs=60,patience=6,min_delta=.001,precision='FP32; cuDNN TF32 true matching historical parents'),
        state=status,results=rows,complete=len(rows)==len(q['cases']),
        limitations=['single_seed','same_dev_selection','different_cohort_and_input_from_Ibex','MMTM_and_attention_are_explicit_identity_initialized_adaptations'])
    # Paired bootstrap for complete core only, never rank an incomplete group.
    if current['complete']:
        counts=np.random.default_rng(20260918).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=10000)
        def f1(prob):
            pred=prob.argmax(1);v=[]
            for label in (0,1):
                tp=counts@((labels==label)&(pred==label));fp=counts@((labels!=label)&(pred==label));fn=counts@((labels==label)&(pred!=label))
                den=2*tp+fp+fn;v.append(np.divide(2*tp,den,out=np.zeros(len(counts)),where=den>0))
            return (v[0]+v[1])/2
        dist={key:(f1(p['cfp'])+f1(p['oct']))/2 for key,p in predictions.items()}
        mean={r['id']:r['mean_f1'] for r in rows};contrasts=[];arrays=[]
        for key in ('none','linear','self','mmtm','attention'):
            d=dist['svd']-dist[key];arrays.append(d)
            contrasts.append(dict(reference=key,difference=mean['svd']-mean[key],ordinary95=np.quantile(d,[.025,.975]).tolist()))
        a=np.stack(arrays);sd=a.std(1,ddof=1);valid=sd>0
        critical=float(np.quantile(np.max(np.abs((a[valid]-a[valid].mean(1,keepdims=True))/sd[valid,None]),axis=0),.95)) if valid.any() else 0.
        for c,se in zip(contrasts,sd):c['simultaneous95']=[c['difference']-critical*se,c['difference']+critical*se]
        current['comparisons']=dict(resamples=10000,unit='participant',metric='mean branch macro-F1',contrasts=contrasts)
    old=json.loads((out/'current.json').read_text()) if (out/'current.json').exists() else None
    if old!=current:write_json(out/'current.json',current)
    lines=['# Radon_Bridge 小队列核心比较','',
        'ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。',
        'CFP为224×224二维；OCT是旧数据32×96×96三维体积。不是Ibex的32×224×224大队列。',
        '两条独立ResNet18专家；同一对父权重，Stage3通信；真实batch16，至少8轮、最多60轮、patience6；停止规则未为周报缩短。',
        'MMTM和交叉注意力为本项目身份初始化的适配实现，不声称复现原论文完整系统。','',
        f'完整核心：{"已齐全" if current["complete"] else "尚未齐全，以下仅逐臂进度，不排名"}；运行状态：{status.get("state")}。','',
        '|方法|眼底分支F1|OCT分支F1|分支均值F1|最佳/停止轮|来源|','|---|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f'|{row["name"]}|{100*row["metrics"]["cfp"]["macro_f1"]:.2f}%|{100*row["metrics"]["oct"]["macro_f1"]:.2f}%|{100*row["mean_f1"]:.2f}%|{row["best_epoch"]}/{row["stop_epoch"]}|{row["provenance"]}|')
    lines+=['','分支均值不是概率融合后的单模型分数，不与LOOK的融合输出F1混排。完整后自动生成10,000次配对bootstrap普通与同时区间；单种子且dev参与选择，不能推出稳定泛化优势。',
        '完整配置/状态和聚合指标：[current.json](current.json)。参与者预测及权重不上传GitHub。']
    if current['complete']:
        lines+=['','## 完整匹配后的差异（百分点）','','均为SVD-Radon减对应对照，越大表示本配置下F1更高。','',
                '|对照|差值|普通95%区间|五项同时95%区间|','|---|---:|---|---|']
        for c in current['comparisons']['contrasts']:
            lo,hi=c['ordinary95'];sl,sh=c['simultaneous95']
            lines.append(f'|{labels_name[c["reference"]]}|{100*c["difference"]:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|[{100*sl:+.2f}, {100*sh:+.2f}]|')
        initial=[r['name'] for r in rows if r['best_epoch']==0]
        if initial:lines+=['','选回初始父模型的设置：'+ '、'.join(initial)+'。它们已按停止规则训练，最终选模回到第0轮；分数相同不能解释成方法等效。']
        lines+=['','区间是固定已选模型下的参与者重采样，未计入训练种子波动及开发集选择偏差；MMTM/注意力只代表此适配配方，不能据此否定原方法。']
    content='\n'.join(lines)+'\n';target=out/'README.md'
    if not target.exists() or target.read_text()!=content:
        temp=out/'.README.tmp';temp.write_text(content);temp.replace(target)
    return current
