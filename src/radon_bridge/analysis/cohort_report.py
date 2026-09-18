"""Allowlisted cumulative single-seed core comparison, separate from large cohort."""
import json
from pathlib import Path
import numpy as np
from radon_bridge.studies.cohort_case import sha,write_json
from radon_bridge.evaluation.metrics import classification_metrics


def report(root,_allow_centered_audit_build=False):
    root=Path(root);q=json.loads((root/'queue.json').read_text());rows=[];predictions={};ids=labels=None
    labels_name={'none':'无通信继续训练','svd':'SVD-Radon','linear':'匹配普通通信','self':'自身处理','mmtm':'MMTM适配','attention':'交叉注意力适配','qr':'随机QR-Radon','qr_linear':'随机QR-普通通信','learned':'可学习通道-Radon','learned_linear':'可学习通道-普通通信'}
    augmentation=q.get('study_kind')=='existing_method_augmentation'
    labels_name.update(host_continue='MMTM继续训练',host_radon='MMTM＋Radon桥',host_linear='MMTM＋普通通信')
    channel=q.get('study_kind')=='channel_compression';grouped=q.get('study_kind')=='grouped_linear';centered=q.get('study_kind')=='centered_basis'
    def display_name(key):
        if key in labels_name:return labels_name[key]
        if grouped and key.startswith('grouped_g'):
            parts=key.split('_');group=parts[1][1:];mode='_'.join(parts[2:])
            return f'G={group} '+('Radon' if mode=='radon' else '普通通信')
        if centered:
            names={'uncentered_radon':'未中心化SVD-Radon','uncentered_linear_resample':'未中心化SVD-普通通信',
                   'centered_radon':'中心化SVD-Radon','centered_linear_resample':'中心化SVD-普通通信'}
            if key in names:return names[key]
        raise KeyError(key)
    def source_label(row):
        if grouped:
            return 'G=1严格复用已接受核心结果' if row['reused'] else '新增单种子3416匹配执行'
        if centered:
            return '严格复用已接受未中心化核心结果' if row['reused'] else '新增中心化SVD单种子3416匹配执行'
        return row['provenance']
    comparisons=q.get('comparisons',[[ 'svd',k] for k in ('none','linear','self','mmtm','attention')])
    entries=list(q.get('references',[]))+list(q['cases'])
    for c in entries:
        p=Path(c['trial']) if 'trial' in c else root/'trials'/c['name'];receipt=p/'accepted.json'
        if not receipt.exists():continue
        a=json.loads(receipt.read_text());cfg=json.loads(Path(c['config']).read_text())
        if c.get('receipt_sha256') and sha(receipt)!=c['receipt_sha256']:raise ValueError('Reference receipt changed')
        if c.get('config_sha256') and sha(c['config'])!=c['config_sha256']:raise ValueError('Reference configuration changed')
        if a['configuration']!=cfg or a['test_used'] is not False or not a['converged_by_policy']:raise ValueError('Unaccepted case')
        for name,digest in a['files'].items():
            if sha(p/name)!=digest:raise ValueError('Evidence changed')
        with np.load(p/'selected_predictions.npz',allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy();labels=z['y'].copy()
            if not np.array_equal(ids,z['ids']) or not np.array_equal(labels,z['y']):raise ValueError('Participants not matched')
            if len(ids)!=296 or len(set(ids.tolist()))!=296:raise ValueError('Cohort count/uniqueness')
            metrics={k:classification_metrics(labels,z[k]) for k in ('cfp','oct')}
            if centered:
                for k in metrics:metrics[k]['calibration_in_the_large']=float(z[k][:,1].mean()-labels.mean())
            for k in metrics:
                if abs(metrics[k]['macro_f1']-a['selected_validation']['tasks'][k]['macro_f1'])>1e-12:raise ValueError('F1 replay differs')
            predictions[c['id']]={k:z[k].copy() for k in metrics}
        rows.append(dict(id=c['id'],name=display_name(c['id']),seed=cfg['seed'],metrics=metrics,reused='trial' in c,
            mean_f1=float(np.mean([metrics[k]['macro_f1'] for k in metrics])),best_epoch=a['best_epoch'],stop_epoch=a['epochs_ran'],
            provenance=c['provenance'],prediction_sha256=a['files']['selected_predictions.npz'],
            selected_model=dict(artifact_ref='Radon_Bridge/'+c.get('source_sequence_id',q['sequence_id'])+'/'+c.get('source_id',c['id'])+'/best.pt',sha256=a['files']['best.pt'])))
    out=root/'publication';out.mkdir(exist_ok=True)
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    status={k:state.get(k) for k in ('state','accepted','planned','updated_at')}
    status['active']=[dict(device=g,name=v['name'],phase=v['phase']) for g,v in state.get('active',{}).items()]
    status['failed']={name:{k:v[k] for k in ('state','exit_code','mode') if k in v} for name,v in state.get('failed',{}).items()}
    matched_results_complete=len(rows)==len(entries)
    audit_accepted=False
    if (grouped or centered) and (root/'independent_final_audit.json').exists():
        final_audit=json.loads((root/'independent_final_audit.json').read_text())
        if centered:
            from radon_bridge.studies.cohort_centered import validate_final_audit
            try:audit_accepted=validate_final_audit(root,final_audit,require_publication=not _allow_centered_audit_build)
            except (ValueError,KeyError,FileNotFoundError):audit_accepted=False
        else:
            audit_accepted=(final_audit.get('passed') is True and final_audit.get('test_used') is False
                and final_audit.get('queue_sha256')==sha(root/'queue.json')
                and final_audit.get('status_sha256')==(sha(root/'status.json') if (root/'status.json').exists() else None))
    if grouped:
        manager_complete=(status.get('state')=='complete' and status.get('planned')==8 and status.get('accepted')==8
            and not status['active'] and not status['failed'])
    elif centered:
        manager_complete=(status.get('state')=='complete' and status.get('planned')==2 and status.get('accepted')==2
            and not status['active'] and not status['failed'])
    else:manager_complete=True
    accepted_complete=matched_results_complete and manager_complete and audit_accepted if (grouped or centered) else matched_results_complete
    statistics_ready=matched_results_complete and (not centered or audit_accepted)
    current=dict(schema='radon_small_cohort_publication_v1',sequence_id=q['sequence_id'],
        test_used=False,seed=3416,train=1264,development=296,
        input=dict(cfp='two eyes x RGB x224x224',oct='two eyes x1x32x96x96'),
        backbone='ResNet18 2D / inflated3D',selection='development mean of branch macro-F1',
        configuration=dict(stage=3,channel_rank=32,M=32,S=64,kernel=3,real_batch=16,
            minimum_epochs=8,maximum_epochs=60,patience=6,min_delta=.001,precision='FP32; cuDNN TF32 true matching historical parents'),
        state=status,results=rows,matched_results_complete=matched_results_complete,complete=accepted_complete,
        reused_references=len(q.get('references',[])),
        limitations=['single_seed','same_dev_selection','different_cohort_and_input_from_Ibex','MMTM_and_attention_are_explicit_identity_initialized_adaptations'])
    if channel:
        current['limitations'] = ['single_seed','same_dev_selection','different_cohort_and_input_from_Ibex','learned_codec_has_additional_trainable_parameters']
    # Paired bootstrap for complete matched groups only.
    if statistics_ready:
        counts=np.random.default_rng(20260918).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=10000)
        def f1(prob):
            pred=prob.argmax(1);v=[]
            for label in (0,1):
                tp=counts@((labels==label)&(pred==label));fp=counts@((labels!=label)&(pred==label));fn=counts@((labels==label)&(pred!=label))
                den=2*tp+fp+fn;v.append(np.divide(2*tp,den,out=np.zeros(len(counts)),where=den>0))
            return (v[0]+v[1])/2
        dist={key:(f1(p['cfp'])+f1(p['oct']))/2 for key,p in predictions.items()}
        mean={r['id']:r['mean_f1'] for r in rows};contrasts=[];arrays=[]
        for method,key in comparisons:
            d=dist[method]-dist[key];arrays.append(d)
            c=dict(reference=key,difference=mean[method]-mean[key],ordinary95=np.quantile(d,[.025,.975]).tolist())
            if channel or augmentation or grouped or centered:c['method']=method
            contrasts.append(c)
        a=np.stack(arrays);sd=a.std(1,ddof=1);valid=sd>0
        critical=float(np.quantile(np.max(np.abs((a[valid]-a[valid].mean(1,keepdims=True))/sd[valid,None]),axis=0),.95)) if valid.any() else 0.
        for c,se in zip(contrasts,sd):c['simultaneous95']=[c['difference']-critical*se,c['difference']+critical*se]
        current['comparisons']=dict(resamples=10000,unit='participant',metric='mean branch macro-F1',contrasts=contrasts)
    if channel:
        current['study_kind']='channel_compression'
        current['limitations']=['single_seed','same_dev_selection','learned_codec_has_extra_parameters','not_centered_or_grouped_ablation']
    if augmentation:
        current['study_kind']='existing_method_augmentation'
        current['augmentation_host']=q['host_summary']
        current['limitations']=['single_seed','same_dev_selection','explicit_identity_initialized_MMTM_adapter','additional_training_stage_all_arms_matched']
    if grouped:
        current['study_kind']='grouped_linear';current['groups']=q['groups']
        current['limitations']=['single_seed','same_dev_selection','different_cohort_and_input_from_Ibex','G1_strict_reuse','grouping_limits_direct_bridge_connectivity_not_whole_network_independence','uncompressed_budget_match_separate']
    if centered:
        current['study_kind']='centered_basis'
        current['basis_definition']=dict(uncentered='FF^T/N',centered='FF^T/N - mu mu^T',runtime_centering=False)
        current['historical_reference']=q['historical_reference']
        current['limitations']=['single_seed','same_dev_selection','different_cohort_and_input_from_Ibex',
            'centered_basis_changes_fit_directions_not_runtime_mean_centering','strict_uncentered_core_reuse']
    old=json.loads((out/'current.json').read_text()) if (out/'current.json').exists() else None
    if old!=current:write_json(out/'current.json',current)
    lines=['# Radon_Bridge 小队列核心比较','',
        '本页只含六臂核心。另见[项目总览与覆盖](../README.md)、[已完成24项线性分解](../factorized/README.md)、[覆盖与后续机制](../coverage.md)。','',
        'ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。',
        'CFP为224×224二维；OCT是旧数据32×96×96三维体积。不是Ibex的32×224×224大队列。',
        '两条独立ResNet18专家；同一对父权重，Stage3通信；真实batch16，至少8轮、最多60轮、patience6；停止规则未为周报缩短。',
        'MMTM和交叉注意力为本项目身份初始化的适配实现，不声称复现原论文完整系统。','',
        f'完整核心：{"已齐全" if current["complete"] else "尚未齐全，以下仅逐臂进度，不排名"}；运行状态：{status.get("state")}。','',
        '|方法|CFP分支F1|OCT分支F1|分支均值F1|最佳/停止轮|来源|','|---|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f'|{row["name"]}|{100*row["metrics"]["cfp"]["macro_f1"]:.2f}%|{100*row["metrics"]["oct"]["macro_f1"]:.2f}%|{100*row["mean_f1"]:.2f}%|{row["best_epoch"]}/{row["stop_epoch"]}|{source_label(row)}|')
    interval_text=('本页已基于296名开发集参与者生成10,000次配对bootstrap普通与同时区间' if statistics_ready
                   else '完整后才生成10,000次配对bootstrap普通与同时区间')
    lines+=['',f'分支均值不是概率融合后的单模型分数，不与LOOK的融合输出F1混排。{interval_text}；单种子且dev参与选择，不能推出稳定泛化优势。',
        '完整配置/状态和聚合指标：[current.json](current.json)。参与者预测及权重不上传GitHub。']
    if statistics_ready:
        lines+=['','## 完整匹配后的差异（百分点）','','均为SVD-Radon减对应对照，越大表示本配置下F1更高。','',
                '|对照|差值|普通95%区间|五项同时95%区间|','|---|---:|---|---|']
        for c in current['comparisons']['contrasts']:
            lo,hi=c['ordinary95'];sl,sh=c['simultaneous95']
            contrast_label = (display_name(c['method'])+' − ') if 'method' in c else ''
            lines.append(f'|{contrast_label}{display_name(c["reference"])}|{100*c["difference"]:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|[{100*sl:+.2f}, {100*sh:+.2f}]|')
        initial=[r['name'] for r in rows if r['best_epoch']==0]
        if initial:lines+=['','选回初始父模型的设置：'+ '、'.join(initial)+'。它们已按停止规则训练，最终选模回到第0轮；分数相同不能解释成方法等效。']
        lines+=['','区间是固定已选模型下的参与者重采样，未计入训练种子波动及开发集选择偏差；MMTM/注意力只代表此适配配方，不能据此否定原方法。']
    if channel:
        lines=[line.replace('小队列核心比较','小队列通道压缩比较').replace('本页只含六臂核心。','本页比较SVD、随机QR和可学习通道映射，各自匹配Radon与普通通信。').replace('ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。','ws02 GPU1；单种子3416。SVD两项精确复用，新增QR和可学习通道各两项，逐臂预检后训练。').replace('完整核心：','完整压缩匹配组：').replace('均为SVD-Radon减对应对照，越大表示本配置下F1更高。','差值按表中左方法减右方法；完整组才给配对区间。').replace('五项同时95%区间',str(len(comparisons))+'项同时95%区间').replace('；MMTM/注意力只代表此适配配方，不能据此否定原方法。','。') for line in lines if 'MMTM和交叉注意力为' not in line]
        lines+=['','SVD按训练特征能量选固定方向；随机QR独立于数据且不按能量排序；可学习通道映射从同一随机QR初始化，但训练时更新编码和解码参数，参数量不同。中心化SVD仍属后续；分组卷积与MMTM宿主加桥已在各自有限包完成，不由本页冒充覆盖。']
    if augmentation:
        lines=[line.replace('小队列核心比较','小队列已有方法加桥比较').replace('本页只含六臂核心。','本页固定同一MMTM适配宿主，比较再次训练、加入Radon桥和加入普通通信。').replace('ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。','ws02 GPU1；单种子3416；三项均从同一已验收MMTM权重重新建立优化器，按同一原停止规则继续训练；不复用第一阶段分数冒充第二阶段对照。').replace('完整核心：','完整加桥匹配组：').replace('均为SVD-Radon减对应对照，越大表示本配置下F1更高。','按左方法减右方法；同时区间覆盖三项预定比较。').replace('五项同时95%区间','三项同时95%区间') for line in lines]
        lines+=['','本包宿主为项目MMTM身份初始化适配，第一阶段选择第0轮；不是作者完整系统。三个新臂都保留同一宿主通信，新增项以并行残差写回，初始预测须严格重放。SVD沿用相同父模型Stage3基（本宿主选中状态与原父状态相同）；不是任意变化宿主都可复用。']
    if grouped:
        lines=[line.replace('小队列核心比较','小队列分组线性通信比较').replace('本页只含六臂核心。','本页固定SVD r32/M32/S64/k3，比较G=1/2/4/8/16；每个G均配同G普通线性重采样。').replace('ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。','ws02 GPU1；单种子3416。G=1两臂仅在配置、父模型、SVD基和接受文件SHA一致时严格复用；G=2/4/8/16为8个新执行臂。').replace('完整核心：','完整分组匹配包：').replace('均为SVD-Radon减对应对照，越大表示本配置下F1更高。','每项为同一G下Radon减普通线性重采样；完整组才给配对区间。').replace('；MMTM/注意力只代表此适配配方，不能据此否定原方法。','。') for line in lines if 'MMTM和交叉注意力为' not in line]
        lines=[line.replace('完整分组匹配包：已齐全','完整分组匹配包：已齐全（manager、profiles 与包内独立 audit 均接受）') if current['complete'] else line for line in lines]
        lines+=['','G表示桥内分组数：G=1是稠密混合；G越大，每组越小，桥内直接连接比例与mixer参数量都约按1/G下降。因此跨G点估计同时改变通信拓扑和参数量；只有同一G下Radon与普通线性重采样是参数匹配的几何对照。',
                '分组只限制桥内的直接连接：每组同时含CFP/OCT来源、保留通道内全部M方向，并在返回前逆排列。跨组直接梯度为零不代表完整网络彼此独立。无压缩分组的等参数/近似等计算比较仍是单独待验收问题。']
    if centered:
        lines=[line.replace('小队列核心比较','小队列中心化SVD基比较')
               .replace('本页只含六臂核心。','本页固定r32/M32/S64/k3/G1，只把通道基从未中心化二阶矩方向换成训练集协方差方向；Radon与普通通信各一新臂，旧未中心化两臂严格复用。')
               .replace('ws02 GPU1；单种子3416。核心六种设置按顺序完成，精确复用已验收的无通信及SVD-Radon，补普通通信、自身处理、MMTM与交叉注意力。','ws02 GPU1；单种子3416。中心化基用同父模型、同1264训练样本、同Stage3特征采样拟合；运行时仍直接Q^T X与Q delta，不做均值减加。')
               .replace('完整核心：','完整中心化匹配包：')
               .replace('均为SVD-Radon减对应对照，越大表示本配置下F1更高。','三项预登记探索性比较按左方法减右方法；完整组才给本三项共同的普通/同时区间。')
               .replace('五项同时95%区间','三项同时95%区间')
               .replace('；MMTM/注意力只代表此适配配方，不能据此否定原方法。','。')
               for line in lines if 'MMTM和交叉注意力为' not in line]
        lines=[line.replace('完整中心化匹配包：已齐全','完整中心化匹配包：两新臂、profiles与包内独立audit已齐全（不等于外部最终科学接受）') if current['complete'] else line for line in lines]
        lines+=['','## 概率质量与校准方向','','校准方向采用 calibration-in-the-large：正类平均预测概率减开发集正类比例；正值表示整体偏高估，负值表示整体偏低估。它只作方向性描述，不新增显著性检验。','',
                '|方法|CFP log-loss|CFP AUROC|CFP校准方向|OCT log-loss|OCT AUROC|OCT校准方向|','|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            a=row['metrics']['cfp'];b=row['metrics']['oct']
            lines.append(f'|{row["name"]}|{a["log_loss"]:.4f}|{a.get("auroc",float("nan")):.4f}|{a["calibration_in_the_large"]:+.4f}|{b["log_loss"]:.4f}|{b.get("auroc",float("nan")):.4f}|{b["calibration_in_the_large"]:+.4f}|')
        h=q['historical_reference'];lo,hi=h['ordinary95'];sl,sh=h['simultaneous95']
        lines+=['','## 原未中心化几何比较（引用原比较族）','',
                f'原核心包的未中心化SVD-Radon − 未中心化SVD-普通通信为 {100*h["difference"]:+.2f} pp；普通95%区间 [{100*lo:+.2f}, {100*hi:+.2f}]，原五项同时95%区间 [{100*sl:+.2f}, {100*sh:+.2f}]。该区间原样引用，不与本包三项同时区间混成同一比较族。','',
                '中心化改变的是训练特征用于拟合Q的方向定义（协方差而不是未中心化二阶矩）；运行时没有输入去均值，因此本包不能单独证明“均值造成SVD优势”。']
    lines+=['','## 阅读图表前：缩写和参数','','CFP（Color Fundus Photography）为彩色眼底照片；OCT（Optical Coherence Tomography）为光学相干断层扫描。Stage3是第3个残差阶段后的通信位置；r=32是每分支保留通道方向数，M=32是投影方向数，S=64是每方向采样格点数，k=3是一维卷积核宽。','SVD用训练特征确定固定通道方向；随机QR不按信息重要性排序；可学习通道映射额外更新编码/解码参数。全局分解中间通道数大写R（另一个研究包）不是这里的小写压缩秩r。','批准范围、未完成项和下一步见[覆盖清单](../coverage.md)，不能把局部包完成当项目所有情况完成。']
    content='\n'.join(lines)+'\n';target=out/'README.md'
    if not target.exists() or target.read_text()!=content:
        temp=out/'.README.tmp';temp.write_text(content);temp.replace(target)
    return current
