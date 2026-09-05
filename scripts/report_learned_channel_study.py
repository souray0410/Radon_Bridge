"""129-result report; learned channel minus matched fixed/learned references."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from scripts.run_three_seed_study import SEEDS,RHOS,key
from scripts.run_centered_study import SVD_ARMS,FILES
from scripts.report_five_seed_study import paired_bootstrap,read,sha,BRANCHES
from scripts.report_three_seed_study import rho_label,save,NAMES as OLD_NAMES
from scripts.report_centered_study import name as previous_name

LABELS={'svd_radon':'Radon','svd_self':'Self only','svd_scrambled':'Scrambled Radon','svd_resample':'Linear resampling'}
def name(arm):return 'Learned channel / '+LABELS[arm[8:]] if arm.startswith('channel_') else previous_name(arm)
RADON=['learned','svd_radon','centered_svd_radon','qr_radon','channel_svd_radon']
COLORS=['#B56D37','#007E87','#7159A5','#A38B4F','#C33E67']
METHOD_COLORS=dict(zip(RADON,COLORS))|{'channel_svd_self':'#9674A6','channel_svd_scrambled':'#71828D','channel_svd_resample':'#5E879A'}
MARKERS=dict(zip(RADON,['o','s','^','D','P']))|{'channel_svd_self':'s','channel_svd_scrambled':'^','channel_svd_resample':'D'}
def contrasts():
    pairs=[('channel_'+a,b+a) for a in SVD_ARMS for b in ['', 'centered_']]
    pairs += [('channel_svd_radon',a) for a in ['no_bridge','learned','qr_radon']+['channel_'+a for a in SVD_ARMS[1:]]]
    return pairs

def figures(out,lookup,stats,boots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ticks=['rho=1/16\nr=16, h=512','rho=1/8\nr=32, h=1024','rho=1/4\nr=64, h=2048']
    for number,arms,title in [(1,RADON,'Radon: channel encoding comparisons'),(2,['channel_'+a for a in SVD_ARMS],'Learned channel: mechanism comparisons')]:
        fig,axes=plt.subplots(1,3,figsize=(17,7))
        for ax,branch in zip(axes,BRANCHES):
            for i,a in enumerate(arms):
                vals=[stats['all3'][rho_label(r)][a][branch] for r in RHOS]
                ax.errorbar(range(3),[v['mean'] for v in vals],yerr=[v['sample_sd'] for v in vals],label=name(a),color=METHOD_COLORS[a],marker=MARKERS[a],capsize=3,lw=1.7)
            ax.axhline(stats['all3']['1/8']['no_bridge'][branch]['mean'],color='#59636B',ls=':',label='No bridge')
            ax.set(xticks=range(3),xticklabels=ticks,xlim=(-.2,2.2),ylabel='Macro-F1 (%)',title=branch.upper());ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
        fig.suptitle('R&B (Radon Bridge) | '+title,x=.06,ha='left',fontsize=19,fontweight='bold')
        fig.text(.06,.9,'Three seeds (3416-3418), mean +/- sample SD | stage3 / M32 / S64 | backbone LR 6e-5, head/bridge 1e-4',fontsize=10)
        fig.subplots_adjust(left=.06,right=.985,top=.81,bottom=.27,wspace=.23);save(fig,out,f'0{number}_comparison');plt.close(fig)
    pairs=contrasts();fig,axes=plt.subplots(3,3,figsize=(20,19),sharex=True,sharey=True)
    bounds=[abs(x) for rs in boots['all3'].values() for bs in rs.values() for v in bs.values() for x in [v['delta_pp']]+v['ci95_pp']];lim=max(bounds+[1])*1.06
    for i,rho in enumerate(RHOS):
        for j,branch in enumerate(BRANCHES):
            ax=axes[i,j]
            for k,(a,b) in enumerate(pairs):
                v=boots['all3'][rho_label(rho)][a+'__minus__'+b][branch];ax.hlines(k,*v['ci95_pp'],color='#C33E67');ax.scatter(v['delta_pp'],k,color='#C33E67',s=22)
            ax.axvline(0,color='#59636B');ax.set(yticks=range(len(pairs)),yticklabels=[name(a)+' - '+name(b) for a,b in pairs],ylim=(len(pairs)-.5,-.5),xlim=(-lim,lim),xlabel='Paired difference (pp)',title=f'rho={rho_label(rho)} | {branch.upper()}');ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2)
    fig.suptitle('R&B | prespecified comparisons: learned channel minus reference',x=.04,ha='left',fontsize=19,fontweight='bold')
    fig.text(.04,.94,'10,000 participant resamples; identical indices across seeds; 95% percentile intervals; no multiplicity correction.\nIntervals condition on existing trained models and development-selected checkpoints.',fontsize=10)
    fig.subplots_adjust(left=.34,right=.985,top=.88,bottom=.06,wspace=.12,hspace=.25);save(fig,out,'03_paired_intervals');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,14))
    for i,seed in enumerate(SEEDS):
        for j,branch in enumerate(BRANCHES):
            ax=axes[i,j]
            for a,color in zip(RADON,COLORS):ax.plot(range(3),[lookup[key(seed,a,r)]['scores'][branch] for r in RHOS],marker='o',color=color,label=name(a))
            ax.axhline(lookup[key(seed,'no_bridge',None)]['scores'][branch],color='#59636B',ls=':')
            ax.set(xticks=range(3),xticklabels=['1/16','1/8','1/4'],xlabel='rho',ylabel='Macro-F1 (%)',title=f'seed{seed} | {branch.upper()}');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('R&B | all individual seeds; seed3418 shown separately',x=.06,ha='left',fontsize=19,fontweight='bold');fig.subplots_adjust(left=.06,right=.98,top=.92,bottom=.12,hspace=.35,wspace=.22);save(fig,out,'04_seeds');plt.close(fig)
    labels=[(a,r) for r in RHOS for a in SVD_ARMS];fig,axes=plt.subplots(1,4,figsize=(19,10))
    for column,(metric,title) in enumerate([('retained_energy_ratio','Encoder row-space energy'),('retained_variance_ratio','Encoder row-space variance'),('encoded_energy_ratio','Actual encoded / input energy'),('delta_over_input_l2','Residual / input L2')]):
        ax=axes[column];values=np.asarray([[np.mean([lookup[key(s,'channel_'+a,r)]['diagnostic']['phases']['selected']['energy'][b+'_stage3'][metric] for s in SEEDS]) for b in ['cfp','oct']] for a,r in labels])
        im=ax.imshow(values,aspect='auto',cmap='viridis');ax.set(xticks=[0,1],xticklabels=['CFP','OCT'],yticks=range(12),yticklabels=[LABELS[a]+' | '+rho_label(r) for a,r in labels] if column==0 else [],title=title)
        for i in range(12):
            for j in range(2):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    fig.suptitle('R&B | learned channel diagnostics after joint training',x=.05,ha='left',fontsize=19,fontweight='bold')
    fig.text(.05,.055,'Row-space retention uses an orthogonal diagnostic projector; actual learned encoder energy can exceed input energy.\nAll1264 training participants; three-seed mean. Initial/selected encoder spectra, decoder mismatch and per-module gradients are in results.json.',fontsize=10)
    fig.subplots_adjust(left=.18,right=.98,top=.88,bottom=.14,wspace=.35);save(fig,out,'05_diagnostics');plt.close(fig)

def build(root,resamples=10000,make_figures=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);m=read(root/'manifest.json');p=read(root/'protocol.json');ledger=read(root/'ledger.json');cost={r['id']:r for r in ledger['jobs']}
    previous=read(Path(p['predecessor'])/'report/results.json');old={r['id']:r for r in previous['rows']};assert len(old)==93
    assert m['seeds']==SEEDS and m['rhos']==RHOS and len(m['rows'])==129
    rows=[]
    for r in m['rows']:
        path=Path(r['directory']);d=read(path/'summary.json');assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and not d['test_used']
        hashes={n:sha(path/n) for n in FILES}
        if r['reused']:
            assert hashes==r['accepted_hashes']==old[r['id']]['result_hashes'];row=dict(old[r['id']])
        else:
            assert d['configuration']['bridges'][0]['compression']=='learned_channel'
            scores={b:100*d['selected']['tasks'][b]['macro_f1'] for b in ['cfp','oct']};scores['mean']=sum(scores.values())/2
            assert abs(scores['mean']-100*d['selected']['mean_task_macro_f1'])<1e-9
            info=read(path/'model.json');diagpath=root/('diagnostic_'+r['id'])/'summary.json';diag=read(diagpath)
            assert diag['passed'] and not diag['preflight'] and diag['selected_sha256']==hashes['selected.pt']
            for phase in diag['phases'].values():
                assert phase['probe_participants']==128 and phase['energy_participants']==1264 and phase['state_parameters_bn_gradients_rng_preserved']
                for energy in phase['energy'].values():assert 'encoded_energy_ratio' in energy and energy['projection_basis']['numerical_rank']>=0
            row=dict(r,summary=d,scores=scores,model=info,result_hashes=hashes,cost=cost[r['id']],diagnostic=diag,diagnostic_sha256=sha(diagpath),diagnostic_cost=cost['diagnostic_'+r['id']],
                stored_bridge_parameters=sum(g['stored_bridge_parameters'] for g in info['groups']),effective_bridge_parameters=sum(g['effective_bridge_parameters'] for g in info['groups']))
        rows.append(row)
    lookup={key(r['seed'],r['arm'],r['rho']):r for r in rows};assert len(lookup)==129
    arms=['no_bridge']+list(dict.fromkeys(r['arm'] for r in rows if r['arm']!='no_bridge'));stats={};boots={}
    subsets={str(s):[s] for s in SEEDS}|{'all3':SEEDS}
    for label,seeds in subsets.items():
        stats[label]={};boots[label]={}
        for rho in RHOS:
            rs=rho_label(rho);stats[label][rs]={};boots[label][rs]={}
            for arm in arms:
                values=[lookup[key(s,arm,None if arm=='no_bridge' else rho)] for s in seeds]
                stats[label][rs][arm]={b:{'mean':float(np.mean([r['scores'][b] for r in values])),'sample_sd':float(np.std([r['scores'][b] for r in values],ddof=1)) if len(seeds)>1 else None} for b in BRANCHES}
            for main,control in contrasts():
                pairs=[(lookup[key(s,control,None if control=='no_bridge' else rho)],lookup[key(s,main,rho)]) for s in seeds]
                for a,b in pairs:
                    assert a['summary']['initial_native_sha256']==b['summary']['initial_native_sha256'] and a['summary']['parent_checkpoints']==b['summary']['parent_checkpoints']
                    assert a['diagnostic']['phases']['selected']['probe_ids_sha256']==b['diagnostic']['phases']['selected']['probe_ids_sha256']
                v=paired_bootstrap([(Path(a['directory']),Path(b['directory'])) for a,b in pairs],resamples)
                for branch in BRANCHES:assert abs(v[branch]['delta_pp']-np.mean([b['scores'][branch]-a['scores'][branch] for a,b in pairs]))<1e-9
                boots[label][rs][main+'__minus__'+control]=v
    result={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'statistics':stats,
        'bootstrap':{'resamples':resamples,'unit':'participant; shared indices across seeds','direction':'learned channel minus matched reference','comparisons':boots},
        'manifest':m,'protocol':p,'ledger':ledger,'gpu_acceptance':read(root/'gpu_acceptance.json'),'study_summary':read(root/'study_summary.json'),
        'previous_report_sha256':sha(Path(p['predecessor'])/'report/results.json'),'test_used':False}
    write_json(out/'results.json',result);write_json(out/'bootstrap.json',result['bootstrap'])
    if make_figures:figures(out,lookup,stats,boots)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','method','rho','channel_rank','h','cfp_f1_percent','oct_f1_percent','mean_f1_percent','cfp_gain_vs_baseline_pp','oct_gain_vs_baseline_pp','mean_gain_vs_baseline_pp','best_epoch','stop_epoch','total_parameters','bridge_stored','bridge_effective','gpu_minutes','peak_process_mib','directory','summary_sha256'])
        for r in rows:
            d=r['summary'];baseline=lookup[key(r['seed'],'no_bridge',None)];rho=r['rho'];c=r['cost']
            w.writerow([r['seed'],name(r['arm']),rho_label(rho),int(256*rho) if rho and r['arm']!='learned' else '',int(8192*rho) if rho else '']+[r['scores'][b] for b in BRANCHES]+[r['scores'][b]-baseline['scores'][b] for b in BRANCHES]+[d['selection']['joint']['best_epoch'],d['epochs_ran'],d['trainable_parameters'],r['stored_bridge_parameters'],r['effective_bridge_parameters'],c['gpu_seconds']/60,c['sampled_peak_process_mib'],r['directory'],r['result_hashes']['summary.json']])
    with (out/'paired_differences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subset','rho','contrast','branch','difference_pp','ci95_low','ci95_high'])
        for subset,rhos in boots.items():
            for rho,pairs in rhos.items():
                for pair,branches in pairs.items():
                    for branch,v in branches.items():w.writerow([subset,rho,pair,branch,v['delta_pp']]+v['ci95_pp'])
    lines=['# R&B：可学习原生通道压缩与恢复卷积的配对补充',
        '新增36项：三个种子3416–3418 × rho1/16、1/8、1/4 × Radon、自身、空间打乱、普通线性重采样。复用已有93项，共129项第二阶段结果；无新增独立预训练或基拟合。',
        '数据流：无偏置逐点卷积C→r → 几何 → kernel3线性通信 → 普通反投影/返回算子 → 无偏置逐点卷积r→C → 原Node ID残差写回。r16/32/64、h512/1024/2048，全部方向和S轴保留。',
        '编码与恢复权重独立训练，初始分别为与随机QR对照相同的Q^T和Q；不绑权、不约束正交、不中心化。中间卷积零初始化。只有中间卷积可学习这一旧约束在本可学习实验臂明确放宽；原固定路径不变。',
        '固定stage3/M32/S64、backbone LR6e-5、head/bridge1e-4、batch16、AdamW WD0.01、分支与桥各裁剪5、全参数训练、BN更新及原平台规则。相同父检查点、训练顺序。',
        '对照解读：与QR标准Radon比较共享初始子空间，但同时放开编码、恢复、正交与转置绑定；与SVD比较还改变初始化。可学习CM同时改变压缩对象和结构。不能把所有差异仅归因于是否训练。',
        '## 标准Radon三种子均值与样本标准差','![主比较](01_comparison.png)','| rho | 方法 | CFP | OCT | 平均 |','|---|---|---:|---:|---:|']
    for rho in RHOS:
        for arm in ['no_bridge']+RADON:
            st=stats['all3'][rho_label(rho)][arm];lines.append('| '+' | '.join([rho_label(rho),name(arm)]+[f"{st[b]['mean']:.2f} ± {st[b]['sample_sd']:.2f}" for b in BRANCHES])+' |')
    lines+=['## 机制与配对差值','![机制](02_comparison.png)','![区间](03_paired_intervals.png)','| rho | 比较：可学习通道减参照 | 平均差 pp [95% CI] | 3416 / 3417 / 3418 |','|---|---|---|---|']
    for rho in RHOS:
        for a,b in contrasts():
            label=a+'__minus__'+b;v=boots['all3'][rho_label(rho)][label]['mean'];ss=' / '.join(f"{boots[str(s)][rho_label(rho)][label]['mean']['delta_pp']:+.2f}" for s in SEEDS)
            lines.append(f"| {rho_label(rho)} | {name(a)} - {name(b)} | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {ss} |")
    lines+=['## 逐种子与诊断','![逐种子](04_seeds.png)','![诊断](05_diagnostics.png)',
        '可学习编码不保持正交。诊断以编码矩阵当前行空间的正交投影计算能量/方差保留率；实际编码能量另记encoded_energy_ratio，允许超过1。后者不是信息保留率。保存编码奇异值、数值秩、编解码SHA及恢复矩阵与编码转置的差异。',
        '初始及选中模型均对全部1264训练参与者统计能量/方差/残差；固定前128训练参与者计算两个CE对stage3、编码卷积、通信卷积、恢复卷积的累积梯度与余弦，零梯度记未定义。诊断不改变参数、BN、梯度或随机状态。',
        '## 完整交付与限制',
        'results.csv含129项分支F1、无桥增益、参数、轮次、训练时间和显存；paired_differences.csv包含全部逐种子和三种子配对区间。配置、父检查点、旧基、新初始化、预测/摘要SHA与诊断在results.json。',
        '10000次参与者配对bootstrap跨种子共享索引；区间条件于已有模型和开发集选优，不包含重新训练不确定性，比较未经多重性校正。原3416/3417参与配置选择，3418单列。仅1264训练/296开发验证，不读取测试；仍为探索性证据。',
        '不要求性能随rho单调，不因平均提升宣称两分支全部提高。自身处理有效参数较少；普通通信仅匹配宽度与算子行范数。总训练时间受停止轮次与共享GPU负载影响，不作为同硬件单步速度证明。',
        f"源码：{p['source_commit']}；运行目录：{root}。每卡项目≤10GiB，GPU时长不限但持续记账。"]
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write_json(out/'report_status.json',{'state':'complete','complete_trials':129,'total_trials':129,'new_trials':36,'new_diagnostics':36,'seeds':SEEDS,'rhos':RHOS,'visual_review':'pending','test_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
