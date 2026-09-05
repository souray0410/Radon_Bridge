"""Centering during basis fitting only: complete corresponding comparisons."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from scripts.run_centered_study import PREVIOUS,SVD_ARMS,FILES,matrix
from scripts.run_three_seed_study import SEEDS,RHOS,key
from scripts.report_three_seed_study import NAMES,rho_label
from scripts.report_five_seed_study import read,sha,paired_bootstrap,BRANCHES


def name(arm):return ('Centered-fit '+NAMES[arm[len('centered_'):]]) if arm.startswith('centered_') else NAMES[arm]
def save(fig,out,stem):
    if 'QA_ONLY' in str(out):fig.text(.5,.5,'QA FIXTURE - NOT EXPERIMENT RESULTS',ha='center',rotation=25,fontsize=28,color='red',alpha=.35)
    for ext in ['png','svg']:fig.savefig(out/(stem+'.'+ext),dpi=180)


def figures(out,lookup,stats,boots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,3,figsize=(16,15))
    for i,a in enumerate(SVD_ARMS):
        for j,b in enumerate(BRANCHES):
            ax=axes[i,j]
            for centered,color,marker in [(False,'#B56D37','o'),(True,'#007E87','s')]:
                arm='centered_'+a if centered else a;values=[stats['all3'][rho_label(r)][arm][b] for r in RHOS]
                line=ax.errorbar(range(3),[v['mean'] for v in values],yerr=[v['sample_sd'] for v in values],marker=marker,markerfacecolor='none' if centered else color,markersize=8 if centered else 5,color=color,capsize=3,lw=1.8,label='Centered-fit' if centered else 'Uncentered-fit')
                assert np.array_equal(line.lines[0].get_xdata(),[0,1,2])
            ax.set(xticks=range(3),xticklabels=['1/16','1/8','1/4'],xlabel='rho (dimension ratio)',ylabel='Macro-F1 (%)',title=f'{NAMES[a]} | {b.upper()}',xlim=(-.2,2.2));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('R&B (Radon Bridge) | centered versus uncentered basis fitting',x=.06,ha='left',fontsize=19,fontweight='bold')
    fig.text(.06,.944,'Three seeds (3416-3418), mean +/- sample SD | stage3 / M32 / S64 | backbone LR 6e-5 | head/bridge LR 1e-4',fontsize=10)
    fig.text(.06,.025,'Only the basis-fitting statistic changes. Both runtime paths use Q^T X and Q delta, with no mean subtraction/addition.\nrho 1/16, 1/8, 1/4 gives channel rank16,32,64 and per-source width512,1024,2048. No energy-threshold rank selection.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.06,right=.98,top=.9,bottom=.1,hspace=.55,wspace=.25);save(fig,out,'01_centering_comparison');plt.close(fig)
    labels=[(a,r) for r in RHOS for a in SVD_ARMS];bounds=[1.]
    for subset in ['all3','3418']:
        for _,rs in boots[subset].items():
            for _,v in rs.items():
                for _,m in v.items():bounds+=list(map(abs,[m['delta_pp']]+m['ci95_pp']))
    lim=max(bounds)*1.08;fig,axes=plt.subplots(2,3,figsize=(16,13),sharex=True,sharey=True)
    for i,subset in enumerate(['all3','3418']):
        for j,b in enumerate(BRANCHES):
            ax=axes[i,j]
            for k,(a,rho) in enumerate(labels):
                v=boots[subset][rho_label(rho)][a][b];ax.hlines(k,*v['ci95_pp'],color='#007E87',lw=1.7);ax.scatter(v['delta_pp'],k,color='#007E87',s=28)
            ax.set(yticks=range(12),yticklabels=[f'{NAMES[a]} | rho={rho_label(r)}' for a,r in labels],ylim=(11.5,-.5),xlim=(-lim,lim),title=f'{"All 3 seeds" if subset=="all3" else "New seed3418"} | {b.upper()}',xlabel='Centered-fit minus uncentered-fit (pp)');ax.axvline(0,color='#60717C');ax.grid(axis='x',alpha=.2)
    fig.suptitle('R&B | matched centering contrasts',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.94,'10,000 paired participant resamples | 95% percentile intervals | identical indices across seeds | no multiplicity adjustment',fontsize=10)
    fig.text(.055,.025,'Original seeds3416/3417 informed earlier configuration selection. Seed3418 is shown separately.\nIntervals condition on trained models and selected development checkpoints; three seeds do not multiply the number of participants.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.23,right=.98,top=.9,bottom=.1,wspace=.15,hspace=.25);save(fig,out,'02_centering_intervals');plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(17,12),sharex=True)
    for i,s in enumerate(SEEDS):
        for j,a in enumerate(SVD_ARMS):
            ax=axes[i,j]
            for centered,color,marker in [(False,'#B56D37','o'),(True,'#007E87','s')]:
                ax.plot(range(3),[lookup[key(s,'centered_'+a if centered else a,r)]['scores']['mean'] for r in RHOS],color=color,marker=marker,label='Centered-fit' if centered else 'Uncentered-fit')
            ax.set(xticks=range(3),xticklabels=['1/16','1/8','1/4'],xlabel='rho',ylabel='Mean of branch macro-F1 (%)',title=f'seed{s} | {NAMES[a]}');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False);fig.suptitle('R&B | every seed and every matched mechanism',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.035,'All individual CFP/OCT scores, baseline gains, costs, epochs and hashes are provided in results.csv / results.json.\nThe mean of branch F1 scores is not pooled classification performance. Development-set exploration.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.06,right=.98,top=.91,bottom=.12,hspace=.4,wspace=.3);save(fig,out,'03_centering_seeds');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,9));metrics=[('retained_energy_ratio','Raw energy retention difference'),('retained_variance_ratio','Centered variance retention difference'),('delta_over_input_l2','Residual / input norm difference')]
    for column,(ax,(metric,title)) in enumerate(zip(axes,metrics)):
        values=np.full((12,2),np.nan)
        for i,(a,rho) in enumerate(labels):
            for j,b in enumerate(['cfp','oct']):
                vals=[]
                for s in SEEDS:
                    un=lookup[key(s,a,rho)]['diagnostic']['phases']['selected']['energy'][b+'_stage3'][metric]
                    ce=lookup[key(s,'centered_'+a,rho)]['diagnostic']['phases']['selected']['energy'][b+'_stage3'][metric]
                    if un is not None and ce is not None:vals.append(ce-un)
                if vals:values[i,j]=np.mean(vals)
        lim=max(.001,float(np.nanmax(abs(values)))) if np.isfinite(values).any() else .001
        im=ax.imshow(np.ma.masked_invalid(values),cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set(yticks=range(12),yticklabels=[f'{NAMES[a]} | {rho_label(r)}' for a,r in labels],xticks=[0,1],xticklabels=['CFP','OCT'],title=title)
        if column>0:ax.set_yticklabels([])
        for i in range(12):
            for j in range(2):ax.text(j,i,'N/A' if not np.isfinite(values[i,j]) else f'{values[i,j]:+.3f}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.8))
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    fig.suptitle('R&B | paired diagnostics after joint training',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.905,'Centered-fit minus uncentered-fit | three-seed mean | all1264 training participants | same diagnostic code for both methods',fontsize=10)
    fig.text(.055,.035,'Raw energy and centered variance are different denominators. Selected-model ratios use each model\'s current training features.\nInitial paired models share features; fitted-basis statistics and initial/selected gradient diagnostics are retained in results.json.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.20,right=.94,top=.84,bottom=.13,wspace=.28);save(fig,out,'04_centering_diagnostics');plt.close(fig)


def build(root,resamples=10000,make_figures=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);m=read(root/'manifest.json');p=read(root/'protocol.json');ledger=read(root/'ledger.json');cost={j['id']:j for j in ledger['jobs']}
    previous=read(Path(p['predecessor'])/'report/results.json');old={r['id']:r for r in previous['rows']};rows=[]
    assert m['seeds']==SEEDS and m['rhos']==RHOS and len(m['rows'])==93
    for r in m['rows']:
        path=Path(r['directory']);d=read(path/'summary.json');assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and not d['test_used']
        hashes={n:sha(path/n) for n in FILES}
        if r['reused']:
            assert hashes==r['accepted_hashes'];row=dict(old[r['id']]);assert hashes==row['result_hashes']
        else:
            scores={b:100*d['selected']['tasks'][b]['macro_f1'] for b in ['cfp','oct']};scores['mean']=sum(scores.values())/2
            for b in ['cfp','oct']:
                cm=np.asarray(d['selected']['tasks'][b]['confusion_matrix']);den=cm.sum(0)+cm.sum(1)
                assert abs(np.divide(2*np.diag(cm),den,out=np.zeros(len(cm)),where=den!=0).mean()*100-scores[b])<1e-9
            info=read(path/'model.json');row=dict(r,summary=d,scores=scores,model=info,result_hashes=hashes,cost=cost[r['id']],
                stored_bridge_parameters=sum(g['stored_bridge_parameters'] for g in info['groups']),effective_bridge_parameters=sum(g['effective_bridge_parameters'] for g in info['groups']))
        arm=r.get('paired_arm',r['arm']);is_pair=arm in SVD_ARMS
        if is_pair:
            dp=root/('diagnostic_'+r['id'])/'summary.json';diag=read(dp)
            assert diag['passed'] and not diag['preflight'] and diag['selected_sha256']==hashes['selected.pt']
            for phase in diag['phases'].values():
                assert phase['probe_participants']==128 and phase['energy_participants']==1264
                assert all('retained_variance_ratio' in e for e in phase['energy'].values())
            row.update(diagnostic=diag,diagnostic_sha256=sha(dp),diagnostic_cost=cost['diagnostic_'+r['id']])
        row.update(fit_centered=not r['reused'] if is_pair else None,runtime_centering=False);rows.append(row)
    lookup={key(r['seed'],r['arm'],r['rho']):r for r in rows};assert len(lookup)==93
    stats={};boots={};subsets={str(s):[s] for s in SEEDS}|{'all3':SEEDS}
    for label,seeds in subsets.items():
        stats[label]={};boots[label]={}
        for rho in RHOS:
            rs=rho_label(rho);stats[label][rs]={};boots[label][rs]={}
            for arm in SVD_ARMS:
                un=[lookup[key(s,arm,rho)] for s in seeds];ce=[lookup[key(s,'centered_'+arm,rho)] for s in seeds]
                for a,b in zip(un,ce):
                    assert a['summary']['initial_native_sha256']==b['summary']['initial_native_sha256'] and a['summary']['parent_checkpoints']==b['summary']['parent_checkpoints']
                    assert a['stored_bridge_parameters']==b['stored_bridge_parameters'] and a['effective_bridge_parameters']==b['effective_bridge_parameters']
                    assert a['diagnostic']['phases']['selected']['probe_ids_sha256']==b['diagnostic']['phases']['selected']['probe_ids_sha256']
                    for source in ['cfp_stage3','oct_stage3']:
                        x=a['diagnostic']['phases']['initial']['energy'][source];y=b['diagnostic']['phases']['initial']['energy'][source]
                        assert abs(x['input_energy']-y['input_energy'])<1e-6*max(1,x['input_energy']) and x['delta_over_input_l2']==y['delta_over_input_l2']==0
                v=paired_bootstrap([(Path(a['directory']),Path(b['directory'])) for a,b in zip(un,ce)],resamples);boots[label][rs][arm]=v
                for b in BRANCHES:assert abs(v[b]['delta_pp']-np.mean([y['scores'][b]-x['scores'][b] for x,y in zip(un,ce)]))<1e-9
                for a,values in [(arm,un),('centered_'+arm,ce)]:stats[label][rs][a]={b:{'mean':float(np.mean([r['scores'][b] for r in values])),'sample_sd':float(np.std([r['scores'][b] for r in values],ddof=1)) if len(values)>1 else None} for b in BRANCHES}
    result={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'statistics':stats,
        'bootstrap':{'resamples':resamples,'unit':'participant; shared indices across seeds','direction':'centered-fit minus corresponding uncentered-fit','comparisons':boots},
        'manifest':m,'protocol':p,'ledger':ledger,'gpu_acceptance':read(root/'gpu_acceptance.json'),'study_summary':read(root/'study_summary.json'),
        'previous_report_sha256':sha(Path(p['predecessor'])/'report/results.json'),'test_used':False}
    write_json(out/'results.json',result);write_json(out/'bootstrap.json',result['bootstrap'])
    if make_figures:figures(out,lookup,stats,boots)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','method','rho','fit_centered','runtime_centering','r_fixed','h','cfp_f1_percent','oct_f1_percent','mean_f1_percent','cfp_gain_vs_baseline_pp','oct_gain_vs_baseline_pp','mean_gain_vs_baseline_pp','best_epoch','stop_epoch','total_parameters','bridge_stored','bridge_effective','gpu_minutes','peak_process_mib','directory','summary_sha256'])
        for r in rows:
            d=r['summary'];base=lookup[key(r['seed'],'no_bridge',None)];rho=r['rho'];c=r['cost']
            w.writerow([r['seed'],name(r['arm']),rho_label(rho),r['fit_centered'],False,int(256*rho) if rho and 'learned' not in r['arm'] else '',int(8192*rho) if rho else '']+[r['scores'][b] for b in BRANCHES]+[r['scores'][b]-base['scores'][b] for b in BRANCHES]+[d['selection']['joint']['best_epoch'],d['epochs_ran'],d['trainable_parameters'],r['stored_bridge_parameters'],r['effective_bridge_parameters'],c['gpu_seconds']/60,c['sampled_peak_process_mib'],r['directory'],r['result_hashes']['summary.json']])
    with (out/'paired_differences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subset','rho','mechanism','branch','centered_minus_uncentered_pp','ci95_low','ci95_high'])
        for label,rhos in boots.items():
            for rho,arms in rhos.items():
                for a,bs in arms.items():
                    for b,v in bs.items():w.writerow([label,rho,a,b,v['delta_pp']]+v['ci95_pp'])
    lines=['# R&B：中心化拟合基与非中心化拟合基的配对补充实验',f"\n生成时间：{result['generated_at']}\n",
        '三个种子3416/3417/3418，四类SVD桥×三档ρ×三个种子，新增36项训练，与既有36项非中心化SVD逐项配对。另复用21项无桥、可学习和随机QR结果，共93项不同第二阶段结果；没有新增独立预训练网络。',
        '\n## 精确比较定义\n','**只改变拟合基所用统计量。** 非中心化使用E[xxᵀ]，中心化使用E[xxᵀ]−μμᵀ。μ按每个来源训练集所有参与者、双眼及空间位置统计，只使用训练集。两者运行时都计算QᵀX、解码QΔ；不在运行时减均值或加均值，不称为完整的中心化PCA编解码器。中心化拟合的基按方差排序；ρ仍是维数比例。',
        '\n同种子同来源共用完整基，不同ρ取嵌套前r列，r16/32/64、每路h512/1024/2048。中心化基拟合核验与原基相同的父检查点、训练参与者顺序、原始二阶矩和原生状态SHA。桥内仍仅kernel3无偏置零初始化卷积学习，两条原生网络全部训练、BN正常更新。',
        '\n其余固定：stage3、M32、S64、backbone LR6e-5、head/bridge LR1e-4、batch16、AdamW WD0.01、分支与桥分别裁剪5；至少8轮，6轮无>0.001改善判平台，3轮停滞LR×0.3，60轮仅保护上限。共同检查点按两任务平均macro-F1选取。',
        '\n## 三种子均值 ± 样本标准差\n','![压缩比较](01_centering_comparison.png)','| ρ | 方法 | CFP | OCT | 平均 |','|---|---|---:|---:|---:|']
    for rho in RHOS:
        for a in SVD_ARMS:
            for arm in [a,'centered_'+a]:
                st=stats['all3'][rho_label(rho)][arm];lines.append('| '+' | '.join([rho_label(rho),name(arm)]+[f"{st[b]['mean']:.2f} ± {st[b]['sample_sd']:.2f}" for b in BRANCHES])+' |')
    lines+=['\n## 配对差值：中心化拟合减非中心化拟合\n','![配对区间](02_centering_intervals.png)','| ρ | 机制 | 三种子平均F1差 pp [95% CI] | 3416 / 3417 / 3418差 pp |','|---|---|---|---|']
    for rho in RHOS:
        for a in SVD_ARMS:
            v=boots['all3'][rho_label(rho)][a]['mean'];vals=' / '.join(f"{boots[str(s)][rho_label(rho)][a]['mean']['delta_pp']:+.2f}" for s in SEEDS)
            lines.append(f"| {rho_label(rho)} | {NAMES[a]} | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {vals} |")
    lines+=['\n![逐种子](03_centering_seeds.png)','![配对诊断](04_centering_diagnostics.png)',
        '\n## 完整结果与解释边界\n','完整93项原始分支F1、相对无桥增益、参数、最佳/停止轮次、显存和训练时间见results.csv。全部逐种子分支配对差及区间见paired_differences.csv。配置、父检查点、实际基、均值、协方差文件SHA、诊断及成本见results.json。',
        '\n原始能量保留率使用未中心化能量作分母，方差保留率使用中心化方差作分母；不混用。基拟合谱为独立最佳网络的特征，选中检查点诊断为联合训练后的特征。两臂均重做全1264训练参与者能量/方差/残差及固定128人梯度诊断，不参与优化或选优。',
        '\n10000次参与者配对bootstrap跨种子共享索引，不将3×296预测当成独立参与者；区间条件于已有模型和选中的开发检查点，不代表训练随机性的置信区间。样本SD另列，单种子SD未定义。比较未经多重性校正。',
        '\n仅用1264训练、296开发验证；测试数据未读取。原3416/3417参与过配置筛选，新增3418单列，仍属开发集探索性证据。保留负结果和种子间反向变化，不因平均提升就宣称两个分支均受益。',
        '\n本补充实验隔离的是基选择统计量；不能从结果推断运行时减均值是否有益。更高的能量或方差保留不等于更高判别性。自身处理有效参数较少；重采样不完全匹配几何/秩/奇异谱。固定与可学习对照还改变了投影结构和可学习性。',
        f"\n源码：{p['source_commit']}；目录：{root}。36项训练、3次基拟合、72项完整诊断以及预检分别记账，每GPU项目≤10GiB。"]
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write_json(out/'report_status.json',{'state':'complete','complete_trials':93,'total_trials':93,'new_trials':36,'paired_diagnostics':72,'seeds':SEEDS,'rhos':RHOS,'visual_review':'pending','test_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
