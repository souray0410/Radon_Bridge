"""Advisor-facing complete 3-seed x 3-width matrix, with participant pairing."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from scripts.run_three_seed_study import SEEDS,RHOS,METHODS,matrix,key
from scripts.report_five_seed_study import paired_bootstrap,read,sha,fmt,BRANCHES

NAMES={'no_bridge':'No bridge','learned':'Learned CM','svd_radon':'SVD / Radon','svd_self':'SVD / self only','svd_scrambled':'SVD / scrambled Radon','svd_resample':'SVD / linear resampling','qr_radon':'Random QR / Radon'}
COLORS={'learned':'#B56D37','svd_radon':'#007E87','svd_self':'#9674A6','svd_scrambled':'#71828D','svd_resample':'#5E879A','qr_radon':'#A38B4F'}
CONTROLS=['no_bridge']+[a for a in METHODS if a!='svd_radon']
def rho_label(r):return 'shared baseline' if r is None else '1/'+str(round(1/r))
def save(fig,out,name):
    if 'QA_ONLY' in str(out):fig.text(.5,.5,'QA FIXTURE - NOT EXPERIMENT RESULTS',ha='center',va='center',rotation=25,fontsize=28,color='red',alpha=.35)
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)


def figures(out,lookup,stats,boots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,6.3))
    for ax,b in zip(axes,BRANCHES):
        for a,marker in [('learned','o'),('svd_radon','s')]:
            values=[stats['all3'][rho_label(r)][a][b] for r in RHOS]
            line=ax.errorbar([0,1,2],[v['mean'] for v in values],yerr=[v['sample_sd'] for v in values],color=COLORS[a],marker=marker,lw=2,capsize=4,label=NAMES[a])
            assert np.array_equal(line.lines[0].get_xdata(),[0,1,2]),'Rho points must align with ticks'
        baseline=stats['all3']['1/8']['no_bridge'][b]['mean'];ax.axhline(baseline,ls=':',color='#56616A',label='No bridge')
        ax.set(xticks=[0,1,2],xticklabels=[r'$\rho=1/16$\n$r=16,\ h=512$',r'$\rho=1/8$\n$r=32,\ h=1024$',r'$\rho=1/4$\n$r=64,\ h=2048$'],title=b.upper(),ylabel='Macro-F1 (%)',xlim=(-.25,2.25))
        ax.set_xticklabels([t.get_text().replace('\\n','\n') for t in ax.get_xticklabels()]);ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='best',frameon=False)
    fig.suptitle('R&B (Radon Bridge) | compression comparison',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.9,'Three seeds (3416-3418) | means +/- sample SD | stage3 / M32 / S64 / backbone LR 6e-5',fontsize=10)
    fig.text(.055,.045,'Both methods use exactly the labeled x coordinates. rho is a dimension ratio, not retained energy.\nr describes the fixed channel projection; learned CM has the same h. Development-set exploration.',fontsize=9.5,color='#52606C')
    fig.subplots_adjust(left=.06,right=.985,top=.81,bottom=.24,wspace=.23);save(fig,out,'01_compression');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,14),sharey=True)
    for i,rho in enumerate(RHOS):
        for j,b in enumerate(BRANCHES):
            ax=axes[i,j]
            for k,a in enumerate(['no_bridge']+METHODS):
                for s,marker in zip(SEEDS,['o','s','^']):
                    record=lookup[key(s,a,None if a=='no_bridge' else rho)]
                    ax.scatter(record['scores'][b],k+(s-3417)*.13,marker=marker,color=COLORS.get(a,'#525E67'),s=33)
                st=stats['all3'][rho_label(rho)][a][b];ax.scatter(st['mean'],k,color='#20323F',marker='|',s=150)
            ax.set(yticks=range(7),yticklabels=[NAMES[a] for a in ['no_bridge']+METHODS],ylim=(6.5,-.5),title=f'rho={rho_label(rho)} | {b.upper()}',xlabel='Macro-F1 (%)')
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('R&B | all 57 results, including negative results',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.94,'Circle: seed3416; square: seed3417; triangle: new seed3418. Black marks: three-seed means. Baselines are shared across rho.',fontsize=10)
    fig.text(.055,.025,'CFP / OCT / mean are reported separately. Only trials reaching the prespecified validation plateau are accepted.\nSeed3418 was not used for the earlier configuration choice; all seeds still use the same previously screened development set.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.17,right=.98,top=.9,bottom=.09,hspace=.35,wspace=.12);save(fig,out,'02_all_seeds');plt.close(fig)
    bounds=[abs(v) for rs in boots['all3'].values() for arm in rs.values() for br in arm.values() for v in [br['delta_pp']]+br['ci95_pp']];limit=max(bounds+[1])*1.1
    fig,axes=plt.subplots(3,3,figsize=(16,13),sharex=True,sharey=True)
    for i,rho in enumerate(RHOS):
        for j,b in enumerate(BRANCHES):
            ax=axes[i,j]
            for k,a in enumerate(CONTROLS):
                v=boots['all3'][rho_label(rho)][a][b];lo,hi=v['ci95_pp'];ax.hlines(k,lo,hi,color='#007E87',lw=2);ax.scatter(v['delta_pp'],k,color='#007E87',s=34)
            ax.axvline(0,color='#697680',lw=1);ax.set(yticks=range(6),yticklabels=[NAMES[a] for a in CONTROLS],ylim=(5.5,-.5),xlim=(-limit,limit),title=f'rho={rho_label(rho)} | {b.upper()}',xlabel='SVD / Radon minus control (pp)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('R&B | matched controls and paired uncertainty',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.94,'Three-seed mean differences | 10,000 participant resamples | 95% percentile intervals | same rho for every bridge comparison',fontsize=10)
    fig.text(.055,.025,'Participant indices are shared across seeds. Intervals condition on the trained models and selected development checkpoints.\nThey do not measure uncertainty from new training seeds or constitute independent-test evidence. No multiplicity correction.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.17,right=.98,top=.9,bottom=.1,wspace=.12,hspace=.34);save(fig,out,'03_paired_intervals');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,14))
    metrics=[('retained_energy_ratio','Selected energy retained',0,1,'viridis'),('delta_over_input_l2','Selected residual / input',0,None,'YlGnBu'),('cosine','Stage3 CE gradient cosine',-1,1,'coolwarm')]
    for i,rho in enumerate(RHOS):
        for j,(metric,title,lo,hi,cmap) in enumerate(metrics):
            ax=axes[i,j];values=np.full((7,2),np.nan)
            for k,a in enumerate(['no_bridge']+METHODS):
                for t,b in enumerate(['cfp','oct']):
                    vals=[]
                    for s in SEEDS:
                        d=lookup[key(s,a,None if a=='no_bridge' else rho)]['diagnostic']['phases']['selected']
                        v=d['gradient_groups'][b+'_stage3'][metric] if metric=='cosine' else d['energy'][b+'_stage3'][metric]
                        if v is not None:vals.append(v)
                    if vals:values[k,t]=np.mean(vals)
            im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',vmin=lo,vmax=hi,cmap=cmap)
            ax.set(xticks=[0,1],xticklabels=['CFP','OCT'],yticks=range(7),yticklabels=[NAMES[a] for a in ['no_bridge']+METHODS],title=f'rho={rho_label(rho)} | {title}')
            for k in range(7):
                for t in range(2):ax.text(t,k,'N/A' if not np.isfinite(values[k,t]) else f'{values[k,t]:.3f}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.8))
            fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    fig.suptitle('R&B | training-set diagnostics',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.94,'All 1264 training participants: energy and residuals. Fixed first 128 participants: accumulated CE gradients. Three-seed means.',fontsize=10)
    fig.text(.055,.025,'Energy for learned CM / no bridge uses a fixed parent-SVD r32 diagnostic subspace, not their actual compression.\nFixed methods use their actual r16/r32/r64 basis. Zero-gradient cosines are undefined; initial and selected details are in results.json.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.17,right=.97,top=.89,bottom=.1,wspace=.8,hspace=.35);save(fig,out,'04_diagnostics');plt.close(fig)


def build(root,resamples=10000,make_figures=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);m=read(root/'manifest.json');p=read(root/'protocol.json');ledger=read(root/'ledger.json');cost={j['id']:j for j in ledger['jobs']};rows=[]
    assert m['seeds']==SEEDS and m['rhos']==RHOS and len(m['rows'])==57
    assert {key(r['seed'],r['arm'],r['rho']) for r in m['rows']}=={key(*x) for x in matrix()}
    for r in m['rows']:
        path=Path(r['directory']);d=read(path/'summary.json');assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and not d['test_used']
        hashes={n:sha(path/n) for n in ['summary.json','configuration.json','selected_predictions.npz','selected.pt']}
        if r['reused']:assert hashes==r['accepted_hashes']
        assert d['configuration']['seed']==r['seed'] and d['configuration']['backbone_lr']==6e-5
        for b in d['configuration']['bridges']:assert b['rho']==r['rho'] and b['M']==32 and b['S']==64
        scores={b:100*d['selected']['tasks'][b]['macro_f1'] for b in ['cfp','oct']};scores['mean']=sum(scores.values())/2
        for b in ['cfp','oct']:
            cm=np.asarray(d['selected']['tasks'][b]['confusion_matrix']);den=cm.sum(0)+cm.sum(1)
            assert abs(np.divide(2*np.diag(cm),den,out=np.zeros(len(cm)),where=den!=0).mean()*100-scores[b])<1e-9
        diag_path=root/('diagnostic_'+r['id'])/'summary.json';diag=read(diag_path);assert diag['passed'] and not diag['preflight'] and diag['selected_sha256']==hashes['selected.pt']
        for phase in ['initial','selected']:assert diag['phases'][phase]['probe_participants']==128 and diag['phases'][phase]['energy_participants']==1264
        info=read(path/'model.json');base_record=next(x for x in m['rows'] if x['seed']==r['seed'] and x['arm']=='no_bridge')
        stored=d['trainable_parameters']-read(Path(base_record['directory'])/'summary.json')['trainable_parameters']
        effective=sum(g.get('effective_bridge_parameters',stored) for g in info['groups'])
        rows.append(dict(r,scores=scores,summary=d,model=info,diagnostic=diag,diagnostic_sha256=sha(diag_path),result_hashes=hashes,
            stored_bridge_parameters=stored,effective_bridge_parameters=effective,cost=r.get('reference_cost',cost.get(r['id'])),diagnostic_cost=cost['diagnostic_'+r['id']]))
    lookup={key(r['seed'],r['arm'],r['rho']):r for r in rows};stats={};boots={}
    subsets={str(s):[s] for s in SEEDS}|{'all3':SEEDS}
    for label,seeds in subsets.items():
        stats[label]={};boots[label]={}
        for rho in RHOS:
            rs=rho_label(rho);stats[label][rs]={};boots[label][rs]={}
            main=[lookup[key(s,'svd_radon',rho)] for s in seeds]
            for arm in ['no_bridge']+METHODS:
                control=[lookup[key(s,arm,None if arm=='no_bridge' else rho)] for s in seeds]
                stats[label][rs][arm]={b:{'mean':float(np.mean([r['scores'][b] for r in control])),
                    'sample_sd':float(np.std([r['scores'][b] for r in control],ddof=1)) if len(seeds)>1 else None} for b in BRANCHES}
                if arm=='svd_radon':continue
                for a,b in zip(control,main):
                    assert a['summary']['initial_native_sha256']==b['summary']['initial_native_sha256'] and a['summary']['parent_checkpoints']==b['summary']['parent_checkpoints']
                    assert a['diagnostic']['phases']['selected']['probe_ids_sha256']==b['diagnostic']['phases']['selected']['probe_ids_sha256']
                v=paired_bootstrap([(Path(a['directory']),Path(b['directory'])) for a,b in zip(control,main)],resamples)
                for branch in BRANCHES:assert abs(v[branch]['delta_pp']-np.mean([b['scores'][branch]-a['scores'][branch] for a,b in zip(control,main)]))<1e-9
                boots[label][rs][arm]=v
    result={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'statistics':stats,
            'bootstrap':{'resamples':resamples,'unit':'participant; shared indices across seeds','comparisons':boots},'manifest':m,'protocol':p,'ledger':ledger,
            'gpu_acceptance':read(root/'gpu_acceptance.json'),'study_summary':read(root/'study_summary.json'),'test_used':False}
    write_json(out/'results.json',result);write_json(out/'bootstrap.json',result['bootstrap'])
    if make_figures:figures(out,lookup,stats,boots)
    fields=['seed','method','rho','r_fixed','h','cfp_f1_percent','oct_f1_percent','mean_f1_percent','cfp_gain_vs_baseline_pp','oct_gain_vs_baseline_pp','mean_gain_vs_baseline_pp','best_epoch','stop_epoch','total_parameters','stored_bridge_parameters','effective_bridge_parameters','gpu_minutes','peak_process_mib','directory','summary_sha256']
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for r in rows:
            base=lookup[key(r['seed'],'no_bridge',None)];d=r['summary'];cost=r['cost'];rho=r['rho']
            w.writerow([r['seed'],NAMES[r['arm']],rho_label(rho),int(256*rho) if rho and r['arm']!='learned' else '',int(8192*rho) if rho else '']+[r['scores'][b] for b in BRANCHES]+[r['scores'][b]-base['scores'][b] for b in BRANCHES]+[d['selection']['joint']['best_epoch'],d['epochs_ran'],d['trainable_parameters'],r['stored_bridge_parameters'],r['effective_bridge_parameters'],cost['gpu_seconds']/60,cost['sampled_peak_process_mib'],r['directory'],r['result_hashes']['summary.json']])
    with (out/'paired_differences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed_subset','rho','control','branch','svd_minus_control_pp','ci95_low_pp','ci95_high_pp'])
        for subset,rhos in boots.items():
            for rho,arms in rhos.items():
                for arm,bs in arms.items():
                    for b,v in bs.items():w.writerow([subset,rho,arm,b,v['delta_pp']]+v['ci95_pp'])
    lines=['# R&B（Radon Bridge）：三种子、三档ρ完整机制验证',f"\n生成时间：{result['generated_at']}\n",
        '本报告包括57项第二阶段结果：3416/3417/3418各有六种桥×三档ρ，以及一项共享无桥基线。所有结果均达到预设开发集平台期。完整原始值和成本见results.csv；配对差值见paired_differences.csv；全部配置、诊断、基和摘要SHA见results.json。',
        '\n## 研究问题与配对设计\n','固定SVD收益是否能在新种子重现？跨来源混合、空间结构和训练导出的通道子空间各有什么贡献？每项比较共享同种子父检查点、训练顺序与优化设置。六种桥均运行ρ=1/16、1/8、1/4，每路h=512、1024、2048。固定通道r=16、32、64；ρ不是能量阈值。',
        '\nstage3、M32、S64、kernel3无偏置零初始化；backbone LR6e-5、head/bridge LR1e-4；全参数训练、BN正常更新。独立预训练主干LR3e-5。batch16、AdamW WD0.01、分支/桥分别裁剪5；至少8轮，6轮无>0.001实质改善判平台，3轮停滞LR×0.3，最多60轮仅保护上限。共同检查点按两任务平均macro-F1选取。',
        '\n## 主要结果：三种子均值 ± 样本标准差\n','![压缩比较](01_compression.png)','| ρ | 方法 | CFP F1 (%) | OCT F1 (%) | 平均 F1 (%) |','|---|---|---:|---:|---:|']
    for rho in RHOS:
        for a in ['no_bridge']+METHODS:
            st=stats['all3'][rho_label(rho)][a]
            lines.append('| '+' | '.join([rho_label(rho),NAMES[a]]+[f"{st[b]['mean']:.2f} ± {st[b]['sample_sd']:.2f}" for b in BRANCHES])+' |')
    lines+=['\n## 预设对比的证据\n','所有差值统一为同ρ的SVD/Radon减对照，不从多档配置中挑最高值作配对。自身处理检验跨来源混合，空间打乱检验空间组织，重采样检验相对于普通线性通信的差异，随机QR检验训练导出子空间相对随机子空间的差异。',
        '| ρ | 对照 | 三种子平均F1差 pp [95% CI] | 各种子平均F1差 pp：3416 / 3417 / 3418 |','|---|---|---|---|']
    for rho in RHOS:
        rs=rho_label(rho)
        for a in CONTROLS:
            v=boots['all3'][rs][a]['mean'];individual=' / '.join(f"{boots[str(s)][rs][a]['mean']['delta_pp']:+.2f}" for s in SEEDS)
            lines.append(f"| {rs} | {NAMES[a]} | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {individual} |")
    lines+=['\n![所有种子](02_all_seeds.png)','![配对区间](03_paired_intervals.png)',
        '\n## 新种子3418单列\n','3416和3417参与过先前配置筛选；3418是本轮新增训练重复。仅一枚新种子，不能单独估计训练随机性的样本标准差。仍使用相同开发集，不能称为独立测试验证。',
        '| ρ | 方法 | CFP | OCT | 平均 |','|---|---|---:|---:|---:|']
    for rho in RHOS:
        for a in ['no_bridge']+METHODS:lines.append('| '+' | '.join([rho_label(rho),NAMES[a]]+[f"{stats['3418'][rho_label(rho)][a][b]['mean']:.2f}" for b in BRANCHES])+' |')
    lines+=['\n## 诊断与计算成本\n','![训练集诊断](04_diagnostics.png)',
        '训练集1264人统计初始及选中检查点的通道能量和残差；按稳定标识排序的前128人，batch16/eval，分别对两CE累积参与者加权梯度，再求范数和余弦。零范数记为未定义。诊断不改变参数、BN或随机状态。固定单桥卷积梯度位于不同目的行块、余弦为0，是结构核验，不等价于整个系统没有梯度冲突。',
        '\n无桥和可学习CM的能量诊断统一使用父SVD的r32参照子空间，并非它们的实际压缩。固定版本使用对应ρ的实际基。初始基拟合能量与联合训练后能量必须区分。',
        '\n全部57项的参数、GPU分钟、显存、最佳/停止轮次见results.csv。预训练、拟合、诊断、预检及失败工作单独记账，不删除成本；当前运行 ledger 和继承累计时间见results.json。每卡项目不超过10GiB，无GPU时长预算。',
        '\n## 解释边界与后续判断\n',
        '1. 1264训练、296开发验证；测试数据未读取。开发集历史参与任务与配置筛选，本报告是探索性证据。',
        '2. 10000次参与者级配对bootstrap在不同种子共享同一重采样索引；不把3×296份预测当成独立参与者。区间条件于现有模型和选中检查点，不包含重新训练的不确定性；另列三种子样本SD。多项比较未经多重性校正。',
        '3. 可学习CM与固定通道投影同时改变投影结构和两端可学习性，不能把所有差异归因于是否训练。SVD与QR比较只检验训练导出子空间相对随机子空间；种子波动含预训练、联合训练及相应基差异。',
        '4. 自身处理屏蔽跨来源卷积块，存储与有效参数分列，不称为有效容量完全匹配。重采样匹配宽度、核长和前后向行L2范数，不匹配几何、秩或奇异谱；打包维度32不代表角度。',
        '5. 单独检查CFP、OCT以及三枚种子；平均值提高不能替代双分支一致改善。跨零区间、负差值和种子间反向结果均完整保留。三种子足以支持当前草稿的初步稳定性检查，不能据此宣称充分复现或临床有效。',
        '\n## 溯源\n',f"源码提交：{p['source_commit']}；实验目录：{root}。协议、接受清单与原始结果均不可覆盖。旧五种子队列因诊断OOM停下，4项已完成机制训练通过配置、平台状态、父检查点及SHA核验后复用；3419/3420取消，未启动。诊断释放计算图的修复经完整128人探针及最大ρ验证。"]
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write_json(out/'report_status.json',{'state':'complete','complete_trials':57,'total_trials':57,'complete_diagnostics':57,'seeds':SEEDS,'rhos':RHOS,'visual_review':'pending','test_used':False})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();build(args.root)
