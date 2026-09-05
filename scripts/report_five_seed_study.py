"""Five-seed paired statistics; shared participant indices across seed repeats."""
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from scripts.run_five_seed_study import ARMS

NAMES={'no_bridge':'No bridge','learned_equal':'Learned CM · ρ=1/8','learned_wide':'Learned CM · ρ=1/4','svd_radon':'SVD · Radon','svd_self':'SVD · self only','svd_scrambled':'SVD · scrambled Radon','svd_resample':'SVD · linear resampling','qr_radon':'Random QR · Radon'}
BRANCHES=['cfp','oct','mean'];COLORS=['#087F8C','#BC703A','#68749A']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v,signed=False):return '—' if v is None else format(v,'+.2f' if signed else '.2f')


def f1(y,pred):
    values=[]
    for c in [0,1]:
        tp=((pred==c)&(y==c)).sum(-1);den=(pred==c).sum(-1)+(y==c).sum(-1)
        values.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0))
    return sum(values)/2


def paired_bootstrap(pairs,resamples=10000):
    labels=None;ids=None;a={b:[] for b in ['cfp','oct']};b={k:[] for k in a}
    for control,main in pairs:
        with np.load(control/'selected_predictions.npz',allow_pickle=False) as x,np.load(main/'selected_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(x['ids'],z['ids']) and np.array_equal(x['y'],z['y']) and len(np.unique(x['ids']))==len(x['ids'])
            if ids is None:ids=x['ids'].copy();labels=x['y'].copy()
            else:assert np.array_equal(ids,x['ids']) and np.array_equal(labels,x['y'])
            for k in a:a[k].append(x[k].argmax(1));b[k].append(z[k].argmax(1))
    a={k:np.stack(v) for k,v in a.items()};b={k:np.stack(v) for k,v in b.items()}
    observed={k:float((f1(labels,b[k])-f1(labels,a[k])).mean()) for k in a};observed['mean']=sum(observed.values())/2
    rng=np.random.default_rng(20260905);samples={k:[] for k in BRANCHES}
    for start in range(0,resamples,100):
        idx=rng.integers(0,len(labels),size=(min(100,resamples-start),len(labels)))
        delta={k:(f1(labels[idx],b[k][:,idx])-f1(labels[idx],a[k][:,idx])).mean(0) for k in a}
        delta['mean']=(delta['cfp']+delta['oct'])/2
        for k in samples:samples[k].extend(delta[k].tolist())
    return {k:{'delta_pp':100*observed[k],'ci95_pp':(100*np.quantile(samples[k],[.025,.975])).tolist()} for k in BRANCHES}


def figures(out,rows,statistics,boot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
    all_values=[v for r in rows for v in r.get('scores',{}).values()];lo=min(all_values+[60])-1;hi=max(all_values+[70])+4.5
    for ax,branch,color in zip(axes,BRANCHES,COLORS):
        for i,arm in enumerate(ARMS):
            rs=[r for r in rows if r['arm']==arm and r['complete']]
            for r in rs:ax.scatter(r['scores'][branch],i+(r['seed']-3418)*.10,color=color,s=26,alpha=.65,marker='o' if r['seed']<3418 else '^')
            stat=statistics.get('all5',{}).get(arm,{}).get(branch)
            if stat:
                ax.scatter(stat['mean'],i,marker='D',s=48,color='#233746',zorder=4)
                ax.text(hi-.05,i,f"{stat['mean']:.2f} ± {stat['sample_sd']:.2f}",ha='right',va='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        ax.axhspan(2.65,3.35,color='#ECF4F5',zorder=-1);ax.set(xticks=np.arange(np.ceil(lo),hi,2),xlim=(lo,hi),title=branch.upper() if branch!='mean' else 'Mean of branches',yticks=range(8),yticklabels=[NAMES[a] for a in ARMS],xlabel='Macro-F1 (%)');ax.invert_yaxis();ax.grid(axis='x',color='#E5E9EC');ax.set_axisbelow(True)
    fig.suptitle('R&B (Radon Bridge) | five-seed mechanism validation',x=.05,ha='left',fontsize=19,fontweight='bold')
    fig.text(.05,.925,'stage3 · M=32 · S=64 · backbone LR=6e-5 · head/bridge LR=1e-4 · ρ=1/8 except learned-wide ρ=1/4',fontsize=10)
    fig.text(.05,.065,'Circles: original seeds 3416/3417. Triangles: new seeds 3418–3420. Diamonds: five-seed means. Labels: mean ± sample SD.\nOnly validation plateaus are shown. Linear resampling uses packing factor32, not Radon angles. Development-set exploration.',fontsize=9.5,color='#52606C')
    fig.subplots_adjust(left=.18,right=.98,top=.86,bottom=.17,wspace=.12)
    for ext in ['png','pdf']:fig.savefig(out/('01_seed_results.'+ext),dpi=180)
    plt.close(fig)
    controls=[a for a in ARMS if a!='svd_radon'];fig,axes=plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True);bounds=[1]
    for i,subset in enumerate(['all5','new3']):
        for j,(branch,color) in enumerate(zip(BRANCHES,COLORS)):
            ax=axes[i,j]
            for k,arm in enumerate(controls):
                v=boot.get(subset,{}).get(arm,{}).get(branch)
                if not v:continue
                lower,upper=v['ci95_pp'];delta=v['delta_pp'];bounds.extend([abs(lower),abs(upper),abs(delta)])
                ax.hlines(k,lower,upper,color=color,lw=1.5);ax.scatter(delta,k,color=color,s=30)
            ax.set(yticks=range(7),yticklabels=[NAMES[a] for a in controls],title=f"{'All 5 seeds' if i==0 else 'New 3 seeds'} | {branch.upper()}",xlabel='SVD-Radon minus control (pp)');ax.invert_yaxis();ax.axvline(0,color='#83909B',lw=1);ax.grid(axis='x',color='#E5E9EC');ax.set_axisbelow(True)
    limit=max(bounds)*1.1;axes[0,0].set_xlim(-limit,limit);axes[0,0].set_ylim(6.5,-.5)
    fig.suptitle('R&B | paired participant uncertainty',x=.05,ha='left',fontsize=19,fontweight='bold')
    fig.text(.05,.925,'10,000 resamples · identical participant indices across seeds · 95% percentile intervals',fontsize=10)
    fig.text(.05,.04,'Intervals condition on the trained models and selected development checkpoints. Seeds do not multiply the participant sample size.\nOriginal seeds informed configuration selection; new-seed results are shown separately. These are not external-test estimates.',fontsize=9.5,color='#52606C')
    fig.subplots_adjust(left=.18,right=.98,top=.86,bottom=.14,wspace=.12,hspace=.27)
    for ext in ['png','pdf']:fig.savefig(out/('02_paired_intervals.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,6.8));specs=[('retained_energy_ratio','Selected channel energy retained',0,1,'viridis'),('delta_over_input_l2','Selected residual / feature norm',0,None,'YlGnBu'),('cosine','Selected stage3 gradient cosine',-1,1,'coolwarm')]
    for ax,(metric,title,vmin,vmax,cmap) in zip(axes,specs):
        matrix=np.full((8,2),np.nan)
        for i,arm in enumerate(ARMS):
            for j,b in enumerate(['cfp','oct']):
                vals=[]
                for r in rows:
                    if r['arm']!=arm or not r.get('diagnostic'):continue
                    diag=r['diagnostic']['phases']['selected'];v=diag['gradient_groups'][b+'_stage3'][metric] if metric=='cosine' else diag['energy'][b+'_stage3'][metric]
                    if v is not None:vals.append(v)
                if vals:matrix[i,j]=np.mean(vals)
        im=ax.imshow(np.ma.masked_invalid(matrix),vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto');ax.set(xticks=[0,1],xticklabels=['CFP','OCT'],yticks=range(8),yticklabels=[NAMES[a] for a in ARMS],title=title)
        for i in range(8):
            for j in range(2):ax.text(j,i,'N/A' if not np.isfinite(matrix[i,j]) else f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
        fig.colorbar(im,ax=ax,fraction=.035,pad=.03)
    fig.suptitle('R&B | read-only training-set diagnostics',x=.05,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.925,'Means across available seeds; full initial/selected values and gradient groups are retained in the report.',fontsize=10)
    fig.text(.05,.035,'Energy: all1264 training participants, pre-writeback features. Gradients: fixed128 training participants, accumulated before cosine.\nFor learned/no-bridge arms, energy is measured in the parent SVD diagnostic subspace. Zero-gradient cosine is undefined, not zero.',fontsize=9,color='#52606C')
    fig.subplots_adjust(left=.19,right=.97,top=.85,bottom=.16,wspace=1.2)
    for ext in ['png','pdf']:fig.savefig(out/('03_diagnostics.'+ext),dpi=180)
    plt.close(fig)


def build(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);manifest=read(root/'manifest.json');p=read(root/'protocol.json');ledger=read(root/'ledger.json');cost={j['id']:j for j in ledger['jobs']};rows=[]
    for record in manifest['rows']:
        path=Path(record['directory']);d=read(path/'summary.json') if (path/'summary.json').exists() else {};complete=d.get('state')=='complete' and d.get('converged_by_policy') is True
        if record['reused']:
            for name,digest in record['accepted_hashes'].items():assert sha(path/name)==digest
        r=dict(record,complete=complete,state=d.get('state','not_completed'),summary=d,cost=record.get('reference_cost',cost.get(record['id'])))
        if complete:
            assert d['test_used'] is False and d['stop_reason']=='validation_plateau'
            r['scores']={b:100*d['selected']['tasks'][b]['macro_f1'] for b in ['cfp','oct']};r['scores']['mean']=(r['scores']['cfp']+r['scores']['oct'])/2
            for b in ['cfp','oct']:
                cm=np.asarray(d['selected']['tasks'][b]['confusion_matrix']);den=cm.sum(0)+cm.sum(1)
                assert abs(np.divide(2*np.diag(cm),den,out=np.zeros(len(cm)),where=den!=0).mean()*100-r['scores'][b])<1e-10
            info=read(path/'model.json');r['model']=info
            r['stored_bridge_parameters']=sum(g.get('stored_bridge_parameters',0) for g in info['groups'])
            if record['reused'] and info['groups']:
                baseline=next(x for x in manifest['rows'] if x['seed']==record['seed'] and x['arm']=='no_bridge')
                r['stored_bridge_parameters']=d['trainable_parameters']-read(Path(baseline['directory'])/'summary.json')['trainable_parameters']
            r['effective_bridge_parameters']=sum(g.get('effective_bridge_parameters',r['stored_bridge_parameters']) for g in info['groups'])
        diag=root/('diagnostic_'+record['id'])/'summary.json'
        if diag.exists():r['diagnostic']=read(diag);assert r['diagnostic']['passed'] and not r['diagnostic']['preflight']
        rows.append(r)
    lookup={(r['seed'],r['arm']):r for r in rows};stats={};boots={};subsets={str(s):[s] for s in range(3416,3421)}|{'all5':list(range(3416,3421)),'new3':[3418,3419,3420]}
    for label,seeds in subsets.items():
        stats[label]={};boots[label]={}
        for arm in ARMS:
            rs=[lookup[(s,arm)] for s in seeds]
            if all(r['complete'] for r in rs):
                stats[label][arm]={b:{'mean':float(np.mean([r['scores'][b] for r in rs])),'sample_sd':float(np.std([r['scores'][b] for r in rs],ddof=1)) if len(rs)>1 else None} for b in BRANCHES}
            if arm=='svd_radon':continue
            pairs=[(lookup[(s,arm)],lookup[(s,'svd_radon')]) for s in seeds]
            if all(a['complete'] and b['complete'] for a,b in pairs):
                for a,b in pairs:
                    assert a['summary']['initial_native_sha256']==b['summary']['initial_native_sha256'] and a['summary']['parent_checkpoints']==b['summary']['parent_checkpoints']
                boots[label][arm]=paired_bootstrap([(Path(a['directory']),Path(b['directory'])) for a,b in pairs])
                for branch in BRANCHES:assert abs(boots[label][arm][branch]['delta_pp']-np.mean([b['scores'][branch]-a['scores'][branch] for a,b in pairs]))<1e-9
    bootstrap={'resamples':10000,'unit':'participant; shared resampling indices across seeds','comparisons':boots,'test_used':False}
    write_json(out/'bootstrap.json',bootstrap)
    result={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'statistics':stats,'bootstrap':bootstrap,
            'protocol':p,'manifest':manifest,'ledger':ledger,'gpu_acceptance':read(root/'gpu_acceptance.json'),'qr_generation':read(root/'qr_generation.json') if (root/'qr_generation.json').exists() else {},'test_used':False}
    write_json(out/'results.json',result);figures(out,rows,stats,boots)
    lines=['# R&B：五种子机制验证与固定基对照',f"\n生成时间：{result['generated_at']}\n",'五个种子3416–3420；第二阶段stage3、M32、S64、backbone LR6e-5、head/bridge LR1e-4。除可学习历史优选参照ρ=1/4外，均为ρ=1/8，固定基r32、每路h1024。ρ表示维数比例，不表示能量阈值。',
    '\n同一种子复用父检查点和数据顺序；共同检查点按两任务平均macro-F1选取。至少8轮、6轮无超过0.001改善判平台，3轮停滞LR×0.3，60轮是保护上限。1264训练、296开发验证；未读取测试。',
    '\n![逐种子结果](01_seed_results.png)\n','![配对区间](02_paired_intervals.png)\n','![诊断](03_diagnostics.png)\n','## 汇总：均值 ± 样本标准差\n','| 种子范围 | 方法 | CFP F1 (%) | OCT F1 (%) | 平均F1 (%) |','|---|---|---:|---:|---:|']
    for subset in ['all5','new3']:
        for arm in ARMS:
            s=stats[subset].get(arm,{})
            lines.append('| '+' | '.join([subset,NAMES[arm]]+[f"{s[b]['mean']:.2f} ± {s[b]['sample_sd']:.2f}" if b in s else '—' for b in BRANCHES])+' |')
    lines+=['\n## 逐种子结果与成本\n','| seed | 方法 | CFP | OCT | 平均 | 最佳/停止轮 | 总参数 | 桥存储/有效参数 | GPU分钟 | 进程峰值MiB | 状态 |','|---:|---|---:|---:|---:|---|---:|---|---:|---:|---|']
    for r in rows:
        d=r['summary'];c=r['cost'] or {};s=r.get('scores',{})
        vals=[str(r['seed']),NAMES[r['arm']]]+[fmt(s.get(b)) for b in BRANCHES]+[f"{d['selection']['joint']['best_epoch']}/{d['epochs_ran']}" if r['complete'] else '—',str(d.get('trainable_parameters','—')),f"{r.get('stored_bridge_parameters','—')}/{r.get('effective_bridge_parameters','—')}",fmt(c['gpu_seconds']/60 if 'gpu_seconds' in c else None),str(c.get('sampled_peak_process_mib','—')),r['state']]
        lines.append('| '+' | '.join(vals)+' |')
    lines+=['\n## 配对bootstrap\n','每项差值均为SVD-Radon减参照。10000次参与者级重采样，同一次索引跨种子共享；区间条件于已有模型与选中的开发集检查点。不将5×296预测当作独立样本。','| 种子范围 | 参照 | 分支 | Δ pp | 95% CI pp |','|---|---|---|---:|---|']
    for subset,comparisons in boots.items():
        for arm,values in comparisons.items():
            for b,v in values.items():lines.append(f"| {subset} | {NAMES[arm]} | {b.upper()} | {v['delta_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] |")
    lines+=['\n## 诊断与解释边界\n','诊断只读取初始和选中检查点，不参与优化。能量/残差统计使用全部训练参与者及写回前特征；梯度固定使用排序后的128名训练参与者，先累积参与者加权梯度再求余弦。零梯度范数记为未定义。固定单桥卷积两项CE作用于不同输出行块，理论余弦为0，不代表桥前主干没有任务冲突。',
    '\n自身处理报告有效参数减少；重采样匹配宽度、核长和前后向行范数，但不匹配秩或奇异谱，打包维度32不是角度。随机QR基没有能量排序，SVD与随机基对照检验数据导出子空间是否有用。可学习历史优选参照宽度更大，不属于等宽比较。',
    '\n原两个种子参与过配置选择，新增三个种子单独展示，但仍使用同一开发集，不能称为独立测试验证。五种子波动包括独立预训练、相应基及联合训练差异，QR臂还含固定随机基差异。',
    '\n完整逐来源初始/选中能量、残差、梯度分组、探针清单摘要、基来源、配置和SHA见[results.json](results.json)。\n','## 新增工作记账\n','| 工作 | GPU分钟 | 进程峰值MiB |','|---|---:|---:|']
    for j in result['gpu_acceptance']['ledger']['jobs']+ledger['jobs']:lines.append(f"| {j['id']} | {j['gpu_seconds']/60:.3f} | {j['sampled_peak_process_mib']} |")
    lines+=['\n随机QR基CPU生成耗时单列，不重复计为GPU训练时间：'+json.dumps(result['qr_generation'],ensure_ascii=False)]
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','arm','complete','cfp_f1_percent','oct_f1_percent','mean_f1_percent','directory'])
        for r in rows:w.writerow([r['seed'],r['arm'],r['complete']]+[r.get('scores',{}).get(b) for b in BRANCHES]+[r['directory']])
    complete=all(r['complete'] and r.get('diagnostic') for r in rows)
    write_json(out/'report_status.json',{'state':'complete' if complete else 'partial','complete_trials':sum(r['complete'] for r in rows),'total_trials':40,'complete_diagnostics':sum(bool(r.get('diagnostic')) for r in rows),'test_used':False})
