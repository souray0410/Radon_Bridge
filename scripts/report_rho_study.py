"""Render the final matched backbone-LR/rho matrix from immutable trial metadata."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import traceback
from datetime import datetime


def read(p): return json.loads(p.read_text())
def write(p, data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def ratio(v): return str(Fraction(v).limit_denominator())

def build(root):
    protocol=read(root/'protocol.json');out=root/'report';out.mkdir(exist_ok=True)
    rows=[]
    ledger={j['id']:j for j in read(root/'ledger.json')['jobs']}
    for group in protocol['groups']:
        for job in group['jobs']:
            path=root/job['id'];cfg=job['config'];summary=path/'summary.json'
            d=read(summary) if summary.exists() else {};failure=read(path/'failure.json') if (path/'failure.json').exists() else {}
            bridge=cfg['bridges'];rho=bridge[0]['rho'] if bridge else None
            complete=d.get('state')=='complete' and d.get('converged_by_policy') is True
            row={'trial':job['id'],'seed':cfg['seed'],'backbone_lr':cfg['backbone_lr'],'rho':rho,'h':int(256*32*rho) if rho else None,'configuration':cfg,'complete':complete,'state':'complete' if complete else failure.get('state',d.get('state','not_completed')),'source':str(summary),'sha256':hashlib.sha256(summary.read_bytes()).hexdigest() if summary.exists() else None,'summary':d,'cost':ledger.get(job['id'])}
            if complete:
                assert d['test_used'] is False
                metrics=d['selected']['tasks'];row['scores']={k:100*metrics[k]['macro_f1'] for k in ['cfp','oct']};row['scores']['mean']=sum(row['scores'].values())/2
                for m in metrics.values():
                    cm=m['confusion_matrix'];f=[]
                    for i in range(len(cm)):
                        den=sum(cm[i])+sum(v[i] for v in cm);f.append(2*cm[i][i]/den if den else 0)
                    assert abs(sum(f)/len(f)-m['macro_f1'])<1e-12
            rows.append(row)
    bases={(r['seed'],r['backbone_lr']):r for r in rows if r['rho'] is None and r['complete']}
    for r in rows:
        b=bases.get((r['seed'],r['backbone_lr']))
        if r['complete'] and b:
            assert r['summary']['parent_checkpoints']==b['summary']['parent_checkpoints']
            r['delta_pp']={k:r['scores'][k]-b['scores'][k] for k in ['cfp','oct','mean']}
    bootstrap=read(root/'bootstrap.json') if (root/'bootstrap.json').exists() else {}
    payload={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'protocol':protocol,'rows':rows,'bootstrap':bootstrap,'status':read(root/'status.json'),'test_used':False}
    write(out/'results.json',payload)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'cfp':'#087F8C','oct':'#BC703A'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    seeds=sorted({r['seed'] for r in rows});lrs=sorted({r['backbone_lr'] for r in rows});ratios=sorted({r['rho'] for r in rows if r['rho'] is not None})
    for change,stem in [(False,'01_rho_absolute'),(True,'02_rho_gain')]:
        fig,axs=plt.subplots(len(seeds),len(lrs),figsize=(12,8),sharey=True,squeeze=False)
        for i,seed in enumerate(seeds):
            for j,lr in enumerate(lrs):
                ax=axs[i,j];subset={r['rho']:r for r in rows if r['seed']==seed and r['backbone_lr']==lr}
                for branch,color in colors.items():
                    xs=[];ys=[]
                    for x,rho in enumerate(ratios):
                        r=subset.get(rho,{})
                        field='delta_pp' if change else 'scores'
                        if field in r:xs.append(x);ys.append(r[field][branch])
                    ax.plot(xs,ys,'o-',color=color,label=branch.upper(),linewidth=1.8)
                    base=subset.get(None,{})
                    if not change and 'scores' in base:ax.axhline(base['scores'][branch],color=color,linestyle='--',linewidth=1,alpha=.65)
                if change:ax.axhline(0,color='#667783',linestyle='--',linewidth=1)
                for x,rho in enumerate(ratios):
                    if not subset.get(rho,{}).get('complete'):ax.text(x,.04,'Incomplete',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#667783')
                ax.set(title=f'Seed {seed} · backbone LR={lr:g}',xticks=range(len(ratios)),xticklabels=[f'ρ={ratio(r)}\nh={int(r*8192)}' for r in ratios],xlim=(-.25,len(ratios)-.75))
                ax.grid(axis='y',color='#E4E8EB');ax.set_axisbelow(True)
                if j==0:ax.set_ylabel('Change vs matched no bridge (pp)' if change else 'Macro-F1 (%)')
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.95,.925),frameon=False,ncol=2)
        fig.suptitle('R&B (Radon Bridge) | '+('gain over no bridge' if change else 'backbone LR × retained width'),x=.08,ha='left',fontsize=18,fontweight='bold')
        fig.text(.08,.91,'stage3 · M=32 · S=64 · head/bridge LR=1e-4 · batch16',fontsize=11)
        fig.text(.08,.04,'Each panel uses its own seed/LR-matched no-bridge control. Dashed lines: '+('zero gain.' if change else 'CFP/OCT no-bridge F1.')+'\nDevelopment-set results; selected validation plateaus only. Missing results are not zero gains.',fontsize=10,color='#52606C')
        fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.83,hspace=.48,wspace=.16)
        for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=180)
        plt.close(fig)
    lines=['# R&B（Radon Bridge）：backbone学习率 × ρ 完整实验',f"\n生成时间：{payload['generated_at']}",
        '\n## 固定条件\n','stage3；M=32；S=64；任务头/桥学习率均1e-4。backbone为3e-5或6e-5；ρ=1/16、1/8、1/4；两来源使用相同ρ。batch16，AdamW，全部参数训练，BN正常更新。',
        '\n至少8轮；连续6轮没有超过0.001的实质改善判定平台；连续3轮停滞LR乘0.3；60轮仅保护上限。两分支平均macro-F1选择共同检查点，各分支分别报告。',
        '\n1264训练、296开发验证；开发集曾用于任务筛选，结果为探索性证据；未读取测试。GPU时长不限，继续记录实际耗时。低原生学习率历史实验不纳入本矩阵。',
        '\n![绝对F1](01_rho_absolute.png)\n','![匹配无桥增益](02_rho_gain.png)\n',
        '## 全部16项结果\n','| seed | backbone LR | 方法 | ρ | h | CFP F1 (%) | OCT F1 (%) | 平均 (%) | ΔCFP (pp) | ΔOCT (pp) | Δ平均 (pp) | 最佳/停止轮次 | 状态 |','|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in sorted(rows,key=lambda r:(r['seed'],r['backbone_lr'],r['rho'] or 0)):
        d=r['summary'];scores=r.get('scores',{});delta=r.get('delta_pp',{})
        vals=[str(r['seed']),f"{r['backbone_lr']:g}",'R&B' if r['rho'] else '无桥',ratio(r['rho']) if r['rho'] else '—',str(r['h'] or '—')]
        vals += [f'{scores[k]:.2f}' if k in scores else '—' for k in ['cfp','oct','mean']]
        vals += [f'{delta[k]:+.2f}' if k in delta else '—' for k in ['cfp','oct','mean']]
        vals += [f"{d['selection']['joint']['best_epoch']}/{d['epochs_ran']}" if r['complete'] else '—',r['state']]
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['\n## 配对bootstrap\n','10000次参与者级配对重采样，条件于已选择的检查点，不校正开发集筛选偏差。两个种子使用同一开发集。','| 比较 | 分支 | Δ (pp) | 95% CI (pp) |','|---|---|---:|---|']
    for name,branches in bootstrap.get('comparisons',{}).items():
        for branch,v in branches.items():lines.append(f"| {name} | {branch.upper()} | {v['delta_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] |")
    lines += ['\n## 完整指标、资源和溯源\n']
    for r in rows:
        cfg=r['configuration'];lines += [f"### {r['trial']}\n",f"seed={r['seed']}；backbone/head/bridge LR={r['backbone_lr']:g}/1e-4/1e-4；"+('无桥。' if r['rho'] is None else f"stage3 · M=32 · S=64 · ρ={ratio(r['rho'])} · h={r['h']}。"),f"来源：`{r['source']}`；SHA256：`{r['sha256']}`。"]
        if r['complete']:
            d=r['summary'];cost=r['cost'];lines += [f"参数量：{d['parameters']:,}；实际GPU分钟：{cost['gpu_seconds']/60:.2f}；峰值进程显存：{cost['sampled_peak_process_mib']} MiB。",'| 分支 | F1 (%) | Precision (%) | Recall (%) | Accuracy (%) | AUROC | 混淆矩阵 |','|---|---:|---:|---:|---:|---:|---|']
            for branch,m in d['selected']['tasks'].items():lines.append(f"| {branch.upper()} | {100*m['macro_f1']:.2f} | {100*m['macro_precision']:.2f} | {100*m['macro_recall']:.2f} | {100*m['accuracy']:.2f} | {m['auroc']:.4f} | {m['confusion_matrix']} |")
    lines += ['\nρ为保留比例，增大ρ增加实际宽度与参数量；不能把容量收益直接归因为Radon几何。缺失/资源受限/未收敛均不解释为零增益。完整配置、父检查点、源码提交、汇总SHA、资源记录见results.json。\n']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write(out/'report_status.json',{'state':'complete' if all(r['complete'] for r in rows) else 'partial','completed_trials':sum(r['complete'] for r in rows),'total_trials':len(rows),'generated_at':payload['generated_at']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true');args=p.parse_args();root=Path(args.root)
    out=root/'report';out.mkdir(exist_ok=True)
    try:
        if args.wait:
            write(out/'report_status.json',{'state':'waiting_for_experiments'})
            while not (root/'status.json').exists() or read(root/'status.json')['state'] not in ['complete','failed','needs_attention','interrupted','budget_complete']:time.sleep(10)
        build(root)
    except Exception:
        write(out/'report_status.json',{'state':'failed','error':traceback.format_exc()});raise
