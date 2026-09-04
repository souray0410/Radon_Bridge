"""Combine the fixed-M32 LR/rho matrix and its matched mechanism controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime


def read(p):return json.loads(p.read_text())
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
NAMES={'independent':'无桥','radon':'R&B','random':'高斯随机投影','scrambled':'空间打乱Radon','self':'仅自身处理','pooled':'普通池化通信'}
ENGLISH={'independent':'No bridge','radon':'R&B','random':'Gaussian projection','scrambled':'Scrambled Radon','self':'Self only','pooled':'Pooled*'}
ORDER=['independent','radon','random','scrambled','self','pooled']


def build(root):
    out=root/'report';out.mkdir(exist_ok=True);protocol=read(root/'protocol.json');prior=Path(protocol['predecessor'])
    original=read(prior/'report/results.json');rows=original['rows'];bases={}
    for r in rows:
        r['mode']='independent' if r['rho'] is None else 'radon'
        if r['rho'] is None and r['complete']:bases[(r['seed'],r['backbone_lr'])]=r
    ledger={j['id']:j for j in read(root/'ledger.json')['jobs']}
    for group in protocol['groups']:
        for job in group['jobs']:
            path=root/job['id'];cfg=job['config'];summary=path/'summary.json';d=read(summary) if summary.exists() else {}
            failure=read(path/'failure.json') if (path/'failure.json').exists() else {}
            complete=d.get('state')=='complete' and d.get('converged_by_policy') is True
            mode=cfg['bridges'][0]['mode'];model=read(path/'model.json') if (path/'model.json').exists() else {}
            widths=[{p['key']:p['retained_channels'] for p in g['participants']} for g in model.get('groups',[])]
            r={'trial':job['id'],'seed':cfg['seed'],'backbone_lr':cfg['backbone_lr'],'mode':mode,'rho':.125,'h':widths,'complete':complete,'state':'complete' if complete else failure.get('state',d.get('state','not_completed')),'source':str(summary),'sha256':hashlib.sha256(summary.read_bytes()).hexdigest() if summary.exists() else None,'configuration':cfg,'summary':d,'cost':ledger.get(job['id'])}
            if complete:
                assert d['test_used'] is False
                base=bases[(r['seed'],r['backbone_lr'])]
                assert d['initial_native_sha256']==base['summary']['initial_native_sha256']
                assert d['parent_checkpoints']==base['summary']['parent_checkpoints']
                m=d['selected']['tasks'];r['scores']={b:100*m[b]['macro_f1'] for b in ['cfp','oct']};r['scores']['mean']=sum(r['scores'].values())/2
                for metrics in m.values():
                    cm=metrics['confusion_matrix'];f=[]
                    for i in range(len(cm)):
                        den=sum(cm[i])+sum(v[i] for v in cm);f.append(2*cm[i][i]/den if den else 0.)
                    assert abs(sum(f)/len(f)-metrics['macro_f1'])<1e-12
                r['delta_pp']={b:r['scores'][b]-base['scores'][b] for b in ['cfp','oct','mean']}
            rows.append(r)
    bootstrap=read(root/'bootstrap.json') if (root/'bootstrap.json').exists() else {}
    data={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'original_protocol':original['protocol'],'mechanism_protocol':protocol,'bootstrap_rho':original['bootstrap'],'bootstrap_mechanisms':bootstrap,'status':read(root/'status.json'),'test_used':False}
    write(out/'results.json',data)
    for name in ['01_rho_absolute','02_rho_gain']:
        for ext in ['png','pdf']:shutil.copy2(prior/'report'/(name+'.'+ext),out/(name+'.'+ext))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,10),sharex=True)
    selected=[r for r in rows if r['mode']=='independent' or r['rho']==.125]
    values=[v for r in selected for k,v in r.get('delta_pp',{}).items() if k!='mean'];lo=min([0]+values)-1;hi=max([0]+values)+1
    for i,seed in enumerate([3416,3417]):
        for j,lr in enumerate([3e-5,6e-5]):
            ax=axes[i,j];lookup={r['mode']:r for r in selected if r['seed']==seed and r['backbone_lr']==lr}
            for offset,branch,color in [(-.16,'cfp','#087F8C'),(.16,'oct','#BC703A')]:
                for k,mode in enumerate(ORDER):
                    r=lookup.get(mode,{})
                    if 'delta_pp' not in r:continue
                    v=r['delta_pp'][branch];ax.barh(k+offset,v,height=.28,color=color,label=branch.upper() if k==0 else None)
                    ax.text(v+(.08 if v>=0 else -.08),k+offset,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
            for k,mode in enumerate(ORDER):
                if not lookup.get(mode,{}).get('complete'):ax.text(0,k,'Incomplete',fontsize=9,color='#667783')
            ax.set(yticks=np.arange(6),yticklabels=[ENGLISH[m] for m in ORDER],title=f'Seed {seed} · backbone LR={lr:g}',xlim=(lo,hi));ax.invert_yaxis();ax.axvline(0,color='#667783',linewidth=.8);ax.grid(axis='x',color='#E4E8EB');ax.set_axisbelow(True)
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
            if i==1:ax.set_xlabel('Change vs matched no bridge (pp)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.97,.925),frameon=False,ncol=2)
    fig.suptitle('R&B (Radon Bridge) | matched mechanism controls',x=.05,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.925,'stage3 · M=32 · S=64 · ρ=1/8 · head/bridge LR=1e-4',fontsize=11)
    fig.text(.05,.04,'*Pooling does not use M/S and retains a different width; it is not capacity matched.\nAll deltas use the same seed/LR no-bridge control. Development-set evidence; completed plateaus only.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.16,right=.96,bottom=.13,top=.86,wspace=.60,hspace=.26)
    for ext in ['png','pdf']:fig.savefig(out/('03_mechanisms.'+ext),dpi=180)
    plt.close(fig)
    lines=['# R&B（Radon Bridge）：32项完整对比',f"\n更新时间：{data['generated_at']}\n",
        '## 统一实验口径\n','两个backbone学习率3e-5、6e-5；head/bridge均1e-4；两个种子3416、3417；stage3、M32、S64。16项无桥/ρ扫描，加16项机制对照。ρ=1/2按用户要求排除。',
        '\n主矩阵只比较ρ=1/16、1/8、1/4。机制对照固定ρ=1/8，以同seed、同学习率的无桥与标准R&B匹配；池化不使用M/S，h不同，不属于严格容量匹配。',
        '\n1264训练、296开发验证；未读取测试。开发集历史参与过筛选，结果为探索性证据。全参数训练、BN正常更新、batch16；GPU时长不限，每卡项目显存≤10 GiB。至少8轮、连续6轮未超过0.001实质改善判定平台；3轮停滞LR乘0.3；60轮只是保护上限。共同检查点按两分支平均macro-F1选择，分别报告CFP/OCT。',
        '\n![绝对F1](01_rho_absolute.png)\n','![ρ增益](02_rho_gain.png)\n','![机制对照](03_mechanisms.png)\n','## 机制对照结果\n',
        '| seed | backbone LR | 方法 | 完整配置 | CFP F1 (%) | OCT F1 (%) | 平均 (%) | Δ平均 (pp) | 最佳/停止轮次 | 状态 |','|---:|---:|---|---|---:|---:|---:|---:|---|---|']
    for r in sorted(selected,key=lambda r:(r['seed'],r['backbone_lr'],ORDER.index(r['mode']))):
        d=r['summary'];score=r.get('scores',{});delta=r.get('delta_pp',{})
        config='—' if r['mode']=='independent' else 'stage3 · M=— · S=— · ρ=1/8' if r['mode']=='pooled' else 'stage3 · M=32 · S=64 · ρ=1/8'
        vals=[str(r['seed']),f"{r['backbone_lr']:g}",NAMES[r['mode']],config]+[f'{score[k]:.2f}' if k in score else '—' for k in ['cfp','oct','mean']]+[f"{delta['mean']:+.2f}" if delta else '—',f"{d['selection']['joint']['best_epoch']}/{d['epochs_ran']}" if r['complete'] else '—',r['state']]
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['\n## 机制配对bootstrap\n','10000次参与者级配对重采样；Δ定义为比较名称中前者减去后者，条件于已选检查点，不校正选择偏差。','| 比较 | 分支 | Δ (pp) | 95% CI (pp) |','|---|---|---:|---|']
    for name,branches in bootstrap.get('comparisons',{}).items():
        for branch,v in branches.items():lines.append(f"| {name} | {branch.upper()} | {v['delta_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] |")
    lines += ['\n## ρ扫描完整报告\n','[查看16项学习率/ρ扫描明细](RHO_REPORT.zh-CN.md)\n','## 机制完整指标及溯源\n']
    for r in rows:
        if r['mode'] in ['independent','radon']:continue
        lines += [f"### {r['trial']}\n",f"原始摘要：`{r['source']}`；SHA256：`{r['sha256']}`；实际h：`{r['h']}`。"]
        if r['complete']:
            d=r['summary'];c=r['cost'];lines += [f"参数量：{d['parameters']:,}；GPU分钟：{c['gpu_seconds']/60:.2f}；峰值进程显存：{c['sampled_peak_process_mib']} MiB。",'| 分支 | F1 (%) | Precision (%) | Recall (%) | Accuracy (%) | AUROC | 混淆矩阵 |','|---|---:|---:|---:|---:|---:|---|']
            for b,m in d['selected']['tasks'].items():lines.append(f"| {b.upper()} | {100*m['macro_f1']:.2f} | {100*m['macro_precision']:.2f} | {100*m['macro_recall']:.2f} | {100*m['accuracy']:.2f} | {m['auroc']:.4f} | {m['confusion_matrix']} |")
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines));shutil.copy2(prior/'report/REPORT.zh-CN.md',out/'RHO_REPORT.zh-CN.md')
    write(out/'report_status.json',{'state':'complete' if all(r['complete'] for r in rows) else 'partial','completed_trials':sum(r['complete'] for r in rows),'total_trials':len(rows),'generated_at':data['generated_at']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=Path(a.root);out=root/'report';out.mkdir(exist_ok=True)
    try:
        if a.wait:
            write(out/'report_status.json',{'state':'waiting_for_experiments'})
            while not (root/'status.json').exists() or read(root/'status.json')['state'] not in ['complete','failed','needs_attention','interrupted','budget_complete']:time.sleep(10)
        build(root)
    except Exception:
        write(out/'report_status.json',{'state':'failed','error':traceback.format_exc()});raise
