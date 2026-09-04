"""Traceable fixed-rank SVD/learned paired report; never promote incomplete trials."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def fmt(v, signed=False):return '—' if v is None else format(v,'+.2f' if signed else '.2f')
def ratio(r):return {0.0625:'1/16',0.125:'1/8',0.25:'1/4'}[r]
BRANCHES=['cfp','oct','mean'];COLORS={'cfp':'#087F8C','oct':'#BC703A','mean':'#626C91'}


def trial(path,cost):
    d=read(path/'summary.json') if (path/'summary.json').exists() else {}
    failure=read(path/'failure.json') if (path/'failure.json').exists() else {}
    complete=d.get('state')=='complete' and d.get('converged_by_policy') is True and d.get('stop_reason')=='validation_plateau'
    r={'path':str(path),'summary_sha256':sha(path/'summary.json') if d else None,'complete':complete,
       'state':'complete' if complete else failure.get('state',d.get('state','not_completed')),'summary':d,'cost':cost}
    if (path/'model.json').exists():r['model']=read(path/'model.json')
    if (path/'basis_manifest.json').exists():r['bases']=read(path/'basis_manifest.json')
    if complete:
        assert d['test_used'] is False
        m=d['selected']['tasks'];scores={b:100*m[b]['macro_f1'] for b in ['cfp','oct']};scores['mean']=(scores['cfp']+scores['oct'])/2
        for metrics in m.values():
            cm=np.asarray(metrics['confusion_matrix']);den=cm.sum(0)+cm.sum(1)
            value=np.divide(2*np.diag(cm),den,out=np.zeros(len(cm)),where=den!=0).mean()
            assert abs(value-metrics['macro_f1'])<1e-12
        r.update(scores=scores,best_epoch=d['selection']['joint']['best_epoch'],stop_epoch=d['epochs_ran'])
    return r


def figures(out,rows,bootstrap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    settings=[(s,lr) for s in [3416,3417] for lr in [3e-5,6e-5]];rhos=[.0625,.125,.25]
    fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True,sharey='row')
    for col,(seed,lr) in enumerate(settings):
        by={r['rho']:r for r in rows if r['seed']==seed and r['backbone_lr']==lr}
        for row,b in enumerate(BRANCHES):
            ax=axes[row,col];color=COLORS[b]
            baseline=next((r['baseline']['scores'][b] for r in by.values() if r['baseline']['complete']),None)
            if baseline is not None:ax.axhline(baseline,color='#9099A0',linestyle=':',linewidth=1.2)
            for method,offset,marker,style in [('learned',-.06,'o','--'),('fixed',.06,'s','-')]:
                values=[by[r][method].get('scores',{}).get(b,np.nan) for r in rhos]
                ax.plot(np.arange(3)+offset,values,color=color,marker=marker,linestyle=style,markersize=6,linewidth=1.5,alpha=.70 if method=='learned' else 1.)
                for x,v in enumerate(values):
                    if np.isfinite(v):ax.annotate(f'{v:.2f}',(x+offset,v),xytext=(0,8 if method=='fixed' else -14),textcoords='offset points',ha='center',fontsize=8)
            if row==0:ax.set_title(f'Seed {seed} | backbone LR={lr:g}',pad=12,fontsize=10)
            if col==0:ax.set_ylabel(('Mean' if b=='mean' else b.upper())+' macro-F1 (%)')
            ax.set_xticks(range(3),['ρ=1/16','ρ=1/8','ρ=1/4']);ax.set_xlim(-.3,2.3);ax.margins(y=.24);ax.grid(axis='y',color='#E5E9EC');ax.set_axisbelow(True)
            if any(not r['fixed']['complete'] for r in by.values()):ax.text(.02,.02,'Incomplete fixed trials omitted',transform=ax.transAxes,fontsize=8,color='#697681')
    handles=[Line2D([],[],color='#344653',marker='o',linestyle='--',label='Learned CM projection'),Line2D([],[],color='#344653',marker='s',label='Fixed SVD channel projection'),Line2D([],[],color='#9099A0',linestyle=':',label='Matched no bridge')]
    fig.suptitle('R&B (Radon Bridge) | matched compression comparison',x=.055,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.935,'stage3 · M=32 · S=64 · head/bridge LR=1e-4 · equal mixer widths h=512 / 1024 / 2048',fontsize=11)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.918),ncol=3,frameon=False)
    fig.text(.055,.025,'Development-set exploration. Fixed rank is set by ρ; retained energy is measured separately. Only validation plateaus count as complete.',fontsize=10,color='#52606C')
    fig.subplots_adjust(left=.07,right=.98,top=.855,bottom=.085,hspace=.26,wspace=.15)
    for ext in ['png','pdf']:fig.savefig(out/('01_compression_f1.'+ext),dpi=180)
    plt.close(fig)
    for suffix,title in [('minus_learned','Fixed SVD minus learned CM'),('minus_no_bridge','Fixed SVD minus matched no bridge')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True);extents=[]
        for ax,(seed,lr) in zip(axes.flat,settings):
            by={r['rho']:r for r in rows if r['seed']==seed and r['backbone_lr']==lr}
            for j,b in enumerate(BRANCHES):
                for i,rho in enumerate(rhos):
                    r=by[rho];v=bootstrap.get('comparisons',{}).get(r['fixed_id']+'_'+suffix,{}).get(b)
                    if v is None:continue
                    lower,upper=v['ci95_pp'];delta=v['delta_pp'];extents.extend([lower,upper,delta])
                    x=i+(j-1)*.18
                    ax.vlines(x,lower,upper,color=COLORS[b],linewidth=1.4);ax.hlines([lower,upper],x-.035,x+.035,color=COLORS[b],linewidth=1.2)
                    ax.scatter([x],[delta],color=COLORS[b],s=34,zorder=3)
            ax.axhline(0,color='#75818B',linewidth=1);ax.set_xticks(range(3),['ρ=1/16','ρ=1/8','ρ=1/4']);ax.grid(axis='y',color='#E5E9EC');ax.set_axisbelow(True)
            ax.set_title(f'Seed {seed} | backbone LR={lr:g}');ax.set_ylabel('Macro-F1 change (pp)')
        lim=max([1]+[abs(x) for x in extents])*1.12
        axes[0,0].set_ylim(-lim,lim)
        fig.suptitle('R&B (Radon Bridge) | '+title,x=.075,ha='left',fontsize=17,fontweight='bold')
        fig.text(.075,.926,'stage3 · M=32 · S=64 · head/bridge LR=1e-4 · 10,000 paired participant bootstrap resamples',fontsize=10)
        fig.legend(handles=[Line2D([],[],color=COLORS[b],marker='o',label=b.upper() if b!='mean' else 'Mean') for b in BRANCHES],loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
        fig.text(.075,.035,'Points: observed changes. Lines: 95% percentile intervals, conditional on selected checkpoints.\nNo correction for development-set selection. Two seeds are reported separately.',fontsize=10,color='#52606C')
        fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.13,hspace=.28,wspace=.23)
        name='02_fixed_minus_learned' if suffix=='minus_learned' else '03_fixed_minus_no_bridge'
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)


def build(root):
    p=read(root/'protocol.json');out=root/'report';out.mkdir(exist_ok=True)
    original=Path(p['learned_reference_root']);oldcost={j['id']:j for j in read(original/'ledger.json')['jobs']}
    costs={j['id']:j for j in read(root/'ledger.json').get('jobs',[])} if (root/'ledger.json').exists() else {}
    configs={j['id']:j['config'] for g in p['groups'] for j in g['jobs']};rows=[]
    for item in p['paired_comparisons']:
        cfg=configs[item['fixed']];r={'fixed_id':item['fixed'],'learned_id':item['learned'],'baseline_id':item['baseline'],
            'seed':cfg['seed'],'backbone_lr':cfg['backbone_lr'],'rho':cfg['bridges'][0]['rho'],'configuration':cfg}
        for method in ['fixed','learned','baseline']:
            name=item[method];path=root/name if method=='fixed' else Path(p['references'][name]['directory'])
            if method!='fixed':
                ref=p['references'][name];assert sha(path/'summary.json')==ref['summary_sha256']
                assert sha(path/'selected_predictions.npz')==ref['predictions_sha256']
            r[method]=trial(path,costs.get(name) if method=='fixed' else oldcost.get(name))
        assert r['learned']['complete'] and r['baseline']['complete']
        if r['fixed']['complete']:
            ds=[r[k]['summary'] for k in ['fixed','learned','baseline']]
            assert len({d['initial_native_sha256'] for d in ds})==1
            assert all(d['parent_checkpoints']==ds[0]['parent_checkpoints'] for d in ds)
            assert r['fixed']['bases']
            r['fixed_minus_learned_pp']={b:r['fixed']['scores'][b]-r['learned']['scores'][b] for b in BRANCHES}
        for method in ['fixed','learned']:
            if r[method]['complete']:r[method]['minus_no_bridge_pp']={b:r[method]['scores'][b]-r['baseline']['scores'][b] for b in BRANCHES}
        rows.append(r)
    assert len(rows)==12
    rows.sort(key=lambda r:(r['seed'],r['backbone_lr'],r['rho']))
    boot=read(root/'bootstrap.json') if (root/'bootstrap.json').exists() else {}
    for r in rows:
        if r['fixed']['complete']:
            for suffix,key in [('minus_learned','fixed_minus_learned_pp'),('minus_no_bridge',None)]:
                expected=r[key] if key else r['fixed']['minus_no_bridge_pp']
                observed=boot.get('comparisons',{}).get(r['fixed_id']+'_'+suffix,{})
                assert all(b in observed and abs(observed[b]['delta_pp']-expected[b])<1e-9 for b in BRANCHES),'Missing or mismatched bootstrap'
    acceptance=read(root/'gpu_acceptance.json');assert sha(root/'gpu_acceptance.json')==p['gpu_acceptance_sha256']
    basis_rows=[]
    for seed,files in p['basis_files'].items():
        for source,artifact in files.items():
            assert sha(artifact['path'])==artifact['sha256']
            with np.load(artifact['path'],allow_pickle=False) as z:
                values=z['eigenvalues'];metadata=json.loads(str(z['metadata']))
            for rho in [.0625,.125,.25]:
                rank=max(1,int(rho*metadata['channels']))
                basis_rows.append({'seed':int(seed),'source':source,'rho':rho,'rank':rank,'h':rank*32,'retained_energy_ratio':float(values[:rank].sum()/values.sum()),'artifact':artifact,'metadata':metadata})
    data={'project':'R&B (Radon Bridge)','generated_at':datetime.now().astimezone().isoformat(),'rows':rows,'basis_rows':basis_rows,'bootstrap':boot,'protocol':p,'gpu_acceptance':acceptance,'status':read(root/'status.json'),'test_used':False}
    write(out/'results.json',data);figures(out,rows,boot)
    lines=['# R&B（Radon Bridge）：固定SVD通道压缩配对报告',f"\n生成时间：{data['generated_at']}\n",
        '本报告比较12项固定SVD通道投影与12项已完成可学习CM投影，复用4项匹配无桥。固定M32、S64、stage3；backbone LR为3e-5/6e-5，head/bridge均1e-4。ρ统一为1/16、1/8、1/4，对应r16/32/64、每路h512/1024/2048。ρ控制维数，不代表能量保留率。',
        '\n1264训练、296开发验证，不使用测试。独立最佳检查点起步、batch16、AdamW WD0.01、分支/桥分别裁剪5、所有原生参数训练且BN更新。至少8轮，连续6轮未超过0.001实质改善判定平台；3轮停滞LR乘0.3；60轮仅保护上限。共同检查点按两分支平均macro-F1选择。',
        '\n固定基由原独立检查点的训练集native stage3特征拟合，eval/no_grad，非中心化FFᵀ/N特征分解。每个种子每个来源拟合一次，按能量降序并规范列符号，跨LR共用、跨ρ嵌套；联合训练不再更新Q。桥内只有中间kernel3无偏置零初始化卷积可学习。',
        '\n![绝对F1](01_compression_f1.png)\n','![固定减可学习](02_fixed_minus_learned.png)\n','![固定减无桥](03_fixed_minus_no_bridge.png)\n',
        '## 完整配对结果\n','数值为百分数；Δ单位pp。缺失或未到平台的数据以—表示，不算完整实验。',
        '| seed | backbone LR | ρ | 压缩 | CFP F1 | OCT F1 | 平均F1 | ΔCFP/无桥 | ΔOCT/无桥 | Δ平均/无桥 | 最佳/停止轮次 | 状态 |',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        for method,label in [('learned','可学习CM投影'),('fixed','固定SVD通道投影')]:
            t=r[method];s=t.get('scores',{});delta=t.get('minus_no_bridge_pp',{})
            vals=[str(r['seed']),f"{r['backbone_lr']:g}",ratio(r['rho']),label]+[fmt(s.get(b)) for b in BRANCHES]+[fmt(delta.get(b),True) for b in BRANCHES]+[f"{t['best_epoch']}/{t['stop_epoch']}" if t['complete'] else '—',t['state']]
            lines.append('| '+' | '.join(vals)+' |')
    lines+=['\n## 固定减可学习与参与者配对区间\n','10000次参与者级配对bootstrap；同一次重采样用于两个分支及均值。95%百分位区间条件于已选择检查点，不校正开发集选择偏差。两个种子分别报告。','| seed | backbone LR | ρ | 分支 | 固定−可学习 (pp) | 95% CI | 固定−无桥 (pp) | 95% CI |','|---:|---:|---|---|---:|---|---:|---|']
    for r in rows:
        for b in BRANCHES:
            vals=[str(r['seed']),f"{r['backbone_lr']:g}",ratio(r['rho']),b.upper()]
            for suffix in ['minus_learned','minus_no_bridge']:
                v=boot.get('comparisons',{}).get(r['fixed_id']+'_'+suffix,{}).get(b)
                vals += [fmt(v['delta_pp'],True),f"[{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]"] if v else ['—','—']
            lines.append('| '+' | '.join(vals)+' |')
    lines+=['\n## 实际能量保留率\n','能量针对拟合时的native stage3训练特征；并非判别信息比例、不是Radon域能量，且不保证训练后仍最优。','| seed | 来源 | ρ | r | h | 保留能量 (%) |','|---:|---|---|---:|---:|---:|']
    for b in basis_rows:lines.append(f"| {b['seed']} | {b['source']} | {ratio(b['rho'])} | {b['rank']} | {b['h']} | {100*b['retained_energy_ratio']:.3f} |")
    lines+=['\n## 参数、时间和显存\n','总参数指可学习参数；Q为buffer，基和运行buffer文件另行保存。GPU耗时用控制器占用时长，进程峰值为采样值，PyTorch保留峰值另列。拟合及预检成本只记一次，不重复计入12项。','| trial | 参数量 | 桥参数量 | GPU分钟 | 进程峰值MiB | PyTorch保留峰值MiB |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for method in ['learned','fixed']:
            t=r[method];d=t['summary'];c=t['cost'] or {};base_params=r['baseline']['summary']['trainable_parameters']
            total=d.get('trainable_parameters',t.get('model',{}).get('trainable_parameters'))
            lines.append('| '+' | '.join([r[method+'_id'],str(total) if total is not None else '—',str(total-base_params) if total is not None else '—',fmt(c['gpu_seconds']/60 if 'gpu_seconds' in c else None),str(c.get('sampled_peak_process_mib','—')),fmt(d.get('peak_reserved_mib'))])+' |')
    lines+=['\n### 一次性拟合与预检\n','| 工作 | GPU分钟 | 进程峰值MiB |','|---|---:|---:|']
    for j in acceptance['ledger']['jobs']:lines.append(f"| {j['id']} | {j['gpu_seconds']/60:.3f} | {j['sampled_peak_process_mib']} |")
    lines+=['\n## 解释边界\n','本比较同时改变两端可学习性及投影结构，不能把差异全部归因于是否训练。两种子的波动包含对应预训练特征/SVD基差异和联合训练随机性，不存在另加的随机QR基。保留高能量方向不保证保留低能量判别信息；相关特征可能传递部分信息，效果由本配对实验检验。全部结果属于历史参与筛选的开发集探索证据。',
        '\n## 溯源\n',f"源提交：`{p['source_commit']}`；GPU验收SHA256：`{p['gpu_acceptance_sha256']}`。完整配置、父检查点摘要、每个结果摘要SHA、基文件及生成规则见[results.json](results.json)。"]
    seen=set()
    for b in basis_rows:
        a=b['artifact']
        if a['path'] in seen:continue
        seen.add(a['path']);lines.append(f"\n- seed{b['seed']} / {b['source']}：`{a['path']}`；SHA256 `{a['sha256']}`。")
    previous=Path(p['predecessor'])/'report'
    if (previous/'report_status.json').exists() and read(previous/'report_status.json')['state']=='complete':
        shutil.copytree(previous,out/'previous32',dirs_exist_ok=True)
        lines+=['\n## 前轮32项完整对比\n','[学习率/ρ扫描及随机、打乱、自身、池化对照](previous32/REPORT.zh-CN.md)。这些机制对照属于可学习版本，本轮没有新增固定版机制对照。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write(out/'report_status.json',{'state':'complete' if all(r['fixed']['complete'] for r in rows) else 'partial','completed_fixed_trials':sum(r['fixed']['complete'] for r in rows),'total_fixed_trials':12,'generated_at':data['generated_at']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=Path(a.root);out=root/'report';out.mkdir(exist_ok=True)
    try:
        if a.wait:
            write(out/'report_status.json',{'state':'waiting_for_experiments'})
            terminal=['complete','failed','needs_attention','interrupted','budget_complete']
            while not (root/'status.json').exists() or read(root/'status.json')['state'] not in terminal:time.sleep(10)
            # The controller writes final summaries in its finally block after terminal status.
            while True:
                pid=read(root/'status.json')['controller_pid']
                try:
                    import os
                    os.kill(pid,0)
                except ProcessLookupError:break
                time.sleep(2)
        build(root)
    except Exception:
        write(out/'report_status.json',{'state':'failed','error':traceback.format_exc()});raise
