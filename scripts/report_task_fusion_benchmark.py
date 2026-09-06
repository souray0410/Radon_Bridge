"""Task-fusion statistics and public figures. Never export participant arrays."""
import csv, hashlib, json, platform
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import classify
from scripts.report_five_seed_study import f1
from scripts.queue_fixed_svd_study import read,sha
from scripts.run_five_seed_study import canonical
from scripts.run_task_fusion_benchmark import ARMS,SEEDS,LRS

NAMES={'concat_mlp':'Concat + MLP','gated_mil':'Gated attention MIL','svd_rho1_16':'R&B SVD | rho=1/16, r16, h512','svd_rho1_8':'R&B SVD | rho=1/8, r32, h1024','svd_rho1_4':'R&B SVD | rho=1/4, r64, h2048','mmtm_r4':'MMTM | reduction4','mmtm_r8':'MMTM | reduction8','attention_d128':'Cross-attention | d128','attention_d256':'Cross-attention | d256'}


def contrasts(rows):
    lookup={(r['seed'],r['backbone_lr'],r['arm']):i for i,r in enumerate(rows)}
    result=[]
    def add(lr,a,b,primary):
        w=np.zeros(len(rows))
        for s in SEEDS:w[lookup[s,lr,a]]+=1/3;w[lookup[s,lr,b]]-=1/3
        result.append(dict(id=f'bb{lr:.0e}_{a}_minus_{b}',backbone_lr=lr,positive=a,negative=b,primary=primary,weights=w))
    for lr in LRS:
        for a in ARMS:
            if a.startswith('svd_'):
                for b in ARMS:
                    if not b.startswith('svd_'):add(lr,a,b,True)
    for lr in LRS:add(lr,'gated_mil','concat_mlp',False)
    assert sum(d['primary'] for d in result)==36
    return result


def statistics(rows, probabilities, labels, resamples=10000):
    assert probabilities.shape==(54,4,len(labels),2)
    preds=probabilities.argmax(-1);point=100*f1(labels,preds)
    rng=np.random.default_rng(20260906);indices=rng.integers(0,len(labels),(resamples,len(labels)))
    samples=np.empty((resamples,54,4))
    for start in range(0,resamples,50):
        ix=indices[start:start+50];samples[start:start+len(ix)]=(100*f1(labels[ix],preds[:,:,ix])).transpose(2,0,1)
    defs=contrasts(rows);w=np.stack([d.pop('weights') for d in defs]);obs=w@point[:,2];draw=samples[:,:,2]@w.T
    sd=draw.std(0,ddof=1);active=sd[:36]>1e-10
    critical=float(np.quantile(np.abs((draw[:,:36][:,active]-obs[:36][active])/sd[:36][active]).max(1),.95)) if active.any() else None
    result=[]
    for i,d in enumerate(defs):
        simultaneous=[float(obs[i]-critical*sd[i]),float(obs[i]+critical*sd[i])] if d['primary'] and sd[i]>1e-10 and critical is not None else None
        auxiliary={}
        for j,name in enumerate(['cfp','oct','fusion','native_probability_average']):
            auxiliary[name]=dict(difference_pp=float(w[i]@point[:,j]),ci95_pp=np.quantile(samples[:,:,j]@w[i],[.025,.975]).tolist())
        result.append(dict(d,difference_pp=float(obs[i]),ci95_pp=np.quantile(draw[:,i],[.025,.975]).tolist(),simultaneous_ci95_pp=simultaneous,
                           zero_variance=bool(sd[i]<=1e-10),classification=classify(simultaneous) if simultaneous else 'unclassified',auxiliary=auxiliary))
    return dict(resamples=resamples,participants=len(labels),shared_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),primary_count=36,
                primary_metric='fusion macro-F1',max_abs_t_critical_95=critical,contrasts=result,zero_variance_tolerance_pp=1e-10,
                interpretation='Conditional on fitted models and selected development checkpoints. Seeds are not additional participants; development data were used historically for screening.')


def figures(out,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    files=[]
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig);files.append(name+'.png')
    for lr in LRS:
        fig,axes=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[1.45,1]})
        for i,a in enumerate(ARMS):
            selected=[r for r in rows if r['arm']==a and r['backbone_lr']==lr]
            v=np.array([r['fusion_f1_percent'] for r in selected]);col='#087f8c' if a.startswith('svd_') else '#745892' if a=='gated_mil' else '#59758b'
            axes[0].errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',color=col,capsize=4)
            axes[0].scatter(v,np.full(3,i),s=20,color=col,alpha=.4)
            for j,r in enumerate(selected):axes[1].scatter(r['parameters']/1e6,r['fusion_f1_percent'],c=col,marker=['o','s','^'][j],s=30)
        axes[0].set(yticks=range(9),yticklabels=[NAMES[a] for a in ARMS],xlabel='Fusion macro-F1 (%)',title='Mean, sample SD and all three seeds');axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.2)
        axes[1].set(xlabel='Total trainable parameters (millions)',ylabel='Fusion macro-F1 (%)',title='Performance and allocated capacity');axes[1].grid(alpha=.2)
        fig.suptitle(f'R&B | Task fusion benchmark | backbone LR={lr:.0e}',x=.04,ha='left',fontsize=19,fontweight='bold')
        fig.text(.04,.04,'Common fusion selection and CE_CFP + CE_OCT + CE_fusion. Historical development set; no test. Baseline rho is not applicable.',fontsize=10)
        fig.subplots_adjust(left=.27,right=.97,top=.85,bottom=.17,wspace=.35);save(fig,f'fusion_bb{lr:.0e}')
        for branch in ['cfp','oct','fusion','native_probability_average']:
            fig,ax=plt.subplots(figsize=(16,9));x=np.arange(9)
            for seed,col in zip(SEEDS,['#087f8c','#c9773e','#73589a']):
                v=[next(r for r in rows if r['seed']==seed and r['backbone_lr']==lr and r['arm']==a)[branch+'_f1_percent'] for a in ARMS]
                ax.scatter(x,v,c=col,label=f'Seed {seed}',s=50)
            ax.set(xticks=x,xticklabels=[NAMES[a].replace(' | ','\n') for a in ARMS],ylabel='Macro-F1 (%)',title=branch+' | all seeds (3418 shown explicitly)');ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2);ax.legend()
            fig.suptitle(f'R&B | Task fusion benchmark | backbone LR={lr:.0e}',x=.055,ha='left',fontsize=18,fontweight='bold')
            fig.text(.055,.03,'Native branch and probability-average metrics are secondary. All methods selected by the separately trained fusion prediction.',fontsize=10)
            fig.subplots_adjust(left=.09,right=.97,top=.85,bottom=.29);save(fig,f'{branch}_seeds_bb{lr:.0e}')
    for lr in LRS:
        part=[d for d in stats['contrasts'] if d['primary'] and d['backbone_lr']==lr]
        for rho in [16,8,4]:
            ds=[d for d in part if d['positive']==f'svd_rho1_{rho}'];fig,ax=plt.subplots(figsize=(16,9))
            for i,d in enumerate(ds):
                ax.hlines(i,*d['ci95_pp'],color='#afd7d5',lw=7)
                if d['simultaneous_ci95_pp']:ax.hlines(i,*d['simultaneous_ci95_pp'],color='#087f8c',lw=2)
                ax.scatter(d['difference_pp'],i,color='#173d4d')
            ax.axvspan(-1,1,color='#eef0f2');ax.axvline(0,color='#59636b',lw=1);ax.set(yticks=range(6),yticklabels=[NAMES[d['negative']] for d in ds],xlabel='R&B minus comparator: fusion macro-F1 (pp)');ax.invert_yaxis()
            fig.suptitle(f'R&B | rho=1/{rho} | backbone LR={lr:.0e}',x=.055,ha='left',fontsize=19,fontweight='bold')
            fig.text(.055,.04,'Thick: ordinary 95% CI. Thin: simultaneous 95% CI across all 36 primary contrasts. 10,000 shared participant resamples.',fontsize=10)
            fig.subplots_adjust(left=.3,right=.96,top=.85,bottom=.18);save(fig,f'contrasts_rho1_{rho}_bb{lr:.0e}')
    return files


def build(root,resamples=10000,plots=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);rows=[];probabilities=[];identity=None
    cost={}
    for path in root.glob('attempt_*/ledger.json'):
        for j in read(path)['jobs']:cost[str(path.parent/j['id'])]=j
    for r in read(root/'manifest.json')['rows']:
        path=Path(r['directory']);s=read(path/'summary.json');model=read(path/'model.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
        assert canonical(s['configuration'])==canonical(r['configuration'])
        assert all(sha(path/n)==h for n,h in r['accepted_hashes'].items())
        with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
            current=(z['ids'].copy(),z['y'].copy())
            if identity is None:identity=current
            else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
            p=np.stack([z['cfp'],z['oct'],z['fusion'],(z['cfp']+z['oct'])/2])
        assert p.shape==(4,296,2) and np.isfinite(p).all() and np.allclose(p.sum(-1),1)
        names=['cfp','oct','fusion','native_probability_average'];metrics={b:classification_metrics(identity[1],p[j]) for j,b in enumerate(names)}
        for j,b in enumerate(names):metrics[b]['brier']=float(np.mean((p[j,:,1]-identity[1])**2))
        for b in names[:3]:assert abs(metrics[b]['macro_f1']-s['selected']['tasks'][b]['macro_f1'])<1e-12
        rho=(1/int(r['arm'].rsplit('_',1)[1])) if r['arm'].startswith('svd_') else None
        rows.append(dict(id=r['id'],seed=r['seed'],backbone_lr=r['backbone_lr'],arm=r['arm'],rho=rho,r=int(256*rho) if rho else None,h=int(8192*rho) if rho else None,
                         **{b+'_f1_percent':100*metrics[b]['macro_f1'] for b in names},mean_native_f1_percent=50*(metrics['cfp']['macro_f1']+metrics['oct']['macro_f1']),
                         parameters=s['parameters'],fusion_parameters=model['fusion_parameters'],communication_parameters=model['communication_parameters'],
                         best_epoch=s['selection']['joint']['best_epoch'],stop_epoch=s['epochs_ran'],gpu_seconds=cost[str(path)]['gpu_seconds'],peak_process_mib=cost[str(path)]['sampled_peak_process_mib'],
                         metrics=metrics,configuration=r['configuration'],accepted_hashes=r['accepted_hashes']))
        probabilities.append(p)
    assert len(rows)==54
    stats=statistics(rows,np.stack(probabilities),identity[1],resamples)
    write_json(out/'statistics.json',stats)
    aggregate=[]
    for lr in LRS:
        for a in ARMS:
            subset=[r for r in rows if r['backbone_lr']==lr and r['arm']==a]
            aggregate.append(dict(arm=a,backbone_lr=lr,**{b:dict(mean=float(np.mean([r[b] for r in subset])),sample_sd=float(np.std([r[b] for r in subset],ddof=1))) for b in ['cfp_f1_percent','oct_f1_percent','fusion_f1_percent','mean_native_f1_percent','native_probability_average_f1_percent']}))
    files=figures(out,rows,stats) if plots else []
    write_json(out/'results.json',dict(rows=rows,aggregate=aggregate,figures=files,training_jobs=54,test_used=False,protocol=read(root/'protocol.json'),accounting=read(root/'study_summary.json'),latency_profiles={k:v.get('latency') for k,v in read(root/'gpu_acceptance.json')['profiles'].items()},environment=dict(python=platform.python_version(),numpy=np.__version__)))
    fields=[k for k in rows[0] if k not in ['metrics','configuration','accepted_hashes']]
    with (out/'results_54.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    with (out/'paired_comparisons.csv').open('w',newline='') as f:
        fs=['id','backbone_lr','positive','negative','primary','difference_pp','ci95_pp','simultaneous_ci95_pp','zero_variance','classification'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows({k:r[k] for k in fs} for r in stats['contrasts'])
    lines=['# R&B：任务融合基准结果','','54项均按融合预测macro-F1选优并达到平台。全部结果来自296人开发验证集，未读test。不能与原双头损失/选优研究混为同一排名。','','| 主干LR | 方法 | 融合F1，均值 ± 种子SD (%) |','|---|---|---:|']
    for r in aggregate:
        v=r['fusion_f1_percent'];lines.append(f"| {r['backbone_lr']:.0e} | {NAMES[r['arm']]} | {v['mean']:.2f} ± {v['sample_sd']:.2f} |")
    lines+=['','## 36项预定比较','普通与36比较max-|t|同时区间见statistics.json和paired_comparisons.csv。±1pp仅为方法研究参考。','']
    for d in stats['contrasts'][:36]:lines.append(f"- {d['id']}: {d['difference_pp']:+.2f} pp；同时区间 {d['simultaneous_ci95_pp']}；{d['classification']}。")
    lines+=['','## 解释边界','同一原生网络、输入、三项CE、两档学习率、三种子和选优标准；参数量、家族配置数、历史筛选预算并不相等。MIL为深层空间token的门控汇聚，不是原论文2D B-scan系统完整复现。R&B在此配通用融合头；原独立预测器机制证据另报告。','本实验不能保证方法优势，也不能代替独立测试和外部验证。包含所有负结果、3418单列和逐种子波动；不按新结果筛选展示配置。']
    (out/'INTERPRETATION.zh-CN.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'artifact_manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'})
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
