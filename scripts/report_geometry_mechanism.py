"""Complete development evidence; fixed contrasts and shared participant resampling."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radonbridge.artifacts import resolve,sha256,ARCHIVE,SOURCE,STUDY
from radonbridge.geometry_study import MECHANISM_POINTS,HS,KS,SEEDS,LRS,PROTOCOLS,FIXED_HOSTS,baseline_structures
from radonbridge.metrics import classification_metrics
from scripts.geometry_evidence import read,write
from scripts.report_five_seed_study import f1


def definitions(rows,cat):
    """Predefined structural comparisons, independent of performance."""
    defs=[]
    lookup={(r['protocol'],r['structure']['id'],r['seed'],r['backbone_lr']):i for i,r in enumerate(rows)}
    def add(name,protocol,pairs,family):
        w=np.zeros(len(rows));missing=[]
        for a,b in pairs:
            for seed in SEEDS:
                for lr in LRS:
                    for sid,sign in ((a,1),(b,-1)):
                        idx=lookup.get((protocol,sid,seed,lr))
                        if idx is None or rows[idx]['state']!='accepted':missing.append((sid,seed,lr))
                        elif pairs:w[idx]+=sign/(len(pairs)*len(SEEDS)*len(LRS))
        defs.append(dict(id=name,protocol=protocol,family=family,pairs=pairs,
                         estimable=bool(pairs) and not missing,missing=missing,weights=w))
    sid=lambda mode,M,S,k,r:f'{mode}_M{M}_S{S}_k{k}_r{r}'
    protocols=cat.get('protocols',PROTOCOLS)
    direct_family='direct56' if len(protocols)==2 else 'direct28'
    augmentation_family='augmentation8' if len(protocols)==2 else 'augmentation4'
    for p in protocols:
        for h in HS:
            for k in KS:
                pairs=[(sid('radon',M,S,k,h//M),sid('linear_resample',M,S,k,h//M)) for M,S,kk in MECHANISM_POINTS if kk==k]
                add(f'{p}_equal_parameters_h{h}_k{k}',p,pairs,direct_family)
        for h in HS:
            cells=[c for c in cat['compute_cells'] if c['budget_h']==h]
            pairs=[(c['radon_id'],c['compute_ordinary_id']) for c in cells if 'compute_ordinary_id' in c]
            add(f'{p}_equal_compute_budget{h}',p,pairs,direct_family)
            for ka,kb in ((3,1),(5,3)):
                a=next(c for c in cells if (c['M'],c['S'],c['k'])==(32,64,ka))
                b=next(c for c in cells if (c['M'],c['S'],c['k'])==(32,64,kb))
                pairs=[(a['radon_id'],b['radon_id'])] if 'radon_id' in a and 'radon_id' in b else []
                add(f'{p}_equal_compute_budget{h}_k{ka}_minus_k{kb}',p,pairs,direct_family)
        for h in HS:
            for baseline in baseline_structures()[1:]:
                add(f'{p}_reference_h{h}_minus_{baseline["id"]}',p,[(sid('radon',32,64,3,h//32),baseline['id'])],direct_family)
        for host in FIXED_HOSTS:
            for arm in ('continue','linear_resample'):
                add(f'{p}_{host}_augmentation_radon_minus_{arm}',p,[(host+'_plus_radon',host+'_plus_'+arm)],augmentation_family)
    assert sum(x['family']==direct_family for x in defs)==28*len(protocols)
    assert sum(x['family']==augmentation_family for x in defs)==4*len(protocols)
    return defs


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def communication_parameter_count(model):
    """Read both metadata generations without editing historical evidence."""
    groups = model['groups']
    counts = []
    for g in groups:
        if 'stored_bridge_parameters' in g:
            counts.append(g['stored_bridge_parameters'])
        else:
            # Legacy fixed-SVD bridges contain only a bias-free dense mixer.
            assert g['compression'] == 'fixed_svd_channel'
            width = sum(p['retained_channels'] for p in g['participants'])
            counts.append(g.get('kernel_size', 3) * width ** 2)
    derived = sum(counts)
    if 'communication_parameters' in model:
        assert model['communication_parameters'] == derived, 'Inconsistent communication parameter counts'
        return model['communication_parameters'], 'model.communication_parameters; checked against groups'
    return derived, 'legacy groups; fixed-SVD fallback k * sum(h_i)^2, historical default k=3'


def build(root):
    root=Path(root);manifest=read(root/'manifest.json');rows=manifest['rows'];cat=manifest['catalog']
    out=root/'report';out.mkdir(exist_ok=True)
    defs=definitions(rows,cat);ids=y=None;probs=[];tables=[];model_metadata=[]
    # Outcomes CFP, OCT, learned fusion (or fixed average placeholder), late average.
    for r in rows:
        item={k:r.get(k) for k in ('id','category','protocol','seed','backbone_lr','state','epochs','best_epoch','seconds','parameters','peak_reserved_mib','reused')}
        item.update({k:r['structure'].get(k) for k in ('id','M','S','k','r','h','rho','mode','family','hidden','attention_dimension') if k!='id'})
        item['structure_id']=r['structure']['id']
        if r['state']!='accepted':probs.append(None);tables.append(item);continue
        directory=resolve(r['directory']);assert sha256(directory/'selected_predictions.npz')==r['accepted_hashes']['selected_predictions.npz']
        with np.load(directory/'selected_predictions.npz',allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy();y=z['y'].copy()
            assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y']),'Participant order must match across all models'
            late=(z['cfp']+z['oct'])/2
            probs.append(np.stack([z['cfp'],z['oct'],z['fusion'] if 'fusion' in z else late,late]))
            for key,p in [('cfp',z['cfp']),('oct',z['oct']),('late_fusion',late)]+([('learned_fusion',z['fusion'])] if 'fusion' in z else []):
                metrics=classification_metrics(y,p);metrics['brier_binary']=float(np.mean((p[:,1]-y)**2))
                for metric,value in metrics.items():item[key+'_'+metric]=value
            item['branch_mean_macro_f1']=(item['cfp_macro_f1']+item['oct_macro_f1'])/2
            item['primary_macro_f1']=item['learned_fusion_macro_f1'] if r['protocol']=='fusion' else item['branch_mean_macro_f1']
        if (directory/'model.json').exists():
            model=read(directory/'model.json')
            count, provenance = communication_parameter_count(model)
            model_metadata.append(dict(id=r['id'],parameters=model['parameters'],communication_parameters=count,communication_parameters_source=provenance,
                trainable_parameters=model['trainable_parameters'],groups=model['groups']))
        tables.append(item)
    assert ids is not None and len(ids)==296
    p=np.stack([x if x is not None else np.full((4,296,2),.5) for x in probs]);pred=p.argmax(-1)
    point=100*f1(y,pred)
    primary=np.array([(v[0]+v[1])/2 if r['protocol']=='branch' else v[2] for r,v in zip(rows,point)])
    rng=np.random.default_rng(20260906);ix=rng.integers(0,296,size=(10000,296))
    boot=np.empty((10000,len(rows),4))
    for start in range(0,10000,20):
        indices=ix[start:start+20];boot[start:start+len(indices)]=(100*f1(y[indices],pred[:,:,indices])).transpose(2,0,1)
    bprimary=np.stack([(boot[:,i,0]+boot[:,i,1])/2 if r['protocol']=='branch' else boot[:,i,2] for i,r in enumerate(rows)],axis=1)
    results=[]
    from radonbridge.benchmark_statistics import classify
    for family in dict.fromkeys(d['family'] for d in defs):
        valid=[d for d in defs if d['family']==family and d['estimable']]
        if valid:
            weights=np.stack([d['weights'] for d in valid]);observed=weights@primary;draws=bprimary@weights.T
            sd=draws.std(0,ddof=1);active=sd>1e-10
            critical=float(np.quantile(np.max(np.abs((draws[:,active]-observed[None,active])/sd[None,active]),axis=1),.95)) if active.any() else None
            for j,d in enumerate(valid):
                ci=[float(observed[j]-critical*sd[j]),float(observed[j]+critical*sd[j])] if active[j] else None
                results.append({k:v for k,v in d.items() if k!='weights'}|dict(difference_pp=float(observed[j]),ci95_pp=np.quantile(draws[:,j],[.025,.975]).tolist(),
                    simultaneous_ci95_pp=ci,bootstrap_sd_pp=float(sd[j]),max_abs_t_critical=critical,
                    classification=classify(ci) if ci else 'undefined_zero_variance'))
        for d in defs:
            if d['family']==family and not d['estimable']:results.append({k:v for k,v in d.items() if k!='weights'}|dict(classification='not_estimable'))
    csv_write(out/'all_results.csv',tables);csv_write(out/'primary_contrasts.csv',results)
    write(out/'statistics.json',dict(resamples=10000,resampling_unit='participant',shared_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
          participants=296,contrasts=results,test_used=False,zero_sd_tolerance_pp=1e-10,
          interpretation='Conditional on fitted models and development-selected checkpoints; no independent test evidence'))
    write(out/'model_costs.json',model_metadata)
    aggregates=[]
    for protocol in cat.get('protocols',PROTOCOLS):
        for sid in dict.fromkeys(r['structure']['id'] for r in rows):
            for lr in LRS:
                group=[t for t in tables if t['protocol']==protocol and t['structure_id']==sid and t['backbone_lr']==lr and t['state']=='accepted']
                if not group:continue
                for metric in ('cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1','primary_macro_f1','late_fusion_macro_f1'):
                    values=np.array([t[metric] for t in group])*100
                    aggregates.append(dict(protocol=protocol,structure=sid,backbone_lr=lr,metric=metric,
                        count=len(group),mean_percent=float(values.mean()),sample_sd_pp=float(values.std(ddof=1)) if len(values)>1 else None))
    csv_write(out/'seed_aggregates.csv',aggregates)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for protocol in cat.get('protocols',PROTOCOLS):
        fig,axes=plt.subplots(1,3,figsize=(16,8))
        for ax,axis,levels in zip(axes,('M','S','k'),((16,32,64),(32,64,128),(1,3,5))):
            for h in HS:
                for mode,color in (('radon','#007e87'),('linear_resample','#b87441')):
                    mean=[];sd=[]
                    for value in levels:
                        coordinates={'M':32,'S':64,'k':3};coordinates[axis]=value
                        group=[t for t in tables if t['protocol']==protocol and t['state']=='accepted' and t.get('mode')==mode and t.get('h')==h and all(t.get(k)==v for k,v in coordinates.items())]
                        v=np.array([t['primary_macro_f1'] for t in group])*100
                        mean.append(v.mean() if len(v)==6 else np.nan);sd.append(v.std(ddof=1) if len(v)==6 else np.nan)
                    ax.errorbar(range(3),mean,yerr=sd,color=color,linestyle='-' if h==512 else '--',marker='o',capsize=3,label=f'{"R&B" if mode=="radon" else "Linear resampling"}, h={h}')
            ax.set(xticks=range(3),xticklabels=levels,xlabel=axis,ylabel='Primary macro-F1 (%)');ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4)
        fig.suptitle('R&B (Radon Bridge) | '+protocol+' protocol | Prespecified geometry contrasts')
        fig.text(.05,.07,'Error bars: SD over 3 seeds × 2 learning rates. Development evidence; M changes channel/direction allocation at fixed h.',fontsize=10)
        fig.subplots_adjust(bottom=.19,top=.9,wspace=.3)
        for ext in ('png','pdf','svg'):fig.savefig(out/f'{protocol}_geometry.{ext}',dpi=180)
        plt.close(fig)
    ledger=[]
    archive_batches=ARCHIVE/'current'/root.relative_to(SOURCE/'runs'/STUDY)/'batches'
    for base in (root/'batches',archive_batches):
        if base.exists():
            for path in base.glob('*/ledger.json'):ledger.extend(read(path)['jobs'])
    write(out/'accounting.json',dict(gpu_minutes=sum(x['gpu_seconds'] for x in ledger)/60,entries=ledger,
          includes_training_fitting_preflight_diagnostics=True,
          scope='This execution directory only; inherited evidence is accounted separately',
          inherited_source_manifest=(str(root/'source_snapshot/manifest.json') if (root/'source_snapshot/manifest.json').exists() else None),
          inherited_training_gpu_minutes=(sum(r['seconds'] for r in rows if r['category']=='direct' and r['state']=='accepted')/60 if cat.get('protocols')==['branch'] else None),
          inherited_shared_preflight_and_diagnostic_accounting='See source snapshot and original full ledgers; do not charge withdrawn fusion training to branch-only execution'))
    report=['# R&B（Radon Bridge）：机制基准结果','',
       f'共{len(rows)}个结果位置，已验收{sum(r["state"]=="accepted" for r in rows)}项；其余技术不可行单列。',
       ('仅分支协议；用户在部分融合结果已揭示后撤销融合协议，其结果保留于历史记录。28项直接与4项增强比较沿用原分支对比定义，比较族缩小属于事后范围修订，不是原56/8项校正或确认性证据。' if cat.get('protocols')==['branch'] else '分支与融合协议分别报告；')+'图中误差条是种子和学习率共同波动，不能当作泛化误差。',
       '全部结果来自296人开发集，参与过历史任务筛选。尚未读取独立test，不能据此宣称临床有效或普遍优于其他方法。','',
       '## 主张与反例','',
       '|预定比较|差值 pp|同时95%区间|判定|','|---|---:|---|---|']
    for d in results:report.append(f'|{d["id"]}|{d.get("difference_pp","—")}|{d.get("simultaneous_ci95_pp","—")}|{d["classification"]}|')
    report+=['','完整逐种子、学习率、两分支及晚期融合见 all_results.csv；模型资源见 model_costs.json 和结构预检。',
       '未来独立验证需扩大队列、审计标签与抽样、跨任务和模型；本轮不确定的结果如实保留，不根据排名继续搜索。',
       '临床表型不等于新增影像专家金标准；病例对照平衡队列的预测值与校准不得直接推广到真实患病率。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(report)+'\n')
    write(root/'study_summary.json',dict(state='complete',report_directory=str(out),accepted=sum(r['state']=='accepted' for r in rows),
        total_positions=len(rows),infeasible=sum(r['state']=='infeasible' for r in rows),test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
