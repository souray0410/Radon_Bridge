"""Separate exploratory families for frozen training and component dependence."""
import json,hashlib
from pathlib import Path
import numpy as np
from radonbridge.artifacts import resolve,sha256,ARCHIVE,SOURCE,STUDY
from scripts.geometry_evidence import read,write
from scripts.report_geometry_mechanism import csv_write
from scripts.report_five_seed_study import f1
from radonbridge.benchmark_statistics import classify


def build(root,parent):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);rows=read(root/'manifest.json')['rows']
    diagnostics=read(root/'diagnostics.json');arrays=[];lookup={};ids=y=None;tables=[]
    def add(identifier,path,metadata):
        nonlocal ids,y
        path=resolve(path)
        with np.load(path,allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy();y=z['y'].copy()
            assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
            arrays.append(np.stack([z['cfp'],z['oct']]))
        lookup[identifier]=len(arrays)-1
        tables.append(dict(id=identifier,**metadata))
    for r in rows:
        assert r['state']=='accepted'
        directory=resolve(r['directory']);assert sha256(directory/'selected_predictions.npz')==r['accepted_hashes']['selected_predictions.npz']
        add(r['id'],directory/'selected_predictions.npz',dict(kind='frozen_training',seed=r['seed'],mode=r['structure']['mode'],best_epoch=r['best_epoch'],stop_epoch=r['epochs'],seconds=r['seconds']))
    for seed in (3416,3417,3418):
        r=next(r for r in rows if r['seed']==seed)
        refs=r['configuration']['parent_checkpoints'];prob={}
        for branch,ref in refs.items():
            with np.load(resolve(Path(ref['path']).parent/'selected_predictions.npz'),allow_pickle=False) as z:
                assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y']);prob[branch]=z[branch]
        path=out/f'parent_seed{seed}.npz';np.savez(path,ids=ids,y=y,**prob)
        add(f'parent_seed{seed}',path,dict(kind='frozen_parent_reference',seed=seed))
    diagnostic_summaries=[]
    for key,meta in diagnostics.items():
        directory=resolve(meta['directory']);assert sha256(directory/'summary.json')==meta['summary_sha256']
        s=read(directory/'summary.json');diagnostic_summaries.append(s)
        for state in ('both_on','host_only','new_only','both_off'):
            path=directory/(state+'.npz');assert sha256(path)==s['prediction_files'][state]['sha256']
            add(key+'_'+state,path,dict(kind='component_diagnostic',host=meta['host'],mode=meta['mode'],seed=meta['seed'],backbone_lr=meta['backbone_lr'],host_best_epoch=meta['host_best_epoch'],state=state))
    pred=np.stack(arrays).argmax(-1);point=100*f1(y,pred);primary=point.mean(1)
    for row,p,v in zip(tables,point,primary):row.update(cfp_f1_percent=float(p[0]),oct_f1_percent=float(p[1]),branch_mean_f1_percent=float(v))
    rng=np.random.default_rng(202609071);ix=rng.integers(0,len(y),size=(10000,len(y)));boot=np.empty((10000,len(arrays)))
    for start in range(0,10000,20):
        indices=ix[start:start+20];boot[start:start+len(indices)]=(100*f1(y[indices],pred[:,:,indices])).mean(1).T
    defs=[]
    def contrast(name,family,pairs):
        w=np.zeros(len(arrays))
        for a,b in pairs:w[lookup[a]]+=1/len(pairs);w[lookup[b]]-=1/len(pairs)
        defs.append(dict(id=name,family=family,w=w))
    for control in ('linear_resample','parent'):
        contrast('frozen_radon_minus_'+control,'frozen2',[(f'frozen_radon_seed{s}',f'frozen_linear_resample_seed{s}' if control!='parent' else f'parent_seed{s}') for s in (3416,3417,3418)])
    for host in ('mmtm_hidden256','attention_d256'):
        for mode in ('radon','linear_resample'):
            keys=[key for key,m in diagnostics.items() if m['host']==host and m['mode']==mode];assert len(keys)==6
            for state in ('host_only','new_only'):
                contrast(f'{host}_{mode}_both_on_minus_{state}','component8',[(k+'_both_on',k+'_'+state) for k in keys])
    results=[]
    for family in ('frozen2','component8'):
        group=[d for d in defs if d['family']==family];w=np.stack([d['w'] for d in group]);obs=w@primary;draw=boot@w.T;sd=draw.std(0,ddof=1);active=sd>1e-10
        critical=float(np.quantile(np.max(np.abs((draw[:,active]-obs[None,active])/sd[None,active]),1),.95)) if active.any() else None
        for j,d in enumerate(group):
            ci=[float(obs[j]-critical*sd[j]),float(obs[j]+critical*sd[j])] if active[j] else None
            results.append(dict(id=d['id'],family=family,difference_pp=float(obs[j]),ci95_pp=np.quantile(draw[:,j],[.025,.975]).tolist(),
                                simultaneous_ci95_pp=ci,classification=classify(ci) if ci else 'undefined_zero_variance'))
    csv_write(out/'all_results.csv',tables);csv_write(out/'paired_differences.csv',results)
    write(out/'statistics.json',dict(participants=296,resamples=10000,indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
          comparisons=results,exploratory=True,test_used=False,interpretation='Frozen-training and checkpoint-intervention families are separate; intervals condition on selected development checkpoints'))
    interactions=[]
    for key,meta in diagnostics.items():
        interaction=primary[lookup[key+'_both_on']]+primary[lookup[key+'_both_off']]-primary[lookup[key+'_host_only']]-primary[lookup[key+'_new_only']]
        interactions.append(dict(**meta,branch_f1_nonadditivity_pp=float(interaction),interpretation='Nonadditivity of F1; not proof of biological or causal synergy'))
    csv_write(out/'component_nonadditivity.csv',interactions)
    full_rows=read(Path(parent)/'manifest.json')['rows'];comparison=[]
    for row in rows:
        for lr in (3e-5,6e-5):
            full=next(x for x in full_rows if x['category']=='direct' and x['seed']==row['seed'] and x['backbone_lr']==lr and x['structure']['id']==row['structure']['mode']+'_M32_S64_k3_r16')
            comparison.append(dict(seed=row['seed'],mode=row['structure']['mode'],full_finetune_backbone_lr=lr,
                                   frozen_mean_f1_percent=100*row['scores']['branch_mean'],full_finetune_mean_f1_percent=100*full['scores']['branch_mean'],
                                   interpretation='Descriptive training-regime comparison; frozen mode has no backbone/head LR and fixes native BN'))
    csv_write(out/'frozen_vs_full_finetune.csv',comparison)
    ledger=[]
    for base in (root/'batches',ARCHIVE/'current'/root.relative_to(SOURCE/'runs'/STUDY)/'batches'):
        for path in base.glob('*/ledger.json'):ledger.extend(read(path)['jobs'])
    write(out/'accounting.json',dict(gpu_minutes=sum(x['gpu_seconds'] for x in ledger)/60,entries=ledger))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,5))
    for mode,color in [('radon','#007e87'),('linear_resample','#b87441')]:
        v=[t['branch_mean_f1_percent'] for t in tables if t['kind']=='frozen_training' and t['mode']==mode]
        ax.scatter([mode]*3,v,color=color);ax.plot([mode],[np.mean(v)],marker='_',markersize=24,color='black')
    ax.set(ylabel='Mean branch macro-F1 (%)',title='R&B | Frozen native networks | 3 seeds');fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/('frozen_training.'+ext),dpi=180)
    plt.close(fig)
    text=['# R&B：功能依赖与冻结原网络补充','',
          '6项冻结训练、24个检查点×4种开关状态完成；原378项不改写，不把96个诊断视图当作96次训练。',
          '冻结主干、原分类头及BN统计，仅中间卷积训练。所有原生参数/缓冲区逐轮SHA保持一致。',
          '关闭诊断保留原生特征路径，两者关闭对应微调后的原生网络；原独立预训练预测另列。推理干预可能造成分布变化，不能当严格因果证明。',
          '冻结实验2项比较与功能依赖8项比较分别校正；不以增加实验消除不利结果。所有统计为296人开发集探索性证据。','']
    text += [f"- {d['id']}: {d['difference_pp']:+.3f} pp, simultaneous CI {d['simultaneous_ci95_pp']}; {d['classification']}" for d in results]
    text += ['','all_results.csv和所有聚合图表可以同步；parent_seed*.npz及诊断参与者预测必须保留授权环境，不得上传GitHub。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n')
    write(root/'study_summary.json',dict(state='complete',training_completed=6,diagnostic_checkpoints=24,evaluation_views=96,report_directory=str(out),test_used=False))
