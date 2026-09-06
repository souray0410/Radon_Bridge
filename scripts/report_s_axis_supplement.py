"""Public aggregates only: S-axis paired report, error transfers and claim lock."""
import csv,hashlib,platform
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import classify
from scripts.queue_fixed_svd_study import read,sha
from scripts.report_qr_nested_supplement import bootstrap_models
from scripts.run_s_axis_supplement import SEEDS,RHOS,BASES,PREDECESSOR,PREVIOUS


def error_transfers(y,old,new):
    result={}
    for name,mask in [('all',np.ones(len(y),bool)),('control',y==0),('case',y==1)]:
        a=old[mask]==y[mask];b=new[mask]==y[mask]
        result[name]=dict(n=int(mask.sum()),old_wrong_new_correct=int((~a&b).sum()),old_correct_new_wrong=int((a&~b).sum()),
                          both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()))
    return result


def primary(rows,point,samples):
    lookup={(r['seed'],r['arm'],r['rho']):i for i,r in enumerate(rows)};weights=[]
    for basis in BASES:
        w=np.zeros_like(point)
        for seed in SEEDS:
            for rho in RHOS:
                w[lookup[(seed,basis+'_radon',rho)]]+=1/18
                w[lookup[(seed,basis+'_s_permuted',rho)]]-=1/18
        weights.append(w)
    weights.append(weights[0]-weights[1]);w=np.stack(weights).reshape(3,-1)
    observed=w@point.ravel();draws=samples.reshape(len(samples),-1)@w.T
    sd=draws.std(0,ddof=1);active=sd>1e-10
    critical=float(np.quantile(np.max(np.abs((draws[:,active]-observed[None,active])/sd[None,active]),1),.95)) if active.any() else None
    result=[]
    for j,name in enumerate(['SVD_ordered_minus_S_permuted','QR_ordered_minus_S_permuted','basis_order_interaction']):
        ci=[float(observed[j]-critical*sd[j]),float(observed[j]+critical*sd[j])] if active[j] else None
        result.append(dict(id=name,difference_pp=float(observed[j]),ci95_pp=np.quantile(draws[:,j],[.025,.975]).tolist(),simultaneous_ci95_pp=ci,
                           bootstrap_sd_pp=float(sd[j]),zero_variance=not bool(active[j]),classification=classify(ci) if ci else 'undefined_zero_variance'))
    return dict(primary_count=3,max_abs_t_critical95=critical,contrasts=result)


def plots(out,rows,stats,fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    lookup={(r['seed'],r['arm'],r['rho']):r for r in rows};files=[]
    def save(fig,name):
        if fixture:fig.text(.98,.985,'SYNTHETIC FIXTURE / NOT EXPERIMENTAL RESULTS',ha='right',va='top',fontsize=9,color='#b44')
        for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=160)
        plt.close(fig);files.append(name+'.png')
    for seeds in [SEEDS]+[[s] for s in SEEDS]:
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for i,(basis,color) in enumerate(zip(BASES,['#007e87','#a38b4f'])):
            for j,branch in enumerate(['cfp','oct','mean']):
                ax=axes[i,j]
                for arm,label,style in [(basis+'_radon','Ordered S','-o'),(basis+'_s_permuted','Permuted S','--s')]:
                    v=np.array([[lookup[(s,arm,rho)]['scores'][branch] for s in seeds] for rho in RHOS])
                    ax.errorbar(range(3),v.mean(1),yerr=v.std(1,ddof=1) if len(seeds)>1 else None,fmt=style,color=color,capsize=3,label=label)
                ax.set(xticks=range(3),xticklabels=['1/16\nr16 / h512','1/8\nr32 / h1024','1/4\nr64 / h2048'],title=basis.upper()+' | '+branch.upper(),ylabel='Macro-F1 (%)');ax.grid(axis='y',alpha=.2)
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
        fig.suptitle('R&B (Radon Bridge) | S-axis adjacency | '+('3-seed mean' if len(seeds)>1 else str(seeds[0])),fontsize=18,fontweight='bold',x=.06,ha='left')
        fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.19,hspace=.5,wspace=.32)
        fig.text(.06,.02,'rho = channel dimension ratio. Seed sample SD. Same basis and mixer size within each pair. Historical development data.',fontsize=9)
        save(fig,'S_axis_'+'_'.join(map(str,seeds)))
    fig,ax=plt.subplots(figsize=(16,9))
    for i,c in enumerate(stats['contrasts']):
        ax.hlines(i,*c['ci95_pp'],lw=8,color='#9ccbd0')
        if c['simultaneous_ci95_pp']:ax.hlines(i,*c['simultaneous_ci95_pp'],lw=2,color='#007e87')
        ax.scatter(c['difference_pp'],i,color='#143d4d',zorder=4)
    ax.axvspan(-1,1,color='#eef0f2');ax.axvline(0,color='#64717b');ax.set(yticks=range(3),yticklabels=[c['id'].replace('_',' ') for c in stats['contrasts']],xlabel='Ordered minus S-permuted difference (pp)');ax.invert_yaxis()
    fig.text(.06,.94,'R&B | Prespecified S-axis contrasts',fontsize=18,ha='left',fontweight='bold');fig.subplots_adjust(left=.38,right=.96,bottom=.2,top=.84)
    fig.text(.06,.06,'Thick: ordinary 95% CI. Thin: max-|t| simultaneous CI for this 3-comparison family. One fixed permutation; no clinical threshold.',fontsize=9)
    save(fig,'S_axis_primary')
    return files


def build(root,resamples=10000,make_plots=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);manifest=read(root/'manifest.json')
    source_rows=manifest['references']+manifest['rows'];rows=[];probs=[];identity=None;costs={}
    for path in root.glob('*/ledger.json'):
        for j in read(path)['jobs']:costs[j['id']]=j
    for r in source_rows:
        d=Path(r['directory']);s=read(d/'summary.json')
        assert s['state']=='complete' and s['converged_by_policy'] and s['stop_reason']=='validation_plateau' and s['test_used'] is False
        assert all(sha(d/n)==h for n,h in r['accepted_hashes'].items())
        with np.load(d/'selected_predictions.npz',allow_pickle=False) as z:
            current=(z['ids'].copy(),z['y'].copy())
            if identity is None:identity=current
            else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
            prob=np.stack([z['cfp'],z['oct']]);assert prob.shape==(2,296,2) and np.isfinite(prob).all() and np.allclose(prob.sum(-1),1)
        metrics={b:classification_metrics(identity[1],prob[i]) for i,b in enumerate(['cfp','oct'])}
        assert all(abs(metrics[b]['macro_f1']-s['selected']['tasks'][b]['macro_f1'])<1e-10 for b in metrics)
        scores={b:100*v['macro_f1'] for b,v in metrics.items()};scores['mean']=float(np.mean(list(scores.values())))
        diag=None
        if r.get('diagnostic'):
            diag=read(Path(r['diagnostic'])/'summary.json');assert diag['passed'] and diag['selected_sha256']==r['accepted_hashes']['selected.pt']
            assert all(v['probe_participants']==128 and v['energy_participants']==1264 and v['state_parameters_bn_gradients_rng_preserved'] for v in diag['phases'].values())
        from scripts.report_mechanism_benchmark import clean_public
        rows.append(dict(id=r['id'],seed=r['seed'],arm=r['arm'],rho=r['rho'],r=round(256*r['rho']) if r['rho'] else None,h=round(8192*r['rho']) if r['rho'] else None,
                         scores=scores,metrics=metrics,best_epoch=s['selection']['joint']['best_epoch'],stop_epoch=s['epochs_ran'],parameters=s['parameters'],peak_reserved_mib=s['peak_reserved_mib'],
                         configuration=r['configuration'],hashes=r['accepted_hashes'],cost=costs.get(r['id'],r.get('reference_cost')),diagnostic=clean_public(diag)))
        probs.append(prob)
    assert len(rows)==39
    probabilities=np.stack(probs);point,samples,ix_sha=bootstrap_models(probabilities,identity[1],resamples);lookup={r['id']:i for i,r in enumerate(rows)}
    stats=primary(rows,point,samples);stats.update(resamples=resamples,participants=296,indices_sha256=ix_sha,permutation=read(root/'s_axis_permutation.json'),
          interpretation='Conditional on selected development models and one fixed permutation. Equal seed/width/branch weights; not independent participants.')
    errors=[];cell_diffs=[]
    for r in manifest['rows']:
        new=lookup[r['id']];old=lookup[r['reference_id']]
        entry=dict(seed=r['seed'],basis=r['basis'],rho=r['rho'],ordered_id=r['reference_id'],permuted_id=r['id'])
        errors.append(entry|dict(direction='old=ordered; new=S-permuted',branches={b:error_transfers(identity[1],probabilities[old,j].argmax(-1),probabilities[new,j].argmax(-1)) for j,b in enumerate(['cfp','oct'])},
                                 ordered_branch_disagreements=int((probabilities[old,0].argmax(-1)!=probabilities[old,1].argmax(-1)).sum()),
                                 permuted_branch_disagreements=int((probabilities[new,0].argmax(-1)!=probabilities[new,1].argmax(-1)).sum())))
        for j,b in enumerate(['cfp','oct','mean']):
            delta=point[old]-point[new];draw=samples[:,old]-samples[:,new]
            value=delta[j] if j<2 else delta.mean();v=draw[:,j] if j<2 else draw.mean(-1)
            cell_diffs.append(entry|dict(branch=b,ordered_minus_permuted_pp=float(value),ci95_pp=np.quantile(v,[.025,.975]).tolist(),secondary=True))
    write_json(out/'statistics.json',stats);write_json(out/'paired_cells.json',cell_diffs);write_json(out/'error_transfers_mechanism.json',errors)
    # Separate outcome and reference protocol for task-fusion error transfer.
    fusion=read(PREDECESSOR/'manifest.json')['rows'];fp={}
    for r in fusion:
        path=Path(r['directory']);assert sha(path/'selected_predictions.npz')==r['accepted_hashes']['selected_predictions.npz']
        with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['ids'],identity[0]) and np.array_equal(z['y'],identity[1]);fp[r['id']]=z['fusion'].argmax(-1)
    ferr=[]
    for r in fusion:
        ref=next(a for a in fusion if a['seed']==r['seed'] and a['backbone_lr']==r['backbone_lr'] and a['arm']=='concat_mlp')
        if ref['id']!=r['id']:ferr.append(dict(seed=r['seed'],backbone_lr=r['backbone_lr'],arm=r['arm'],outcome='fusion argmax',reference='concat_mlp',counts=error_transfers(identity[1],fp[ref['id']],fp[r['id']])))
    write_json(out/'error_transfers_task_fusion.json',ferr)
    aggregate=[]
    for arm in sorted({r['arm'] for r in rows}):
        for rho in ([None] if arm=='no_bridge' else RHOS):
            v=[r for r in rows if r['arm']==arm and r['rho']==rho];assert len(v)==3
            aggregate.append(dict(arm=arm,rho=rho,scores={b:dict(mean=float(np.mean([r['scores'][b] for r in v])),sample_sd=float(np.std([r['scores'][b] for r in v],ddof=1))) for b in ['cfp','oct','mean']}))
    with (out/'all_39_results.csv').open('w',newline='') as f:
        fields=['id','seed','arm','rho','r','h','cfp','oct','mean','best_epoch','stop_epoch','peak_reserved_mib'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields}|r['scores'])
    figures=plots(out,rows,stats,fixture=read(root/'protocol.json').get('synthetic_fixture',False)) if make_plots else []
    write_json(out/'results.json',dict(rows=rows,aggregate=aggregate,figures=figures,new_training_jobs=18,reused_results=21,test_used=False,accounting=read(root/'study_summary.json')))
    lock=dict(state='candidate_inventory_and_claims_locked',automatic_further_search=False,unique_ukb_second_stage_trainings=285,evaluation_views=303,test_used=False,
              final_s_axis_source=read(root/'source_acceptance.json'),s_axis_manifest_sha256=sha(root/'manifest.json'),historical_manifest_sha256=sha(PREVIOUS/'manifest.json'),
              task_fusion_manifest_sha256=sha(PREDECESSOR/'manifest.json'),s_axis_contrasts=stats['contrasts'],task_fusion_statistics_sha256=sha(PREDECESSOR/'report/statistics.json'),
              candidate_rule='Freeze all prespecified candidates; do not automatically pick a universal winner from development ranking. Independent study must name its primary method before outcome access.',
              claims=dict(structured_linear_bridge='Implemented and numerically verified; entire CNN/fusion system is nonlinear.',
                          communication='Interpret using historical paired controls and uncertainty; S-axis results supplement rather than replace them.',
                          spatial_order='Use the three listed contrasts and counterexamples, conditional on one permutation and boundary changes.',
                          clinical_generalization='Not established: historically selected development cohort; no independent UKB test or external clinical validation.',
                          rho_monotonicity='Not mathematically guaranteed; report all cells; do not select configurations to manufacture monotonicity.',
                          nonlinear_baseline_superiority='Consult separate 54-task fusion contrasts, capacity and cost. No universal superiority claim.'))
    write_json(out/'METHOD_AND_CLAIM_LOCK.json',lock)
    note='''# R&B：S轴补充与方法/主张锁定\n\n18项新训练、18项有序配对参照及3项无桥均验收平台；原54项融合基准单列。S轴置换保留宽度和参数，改变局部顺序与零填充边界位置；结论条件于一个固定置换。\n\n主要差值为有序减置换，三项比较单独同时校正。全部是296名历史开发参与者，三种子/三宽度不扩充样本量。合成场传递是算子回归探针，不是疾病分类或临床证据。错误转移列出全部病例/对照，无事后筛选；表中“新”表示置换方法，不表示更优。\n\n冻结候选方法集合及当前证据，停止自动算法搜索。若不能区分方法，明确保留不确定性；最终确认性主方法由后续独立协议预先选定，不据开发排名宣称通用最佳。完整285项独特第二阶段训练加18个联合宽度视图，共303评估视图；预检、诊断与合成拟合另列。\n\n当前基准覆盖线性机制、拼接MLP、MIL、MMTM及交叉注意力；足够完成这轮受控问题，但不等于覆盖所有医学SOTA或已具备NBE独立临床验证。\n\n'''
    for c in stats['contrasts']:note+=f"- {c['id']}: {c['difference_pp']:+.3f} pp；同时95%区间 {c['simultaneous_ci95_pp']}；{c['classification']}。\n"
    (out/'INTERPRETATION.zh-CN.md').write_text(note)
    write_json(out/'provenance.json',dict(protocol=read(root/'protocol.json'),environment=dict(python=platform.python_version(),numpy=np.__version__),participant_data_exported=False))
    write_json(out/'artifact_manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'})
    write_json(out/'report_status.json',dict(state='complete',test_used=False,rendered_figure_inspection='pending_delivery',new_trainings=18,claim_inventory_locked=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
