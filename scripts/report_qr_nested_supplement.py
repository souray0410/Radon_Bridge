"""Separate QR mechanism and shared-width study reports; public aggregates only."""
import csv,json,hashlib,platform
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import summarize_bootstrap
from scripts.report_five_seed_study import f1
from scripts.queue_fixed_svd_study import read,sha
from scripts.run_three_seed_study import SEEDS,RHOS
from scripts.run_qr_nested_supplement import NESTED_ARMS,REF_ARMS
from scripts.report_mechanism_benchmark import clean_public

NAMES={'svd_radon':'SVD / Radon','svd_self':'SVD / Self','svd_scrambled':'SVD / Scrambled','svd_resample':'SVD / Resampling',
       'qr_radon':'Random QR / Radon','qr_self':'Random QR / Self','qr_scrambled':'Random QR / Scrambled','qr_resample':'Random QR / Resampling',
       'nested_svd':'SVD','nested_qr':'Random QR','nested_learned_channel':'Learned channel'}


def make_weights(rows,group):
    lookup={(r['seed'],r['arm'],r['rho']):i for i,r in enumerate(rows)};defs=[]
    terms=([('qr_radon_minus_self',[('qr_radon',1),('qr_self',-1)]),
            ('qr_radon_minus_scrambled',[('qr_radon',1),('qr_scrambled',-1)]),
            ('basis_spatial_interaction',[('svd_radon',1),('svd_scrambled',-1),('qr_radon',-1),('qr_scrambled',1)])] if group=='A' else
           [(a+'_joint_minus_independent',[(a,1),(REF_ARMS[a],-1)]) for a in NESTED_ARMS])
    for name,ts in terms:
        w=np.zeros((len(rows),2))
        for arm,coef in ts:
            for seed in SEEDS:
                for rho in RHOS:w[lookup[(seed,arm,rho)]]+=coef/18
        defs.append({'id':name,'weights':w,'terms':ts,'seeds':SEEDS,'rhos':RHOS,'branches':['cfp','oct']})
    return defs


def bootstrap_models(probabilities,labels,resamples):
    preds=probabilities.argmax(-1);rng=np.random.default_rng(20260905);ix=rng.integers(0,len(labels),size=(resamples,len(labels)))
    point=100*f1(labels,preds);samples=np.empty((resamples,*point.shape))
    for start in range(0,resamples,50):
        selected=ix[start:start+50];samples[start:start+len(selected)]=(100*f1(labels[selected],preds[:,:,selected])).transpose(2,0,1)
    return point,samples,hashlib.sha256(ix.tobytes()).hexdigest()


def contrast_report(rows,point,samples,group):
    defs=make_weights(rows,group);w=np.stack([d.pop('weights') for d in defs]).reshape(3,-1)
    observed=w@point.reshape(-1);draws=samples.reshape(len(samples),-1)@w.T
    # Exact zero or numerical rounding around constant differences is undefined.
    sd=draws.std(0,ddof=1);active=sd>1e-10
    critical=float(np.quantile(np.max(np.abs((draws[:,active]-observed[None,active])/sd[None,active]),axis=1),.95)) if active.any() else None
    result=[]
    from radonbridge.benchmark_statistics import classify
    for i,d in enumerate(defs):
        ci=[float(observed[i]-critical*sd[i]),float(observed[i]+critical*sd[i])] if active[i] and critical is not None else None
        result.append(dict(d,difference_pp=float(observed[i]),ci95_pp=np.quantile(draws[:,i],[.025,.975]).tolist(),simultaneous_ci95_pp=ci,
                           bootstrap_sd_pp=float(sd[i]),zero_variance=not bool(active[i]),classification=classify(ci) if ci else 'undefined_zero_variance'))
    return {'group':group,'primary_count':3,'max_abs_t_critical_95':critical,'contrasts':result,'zero_sd_tolerance_pp':1e-10}


def make_figures(out,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    lookup={(r['seed'],r['arm'],r['rho']):r for r in rows};files=[]
    def values(arm,rho,branch,seeds):return np.array([lookup[(s,arm,rho)]['scores'][branch] for s in seeds])
    def save(fig,stem,title,note):
        fig.suptitle('R&B (Radon Bridge) | '+title,x=.055,ha='left',fontweight='bold',fontsize=18)
        fig.text(.055,.025,note,fontsize=9,color='#53616A');fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.23,wspace=.3)
        for ext in ['png','svg','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=180)
        plt.close(fig);files.append({'file':stem+'.png','title':title,'note':note})
    for seeds in [SEEDS]+[[s] for s in SEEDS]:
        fig,axes=plt.subplots(1,3,figsize=(16,9))
        for ax,branch in zip(axes,['cfp','oct','mean']):
            for arm,col,marker in [('qr_radon','#a38b4f','o'),('qr_scrambled','#7159a5','s'),('qr_resample','#5978a6','^'),('qr_self','#53616a','D')]:
                v=np.array([values(arm,r,branch,seeds) for r in RHOS]);ax.errorbar(range(3),v.mean(1),yerr=v.std(1,ddof=1) if len(seeds)>1 else None,color=col,marker=marker,capsize=3,label=NAMES[arm])
            ax.set(xticks=range(3),xticklabels=['1/16\nr16, h512','1/8\nr32, h1024','1/4\nr64, h2048'],xlabel='rho (channel dimension ratio)',ylabel='Macro-F1 (%)',title=branch.upper());ax.grid(axis='y',alpha=.2)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.09),ncol=4,frameon=False)
        save(fig,'A_QR_'+('_'.join(map(str,seeds))),'QR mechanism controls: '+('three-seed mean' if len(seeds)>1 else str(seeds[0])),'All curves share the same QR basis per seed. Error bars: seed sample SD. Self control has fewer effective mixer parameters.')
    for seeds in [SEEDS]+[[s] for s in SEEDS]:
        for branch in ['cfp','oct','mean']:
            fig,axes=plt.subplots(1,3,figsize=(16,9))
            for ax,arm,col in zip(axes,NESTED_ARMS,['#c33e67','#007e87','#a38b4f']):
                for method,label,ls,marker in [(arm,'Joint widths','-','o'),(REF_ARMS[arm],'Independent widths','--','s')]:
                    v=np.array([values(method,r,branch,seeds) for r in RHOS]);ax.errorbar(range(3),v.mean(1),yerr=v.std(1,ddof=1) if len(seeds)>1 else None,color=col,ls=ls,marker=marker,label=label,capsize=3)
                ax.set(xticks=range(3),xticklabels=['1/16\nr16, h512','1/8\nr32, h1024','1/4\nr64, h2048'],xlabel='rho (channel dimension ratio)',ylabel='Macro-F1 (%)',title=NAMES[arm]);ax.grid(axis='y',alpha=.2)
            h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False)
            save(fig,'B_'+branch+'_'+'_'.join(map(str,seeds)),branch.upper()+': '+('three-seed mean' if len(seeds)>1 else str(seeds[0])), 'Joint curves are three views of ONE checkpoint per seed/method. Independent widths have separate checkpoints. Training regimes differ.')
    for group in ['A','B']:
        fig,ax=plt.subplots(figsize=(16,9));part=stats[group]['contrasts']
        for i,d in enumerate(part):
            ax.hlines(i,*d['ci95_pp'],color='#99c9cd',lw=7)
            if d['simultaneous_ci95_pp']:ax.hlines(i,*d['simultaneous_ci95_pp'],color='#007e87',lw=2)
            ax.scatter(d['difference_pp'],i,color='#143d4d')
        ax.axvspan(-1,1,color='#edeff2');ax.axvline(0,color='#53616a');ax.set(yticks=range(3),yticklabels=[d['id'].replace('_',' ') for d in part],xlabel='Paired difference (pp)');ax.invert_yaxis()
        fig.subplots_adjust(left=.45);fig.suptitle('R&B | Study '+group+' primary contrasts',x=.055,ha='left',fontsize=18,fontweight='bold');fig.text(.055,.03,'Thick: ordinary95% CI. Thin: simultaneous95% CI within this three-comparison supplement family. Historical development data.',fontsize=9)
        for ext in ['png','svg','pdf']:fig.savefig(out/(group+'_primary.'+ext),dpi=180)
        plt.close(fig);files.append({'file':group+'_primary.png','title':'Study '+group+' primary contrasts'})
    return files


def build(root,resamples=10000,plots=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);manifest=read(root/'manifest.json');ledger=read(root/'ledger.json');costs={j['id']:j for j in ledger['jobs']}
    rows=[];probs=[];identity=None
    for r in manifest['rows']:
        trial=Path(r['directory']);summary=read(trial/'summary.json')
        assert summary['state']=='complete' and summary['converged_by_policy'] and summary['stop_reason']=='validation_plateau' and not summary['test_used']
        if r['reused']:assert all(sha(trial/n)==h for n,h in r['accepted_hashes'].items())
        views=read(trial/'width_exports.json') if r['arm'] in NESTED_ARMS else [{'rho':r['rho'],'directory':str(trial)}]
        for v in views:
            path=Path(v['directory']);prediction=path/'selected_predictions.npz'
            with np.load(prediction,allow_pickle=False) as z:
                current=(z['ids'].copy(),z['y'].copy())
                if identity is None:identity=current
                else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
                probability=np.stack([z['cfp'],z['oct']]);assert probability.shape==(2,296,2) and np.isfinite(probability).all() and np.allclose(probability.sum(-1),1)
            metrics={b:classification_metrics(identity[1],probability[i]) for i,b in enumerate(['cfp','oct'])};scores={b:100*d['macro_f1'] for b,d in metrics.items()};scores['mean']=np.mean(list(scores.values()))
            nested=r['arm'] in NESTED_ARMS
            view_cfg=read(path/'configuration.json');bridges=view_cfg['bridges'];channel_rank=(round(256*v['rho']) if bridges and bridges[0].get('compression') in ('learned_channel','fixed_svd_channel','fixed_centered_svd_channel','fixed_random_orthogonal_channel') else None)
            expected=summary['selected']['per_width']['rho1_'+str(round(1/v['rho']))] if nested else summary['selected']
            assert all(abs(metrics[b]['macro_f1']-expected['tasks'][b]['macro_f1'])<1e-10 for b in metrics)
            diag_id='diagnostic_'+r['id']+('_'+str(round(1/v['rho'])) if nested else '')
            diag=read(root/diag_id/'summary.json') if not r['reused'] else None
            if diag:
                assert diag['passed'] and all(d['probe_participants']==128 and d['energy_participants']==1264 and d['state_parameters_bn_gradients_rng_preserved'] for d in diag['phases'].values())
                assert diag['selected_sha256']==sha(path/'selected.pt')
            rows.append({'id':r['id']+('_view'+str(round(1/v['rho'])) if nested else ''),'training_id':r['id'],'seed':r['seed'],'arm':r['arm'],'rho':v['rho'],
                'regime':'joint_width' if nested else 'independent_width','r':channel_rank,'h':round(8192*v['rho']) if v['rho'] else None,
                'scores':scores,'metrics':metrics,'best_epoch':summary['selection']['joint']['best_epoch'],'stop_epoch':summary['epochs_ran'],
                'prediction_sha256':sha(prediction),'selected_sha256':sha(path/'selected.pt'),'training_summary_sha256':sha(trial/'summary.json'),
                'parent_checkpoints':summary['parent_checkpoints'],'configuration':view_cfg,'allocated_training_parameters':summary['parameters'],'peak_training_reserved_mib':summary['peak_reserved_mib'],'diagnostic':clean_public(diag),
                'cost_training_shared_across_views':r.get('reference_cost') if r['reused'] else costs[r['id']]})
            probs.append(probability)
    assert len(rows)==231 and len({r['training_id'] for r in rows})==213
    point,samples,indices_sha=bootstrap_models(np.stack(probs),identity[1],resamples)
    stats={g:contrast_report(rows,point,samples,g) for g in ['A','B']}
    for g in stats:stats[g].update(resamples=resamples,participants=296,shared_indices_sha256=indices_sha,interpretation='Exploratory follow-up designed after earlier results. Conditional on fitted models and selected development checkpoints. Views, widths and seeds are not additional participants.')
    write_json(out/'A_QR_mechanism_statistics.json',stats['A']);write_json(out/'B_joint_width_statistics.json',stats['B'])
    def csv_rows(path,subset):
        fields=['id','training_id','seed','arm','regime','rho','r','h','cfp_f1_percent','oct_f1_percent','mean_f1_percent','best_epoch','stop_epoch','prediction_sha256','selected_sha256']
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in subset:writer.writerow({k:r.get(k) for k in fields if not k.endswith('_f1_percent')}|{b+'_f1_percent':r['scores'][b] for b in ['cfp','oct','mean']})
    csv_rows(out/'A_single_width_204_results.csv',[r for r in rows if r['regime']=='independent_width'])
    csv_rows(out/'B_joint_width_27_views.csv',[r for r in rows if r['regime']=='joint_width'])
    csv_rows(out/'all_231_evaluation_views.csv',rows)
    write_json(out/'results.json',{'rows':rows,'unique_training_configurations':213,'evaluation_views':231,'new_training_jobs':27,'test_used':False})
    files=make_figures(out,rows,stats) if plots else []
    write_json(out/'figures.json',files)
    write_json(out/'provenance.json',{'protocol':read(root/'protocol.json'),'training_and_diagnostic_ledger':ledger,'study_summary':read(root/'study_summary.json'),'environment':{'python':platform.python_version(),'numpy':np.__version__},'historical_report_preserved':True,'test_used':False})
    (out/'INTERPRETATION.zh-CN.md').write_text('''# R&B 两组补充研究\n\nA：随机QR自身处理与空间打乱，检验跨来源通信及空间结构解释能否跨基复现。B：可学习通道、固定非中心化SVD和随机QR的联合宽度训练，对比原独立宽度训练。两组分别统计，不混成新的降维排行榜。\n\n新增27次训练。A含历史共204个独立宽度结果；B为9个训练模型的27个宽度视图。合计213个训练配置、231个评估视图，绝非231次独立训练。\n\nB同时改变参数共享、梯度来源、BN统计更新规则、每轮计算量及共同检查点的选优目标。其差异不能完全归因于通道排序，更不能预设优于SVD或F1单调。共享BN为各宽度运行统计更新的算术平均，并非混合分布的总方差。每个宽度的诊断使用同一个共享选中检查点导出的视图。\n\n仅1264训练与296开发参与者。原两个种子及历史开发结果参与过方案选择，3418仍非独立数据验证。10000次参与者配对bootstrap共用索引，三种子与三宽度不增加参与者样本量。A、B各三项主要比较分别校正，原31项比较族不改写。未知伦理/许可等信息继续列为待补。原报告保留，最终导师汇报需逐页渲染验收后交付。\n''')
    write_json(out/'report_status.json',{'state':'complete','new_trainings':27,'single_width_results':204,'joint_models':9,'joint_views':27,'total_training_configurations':213,'total_views':231,'resamples':resamples,'test_used':False,'pdf_visual_qa':'pending_local_authoring'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
