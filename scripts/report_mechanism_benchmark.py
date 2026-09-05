"""Complete 186-result evidence; primary max-|t| intervals and honest costs."""
import csv,json,platform,sys
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import bootstrap
from scripts.run_three_seed_study import SEEDS,RHOS,key
from scripts.run_mechanism_benchmark import BASELINES,MECHANISMS
from scripts.run_centered_study import FILES
from scripts.queue_fixed_svd_study import read,sha
from scripts.report_learned_channel_study import name as old_name

NAMES={'qr_resample':'Random QR / Linear resampling','svd_oct_to_cfp':'SVD / OCT to CFP','svd_cfp_to_oct':'SVD / CFP to OCT','qr_oct_to_cfp':'Random QR / OCT to CFP','qr_cfp_to_oct':'Random QR / CFP to OCT','mmtm_r4':'MMTM / ratio 4, hidden 256','mmtm_r8':'MMTM / ratio 8, hidden 128','attention_d128':'Cross-attention / d128, 4 heads','attention_d256':'Cross-attention / d256, 4 heads'}
COLORS={'svd':'#007E87','qr':'#A38B4F','centered':'#7159A5','channel':'#C33E67','learned':'#B56D37','no_bridge':'#53616A','mmtm':'#5978A6','attention':'#8D6A9F'}
BRANCHES=['cfp','oct','mean']
def name(arm):return NAMES[arm] if arm in NAMES else old_name(arm)
def color(arm):return COLORS[next((k for k in COLORS if arm.startswith(k)),'no_bridge')]
def rho_label(rho):return 'N/A' if rho is None else '1/'+str(round(1/rho))
def clean_public(value):
    if isinstance(value,dict):return {k:clean_public(v) for k,v in value.items() if k not in ('probe_ids','participant_ids','ids','private_predictions','private_features')}
    if isinstance(value,list):return [clean_public(v) for v in value]
    return value

def dataset_card(data):
    import pandas as pd
    f=pd.read_csv(Path(data)/'selected.csv',dtype={'participant_id':str});f=f[f['split'].isin(['train','validation'])]
    assert f.groupby(['split','label_id']).size().to_dict()=={('train',0):632,('train',1):632,('validation',0):148,('validation',1):148}
    result={'counts':{'train':1264,'development':296},'case_control_counts':{'train':[632,632],'development':[148,148]},'selected_csv_sha256':sha(Path(data)/'selected.csv'),
        'endpoint':'Same record-derived glaucoma phenotype for CFP and OCT branches','label_source':'UKB records; verify specific source and timing columns below. No newly created expert image gold standard.',
        'development_role':'Historically used for configuration selection; this is exploratory controlled benchmarking.',
        'limitations':['Balanced case-control sampling; predictive values and calibration do not represent population prevalence.','Ethics approval, UKB application/license details and any missing exclusion/timing information require author verification.'], 'columns':{},'test_used':False}
    tokens=['age','sex','gender','centre','center','label','reference','phenotype','candidate','task_profile','diagnosis','timing','exclusion']
    for col in f.columns:
        if col=='participant_id' or any(t in col.lower() for t in ['path','file','id','order']):continue
        if not any(t in col.lower() for t in tokens):continue
        result['columns'][col]={}
        for split,frame in f.groupby('split'):
            v=frame[col];entry={'available':int(v.notna().sum()),'missing':int(v.isna().sum())}
            if pd.api.types.is_numeric_dtype(v) and v.nunique()>12:
                entry.update(mean=float(v.mean()),sample_sd=float(v.std()),median=float(v.median()),q25=float(v.quantile(.25)),q75=float(v.quantile(.75)),min=float(v.min()),max=float(v.max()))
            elif v.nunique()<=30:entry['counts']={str(k):int(n) for k,n in v.value_counts(dropna=True).items()}
            else:entry['description']='High-cardinality values withheld; source needs dedicated aggregate verification.'
            result['columns'][col][split]=entry
    result['unverified_items']=['Ethics committee and approval number','UKB approved application number and data license','Full source cohort denominator and exclusion flow','Label timing relative to imaging if not recoverable from provided columns']
    return result


def make_figures(out,rows,primary,stats,pairing,latencies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    lookup={key(r['seed'],r['arm'],r['rho']):r for r in rows};files=[]
    def save(fig,stem,title,note):
        fig.suptitle('R&B (Radon Bridge) | '+title,x=.05,ha='left',fontsize=18,fontweight='bold')
        fig.text(.05,.025,note,fontsize=9,color='#53616A');fig.subplots_adjust(left=.38 if len(fig.axes)==1 else .27 if stem.startswith('11_') else .1,right=.97,bottom=.19,top=.83,wspace=1.1 if stem.startswith('11_') else .3)
        for ext in ['png','pdf','svg']:fig.savefig(out/(stem+'.'+ext),dpi=180)
        files.append({'file':stem+'.png','title':title,'note':note});plt.close(fig)
    def scores(arm,rho,branch,seeds=SEEDS):return np.array([lookup[key(s,arm,rho)]['scores'][branch] for s in seeds])
    # Baseline configurations are categorical, never points on a rho axis.
    configs=[('no_bridge',None)]+[(a,None) for a in BASELINES]+[(a,r) for r in RHOS for a in ['svd_radon','qr_radon']]
    for branch in BRANCHES:
        fig,ax=plt.subplots(figsize=(16,9));labels=[]
        for i,(arm,rho) in enumerate(configs):
            v=scores(arm,rho,branch);ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',color=color(arm),capsize=3)
            for j,s in enumerate(SEEDS):ax.scatter(v[j],i+(j-1)*.12,s=19,marker='^' if s==3418 else '.',color=color(arm))
            labels.append(name(arm)+(f' | rho={rho_label(rho)}' if rho else ''))
        ax.set(yticks=range(len(configs)),yticklabels=labels,xlabel='Macro-F1 (%)',title=branch.upper());ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        fig.subplots_adjust(left=.31);save(fig,'01_common_'+branch,'Common communication methods: '+branch.upper(),'Mean +/- sample SD over three seeds. Points show individual seeds; triangles mark 3418. Parameters and search histories differ.')
    # Factorial geometry and direction, every branch with uniform width ticks.
    groups=[('02_geometry',['svd_radon','svd_resample','qr_radon','qr_resample'],'Fixed basis x communication geometry'),
            ('03_svd_direction',['svd_radon','svd_oct_to_cfp','svd_cfp_to_oct'],'SVD: directional training'),
            ('04_qr_direction',['qr_radon','qr_oct_to_cfp','qr_cfp_to_oct'],'Random QR: directional training'),
            ('05_historical',['learned','svd_radon','centered_svd_radon','qr_radon','channel_svd_radon'],'Historical compression exploration')]
    for stem,arms,title in groups:
        fig,axes=plt.subplots(1,3,figsize=(16,9))
        for ax,b in zip(axes,BRANCHES):
            for j,a in enumerate(arms):
                v=np.stack([scores(a,r,b) for r in RHOS]);ax.errorbar(range(3),v.mean(1),yerr=v.std(1,ddof=1),label=name(a),color=color(a),marker=['o','s','^','D','P'][j],linestyle='--' if 'resample' in a else '-.' if 'oct_to_cfp' in a else ':' if 'cfp_to_oct' in a else '-',capsize=3)
            ax.axhline(scores('no_bridge',None,b).mean(),color=color('no_bridge'),ls=':');ax.set(xticks=range(3),xticklabels=['rho=1/16\nr16, h512','rho=1/8\nr32, h1024','rho=1/4\nr64, h2048'],ylabel='Macro-F1 (%)',title=b.upper());ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
        save(fig,stem,title,'Three-seed mean +/- sample SD. CM rank r is not applicable; its h matches. Resampling factor32 is packaging, not an angle count.')
    # Primary intervals split across legible pages, all 31 retained.
    for start in [0,7,19]:
        end=7 if start==0 else min(start+12,31);part=primary[start:end];fig,ax=plt.subplots(figsize=(16,9))
        for i,d in enumerate(part):
            ci=d['simultaneous_ci95_pp'];ordinary=d['ci95_pp'];ax.hlines(i,*ordinary,color='#99C9CD',linewidth=6)
            if ci is not None:ax.hlines(i,*ci,color='#007E87',linewidth=1.5)
            ax.scatter(d['difference_pp'],i,color='#143D4D',s=25)
        ax.axvspan(-1,1,color='#EDEFF2');ax.axvline(0,color='#53616A',lw=.8)
        ax.set(yticks=range(len(part)),yticklabels=[d['id'].replace('_',' ') for d in part],xlabel='Prespecified paired difference (pp)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        save(fig,f'06_primary_{start:02d}','Primary contrasts with simultaneous uncertainty','Thin: max-|t| simultaneous 95% interval over 31 contrasts. Thick: ordinary 95% interval. Gray band: +/-1 pp research margin.')
    for seed in SEEDS:
        fig,axes=plt.subplots(1,3,figsize=(16,9));arms=['svd_radon','qr_radon','svd_resample','qr_resample']
        for ax,b in zip(axes,BRANCHES):
            for j,a in enumerate(arms):ax.plot(range(3),[scores(a,r,b,[seed])[0] for r in RHOS],label=name(a),color=color(a),marker=['o','s','^','D'][j],ls='--' if 'resample' in a else '-')
            ax.set(xticks=range(3),xticklabels=['1/16','1/8','1/4'],xlabel='rho',ylabel='Macro-F1 (%)',title=b.upper());ax.grid(axis='y',alpha=.2)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
        save(fig,f'07_seed{seed}',f'Individual training repeat: seed {seed}','Seeds3416/3417 participated in earlier screening. Seed3418 is separately shown, but uses the same historical development participants.')
    # Late fusion is a distinct prediction, not the arithmetic mean of branch F1.
    fig,ax=plt.subplots(figsize=(16,9))
    for i,(a,rho) in enumerate(configs):
        v=[100*lookup[key(s,a,rho)]['late_fusion']['macro_f1'] for s in SEEDS];ax.errorbar(np.mean(v),i,xerr=np.std(v,ddof=1),fmt='o',color=color(a),capsize=3)
    ax.set(yticks=range(len(configs)),yticklabels=[name(a)+(f' | rho={rho_label(r)}' if r else '') for a,r in configs],xlabel='F1 of equal-probability late fusion (%)');ax.invert_yaxis()
    save(fig,'08_late_fusion','Late fusion: a separate output','p_fused = 0.5 p_CFP + 0.5 p_OCT; fixed argmax; no weight fitting, calibration or threshold search. Secondary analysis.')
    # Pairing effects for each basis/width; correct minus mismatched average.
    for basis in ['svd','qr']:
        fig,axes=plt.subplots(1,2,figsize=(16,9))
        for ax,b in zip(axes,['cfp','oct']):
            for j,condition in enumerate(['shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both']):
                means=[];sds=[]
                for rho in RHOS:
                    values=[]
                    for s in SEEDS:
                        d=pairing[lookup[key(s,basis+'_radon',rho)]['id']]['result'];paired_row=d['conditions'][0]
                        values.append(100*(paired_row['branches'][b]['metrics']['macro_f1']-d['permutation_aggregates'][condition][b]['macro_f1']['mean']))
                    means.append(np.mean(values));sds.append(np.std(values,ddof=1))
                ax.errorbar(range(3),means,yerr=sds,label=condition.replace('_',' '),marker=['o','s','^'][j],capsize=3)
            ax.axhline(0,color='#53616A',lw=.8);ax.set(xticks=range(3),xticklabels=['1/16','1/8','1/4'],xlabel='rho',ylabel='Correct minus mismatched pairing (pp)',title=b.upper());ax.grid(axis='y',alpha=.2)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,frameon=False)
        save(fig,'09_pairing_'+basis,basis.upper()+': use of correctly paired information','Twenty global derangements per checkpoint, both eyes together. Means over permutations, then seeds. Error bars: seed SD, not extra participants.')
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    for config_index,(a,rho) in enumerate(configs):
        subset=[lookup[key(s,a,rho)] for s in SEEDS];v=np.mean([r['scores']['mean'] for r in subset]);label=name(a)+(f' / {rho_label(rho)}' if rho else '')
        params=np.mean([r['effective_bridge_parameters'] for r in subset]);times=[latencies[r['id']]['result'] for r in subset]
        axes[0].scatter(params/1e6,v,color=color(a));axes[1].scatter(np.mean([t['median_ms'] for t in times]),v,color=color(a),marker='x' if any(t['timing_interfered'] for t in times) else 'o')
        for ax in axes:ax.annotate(str(config_index+1),ax.collections[-1].get_offsets()[0],fontsize=9,xytext=(4,4),textcoords='offset points')
    axes[0].set(xlabel='Effective communication parameters (million)',ylabel='Mean of branch F1 (%)');axes[1].set(xlabel='Median full forward latency (ms), batch16',ylabel='Mean of branch F1 (%)')
    fig.text(.1,.09,'; '.join(f'{i+1}: '+name(a)+(f' / {rho_label(rho)}' if rho else '') for i,(a,rho) in enumerate(configs[:6])),fontsize=7)
    fig.text(.1,.065,'; '.join(f'{i+7}: '+name(a)+(f' / {rho_label(rho)}' if rho else '') for i,(a,rho) in enumerate(configs[6:])),fontsize=7)
    save(fig,'10_cost','Performance and actual computational cost','10 warmup +50 synchronized timings per model on GPU1. Crosses mark other GPU processes; contended timings do not support speed advantages.')
    # Train-only residual and gradient diagnostics for every new configuration.
    new_configs=[(a,r) for r in RHOS for a in MECHANISMS]+[(a,None) for a in BASELINES]
    for start in [0,10]:
        part=new_configs[start:start+10];fig,axes=plt.subplots(1,2,figsize=(16,9))
        for i,(a,rho) in enumerate(part):
            for j,b in enumerate(['cfp','oct']):
                rr=[lookup[key(s,a,rho)] for s in SEEDS]
                residual=[r['diagnostic']['phases']['selected']['energy'][b+'_stage3']['delta_over_input_l2'] for r in rr]
                cosine=[r['diagnostic']['phases']['selected']['gradient_groups'][b+'_stage3']['cosine'] for r in rr]
                axes[0].scatter(np.mean(residual),i+(j-.5)*.15,color=['#007E87','#C33E67'][j],label=b.upper() if i==0 else None)
                defined=[v for v in cosine if v is not None]
                if defined:axes[1].scatter(np.mean(defined),i+(j-.5)*.15,color=['#007E87','#C33E67'][j])
        for ax in axes:ax.set(yticks=range(len(part)),yticklabels=[name(a)+(f' / {rho_label(r)}' if r else '') for a,r in part]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        axes[0].set_xlabel('Residual / input L2; all1264 train participants');axes[1].set(xlabel='Two-CE gradient cosine at native stage3',xlim=(-1.05,1.05));axes[0].legend()
        save(fig,f'11_diagnostics_{start}','Train-only diagnostics: residuals and task gradients','128 fixed training participants for gradients, accumulated before cosine. Undefined gradients omitted; exact per-seed values remain in JSON.')
    return files


def build(root,resamples=10000,make_plots=True):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);m=read(root/'manifest.json');p=read(root/'protocol.json');ledger=read(root/'ledger.json');cost={j['id']:j for j in ledger['jobs']}
    old={r['id']:r for r in read(Path(p['predecessor'])/'report/results.json')['rows']};assert len(old)==129 and len(m['rows'])==186
    rows=[];probs=[];identity=None
    for r in m['rows']:
        path=Path(r['directory']);d=read(path/'summary.json');assert d['state']=='complete' and d['converged_by_policy'] and d['stop_reason']=='validation_plateau' and not d['test_used']
        hashes={n:sha(path/n) for n in FILES}
        if r['reused']:
            assert hashes==r['accepted_hashes']==old[r['id']]['result_hashes'];row=dict(old[r['id']])
        else:
            info=read(path/'model.json');diagpath=root/('diagnostic_'+r['id'])/'summary.json';diag=read(diagpath)
            assert diag['passed'] and not diag['preflight'] and diag['selected_sha256']==hashes['selected.pt']
            assert all(v['probe_participants']==128 and v['energy_participants']==1264 and v['state_parameters_bn_gradients_rng_preserved'] for v in diag['phases'].values())
            scores={b:100*d['selected']['tasks'][b]['macro_f1'] for b in ['cfp','oct']};scores['mean']=sum(scores.values())/2
            row=dict(r,summary=d,scores=scores,model=info,result_hashes=hashes,cost=cost[r['id']],diagnostic=diag,diagnostic_sha256=sha(diagpath),diagnostic_cost=cost['diagnostic_'+r['id']],stored_bridge_parameters=sum(g['stored_bridge_parameters'] for g in info['groups']),effective_bridge_parameters=sum(g['effective_bridge_parameters'] for g in info['groups']))
        with np.load(path/'selected_predictions.npz',allow_pickle=False) as z:
            current=(z['ids'].copy(),z['y'].copy())
            if identity is None:identity=current
            else:assert all(np.array_equal(a,b) for a,b in zip(identity,current))
            probability=np.stack([z['cfp'],z['oct']]);assert probability.shape==(2,296,2) and np.isfinite(probability).all() and np.allclose(probability.sum(-1),1)
            probs.append(probability);row['auxiliary']={}
            for i,b in enumerate(['cfp','oct']):
                metrics=classification_metrics(z['y'],probability[i]);metrics['brier_binary']=float(np.mean((probability[i,:,1]-z['y'])**2));row['auxiliary'][b]=metrics
                assert abs(metrics['macro_f1']*100-row['scores'][b])<1e-9
            fused=probability.mean(0);row['late_fusion']=classification_metrics(z['y'],fused);row['late_fusion']['brier_binary']=float(np.mean((fused[:,1]-z['y'])**2))
        row['method_name']=name(r['arm']);rows.append(row)
    stats,boot,fusion_boot,indices=bootstrap(rows,np.stack(probs),identity[1],resamples)
    np.savez_compressed(root/'private_bootstrap_samples.npz',indices=indices,model_f1=boot,fusion_f1=fusion_boot)
    write_json(out/'bootstrap.json',stats)
    pairing={};latencies={}
    for kind,store,count in [('pairing',pairing,54),('latency',latencies,33)]:
        for directory in root.glob(kind+'_*'):
            d=read(directory/'summary.json');assert d['passed'] and not d['test_used'];identifier=directory.name[len(kind)+1:]
            row=next(r for r in rows if r['id']==identifier);assert d['selected_sha256']==row['result_hashes']['selected.pt']
            store[identifier]=dict(d,cost=cost[directory.name],summary_sha256=sha(directory/'summary.json'))
        assert len(store)==count
    write_json(out/'pairing_diagnostics.json',pairing);write_json(out/'latency.json',latencies)
    summary=[];lookup={key(r['seed'],r['arm'],r['rho']):r for r in rows}
    for arm,rho in dict.fromkeys((r['arm'],r['rho']) for r in rows):
        rr=[lookup[key(s,arm,rho)] for s in SEEDS]
        summary.append({'arm':arm,'method':name(arm),'rho':rho,'branches':{b:{'mean':float(np.mean([r['scores'][b] for r in rr])),'sample_sd':float(np.std([r['scores'][b] for r in rr],ddof=1)),'seed3418':rr[2]['scores'][b]} for b in BRANCHES}})
    write_json(out/'group_summary.json',summary)
    card=dataset_card(p['data']);write_json(out/'data_card.json',card)
    # Complete branch/fusion metrics, uncertainty and provenance, no participant IDs.
    with (out/'results.csv').open('w',newline='') as f:
        fields=['id','seed','method','rho','r','h','M','S','cfp_f1_percent','oct_f1_percent','mean_branch_f1_percent','fused_f1_percent','cfp_gain_pp','oct_gain_pp','mean_gain_pp','best_epoch','stop_epoch','total_parameters','stored_bridge_parameters','effective_bridge_parameters','training_gpu_minutes','peak_process_mib','summary_sha256','predictions_sha256','checkpoint_sha256','parent_cfp_sha256','parent_oct_sha256']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            rho=r['rho'];cfg=r['configuration'];b=cfg['bridges'][0] if cfg['bridges'] else {};base=lookup[key(r['seed'],'no_bridge',None)];d=r['summary'];parents=cfg['parent_checkpoints']
            writer.writerow(dict(id=r['id'],seed=r['seed'],method=name(r['arm']),rho=rho_label(rho),r=int(256*rho) if rho and r['arm']!='learned' else 'N/A',h=int(8192*rho) if rho else 'N/A',M=b.get('M','N/A') if b.get('mode')!='linear_resample' else 'N/A (packing32)',S=b.get('S','N/A'),cfp_f1_percent=r['scores']['cfp'],oct_f1_percent=r['scores']['oct'],mean_branch_f1_percent=r['scores']['mean'],fused_f1_percent=100*r['late_fusion']['macro_f1'],cfp_gain_pp=r['scores']['cfp']-base['scores']['cfp'],oct_gain_pp=r['scores']['oct']-base['scores']['oct'],mean_gain_pp=r['scores']['mean']-base['scores']['mean'],best_epoch=d['selection']['joint']['best_epoch'],stop_epoch=d['epochs_ran'],total_parameters=d['trainable_parameters'],stored_bridge_parameters=r['stored_bridge_parameters'],effective_bridge_parameters=r['effective_bridge_parameters'],training_gpu_minutes=r['cost']['gpu_seconds']/60,peak_process_mib=r['cost']['sampled_peak_process_mib'],summary_sha256=r['result_hashes']['summary.json'],predictions_sha256=r['result_hashes']['selected_predictions.npz'],checkpoint_sha256=r['result_hashes']['selected.pt'],parent_cfp_sha256=parents['cfp']['sha256'],parent_oct_sha256=parents['oct']['sha256']))
    with (out/'all_prespecified_paired_differences.csv').open('w',newline='') as f:
        fields=['id','primary','seeds','rhos','branches','difference_pp','ci95_low','ci95_high','simultaneous95_low','simultaneous95_high','bootstrap_sd_pp','classification','zero_variance'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in stats['contrasts']:
            ci=d['simultaneous_ci95_pp'] or [None,None];w.writerow({k:d[k] for k in ['id','primary','seeds','rhos','branches','difference_pp','bootstrap_sd_pp','classification','zero_variance']}|dict(ci95_low=d['ci95_pp'][0],ci95_high=d['ci95_pp'][1],simultaneous95_low=ci[0],simultaneous95_high=ci[1]))
    # Secondary late-fusion differences vs same-seed no bridge, using shared indices.
    with (out/'late_fusion_vs_no_bridge.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','rho','seed_subset','difference_pp','ci95_low','ci95_high'])
        indices_by_key={key(r['seed'],r['arm'],r['rho']):i for i,r in enumerate(rows)}
        for arm,rho in dict.fromkeys((r['arm'],r['rho']) for r in rows):
            for seeds in [SEEDS]+[[s] for s in SEEDS]:
                ix=[indices_by_key[key(s,arm,rho)] for s in seeds];bi=[indices_by_key[key(s,'no_bridge',None)] for s in seeds];delta=(fusion_boot[:,ix]-fusion_boot[:,bi]).mean(1);point=np.mean([100*(rows[i]['late_fusion']['macro_f1']-rows[j]['late_fusion']['macro_f1']) for i,j in zip(ix,bi)])
                w.writerow([name(arm),rho_label(rho),seeds,point,*np.quantile(delta,[.025,.975])])
    primary=[d for d in stats['contrasts'] if d['primary']]
    plots=make_figures(out,rows,primary,summary,pairing,latencies) if make_plots else []
    result={'project':'R&B (Radon Bridge)','rows':rows,'group_summary':summary,'bootstrap':stats,'pairing':pairing,'latency':latencies,'data_card':card,'figures':plots,'protocol':p,'manifest':m,'ledger':ledger,'study_summary':read(root/'study_summary.json'),'gpu_acceptance':read(root/'gpu_acceptance.json'),'source_commit':p['source_commit'],'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},'test_used':False}
    write_json(out/'results.json',clean_public(result));write_json(out/'protocol.json',p)
    lines=['# R&B：结构化跨来源通信的受控机制基准','保留129项历史结果，新增57项训练，共186项；三种子3416/3417/3418，三档维数比例ρ=1/16、1/8、1/4。无重新预训练、无重复基拟合。',
        '所有方法复用同种子独立训练最佳父检查点，两条原生网络全参数训练、BN正常更新。新增单向保留自身块，仅禁止一个跨来源块；MMTM/注意力为单stage适配，参数、计算量和历史搜索预算不声称匹配。',
        '## 31项预定主要比较','差值均以表中方法顺序计算。普通区间与31项max-|t|近似同时区间并列；10000次参与者级配对bootstrap共享296人索引，先逐模型算F1，再对种子及适用的ρ等权平均。',
        '| 对比 | 差值pp | 普通95%区间 | 同时95%区间 | ±1pp研究范围分类 |','|---|---:|---|---|---|']
    labels={'supports_substantive_improvement':'支持实质提升','supports_substantive_decline':'支持实质下降','supports_practical_similarity':'支持实际接近','unresolved':'尚不能分辨','undefined_zero_variance':'零方差，区间未定义'}
    for d in primary:lines.append(f"| {d['id']} | {d['difference_pp']:+.2f} | {d['ci95_pp']} | {d['simultaneous_ci95_pp']} | {labels[d['classification']]} |")
    lines+=['## 解释边界','不能把尚不能分辨说成相同，也不能把描述性排名说成确定优势。±1pp是方法研究约定，不是临床阈值；区间条件于当前模型、父检查点与历史开发集选优。',
        '原3416/3417参与过配置筛选，3418单独展示；三个种子不把296位参与者扩充为888位。置换20次同样不增加参与者样本量。正确配对扰动只修改跨来源项，两个眼保持整体，双方同时打乱使用置换与逆置换。',
        '晚期融合F1是平均概率形成的新预测F1，与两分支F1平均不同。全部186项提供AUROC、precision/recall、混淆矩阵、NLL、二分类Brier和融合指标。',
        '## 后续独立验证','本轮回答机制与常用方法比较，不能用实验数量替代独立临床证据。后续另立协议扩展外部数据、任务、模型、临床终点与更强任务特定基线，锁定分析后验证泛化与生物医学价值；不承诺NBE录用。',
        '受限参与者概率、置换索引与特征不进入GitHub。公开结果为聚合指标、SHA、代码、协议及图源数据。UKB标签来自记录临床表型，不是新增专家影像金标准；平衡病例对照不能推广为真实患病率下预测值和校准。伦理、许可、来源队列与未核实信息见data_card.json待补清单。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines))
    write_json(out/'report_status.json',{'state':'complete','complete_trials':186,'total_trials':186,'new_trials':57,'new_diagnostics':57,'pairing_checkpoints':54,'latency_checkpoints':33,'seeds':SEEDS,'rhos':RHOS,'primary_comparisons':31,'visual_review':'pending','pdf_delivery':'pending_local_authoring_and_visual_review','test_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
