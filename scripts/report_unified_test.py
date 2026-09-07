"""All locked models, shared participant bootstrap, family/global simultaneous CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.sparse import csr_matrix
from radonbridge.artifacts import sha256, resolve
from radonbridge.metrics import classification_metrics
from radonbridge.benchmark_statistics import classify
from scripts.geometry_evidence import read, write
from scripts.report_geometry_mechanism import csv_write


def bootstrap_f1(y,pred,indices):
    """Binary macro-F1; explicit two classes, seeds/models never form extra participants."""
    flat=pred.reshape(-1,len(y))
    point=[]
    for c in (0,1):
        tp=((flat==c)&(y==c)).sum(1);den=(flat==c).sum(1)+(y==c).sum()
        point.append(np.divide(2*tp,den,out=np.zeros(len(flat),dtype=float),where=den>0))
    point=50*(point[0]+point[1])
    draws=np.zeros((len(indices),len(flat)),dtype=np.float64)
    for start in range(0,len(indices),100):
        ix=indices[start:start+100];counts=np.zeros((len(ix),len(y)),dtype=np.float64)
        np.add.at(counts,(np.arange(len(ix))[:,None],ix),1)
        for c in (0,1):
            tp=counts@((flat==c)&(y==c)).T.astype(float)
            den=counts@(flat==c).T.astype(float)+(counts@(y==c))[:,None]
            draws[start:start+len(ix)]+=50*np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    return point,draws


def intervals(defs,observed,draws):
    sd=draws.std(0,ddof=1);active=sd>1e-10
    primary=np.array([d['primary'] and d['estimable'] for d in defs])
    def critical(mask):
        mask=mask&active
        return float(np.quantile(np.max(np.abs((draws[:,mask]-observed[None,mask])/sd[None,mask]),axis=1),.95)) if mask.any() else None
    global_t=critical(primary);families=sorted({a['family'] for d in defs for a in d['aliases'] if a['primary']})
    masks={f:np.array([d['estimable'] and any(a['primary'] and a['family']==f for a in d['aliases']) for d in defs]) for f in families}
    family_t={f:critical(mask) for f,mask in masks.items()}
    rows=[]
    for j,d in enumerate(defs):
        if not d['estimable']:
            rows.append(dict(contrast_id=d['contrast_id'],classification='not_estimable',aliases=d['aliases'],primary=d['primary']));continue
        ci=lambda t:[float(observed[j]-t*sd[j]),float(observed[j]+t*sd[j])] if active[j] and t is not None else None
        simultaneous=ci(global_t) if d['primary'] else None
        fs={f:ci(family_t[f]) for f in families if masks[f][j]}
        rows.append(dict(contrast_id=d['contrast_id'],aliases=d['aliases'],primary=d['primary'],difference_pp=float(observed[j]),
             ci95_pp=np.quantile(draws[:,j],[.025,.975]).tolist(),global_simultaneous_ci95_pp=simultaneous,
             family_simultaneous_ci95_pp=fs,bootstrap_sd_pp=float(sd[j]),zero_variance=not bool(active[j]),
             classification=classify(simultaneous) if simultaneous else 'undefined_zero_variance' if not active[j] else 'secondary'))
    return rows,dict(global_max_abs_t=global_t,family_max_abs_t=family_t,zero_sd_tolerance_pp=1e-10)


def build(lock,run):
    start=time.monotonic();out=run/'report';out.mkdir(exist_ok=True)
    cfg=read(lock/'candidate_lock.json');defs=read(lock/'comparisons.json');views=read(lock/'model_views.json')
    jobs=read(lock/'jobs.json');accepted=read(run/'test/accepted_jobs.json')
    assert len(accepted)==len(jobs)==855 and read(run/'test/status.json')['state']=='complete'
    for name,digest in cfg['files'].items():assert sha256(lock/name)==digest
    summaries={}
    for job in jobs:
        a=accepted[job['job_id']];s=read(a['summary_path'])
        assert sha256(a['summary_path'])==a['summary_sha256'] and s['state']=='accepted' and s['test_used'] and s['split']=='test'
        for name,digest in s['prediction_files'].items():assert sha256(Path(a['summary_path']).parent/name)==digest
        summaries[job['job_id']]=s
    lookup={v['view_id']:i for i,v in enumerate(views)};wi=[];wj=[];wv=[]
    for j,d in enumerate(defs):
        for t in d['terms']:wi.append(j);wj.append(2*lookup[t['view_id']]+(t['branch']=='oct'));wv.append(t['weight'])
    weights=csr_matrix((wv,(wi,wj)),shape=(len(defs),len(views)*2))
    metrics_table=[];results={};boots={};bootstrap_hashes={};point_arrays={}
    for split,n in (('development',296),('test',290)):
        ids=y=None;ps=[]
        for v in views:
            path=Path(v['development_prediction']) if split=='development' else Path(accepted[v['job_id']]['summary_path']).parent/v['file']
            with np.load(resolve(path),allow_pickle=False) as z:
                if ids is None:ids=z['ids'].copy();y=z['y'].copy()
                assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y']) and len(y)==n
                ps.append(np.stack([z['cfp'],z['oct']]))
        p=np.stack(ps);indices=np.random.default_rng(cfg['statistics']['seed']+(split=='test')).integers(0,n,(10000,n))
        # Restricted index arrays stay outside the public aggregate directory.
        np.save(run/(split+'_bootstrap_indices.npy'),indices);bootstrap_hashes[split]=hashlib.sha256(indices.tobytes()).hexdigest()
        point,draw=bootstrap_f1(y,p.argmax(-1),indices)
        contrast_point=weights@point;contrast_draw=np.asarray(weights@draw.T).T
        rows,critical=intervals(defs,contrast_point,contrast_draw);results[split]=dict(contrasts=rows,**critical)
        boots[split]=contrast_draw;point_arrays[split]=contrast_point
        for i,v in enumerate(views):
            row=dict(view_id=v['view_id'],job_id=v['job_id'],split=split,participants=n)
            for name,prob in [('cfp',p[i,0]),('oct',p[i,1]),('fixed_probability_fusion',p[i].mean(0))]:
                m=classification_metrics(y,prob);m['brier_binary']=float(np.mean((prob[:,1]-y)**2))
                row.update({name+'_'+k:value for k,value in m.items()})
            row['branch_mean_macro_f1']=(row['cfp_macro_f1']+row['oct_macro_f1'])/2
            metrics_table.append(row)
        write(out/(split+'_statistics.json'),dict(resamples=10000,participants=n,shared_indices_sha256=bootstrap_hashes[split],**results[split]))
        write(run/'report_progress.json',dict(state=split+'_statistics_complete',updated_at=time.time()))
    # Independent cohort resampling: never pair development person i with test person i.
    delta_rows,critical=intervals(defs,point_arrays['test']-point_arrays['development'],boots['test']-boots['development'])
    write(out/'test_minus_development_effect_change.json',dict(interpretation='Independent participant draws in the two cohorts; fixed models and fixed contrast weights',contrasts=delta_rows,**critical))
    csv_write(out/'all_model_metrics.csv',metrics_table)
    for split in results:csv_write(out/(split+'_all_contrasts.csv'),results[split]['contrasts'])
    refs=read(lock/'model_reference_mapping.json');by_metric={(r['view_id'],r['split']):r for r in metrics_table};combined=[]
    for ref in refs:
        row={k:ref.get(k) for k in ('reference_id','source_group','source_row_id','seed','arm','structure','rho','category','regime','model_view_id')}
        for split in ('development','test'):
            row.update({split+'_'+k:v for k,v in by_metric[(ref['model_view_id'],split)].items() if k not in ('view_id','job_id','split')})
        if ref.get('summary_path'):
            s=read(resolve(ref['summary_path']));c=s['configuration']
            row.update(backbone_lr=c.get('backbone_lr'),training_stage=c.get('training_stage'),parameters=s.get('parameters'),
                best_epoch=s.get('selection',{}).get('joint',{}).get('best_epoch'),stop_epoch=s.get('epochs_ran'),
                training_seconds=s.get('seconds'),peak_training_reserved_mib=s.get('peak_reserved_mib'),checkpoint_sha256=ref['training_checkpoint_sha256'])
        combined.append(row)
    csv_write(out/'all_references_development_test.csv',combined)
    primary=[]
    for a,b in zip(results['development']['contrasts'],results['test']['contrasts']):
        if b['primary']:
            primary.append(dict(contrast_id=b['contrast_id'],names=[x['id'] for x in b['aliases']],development=a,test=b,
                interpretation='Judge test from simultaneous intervals and branch/seed strata; non-significance is not proof of absence'))
    write(out/'claims_development_test.json',primary)
    pairing=[]
    for job in jobs:
        if job['kind']=='pairing':pairing.append(dict(model_view_id=job['model']['model_view_id'],**summaries[job['job_id']]['detail']['pairing']))
    write(out/'pairing_diagnostics.json',pairing)
    write(out/'accounting.json',dict(inference_and_diagnostic_gpu_minutes=sum(s['seconds'] for s in summaries.values())/60,
        per_job=[{k:s.get(k) for k in ('job_id','kind','seconds','peak_reserved_mib','source_commit')} for s in summaries.values()],
        training_ledger_policy='Historical and current training ledgers retained at their original sources; do not double-count reused models'))
    # Primary confidence-interval panels preserve registration order and include negative findings.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    valid=[r for r in results['test']['contrasts'] if r['primary'] and 'difference_pp' in r]
    for start in range(0,len(valid),16):
        part=valid[start:start+16];fig,ax=plt.subplots(figsize=(16,9))
        for i,d in enumerate(part):
            ax.hlines(i,*d['ci95_pp'],color='#9bc5c9',lw=6)
            if d['global_simultaneous_ci95_pp']:ax.hlines(i,*d['global_simultaneous_ci95_pp'],color='#007e87',lw=1.5)
            ax.scatter(d['difference_pp'],i,color='#183c47',s=18)
        ax.axvspan(-1,1,color='#eceff1');ax.axvline(0,color='#65747b',lw=.8)
        ax.set(yticks=range(len(part)),yticklabels=[d['aliases'][0]['id'] for d in part],xlabel='Paired difference in macro-F1 (percentage points)')
        ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
        fig.suptitle('R&B (Radon Bridge) | Locked UKB test contrasts | '+str(start//16+1))
        fig.text(.04,.025,'290 participants. Thick: ordinary 95% CI; thin: global simultaneous 95% CI. ±1 pp is a research convention, not a clinical threshold.',fontsize=9)
        fig.subplots_adjust(left=.48,right=.97,top=.90,bottom=.12)
        for ext in ('png','svg','pdf'):fig.savefig(out/f'test_primary_{start//16+1:02d}.{ext}',dpi=180)
        plt.close(fig)
    text=['# R&B（Radon Bridge）：统一 development/test 核验','',
        '777个锁定模型视图；24个宿主四状态诊断；54个配对扰动检查点。训练完成并锁定模型/比较后才执行test，没有根据test选epoch、配置或阈值。',
        'Train1264、development296、test290。三种子与不同配置是固定模型的重复条件，不是额外参与者。早期LOOK使用记录保留，不能宣称全研究历史从未触及该test队列。',
        '主指标为CFP/OCT macro-F1的平均；固定概率平均融合单列。历史探索、当前模型条件下的test核验与外部临床验证是不同证据层级。',
        '10000次参与者配对bootstrap；同一集合内所有模型共享索引。提供普通、原比较族及全局max-|t|同时区间。两集合增益变化使用独立重采样。',
        '全局区间完全高于+1 pp／低于−1 pp／位于±1 pp内分别支持实质提升／下降／实际接近；其余尚不能分辨。零方差与不可估计项明确保留。',
        '配对删除/打乱及宿主开关说明当前模型的功能依赖，不能解释成重新训练增益或临床因果效应。所有逐种子、学习率和原研究引用见完整表。',
        '', '|主要比较|Dev差值 pp|Test差值 pp|Test全局95%区间|判定|','|---|---:|---:|---|---|']
    for item in primary:
        a=item['development'];b=item['test']
        text.append(f"|{item['names'][0]}|{a.get('difference_pp','—')}|{b.get('difference_pp','—')}|{b.get('global_simultaneous_ci95_pp','—')}|{b['classification']}|")
    text+=['','接下来扩大队列、替换骨干和扩展疾病/器官另立协议，不能把当前test重新用于调参。标签来源、临床表型、抽样平衡及独立外部验证限制沿用已锁定协议。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n')
    write(out/'provenance.json',dict(candidate_lock_sha256=sha256(lock/'candidate_lock.json'),authorization=read(lock/'test_authorization.json'),
        source_manifest_hashes=read(lock/'source_manifests.json'),environment=dict(python=platform.python_version(),numpy=np.__version__),
        primary_comparisons=cfg['primary_comparisons'],no_participant_level_export=True,seconds=time.monotonic()-start))
    write(out/'artifact_manifest.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'})
    write(out/'report_status.json',dict(state='complete',test_used=True,models=777,component_checkpoints=24,pairing_checkpoints=54,
        figure_visual_review='pending_local_delivery',report_directory=str(out),updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();build(a.lock,a.run)
