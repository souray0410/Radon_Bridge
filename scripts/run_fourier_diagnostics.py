"""Post-atlas CPU numerical audit; no training, inference, or test selection."""
import argparse,csv,fcntl,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radonbridge.artifacts import SOURCE,ARCHIVE,sha256,resolve
from radonbridge.projector import Projector,raw_operator,geometry
from radonbridge.fourier_diagnostics import reference_kernels,slice_errors,summarize,kernel_gain
from scripts.geometry_evidence import read,write
from scripts.report_sinogram_atlas import save,spatial,COLORS

SHAPES={2:dict(cfp=(28,28),oct=(16,12,12)),3:dict(cfp=(14,14),oct=(8,6,6)),4:dict(cfp=(7,7),oct=(4,3,3))}


def storage():
    if min(shutil.disk_usage(SOURCE).free,shutil.disk_usage(ARCHIVE).free)<100*1024**3:raise RuntimeError('Keep 100 GiB free on both disks')


def csv_rows(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)


def synthetic(out):
    configs=[(stage,source,32,S) for stage in SHAPES for source in ('cfp','oct') for S in (32,64,128)]
    configs += [(3,source,M,64) for source in ('cfp','oct') for M in (16,64)]
    rows=[];profiles=[];impulses=[];infeasible=[]
    for stage,source,M,S in configs:
        storage();shape=SHAPES[stage][source];identity=dict(stage=stage,source=source,M=M,S=S,shape=str(shape))
        try:p=Projector(shape,M,S).double()
        except ValueError as e:
            if 'budget exceeded' not in str(e):raise
            infeasible.append(dict(**identity,reason=str(e)));continue
        k=reference_kernels(shape,M,S);coords=k['coords'];gaussian=np.exp(-np.square(coords).sum(1)/(2*.18**2))
        fields=dict(gaussian=gaussian,modulated=gaussian*np.cos(2*np.pi*.3*k['native_nyquist']*coords[:,0]),
            signed_pair=np.exp(-np.square(coords-np.eye(len(shape))[0]*.2).sum(1)/(2*.13**2))-np.exp(-np.square(coords+np.eye(len(shape))[0]*.2).sum(1)/(2*.13**2)))
        for name,x in fields.items():
            z=p(torch.tensor(x.reshape(1,1,*shape)))[0].numpy().reshape(1,1,M,S)
            metrics,profile=slice_errors(x.reshape(1,1,-1),z,k)
            rows.append(dict(**identity,field=name,**{name:float(v[0]) if np.isfinite(v[0]) else None for name,v in metrics.items()}))
            profiles.append(dict(**identity,field=name,**profile))
        if M==32 and S==64:
            images=[]
            for offset in (0,1):
                index=[n//2 for n in shape];index[0]=min(shape[0]-1,index[0]+offset)
                x=torch.zeros((1,1,*shape),dtype=torch.float64);x[(0,0,*index)]=1
                y=p.backproject(p(x))[0,0].numpy();energy=y.reshape(-1)**2;origin=k['coords'][np.argmax(np.abs(y))]
                spread=float(np.sqrt(np.sum(energy*np.square(k['coords']-origin).sum(1))/energy.sum()))
                impulses.append(dict(**identity,offset=offset,input_index=index,peak_index=[int(v) for v in np.unravel_index(np.argmax(np.abs(y)),shape)],
                    maximum=float(np.abs(y).max()),l2_norm=float(np.linalg.norm(y)),energy_radius_about_peak=spread))
                images.append((x[0,0].numpy(),y))
            fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
            limit=max(float(np.abs(y).max()) for x,y in images)
            for row,(x,y) in enumerate(images):
                spatial(axes[row,0],x,f'Fixed input impulse | offset {row}',1)
                im=spatial(axes[row,1],y,f'Ordinary B A response | offset {row}',limit);fig.colorbar(im,ax=axes[row,1],shrink=.7)
            fig.suptitle(f'R&B | {source.upper()} stage{stage} | ordinary B A is not an inverse',fontsize=16)
            save(fig,out/'figures',f'BA_{source}_stage{stage}')
        del p;raw_operator.cache_clear()
        write(out.parent/'progress.json',dict(phase='synthetic_operators',completed_configs=len(rows)//3,infeasible=len(infeasible)))
    csv_rows(out/'synthetic_errors.csv',rows);write(out/'synthetic_profiles.json',profiles)
    write(out/'impulse_responses.json',impulses);write(out/'infeasible_geometries.json',infeasible)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,source in enumerate(('cfp','oct')):
        for col,stage in enumerate((2,3,4)):
            ax=axes[row,col]
            for field,color in [('gaussian','#3569a3'),('modulated','#c88322'),('signed_pair','#259681')]:
                values=sorted([r for r in rows if r['source']==source and r['stage']==stage and r['M']==32 and r['field']==field],key=lambda r:r['S'])
                ax.plot([r['S'] for r in values],[r['full_band_nrmse'] for r in values],'o-',color=color,label=field)
            ax.set(title=f'{source.upper()} stage{stage}',xlabel='S samples',ylabel='Fourier-slice relative L2 error');ax.grid(alpha=.2)
            if row==col==0:ax.legend(fontsize=8)
    fig.suptitle('R&B | Discrete quadrature versus exact Fourier transform of the multilinear interpolant\nSynthetic fields; gaps denote infeasible geometry, not poor predictive performance',fontsize=15)
    save(fig,out/'figures','fourier_synthetic_errors')
    return dict(attempted=len(configs),accepted=len(rows)//3,infeasible=len(infeasible))


def features(jobs,accepted,out):
    chosen=[j for j in jobs if j['display']['arm']=='svd_radon' and j['display']['r']==16 and j['display']['stages']==[3]]
    if len(chosen)!=3:raise ValueError('Expected prespecified three feature probes')
    rows=[];profiles=[];evidence=[]
    kernels={source:reference_kernels(shape,32,64) for source,shape in SHAPES[3].items()}
    for j in chosen:
        directory=Path(accepted[j['job_id']]['summary_path']).parent
        for phase in ('constructed_initial','selected'):
            for split in ('train','validation','test'):
                private=directory/'private'/phase/split;manifest=read(private/'artifact_manifest.json')
                for file,digest in manifest.items():
                    if sha256(directory/file)!=digest:raise ValueError('Private atlas artifact changed')
                sample=read(private/'sample_manifest.json');ids=sample['ids']
                if len(ids)!=16 or ids!=sorted(ids):raise ValueError('Wrong prespecified sample')
                evidence.append(dict(seed=j['display']['seed'],phase=phase,split=split,sample_manifest_sha256=sha256(private/'sample_manifest.json'),participants=16,eyes=2,rank_channels=[0,1,2]))
                for source in ('cfp','oct'):
                    xs=[];zs=[]
                    for identifier in ids:
                        with np.load(private/(identifier+'__bridge0_'+source+'_stage3.npz'),allow_pickle=False) as a:
                            xs.append(a['compressed_native'].reshape(6,-1));zs.append(a['projection'].reshape(6,32,64))
                    metrics,profile=slice_errors(np.asarray(xs,dtype=np.float64),np.asarray(zs,dtype=np.float64),kernels[source])
                    identity=dict(seed=j['display']['seed'],phase=phase,split=split,source=source)
                    for metric,v in summarize(metrics).items():rows.append(dict(**identity,metric=metric,**v))
                    profiles.append(dict(**identity,**profile))
        write(out.parent/'progress.json',dict(phase='fixed_feature_probes',completed_seeds=len(evidence)//6))
    csv_rows(out/'feature_probe_errors.csv',rows);write(out/'feature_probe_profiles.json',profiles);write(out/'probe_provenance.json',evidence)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,source in enumerate(('cfp','oct')):
        for col,split in enumerate(('train','validation','test')):
            ax=axes[row,col]
            for phase,color in [('constructed_initial','#8a96a7'),('selected',COLORS[split])]:
                ps=[p for p in profiles if p['source']==source and p['split']==split and p['phase']==phase]
                curves=[]
                for p in ps:
                    denom=np.mean(p['reference_squared']);curves.append(np.sqrt(np.asarray(p['error_squared'])/denom) if denom>0 else np.full(33,np.nan))
                a=np.asarray(curves);freq=ps[0]['frequency_fraction'];mu=a.mean(0);sd=a.std(0,ddof=1)
                ax.plot(freq,mu,color=color,label=phase.replace('_',' '));ax.fill_between(freq,np.maximum(0,mu-sd),mu+sd,color=color,alpha=.15)
            ax.set(title=source.upper()+' | '+split,xlabel='Fraction of common conservative Nyquist',ylabel='Error amplitude / whole-band reference RMS');ax.grid(alpha=.2)
            if row==col==0:ax.legend(fontsize=8)
    fig.suptitle('R&B | Actual feature probes: Fourier-slice numerical discrepancy\nFixed 16 people per split, both eyes, channels0/1/2; mean and SD over 3 seeds; no new inference',fontsize=15)
    save(fig,out/'figures','fourier_feature_probes')


def weights(jobs,out):
    rows=[];profiles=[]
    for count,j in enumerate(jobs):
        storage();m=j['model'];path=resolve(m['checkpoint_path'])
        if sha256(path)!=m['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        for index,b in enumerate(m['configuration']['bridges']):
            module=state[f'bridge_{index}_exchange'];full=module['mixer.conv.weight'];mask=module['mixer.mask']
            widths=[module[f'channel_bases.{i}.q'].shape[1]*b['M'] for i in (0,1)];edges=np.cumsum([0]+widths)
            for dst in (0,1):
                for src in (0,1):
                    w=full[edges[dst]:edges[dst+1],edges[src]:edges[src+1]]*mask[edges[dst]:edges[dst+1],edges[src]:edges[src+1]]
                    response=kernel_gain(w);identity=dict(**j['display'],bridge=index,sender=b['nodes'][src],receiver=b['nodes'][dst],role='self' if src==dst else 'cross')
                    profiles.append(dict(**identity,**response));gain=np.asarray(response['frobenius_gain'])
                    rows.append(dict(**identity,dc_gain=float(gain[0]),nyquist_gain=float(gain[-1]),minimum_gain=float(gain.min()),maximum_gain=float(gain.max()),
                        zero_operator=response['zero_operator'],width_in=widths[src],width_out=widths[dst],kernel=full.shape[-1],checkpoint_sha256=m['checkpoint_sha256']))
                    del w
        del state,full,mask,module
        write(out.parent/'progress.json',dict(phase='selected_kernel_response',accepted=count+1,total=len(jobs)))
    csv_rows(out/'kernel_gain_summary.csv',rows);write(out/'kernel_response_profiles.json',profiles)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for dst in (0,1):
        for src in (0,1):
            ax=axes[dst,src];sender=('cfp_stage3','oct_stage3')[src];receiver=('cfp_stage3','oct_stage3')[dst]
            for r,color in [(16,'#3569a3'),(32,'#c88322'),(64,'#259681')]:
                chosen=[p for p in profiles if p['arm']=='svd_radon' and p['r']==r and p['sender']==sender and p['receiver']==receiver]
                a=np.asarray([p['rms_per_channel_pair_gain'] for p in chosen]);mu=a.mean(0);sd=a.std(0,ddof=1);f=chosen[0]['cycles_per_sample']
                ax.plot(f,mu,color=color,label=f'r{r} / h{r*32}');ax.fill_between(f,np.maximum(0,mu-sd),mu+sd,color=color,alpha=.15)
            ax.set(title=sender+' -> '+receiver,xlabel='Cycles / S sample',ylabel='RMS gain per channel pair');ax.grid(alpha=.2)
            if dst==src==0:ax.legend()
    fig.suptitle('R&B | Learned matrix-valued kernel frequency gain\nMean and SD over 3 seeds; kernel response is not the observed signal power or a scalar ramp filter',fontsize=15)
    save(fig,out/'figures','kernel_frequency_gains')


def run(a):
    a.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (a.output/'controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'status.json').exists() and read(a.output/'status.json')['state']=='complete':raise ValueError('Do not rerun accepted diagnostics')
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Use immutable code')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        while read(a.atlas/'status.json').get('state')!='complete':
            write(a.output/'status.json',dict(state='waiting_for_atlas_report',controller_pid=os.getpid(),source_commit=commit,updated_at=time.time()));time.sleep(30)
        status=read(a.atlas/'status.json');lock=Path(status['lock_directory']);registration=read(lock/'candidate_lock.json')
        for file,digest in registration['files'].items():
            if sha256(lock/file)!=digest:raise ValueError('Atlas lock changed')
        jobs=read(lock/'jobs.json');accepted=read(a.atlas/'all/accepted_jobs.json')
        if len(accepted)!=21 or len(jobs)!=21:raise ValueError('Incomplete atlas')
        for j in jobs:
            entry=accepted[j['job_id']]
            if sha256(entry['summary_path'])!=entry['summary_sha256']:raise ValueError('Atlas acceptance changed')
            s=read(entry['summary_path'])
            if s['state']!='accepted':raise ValueError('Unaccepted atlas')
            for file,digest in s['prediction_files'].items():
                if sha256(Path(entry['summary_path']).parent/file)!=digest:raise ValueError('Atlas output changed')
        os.umask(0o077);torch.set_num_threads(3);start=time.monotonic();out=a.output/'report';out.mkdir(mode=0o700)
        write(a.output/'status.json',dict(state='running',controller_pid=os.getpid(),source_commit=commit,updated_at=time.time()))
        geometry_count=synthetic(out);features(jobs,accepted,out);weights(jobs,out)
        shutil.copyfile('experiments/geometry_mechanism/FOURIER_DIAGNOSTICS.zh-CN.md',out/'protocol.zh-CN.md')
        (out/'README.zh-CN.md').write_text('''# R&B：傅里叶与返回算子诊断\n\n本报告属于数值验证与事后描述性分析，不增加训练或主要test假设。\n\nsynthetic_errors.csv给出22个预定几何配置中可行项的切片关系误差，infeasible_geometries.json保留受原实现规模保护的项目。插值场解析傅里叶参考与离散投影不必相等，误差反映有限网格、积分及采样，不是定理失效。\n\nfeature_probe_errors.csv仅针对事先固定的每集合16名参与者、双眼、前3个基通道；不是全队列估计。频率范围同时受原空间网格及S采样限制。不能把高频小能量处的相对误差直接当作任务信息损失。\n\nkernel_response_profiles.json包含21个选中模型全部位置和来源块的总增益与每通道对RMS增益。卷积沿S局部平移共享，方向与通道之间仍做混合，不能据此宣称原图空间平移或旋转等变，也不能将该响应叫作整个网络频率响应。\n\nBA图是普通反投影对投影的返回：扩散或模糊不意味着分类效果必然下降。它与经典CT滤波反投影的目标不同。后续大数据与其他网络需重新报告数值误差、尺度定义和经验效应，不能预先承诺核心性能结论不变。\n''')
        write(out/'provenance.json',dict(source_commit=commit,source_atlas=str(a.atlas),source_atlas_lock_sha256=sha256(lock/'candidate_lock.json'),
            source_acceptance_sha256=sha256(a.atlas/'all/accepted_jobs.json'),geometry=geometry_count,checkpoints=21,
            feature_probe_models=3,feature_probe_participants_per_split=16,training_tasks=0,gpu_used=False,new_model_inference=False,
            post_hoc_descriptive=True,seconds=time.monotonic()-start))
        write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
        write(a.output/'status.json',dict(state='complete',controller_pid=os.getpid(),source_commit=commit,report=str(out),visual_review_required=True,updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException:
        write(a.output/'failure.json',dict(state='needs_attention',error=traceback.format_exc(),updated_at=time.time()));raise
