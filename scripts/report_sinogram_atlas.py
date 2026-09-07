"""Aggregate-only scientific figures, with a separate restricted case appendix."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radonbridge.artifacts import sha256
from radonbridge.projector import Projector,geometry
from radonbridge.sinogram_atlas import angle_display,PACKETS,SPATIAL
from scripts.geometry_evidence import read,write

COLORS={'train':'#3569a3','validation':'#c88322','test':'#259681'}
SPLITS=('train','validation','test')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'medium','figure.facecolor':'#fafbfD','axes.facecolor':'white','savefig.facecolor':'#fafbfD',
    'svg.fonttype':'none','pdf.fonttype':42})


def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True);fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight')
    for ext in ('svg','pdf'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)


def packet(ax,value,meta,title,vmax=None,signed=False):
    a=np.asarray(value);vmax=max(float(vmax if vmax is not None else np.abs(a).max()),1e-12)
    geom=meta['geometry'];kw=dict(cmap='RdBu_r' if signed else 'magma',vmin=-vmax if signed else 0,vmax=vmax)
    s=np.linspace(*geom['support'],meta['S'])
    if meta['mode']=='radon' and len(meta['shape'])==2:
        a,angles,_=angle_display(a,geom['directions']);im=ax.pcolormesh(s,np.degrees(angles),a,shading='nearest',**kw)
        ax.set_ylabel('Normal angle (degrees)');ax.set_xlabel('Signed distance (feature-grid units)')
    else:
        im=ax.imshow(a,origin='lower',aspect='auto',extent=[s[0],s[-1],-.5,meta['M']-.5],**kw)
        ax.set_ylabel('EEM direction index' if meta['mode']=='radon' else 'Packing index (not an angle)')
        ax.set_xlabel('Signed distance (feature-grid units)' if meta['mode']=='radon' else 'Packed S sample (rescaled index)')
        if meta['mode']!='radon':ax.set_xticks(np.linspace(s[0],s[-1],3),['0',str((meta['S']-1)//2),str(meta['S']-1)])
    clipped=float((np.abs(a)>vmax).mean())*100
    ax.set_title(textwrap.fill(title,52)+f'\nclipped {clipped:.1f}%',fontsize=9)
    return im


def spatial(ax,a,title,vmax=None,signed=False):
    a=np.asarray(a);is3d=a.ndim==3
    if is3d:
        # Three central orthogonal planes, with visible gaps; no anatomical axis claim.
        planes=[np.take(a,a.shape[d]//2,axis=d) for d in range(3)]
        h=max(p.shape[0] for p in planes);w=sum(p.shape[1] for p in planes)+2*2
        canvas=np.full((h,w),np.nan);x=0
        for p in planes:canvas[:p.shape[0],x:x+p.shape[1]]=p;x+=p.shape[1]+2
        a=canvas;title+=' | central axes 0 / 1 / 2'
    vmax=max(float(vmax if vmax is not None else np.nanmax(np.abs(a))),1e-12)
    im=ax.imshow(a,origin='lower',cmap='RdBu_r' if signed else 'magma',vmin=-vmax if signed else 0,vmax=vmax,interpolation='nearest')
    ax.set_title(textwrap.fill(title,48)+f'\nclipped {np.mean(np.abs(a[np.isfinite(a)])>vmax)*100:.1f}%',fontsize=9)
    ax.set_xlabel('Native feature-grid index');ax.set_ylabel('Native feature-grid index')
    if is3d:
        ax.set_xticks([]);ax.set_xlabel('Three central feature-grid views, separated by gaps')
    return im


def synthetic(out):
    """Use the actual R&B operators on a public, synthetic Gaussian point."""
    record={}
    for dim,shape in [(2,(40,40)),(3,(12,12,12))]:
        p=Projector(shape,32,64).double();steps,radius,s,_=geometry(shape,64)
        coords=np.stack(np.meshgrid(*[(np.arange(n)-(n-1)/2)*h for n,h in zip(shape,steps)],indexing='ij'),axis=-1)
        center=np.array([.25,-.2,.15][:dim]);width=.06 if dim==2 else .11
        x=np.exp(-np.square(coords-center).sum(-1)/(2*width**2));t=torch.tensor(x)[None,None]
        z=p(t)[0].numpy();returned=p.backproject(torch.tensor(z)[None])[0,0].numpy()
        dirs=np.asarray(p.metadata['directions']);expected=dirs@center
        peak=s[z.argmax(1)];error=float(np.max(np.abs(peak-expected)))
        if error>2*max(steps)+2*(s[1]-s[0]):raise ValueError('Synthetic point projection does not follow n dot x')
        meta=dict(geometry=p.metadata,M=32,S=64,shape=list(shape),mode='radon')
        fig=plt.figure(figsize=(15,4.7),layout='constrained');axs=[fig.add_subplot(1,3,i+1) for i in range(3)]
        spatial(axs[0],x,'Synthetic Gaussian point')
        im=packet(axs[1],z,meta,'Actual R&B projection')
        if dim==2:
            _,ang,_=angle_display(z,dirs);axs[1].plot(np.cos(ang)*center[0]+np.sin(ang)*center[1],np.degrees(ang),'c--',lw=1.3,label='s = n dot x0');axs[1].legend(fontsize=8)
        else:axs[1].plot(expected,np.arange(32),'c.',ms=3)
        fig.colorbar(im,ax=axs[1],shrink=.7,label='Projection amplitude')
        spatial(axs[2],returned,'Ordinary backprojection (blurred; not an inverse)')
        fig.suptitle(f'R&B | {dim}D projection geometry on synthetic data',fontsize=17)
        save(fig,out,f'01_synthetic_{dim}d')
        record[str(dim)]=dict(center=center.tolist(),shape=shape,peak_distance_max_error=error,directions=dirs.tolist(),support=p.metadata['support'])
    write(out/'synthetic_geometry.json',record)


def report(lock,run):
    out=run/'report';out.mkdir(exist_ok=True,mode=0o700);plots=out/'figures';plots.mkdir(exist_ok=True)
    accepted=read(run/'all/accepted_jobs.json');jobs=read(lock/'jobs.json')
    if len(accepted)!=21:raise ValueError('Do not publish an incomplete atlas as complete')
    records=[];rows=[]
    for j in jobs:
        a=accepted[j['job_id']];summary=Path(a['summary_path']);s=read(summary)
        if sha256(summary)!=a['summary_sha256'] or s['state']!='accepted':raise ValueError('Atlas acceptance changed')
        for name,h in s['prediction_files'].items():
            if sha256(summary.parent/name)!=h:raise ValueError('Atlas evidence changed')
        agg=read(summary.parent/'aggregates.json');records.append(dict(job=j,aggregate=agg,directory=summary.parent))
        # Public evidence contains model hashes, numerical summaries and structural metadata only.
        target=out/'aggregate_data'/j['job_id'];target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(summary.parent/'aggregates.json',target/'aggregates.json')
        write(target/'model.json',dict(display=j['display'],model_view_id=j['model']['model_view_id'],
            checkpoint_sha256=j['model']['checkpoint_sha256'],source_reference=j['source_reference']))
        for phase_split,groups in agg.items():
            phase,split=phase_split.split('/')
            for key,g in groups.items():
                m=g['metadata']
                for metric,v in g['scalars'].items():rows.append(dict(**j['display'],phase=phase,split=split,source=m['source'],bridge=m['bridge'],
                    C=m['C'],rho=m['r']/m['C'],h=m['r']*m['M'],M=m['M'],S=m['S'],k=m['k'],group=g['group'],participants=g['participants'],metric=metric,**v))
    with (out/'scalar_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # All selection below uses declared structure, never metrics or predictions.
    def chosen(arm='svd_radon',r=16,stages=(3,)):
        return sorted([v for v in records if v['job']['display']['arm']==arm and v['job']['display']['r']==r and tuple(v['job']['display']['stages'])==tuple(stages)],key=lambda v:v['job']['display']['seed'])
    def group(v,split='train',phase='selected',source='cfp_stage3',bridge=0,label='all'):
        return v['aggregate'][phase+'/'+split][f'bridge{bridge}_{source}/{label}']
    def mean_array(vs,field,**kw):return np.mean([np.asarray(group(v,**kw)['arrays'][field]) for v in vs],axis=0)
    base=chosen();synthetic(plots)
    for source in ('cfp_stage3','oct_stage3'):
        meta=group(base[0],source=source)['metadata']
        fig,axes=plt.subplots(4,3,figsize=(15,13),layout='constrained')
        for i,field in enumerate(PACKETS):
            train=np.sqrt(mean_array(base,field+'_ms',source=source));limit=float(train.max())
            for col,split in enumerate(SPLITS):
                z=np.sqrt(mean_array(base,field+'_ms',source=source,split=split))
                im=packet(axes[i,col],z,meta,f'{split} | {field.replace("_"," ")}',limit)
                fig.colorbar(im,ax=axes[i,col],shrink=.75,label='RMS amplitude')
        fig.suptitle(f'R&B | {source.upper()} projected messages\nSVD r16 / h512 / M32 / S64 / k3 | RMS over people, eyes, channels and 3 seeds',fontsize=16)
        save(fig,plots,'02_projected_flow_'+source)
        fig,axes=plt.subplots(4,3,figsize=(15,12),layout='constrained')
        for i,field in enumerate(('input','self_return','cross_return','total_return')):
            limit=float(np.sqrt(mean_array(base,field+'_ms',source=source)).max())
            for col,split in enumerate(SPLITS):
                z=np.sqrt(mean_array(base,field+'_ms',source=source,split=split));im=spatial(axes[i,col],z,f'{split} | {field}',limit)
                fig.colorbar(im,ax=axes[i,col],shrink=.7,label='RMS amplitude')
        fig.suptitle(f'R&B | {source.upper()} native feature and returned residual\nFixed train color limit per row; maps are feature grids, not lesion localization',fontsize=16)
        save(fig,plots,'03_return_maps_'+source)
        # Label differences are descriptive phenotype associations, not saliency or causality.
        fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
        for row,field in enumerate(('projection_ms','cross_message_ms')):
            def diff(split):return np.sqrt(mean_array(base,field,source=source,split=split,label='label1'))-np.sqrt(mean_array(base,field,source=source,split=split,label='label0'))
            limit=float(np.abs(diff('train')).max())
            for col,split in enumerate(SPLITS):
                im=packet(axes[row,col],diff(split),meta,f'{split} | label1 minus label0 | {field}',limit,signed=True)
                fig.colorbar(im,ax=axes[row,col],shrink=.7,label='Difference of group RMS')
        fig.suptitle(f'R&B | {source.upper()} class-conditional descriptive differences\nNo diagnostic threshold, no statistical or anatomical interpretation from these maps alone',fontsize=15)
        save(fig,plots,'04_label_difference_'+source)
    meta=group(base[0],source='oct_stage3')['metadata'];directions=np.array(meta['geometry']['directions'])
    fig=plt.figure(figsize=(14,4.5),layout='constrained')
    train=np.sqrt(mean_array(base,'projection_ms',source='oct_stage3').mean(-1));limit=float(train.max())
    for i,split in enumerate(SPLITS):
        ax=fig.add_subplot(1,3,i+1,projection='3d');e=np.sqrt(mean_array(base,'projection_ms',source='oct_stage3',split=split).mean(-1))
        q=ax.scatter(*directions.T,c=e,cmap='magma',vmin=0,vmax=limit,s=45)
        for n,xyz in enumerate(directions):ax.text(*xyz,str(n),fontsize=6)
        ax.set(xlabel='Normal component 0',ylabel='Normal component 1',zlabel='Normal component 2',title=split)
        ax.set_box_aspect((1,1,1));fig.colorbar(q,ax=ax,shrink=.6,label='Direction RMS')
    fig.suptitle('R&B | OCT directions occupy a sphere, not a single angle axis',fontsize=17)
    save(fig,plots,'05_oct_spherical_directions')
    # Seed-level points are shown separately, and SD is over three models, not participants.
    def metricplot(name,sets,labels):
        metrics=('channel_energy_retained','total_relative','self_cross_cosine')
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for row,source in enumerate(('cfp_stage3','oct_stage3')):
            for col,metric in enumerate(metrics):
                ax=axes[row,col]
                for si,split in enumerate(SPLITS):
                    vals=np.array([[group(v,split=split,source=source)['scalars'][metric]['mean'] for v in vs] for vs in sets],float)
                    x=np.arange(len(sets))+(si-1)*.14
                    ax.errorbar(x,vals.mean(1),yerr=vals.std(1,ddof=1),color=COLORS[split],marker='o',capsize=3,label=split)
                    for k in range(vals.shape[1]):ax.scatter(x,vals[:,k],s=14,color=COLORS[split],alpha=.5)
                ax.set_xticks(np.arange(len(sets)),labels);ax.set_title(source+' | '+metric.replace('_',' '));ax.grid(axis='y',alpha=.2)
                if row==0 and col==0:ax.legend()
        fig.suptitle('R&B | '+name+'\nMean and sample SD over 3 seeds; points show all seeds',fontsize=17)
        save(fig,plots,name.lower().replace(' ','_'))
    metricplot('06 Width robustness',[chosen(r=r) for r in (16,32,64)],['r16 / h512\nrho1/16','r32 / h1024\nrho1/8','r64 / h2048\nrho1/4'])
    metricplot('07 Basis and ordinary communication',[base,chosen('qr_radon'),chosen('svd_resample')],['SVD Radon','QR Radon','SVD resampling'])
    fig,axes=plt.subplots(2,4,figsize=(16,7.5),layout='constrained')
    for row,source in enumerate(('cfp_stage3','oct_stage3')):
        for col,field in enumerate(PACKETS):
            ax=axes[row,col]
            for split in SPLITS:
                p=mean_array(base,field+'_spectrum',source=source,split=split);total=p.sum()
                ax.plot(np.fft.rfftfreq(64),p/total if total>0 else np.zeros_like(p),color=COLORS[split],label=split)
            ax.set(title=source+' | '+field,xlabel='Cycles / S sample',ylabel='Fraction of packet energy');ax.grid(alpha=.2)
            if row==col==0:ax.legend()
    fig.suptitle('R&B | 08 S-axis spectrum (one-sided Parseval energy; includes DC)',fontsize=16)
    save(fig,plots,'08_s_axis_spectra')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,modality in enumerate(('cfp','oct')):
        for col,stages in enumerate(((3,),(2,3),(2,3,4))):
            vs=base if stages==(3,) else chosen('svd_multidepth',stages=stages)
            ax=axes[row,col]
            for si,split in enumerate(SPLITS):
                vals=np.array([[group(v,split=split,source=f'{modality}_stage{stage}',bridge=bi)['scalars']['total_relative']['mean'] for v in vs] for bi,stage in enumerate(stages)])
                ax.errorbar(np.arange(len(stages))+(si-1)*.1,vals.mean(1),yerr=vals.std(1,ddof=1),color=COLORS[split],marker='o',capsize=3,label=split)
            ax.set_xticks(np.arange(len(stages)),[f'stage{s}' for s in stages]);ax.set_title(modality.upper()+' | stages '+','.join(map(str,stages)))
            ax.set_ylabel('Mean participant ||delta X|| / ||X||');ax.grid(axis='y',alpha=.2)
            if row==col==0:ax.legend()
    fig.suptitle('R&B | 09 Residuals through one, two and three bridges\nEach bridge r16 / h512; later inputs already include earlier communication',fontsize=16)
    save(fig,plots,'09_multidepth_returns')
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for row,source in enumerate(('cfp_stage3','oct_stage3')):
        for col,split in enumerate(SPLITS):
            ax=axes[row,col]
            for phase,color in [('constructed_initial','#8a96a7'),('selected',COLORS[split])]:
                p=np.sqrt(mean_array(base,'projection_ms',source=source,split=split,phase=phase).mean(0))
                ax.plot(np.linspace(-1,1,64),p,color=color,label=phase.replace('_',' '))
            ax.set(title=source+' | '+split,xlabel='Distance / projection support radius',ylabel='Input packet RMS over directions')
            ax.legend(fontsize=8)
    fig.suptitle('R&B | 10 Representation before joint adaptation and at selected checkpoint\nInitial messages and returned residuals are exactly zero; inputs may change during training',fontsize=16)
    save(fig,plots,'10_initial_selected_inputs')
    private_figures(records,run)
    write(out/'provenance.json',dict(candidate_lock_sha256=sha256(lock/'candidate_lock.json'),accepted_jobs_sha256=sha256(run/'all/accepted_jobs.json'),
        checkpoints=21,training_tasks=0,post_hoc_descriptive=True,primary_hypotheses_unchanged=True,private_examples_exported=False,
        public_aggregation='full-cohort participant means, two eyes; no participant rows, images, predictions or identifiers',
        color_limits='corresponding train maps, shared across splits; clipping printed; seed metrics show mean and sample SD',
        figure_limitations=['feature-grid units, not physical mm','SVD signs/channels not matched across seeds','no claim of semantic alignment','mixed packet need not be a valid Radon transform','ordinary backprojection is not an inverse','class association is not lesion localization']))
    text='''# R&B：Radon空间只读图谱\n\n本图谱独立于主要test统计，属于追加的描述性诊断。21个已有检查点、零新增训练；按结构选取，不按test表现挑选。完整训练1264、开发296、测试290参与者分别计算，三种子等权汇总。\n\n图01通过合成点解释二维正弦轨迹与三维球面方向；图02展示投影、自身消息、跨来源消息及总消息；图03展示写回前特征与返回残差；图04为标签组RMS差异，仅表示当前模型的描述性关联。图05为OCT真实方向向量；图06—07展示宽度、基与普通通信下的数值变化和种子波动；图08为S轴含直流项的能量谱；图09展示多桥逐位置残差；图10比较零桥初始与选中状态的输入投影。\n\n所有公开空间图均为全队列聚合RMS，不能解释为某个病人的病灶位置。自身和跨来源残差可能相消或相长，能量包含交叉内积，不能把两者RMS简单相加。scalar_summary.csv含各参与者等权的能量保留率、残差比、余弦和零范数未定义计数。数据不支持的结构或效应不能靠图形证明。\n\n二维正弦图仅对应方向角与有符号距离；三维方向是球面向量，方向索引邻接不代表角度相邻。普通重采样的32是打包维度，不是Radon角度。卷积后的数据称为投影域消息，未必满足Radon一致性条件；普通反投影不是精确逆变换。\n\n初始状态从同一父模型和固定基构造，并将桥卷积置零；不是早期训练轨迹。跨种子的通道没有强行做语义匹配，图像用平方能量汇总避免符号翻转制造抵消。CFP与OCT的坐标、通道、强度不能解释成已完成物理或语义配准。\n\n固定前16人的原始带符号特征数组，以及前4人的双眼、前3通道案例图，保留在ws02 private_figures与各作业private目录；不进入本目录、不上传GitHub、不导出本地。案例只是预定索引示例，不宣称代表临床人群。\n\n此文件为图谱读图指南，定量效能结论仍须引用统一test报告及其配对区间；不把观察性可视化当成新增显著性检验。\n'''
    (out/'README.zh-CN.md').write_text(text)
    shutil.copyfile(lock/'protocol.zh-CN.md',out/'protocol.zh-CN.md')
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    write(run/'report_status.json',dict(state='generated',public_report=str(out),private_report=str(run/'private_figures'),visual_review_required=True))


def private_figures(records,run):
    """Prespecified rank channels and cases; all outputs remain restricted."""
    v=next(v for v in records if v['job']['display']==dict(arm='svd_radon',seed=3416,r=16,stages=[3]))
    dst=run/'private_figures';dst.mkdir(exist_ok=True,mode=0o700)
    for source in ('cfp_stage3','oct_stage3'):
        key='bridge0_'+source;meta=v['aggregate']['selected/train'][key+'/all']['metadata']
        train=v['directory']/'private/selected/train';ids=read(train/'sample_manifest.json')['ids']
        scales={}
        for field in PACKETS:
            values=[]
            for identifier in ids:
                with np.load(train/(identifier+'__'+key+'.npz'),allow_pickle=False) as a:values.append(np.abs(a[field]).reshape(-1))
            scales[field]=float(np.max(np.concatenate(values)))
        native_scales={}
        for field in SPATIAL:
            scale=0.
            for identifier in ids:
                with np.load(train/(identifier+'__'+key+'.npz'),allow_pickle=False) as z:scale=max(scale,float(z[field+'_rms'].max()))
            native_scales[field]=scale
        for split in SPLITS:
            path=v['directory']/'private/selected'/split;ids=read(path/'sample_manifest.json')['ids'][:4]
            for case,identifier in enumerate(ids):
                with np.load(path/(identifier+'__'+key+'.npz'),allow_pickle=False) as a:
                    fig,axes=plt.subplots(6,4,figsize=(17,19),layout='constrained')
                    for eye in range(2):
                        for rank in range(3):
                            for col,field in enumerate(PACKETS):
                                im=packet(axes[eye*3+rank,col],a[field][eye,rank],meta,f'eye{eye} / channel{rank} | {field}',scales[field],signed=True)
                                fig.colorbar(im,ax=axes[eye*3+rank,col],shrink=.6)
                    fig.suptitle(f'RESTRICTED | R&B {source} | {split} fixed case {case+1}\nBoth eyes; prespecified rank prefix; fixed train-example scales; not a clinical localization',fontsize=16)
                    save(fig,dst,f'{source}_{split}_case{case+1}_packets')
                    fig,axes=plt.subplots(2,5,figsize=(18,7),layout='constrained')
                    for eye in range(2):
                        for col,field in enumerate(SPATIAL):
                            # Each field's scale from the same predeclared train example panel set.
                            scale=native_scales[field]
                            im=spatial(axes[eye,col],a[field+'_rms'][eye],f'eye{eye} | {field}',scale);fig.colorbar(im,ax=axes[eye,col],shrink=.6)
                    fig.suptitle(f'RESTRICTED | {source} {split} fixed case {case+1} | native RMS feature maps',fontsize=16)
                    save(fig,dst,f'{source}_{split}_case{case+1}_native')
    write(dst/'artifact_manifest.json',{str(p.relative_to(dst)):sha256(p) for p in sorted(dst.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();report(a.lock,a.run)
