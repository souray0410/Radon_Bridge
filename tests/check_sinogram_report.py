# Synthetic integration fixture only; never used as research evidence.
import tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from scripts.geometry_evidence import write
from radonbridge.artifacts import sha256
from radonbridge.projector import eem_directions
from radonbridge.sinogram_atlas import PACKETS,SPATIAL
import scripts.report_sinogram_atlas as report
with tempfile.TemporaryDirectory(prefix='synthetic-atlas-test-') as tmp:
 root=Path(tmp);lock=root/'lock';run=root/'run';lock.mkdir();run.mkdir();jobs=[];accepted={};count=[]
 for seed in (3416,3417,3418):
  for arm,r,stages in [('svd_radon',16,[3]),('svd_radon',32,[3]),('svd_radon',64,[3]),('qr_radon',16,[3]),('svd_resample',16,[3]),('svd_multidepth',16,[2,3]),('svd_multidepth',16,[2,3,4])]:
   jid=f'{seed}_{arm}_{r}_{stages}';d=run/jid;d.mkdir();agg={}
   for phase in ('constructed_initial','selected'):
    for split,n in [('train',1264),('validation',296),('test',290)]:
     groups={}
     for bridge,stage in enumerate(stages):
      for modality,dim in [('cfp',2),('oct',3)]:
       source=f'{modality}_stage{stage}';shape=[2]*dim
       meta=dict(source=source,bridge=bridge,shape=shape,C=2**(stage+5),r=r,M=32,S=64,k=3,mode='linear_resample' if arm=='svd_resample' else 'radon',compression='fixed_svd_channel',geometry=dict(support=[-1,1],directions=eem_directions(dim,32)[0].tolist()))
       for group in ('all','label0','label1'):
        arr={field+'_ms':(np.ones((32,64))*.1).tolist() for field in PACKETS}
        arr.update({field+'_spectrum':(np.ones(33)*.1).tolist() for field in PACKETS});arr.update({field+'_ms':np.ones(shape).tolist() for field in SPATIAL})
        if group=='label1':arr['projection_ms']=(np.ones((32,64))*.2).tolist()
        scalar={field:dict(mean=.1,sample_sd=.01,defined_participants=n,undefined_participants=0) for field in ['channel_energy_retained','total_relative','self_cross_cosine']}
        groups[f'bridge{bridge}_{source}/{group}']=dict(metadata=meta,arrays=arr,scalars=scalar,group=group,participants=n)
     agg[phase+'/'+split]=groups
   write(d/'aggregates.json',agg);write(d/'summary.json',dict(state='accepted',prediction_files={'aggregates.json':sha256(d/'aggregates.json')}))
   accepted[jid]=dict(summary_path=str(d/'summary.json'),summary_sha256=sha256(d/'summary.json'))
   jobs.append(dict(job_id=jid,display=dict(seed=seed,arm=arm,r=r,stages=stages),model=dict(model_view_id=jid,checkpoint_sha256='synthetic'),source_reference='synthetic'))
 write(lock/'candidate_lock.json',dict(synthetic=True));write(lock/'jobs.json',jobs);(lock/'protocol.zh-CN.md').write_text('Synthetic fixture')
 write(run/'all/accepted_jobs.json',accepted)
 def save(fig,out,name,pad_inches=.1):
  assert pad_inches==(.4 if name=='05_oct_spherical_directions' else .1)
  count.append(name);assert fig.axes;plt.close(fig)
 with patch.object(report,'save',save),patch.object(report,'private_figures') as private,patch.object(report,'synthetic'):
  report.report(lock,run);assert private.call_count==1
 assert len(count)==12,len(count)
 assert (run/'report/scalar_summary.csv').exists()
 print('PASS: synthetic complete 21-checkpoint report integration, all 12 aggregate figures, scalar CSV, provenance and public artifact manifest')
