"""Synthetic observed-eye epoch loop, pause/reload, plateau and read-only diagnostics."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from mhd_framework.models import create_model
from radon_bridge.models.observed_participant import ObservedParticipantModel
from radon_bridge.studies.project_build import build
from radon_bridge.studies.research_matrix import arms
from radon_bridge.training.paired_native import train,DEFAULTS
from radon_bridge.methods.native_basis import fit
from radon_bridge.analysis.native_diagnostics import diagnose
from radon_bridge.runtime.state import atomic_write_json

class Fixture(Dataset):
    def __init__(self,split):
        self.split=split;self.augment=False;self.participant_ids=[split+str(i) for i in range(4)];self.counts=[1,2,1,2]
        self.first=self;self.rows=[dict(id=i,eyes=['L'] if n==1 else ['L','R'],label=j%2) for j,(i,n) in enumerate(zip(self.participant_ids,self.counts))]
    def __len__(self):return 4
    def set_epoch(self,epoch):pass
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i+(100 if self.split=='train' else 200));n=self.counts[i]
        return dict(cfp=torch.rand(n,3,64,64,generator=g),oct=torch.rand(n,1,16,64,64,generator=g),label=i%2,participant_id=self.participant_ids[i])

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0')
    torch.set_num_threads(2);torch.manual_seed(3416);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(device).total_memory)
    parents={k:ObservedParticipantModel(create_model(dict(name='resnet18',spatial_dims=d,in_channels=c,implementation='torchvision_2d' if d==2 else 'inflated_3d'))) for k,d,c in [('cfp',2,3),('oct',3,1)]}
    shapes={'cfp':(3,64,64),'oct':(1,16,64,64)};tr=Fixture('train');dev=Fixture('development')
    cfg=dict(DEFAULTS,microbatch=2,effective_batch=2)
    base=build(parents,shapes,arms('cataract')[0],{},3416,device)
    bases=fit(base,tr,[3],out/'bases','synthetic',3416,device,lambda:False);del base
    a=dict(arms('cataract')[1],M=4,S=8,r=4);spec=dict(seed=3416,arms=[a],scope='synthetic',test_access=False)
    model=build(parents,shapes,a,bases,3416,device)
    r=train(model,tr,dev,cfg,3416,out/'arms'/a['id'],'synthetic',device,preflight_updates=1);assert r['state']=='paused'
    del model
    model=build(parents,shapes,a,bases,3416,device)
    r=train(model,tr,dev,cfg,3416,out/'arms'/a['id'],'synthetic',device);assert r['state']=='accepted',r
    assert r['plateau'] and r['stop_epoch']>=8
    del model
    d=diagnose(spec,out,parents,shapes,bases,tr,dev,device,lambda:False);assert d['state']=='accepted'
    atomic_write_json(dict(status='accepted',scope='synthetic_training_and_diagnostic_runtime_not_scientific_results',plateau=True,resume=True,read_only=True,test_access=False),out/'gate.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
