"""Bounded synthetic MHD, every mechanism family, masks and exact next-update checks."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
from mhd_framework.models import create_model
from radon_bridge.models.observed_participant import ObservedParticipantModel
from radon_bridge.models.native_pair import NativePair
from radon_bridge.methods.basis import save_basis,save_centered_basis,save_random_basis
from radon_bridge.studies.project_build import build
from radon_bridge.studies.research_matrix import arms,coverage
from radon_bridge.studies.project_profile import profile_model
from radon_bridge.training.paired_native import DEFAULTS
from radon_bridge.runtime.state import atomic_write_json


def run(output,device,architecture='resnet18'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);device=torch.device(device)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(904)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    parents={k:ObservedParticipantModel(create_model(dict(name=architecture,spatial_dims=d,in_channels=c,implementation='torchvision_2d' if d==2 else 'inflated_3d'))) for k,d,c in [('cfp',2,3),('oct',3,1)]}
    shapes={'cfp':(3,64,64),'oct':(1,16,64,64)}
    batch={k:torch.rand((3,*s),device=device) for k,s in shapes.items()};batch.update(label=torch.tensor([0,1],device=device),counts=[1,2])
    base=NativePair(parents,shapes,device=device);base.eval()
    with torch.no_grad():
        expected,_=base(batch)
        for role,parent in parents.items():
            reference_parent=ObservedParticipantModel(create_model(parent.graph.configuration(),device=device))
            reference_parent.load_state_dict(parent.state_dict(),strict=True);reference_parent.eval()
            assert torch.equal(expected[role],reference_parent(batch[role],batch['counts']))
            del reference_parent
    ids=base.node_identity();basis={c:{} for c in ('fixed_svd_channel','fixed_centered_svd_channel','fixed_random_orthogonal_channel')}
    for stage in (2,3,4):
        for role,parent in parents.items():
            key=role+'_stage'+str(stage);c=parent.graph.feature_channels['stage'+str(stage)];moment=torch.eye(c,dtype=torch.float64)
            dest=out/'bases'/key
            svd=save_basis(moment,1,dest,key,3416,{'scope':'synthetic_acceptance_only'})
            center=save_centered_basis(moment,torch.zeros(c,dtype=torch.float64),1,dest,key,3416,{'scope':'synthetic_acceptance_only'})
            qr=save_random_basis(svd,dest)
            for kind,ref in zip(basis,(svd,center,qr)):basis[kind][key]=ref
    results=[]
    chosen=[];seen=set()
    for a in arms('glaucoma'):
        key=(a['family'],a['compression'],a['mode'],a['frozen'],tuple(a['stages']),tuple(a.get('direction',[])),a.get('s_axis_scramble',False),a.get('addition'),a.get('host'))
        if key in seen:continue
        seen.add(key);chosen.append(a)
    for original in chosen:
        a=dict(original,M=4,S=8,r=4)
        model=build(parents,shapes,a,basis,3416,device);model.eval()
        with torch.no_grad():
            actual,_=model(batch)
            assert all(torch.equal(actual[k],expected[k]) for k in expected),(a['id'],'zero initialization differs')
        assert all((i,n) in model.node_identity() for i,n in ids),'Native node identity changed'
        for name,m in model.task.modules_by_name().items():
            if name.endswith('_exchange') and hasattr(m,'mixer'):
                with torch.no_grad():m.mixer.conv.weight.normal_(0,.001);m.mixer.conv.weight.mul_(m.mixer.mask)
        model.zero_grad(set_to_none=True);z,loss=model(batch);model.backward()
        mhd={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True);_,reference=model.native_forward(batch);reference.backward()
        torch.testing.assert_close(loss,reference,atol=1e-6,rtol=1e-5)
        for n,p in model.named_parameters():
            if n in mhd:
                assert p.grad is not None, (a['id'], n)
                torch.testing.assert_close(p.grad,mhd[n],atol=1e-6,rtol=1e-5,msg=lambda detail:f'{a["id"]}/{n}: {detail}')
        opt=torch.optim.AdamW(model.groups(3e-5,1e-4,1e-4),weight_decay=.01);opt.step();opt.zero_grad(set_to_none=True)
        for name,m in model.task.modules_by_name().items():
            if hasattr(m,'mixer'):assert torch.equal(m.mixer.conv.weight[m.mixer.mask==0],torch.zeros_like(m.mixer.conv.weight[m.mixer.mask==0]))
        results.append(a['id']);del model
    cfg=dict(DEFAULTS,microbatch=2,effective_batch=2)
    model=build(parents,shapes,dict(arms('glaucoma')[1],M=4,S=8,r=4),basis,3416,device)
    profile=profile_model(model,batch,cfg,'synthetic_next_update',out/'resume',device,production=False)
    assert profile['exact_next_update']
    result=dict(status='accepted',scope='synthetic_structure_and_recovery_not_clinical_training',device=str(device),
        checked_families=results,exact_native_outputs=True,mhd_native_gradients=True,masked_weights_preserved=True,
        exact_next_update=True,real_production_profiles_required=True,test_access=False,
        planned_positions=sum(len(v) for v in coverage()['arms'].values())*3)
    atomic_write_json(result,out/'gate.json');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--architecture',default='resnet18');a=p.parse_args()
    print(json.dumps(run(a.output,a.device,a.architecture)))
