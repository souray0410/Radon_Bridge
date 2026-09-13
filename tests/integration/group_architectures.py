"""Real architecture compatibility on synthetic inputs; not UKB resource acceptance."""
import argparse
import gc
import json
import torch

from mhd_framework.models import create_model
from radon_bridge.models.observed_participant import ObservedParticipantModel
from radon_bridge.models.native_group import NativeGroup
from radon_bridge.studies.complete_matrix import groups


def check(family,device):
    torch.manual_seed(47);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    g=next(g for g in groups() if g['package']=='same_family_mechanisms' and g['sources'][0]['architecture']==family)
    sources=g['sources'];parents={};shapes={};inputs={};labels={}
    for s in sources:
        config=dict(name=s['model'],spatial_dims=s['spatial_dims'],in_channels=3 if s['modality']=='cfp' else 1,num_classes=2,views=1)
        if family=='resnet50':config['implementation']='torchvision_2d' if s['modality']=='cfp' else 'inflated_3d'
        parents[s['key']]=ObservedParticipantModel(create_model(config,device='cpu')).eval()
        shapes[s['key']]=(3,64,64) if s['modality']=='cfp' else (1,32,32,32)
        inputs[s['key']]=torch.randn(2,*shapes[s['key']],device=device);labels[s['key']]=torch.tensor([0,1],device=device)
    batch=dict(inputs=inputs,labels=labels,counts=[1,1],participant_id=['synthetic_0','synthetic_1'])
    config=[dict(nodes=[s['key']+'_stage3' for s in sources],family='mmtm',hidden_dimension=8)]
    model=NativeGroup(parents,shapes,sources,config,device=device).eval()
    with torch.no_grad():
        logits,_=model(batch)
        for key,parent in parents.items():
            parent.to(device);expected=parent(inputs[key],[1,1]);parent.cpu()
            if not torch.equal(logits[key],expected):
                raise ValueError('Initial parent prediction differs: '+key+' max error '+str(float(abs(logits[key]-expected).max())))
    optimizer=torch.optim.AdamW(model.groups(1e-5,1e-4,1e-4))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);_,loss=model(batch);model.backward()
        expected={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True);_,native=model.native_forward(batch);native.backward()
        if not torch.allclose(loss,native,atol=1e-6,rtol=1e-5):raise ValueError('Native loss differs')
        for name,p in model.named_parameters():
            if name in expected and not torch.allclose(p.grad.cpu(),expected[name],atol=1e-5,rtol=1e-4):
                delta=(p.grad.cpu()-expected[name]).abs()
                raise ValueError('MHD gradient differs: '+name+' max_abs='+str(float(delta.max()))+
                    ' relative_l2='+str(float(delta.norm()/expected[name].norm().clamp_min(1e-30)))+
                    ' expected_max='+str(float(expected[name].abs().max())))
        model.clip(5);optimizer.step()
    result=dict(family=family,initial_predictions_exact=True,updates=2,mhd_native_gradient_equivalent=True,
                test_access=False,production_resource_acceptance=False,fixture='reduced synthetic spatial shapes',
                stages={s['key']:{f'stage{i}':list(model.task.by_name[s['key']+f'_stage{i}'].feature_message.current_state.shape) for i in (2,3,4)} for s in sources})
    del model,parents,optimizer;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(check(a.family,a.device),indent=2))
