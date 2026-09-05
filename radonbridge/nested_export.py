"""Materialize width views of one shared checkpoint; never count views as training."""
import copy,hashlib,json,math,shutil
from pathlib import Path
import torch


def name(rho):return 'rho1_'+str(round(1/rho))


def width_state(state,cfg,rho):
    b=cfg['bridges'][0];assert rho in b['nested_rhos']
    out={k:dict(v) for k,v in state.items()};ex=dict(out['bridge_0_exchange']);out['bridge_0_exchange']=ex
    maximum=[];retained=[]
    for i in range(len(b['nodes'])):
        if b['compression']=='learned_channel':
            ek=f'channel_codecs.{i}.encoder.weight';dk=f'channel_codecs.{i}.decoder.weight'
            channels=ex[ek].shape[1];r=max(1,math.floor(channels*rho));maximum.append(ex[ek].shape[0]*b['M']);retained.append(r*b['M'])
            ex[ek]=ex[ek][:r].clone();ex[dk]=ex[dk][:,:r].clone()
        else:
            q=f'channel_bases.{i}.q';r=max(1,math.floor(ex[q].shape[0]*rho));maximum.append(ex[q].shape[1]*b['M']);retained.append(r*b['M']);ex[q]=ex[q][:,:r].clone()
    offsets=[0]
    for w in maximum:offsets.append(offsets[-1]+w)
    ix=torch.cat([torch.arange(o,o+w) for o,w in zip(offsets,retained)])
    for k in ('mixer.conv.weight','mixer.mask'):ex[k]=ex[k].index_select(0,ix).index_select(1,ix).clone()
    return out


def export_widths(state,cfg,out,selection=None):
    from .experiment import write_json
    out=Path(out);records=[]
    for rho in cfg['bridges'][0]['nested_rhos']:
        directory=out/'width_exports'/name(rho);directory.mkdir(parents=True,exist_ok=False)
        small=copy.deepcopy(cfg);small['bridges'][0].pop('nested_rhos');small['bridges'][0]['rho']=rho
        small['width_training_provenance']={'regime':'joint_nested','training_directory':str(out),'widths':cfg['bridges'][0]['nested_rhos'],
            'selection':'same joint checkpoint for every width; no per-width selection','bn':'shared averaged running updates'}
        write_json(directory/'configuration.json',small)
        torch.save({'model':width_state(state,cfg,rho),'configuration':small,'selection':selection,'derived_view':True},directory/'selected.pt')
        for phase in ('initial','selected','last'):
            src=out/(phase+'_predictions_'+name(rho)+'.npz')
            if src.exists():shutil.copyfile(src,directory/(phase+'_predictions.npz'))
        records.append({'rho':rho,'directory':str(directory),'selected_sha256':hashlib.sha256((directory/'selected.pt').read_bytes()).hexdigest(),'derived_view_not_training':True})
    write_json(out/'width_exports.json',records)
    return records
