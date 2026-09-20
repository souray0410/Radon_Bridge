"""Build prespecified single/multiple/parallel bridge configurations on real endpoints."""
import numpy as np
from radon_bridge.models.native_pair import NativePair,definition
from radon_bridge.methods.projection import geometry_budget


def bridges_for(parents,shapes,arm,bases,seed):
    if arm['family']=='none':return []
    channels={k:parents[k].graph.feature_channels for k in ('cfp','oct')}
    if arm.get('host'):
        host_families={'mmtm256':'mmtm','attention256':'cross_attention','cmx_frm64':'cmx_frm','cmx_full64':'cmx_full'}
        if arm['host'] not in host_families:raise ValueError('Unknown existing-method host')
        host=dict(arm,id=arm['host'],family=host_families[arm['host']])
        host.pop('host');host.pop('addition')
        configs=bridges_for(parents,shapes,host,bases,seed)
        if arm['addition']=='continue':return configs
        extra=dict(arm);extra.pop('host');extra.pop('addition');extra['mode']=arm['addition']
        configs+=bridges_for(parents,shapes,extra,bases,seed)
        configs[1]['parallel_to']=0;return configs
    configs=[]
    for stage in arm['stages']:
        names=[f'{source}_stage{stage}' for source in ('cfp','oct')]
        cs={n:channels[source]['stage'+str(stage)] for n,source in zip(names,('cfp','oct'))}
        if arm['family']=='mmtm':
            ratio=2*sum(cs.values())/256
            if int(ratio)!=ratio:raise ValueError('MMTM hidden256 cannot be exactly represented')
            config=dict(nodes=names,family='mmtm',reduction_ratio=int(ratio))
        elif arm['family']=='cross_attention':config=dict(nodes=names,family='cross_attention',attention_dimension=256,heads=4)
        elif arm['family']=='cmx_frm':
            if len(set(cs.values()))!=1:raise ValueError('CMX-FRM requires equal channels across participants')
            config=dict(nodes=names,family='cmx_frm',alignment_tokens=arm.get('alignment_tokens',64))
        elif arm['family']=='cmx_full':
            if len(set(cs.values()))!=1:raise ValueError('CMX full core requires equal channels across participants')
            heads=int(arm.get('heads',4))
            if next(iter(cs.values()))%heads:raise ValueError('CMX full core channels must divide into heads')
            config=dict(nodes=names,family='cmx_full',alignment_tokens=arm.get('alignment_tokens',64),heads=heads)
        else:
            r=arm['r'];M=arm['M'];S=arm['S']
            if any(r>c for c in cs.values()):raise ValueError('Channel rank exceeds source channels')
            config=dict(nodes=names,M=M,S=S,rho={n:r/c for n,c in cs.items()},mode=arm['mode'],compression=arm['compression'],kernel_size=arm['k'])
            if arm['compression'].startswith('fixed_'):
                config['basis_files']={n:bases[arm['compression']][n] for n in names}
            if arm['compression'] in ('fixed_svd_channel','fixed_random_orthogonal_channel'):
                config.update(r=r,h=r*M)
            if arm.get('group_count',1)!=1:config['group_count']=arm['group_count']
            if arm.get('direction'):config['cross_edges']=[[source+'_stage'+str(stage) for source in arm['direction']]]
            if arm.get('s_axis_scramble'):config['s_axis_permutation']=np.random.default_rng(seed+907).permutation(S).tolist()
        configs.append(config)
    return configs


def build(parents,shapes,arm,bases,seed,device='cpu'):
    import torch
    # Initialization is paired across arms, and independent from previously
    # executed arms. Training order has its own participant/epoch generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with geometry_budget(128_000_000):
            model=NativePair(parents,shapes,bridges_for(parents,shapes,arm,bases,seed),device=device,frozen=arm['frozen'])
    return model
