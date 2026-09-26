"""Attach communication to declared native feature nodes without owning a backbone."""
import torch
from radon_bridge.methods.operator import attach_to_nodes, attach_parallel_to_nodes

def attach_communications(native,configs):
    builder=native.builder;configs=list(configs)
    if not configs:return [],False
    if native.metadata.get('task_fusion') is not None and any('nested_rhos' in c for c in configs):
        raise ValueError('Task fusion protocol does not support joint-width training')
    parallel=any('parallel_to' in c for c in configs)
    if parallel and (len(configs)!=2 or configs[1].get('parallel_to')!=0 or configs[0].get('family') not in ('mmtm','mmtm_author','cross_attention','cmx_frm') or configs[1].get('family','radon')!='radon' or configs[0]['nodes']!=configs[1]['nodes']):
        raise ValueError('Parallel addition requires one intact nonlinear host and one Radon-family addition on identical nodes')
    for c in configs:
        family=c.get('family','radon')
        required={'nodes','M','S','rho','mode'} if family=='radon' else {'nodes','family','reduction_ratio'} if family in ('mmtm','mmtm_author') else {'nodes','family','attention_dimension','heads'} if family=='cross_attention' else {'nodes','family','alignment_tokens'}
        optional={'family','compression','basis_files','cross_edges','nested_rhos','s_axis_permutation','kernel_size','r','h','group_count','bottleneck_rank','parallel_to'} if family=='radon' else set()
        if family not in ('radon','mmtm','mmtm_author','cross_attention','cmx_frm') or not required<=set(c) or set(c)-required-optional:
            raise ValueError('Invalid communication configuration for '+str(family))
    modules=[e.edge_operations[0].function for e in builder.edges]
    training={m:m.training for module in modules for m in module.modules()}
    try:
        for m in training:m.training=False
        # Probing discovers feature shapes; it must not consume a future model's RNG.
        with torch.random.fork_rng(devices=[]),torch.no_grad():samples=builder.native_forward(dict(native.probe_inputs))
    finally:
        for m,value in training.items():m.training=value
    if parallel:return attach_parallel_to_nodes(builder,configs,samples),True
    groups=[]
    for index,c in enumerate(configs):
        _,meta=attach_to_nodes(builder,c['nodes'],prefix=f'bridge_{index}_',samples=samples,**{k:v for k,v in c.items() if k!='nodes'})
        groups.append(meta)
    return groups,False
