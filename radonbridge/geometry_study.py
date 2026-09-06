"""Prespecified candidate enumeration and executed bridge arithmetic convention.

One multiply or add is one operation (MAC=2); padded convolution counts dense
MACs. Gather/index/copy and memory traffic are NOT arithmetic FLOPs. Mask
multiplication is charged once per batch. Counts are per two-eye participant.
"""
import copy
import hashlib
import itertools
import json

SEEDS=(3416,3417,3418)
LRS=(3e-5,6e-5)
MS=(16,32,64)
SS=(32,64,128)
KS=(1,3,5)
HS=(512,1024)
PROTOCOLS=('branch','fusion')
POLICY=dict(min_epochs=8,max_epochs=60,patience=6,min_delta=.001,lr_patience=3,lr_factor=.3)
FUSION=dict(pooling='mean',hidden_dimension=256,attention_dimension=128)
NODES=['cfp_stage3','oct_stage3']
VERSION='geometry_resource_protocol_v1'

def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def cost(M,S,k,r,mode,batch=16):
    if mode not in ('radon','linear_resample'): raise ValueError(mode)
    if any(type(v)!=int or v<1 for v in (M,S,k,r,batch)) or k%2==0: raise ValueError('Invalid integers')
    C=256;L=196+288;H=2*r*M
    parts=dict(channel_encode=2*C*r*L,channel_decode=2*C*r*L,
               forward_projection=2*r*M*S*L,
               return_projection=4*r*M*L+r*L if mode=='radon' else 2*r*M*S*L,
               mixer=2*k*H*H*S,mask_multiply=k*H*H/(2*batch),residual=C*L)
    return dict(parameters=k*H*H,flops_per_participant=2*sum(parts.values()),
                components_per_participant={name:2*v for name,v in parts.items()},
                convention='dense multiply/add=1 each; padded MAC=2; two eyes; mask amortized over batch16; excludes memory/indexing',
                r=r,h=r*M,rho=r/C)

def matched_rank(M,S,k,mode,target):
    candidates=[(abs(cost(M,S,k,r,mode)['flops_per_participant']/target-1),
                 cost(M,S,k,r,mode)['flops_per_participant'],cost(M,S,k,r,mode)['parameters'],r) for r in range(8,65)]
    error,flops,parameters,r=min(candidates)
    return dict(r=r,relative_error=error,flops_per_participant=flops,parameters=parameters,feasible=error<=.05)

def catalog():
    structures={};cells=[]
    def add(M,S,k,r,mode,member):
        key=f'{mode}_M{M}_S{S}_k{k}_r{r}'
        row=structures.setdefault(key,dict(id=key,M=M,S=S,k=k,r=r,h=r*M,rho=r/256,mode=mode,
                                        cost=cost(M,S,k,r,mode),memberships=[]))
        if member not in row['memberships']:row['memberships'].append(member)
        return key
    for M,S,k,h,mode in itertools.product(MS,SS,KS,HS,('radon','linear_resample')):
        add(M,S,k,h//M,mode,dict(kind='equal_parameters',budget_h=h))
    for h in HS:
        target=cost(32,64,3,h//32,'radon')['flops_per_participant']
        for M,S,k in itertools.product(MS,SS,KS):
            a=matched_rank(M,S,k,'radon',target);b=matched_rank(M,S,k,'linear_resample',target)
            cell=dict(budget_h=h,M=M,S=S,k=k,target_flops=target,radon_solution=a,ordinary_solution=b)
            if a['feasible']:
                tag=dict(kind='equal_compute',budget_h=h,role='radon')
                cell['radon_id']=add(M,S,k,a['r'],'radon',tag)
                cell['same_width_id']=add(M,S,k,a['r'],'linear_resample',dict(tag,role='same_width_ordinary'))
                if b['feasible']:
                    cell['compute_ordinary_id']=add(M,S,k,b['r'],'linear_resample',dict(tag,role='compute_ordinary'))
            cells.append(cell)
    rows=list(structures.values())
    # Reference cells first, then fixed lexicographic geometry order. No score used.
    rows.sort(key=lambda x:(not(x['M']==32 and x['S']==64 and x['k']==3 and x['h'] in HS),x['M'],x['S'],x['k'],x['r'],x['mode']))
    return dict(version=VERSION,execution_authorized=False,scope_status='superseded full-grid proposal; revised mechanism subset pending',structures=rows,compute_cells=cells,equal_parameter_positions=1296,
                geometry_positions=len(rows)*12,baseline_positions=108,
                prospective_augmentation_positions=72,final_new_training_tasks=None,
                lock_condition='CPU/counter acceptance, GPU feasibility, and matching historical acceptance',test_used=False)

def baseline_structures():
    return [dict(id='no_communication',family='none')]+[
        dict(id=f'mmtm_hidden{d}',family='mmtm',reduction_ratio=1024//d,hidden=d) for d in (128,256,512,1024)]+[
        dict(id=f'attention_d{d}',family='cross_attention',attention_dimension=d,heads=4) for d in (128,256,512,1024)]

def configuration(structure,seed,lr,protocol,parents,bases):
    cfg=dict(seed=seed,backbone_lr=lr,head_lr=1e-4,bridge_lr=1e-4,bridges=[],
             microbatch=16,effective_batch=16,convergence=copy.deepcopy(POLICY),
             training_stage='communication',parent_checkpoints=copy.deepcopy(parents))
    if protocol=='fusion':cfg.update(task_fusion=copy.deepcopy(FUSION),selection_metric='fusion_macro_f1')
    elif protocol!='branch':raise ValueError(protocol)
    if 'M' in structure:
        cfg['bridges']=[dict(nodes=NODES.copy(),M=structure['M'],S=structure['S'],kernel_size=structure['k'],
                            r=structure['r'],h=structure['h'],rho=structure['rho'],mode=structure['mode'],
                            compression='fixed_svd_channel',basis_files=copy.deepcopy(bases))]
    elif structure['family']=='mmtm':cfg['bridges']=[dict(nodes=NODES.copy(),family='mmtm',reduction_ratio=structure['reduction_ratio'])]
    elif structure['family']=='cross_attention':cfg['bridges']=[dict(nodes=NODES.copy(),family='cross_attention',attention_dimension=structure['attention_dimension'],heads=4)]
    return cfg

def semantic(cfg):
    """Strict configuration identity; only proven legacy defaults/path aliases normalize."""
    c=copy.deepcopy(cfg)
    for key in ('source_commit','budget_estimate_epochs','measure_latency','profile','save_profile_checkpoint'):
        c.pop(key,None)
    c.setdefault('selection_metric','mean_branch_macro_f1')
    c.setdefault('head_lr',1e-4);c.setdefault('bridge_lr',1e-4)
    for b in c['bridges']:
        if b.get('family','radon')=='radon':
            b.setdefault('family','radon');b.setdefault('kernel_size',3);b.setdefault('compression','learned_projected')
            if b['compression']=='fixed_svd_channel' and isinstance(b['M'],int) and isinstance(b['rho'],(int,float)):
                r=max(1,int(b['rho']*256));b.setdefault('r',r);b.setdefault('h',r*b['M'])
        for ref in b.get('basis_files',{}).values(): ref.pop('path',None)
    for ref in c.get('parent_checkpoints',{}).values():ref.pop('path',None)
    if 'host_checkpoint' in c:c['host_checkpoint'].pop('path',None)
    return c

def fingerprint(cfg):return stable_hash(semantic(cfg))

