"""Prespecified depth factorial and parameter-bracket controls; no outcome selection."""
import copy,itertools
from .geometry_study import configuration,SEEDS,LRS
from .pretest_study import row

STAGES=(2,3,4)
SUBSETS=tuple(t for n in (1,2,3) for t in itertools.combinations(STAGES,n))
MODES=('radon','linear_resample')
VERSION='multidepth96_v1'

def arithmetic(stages,rank,mode,batch=16):
    """Same dense arithmetic convention as geometry_study.cost, per participant."""
    parts=[]
    for stage in stages:
        C=64*2**(stage-1);cfp=(224//2**(stage+1),)*2
        oct_shape=(32//2**(stage-1),96//2**(stage+1),96//2**(stage+1))
        import math
        L=math.prod(cfp)+math.prod(oct_shape);H=64*rank
        terms=dict(channel_encode=2*C*rank*L,channel_decode=2*C*rank*L,
            forward_projection=2*rank*32*64*L,
            return_projection=4*rank*32*L+rank*L if mode=='radon' else 2*rank*32*64*L,
            mixer=2*3*H*H*64,mask_multiply=3*H*H/(2*batch),residual=C*L)
        parts.append(dict(stage=stage,C=C,cfp_shape=list(cfp),oct_shape=list(oct_shape),
            components_per_participant={k:2*v for k,v in terms.items()},flops_per_participant=2*sum(terms.values())))
    return dict(bridges=parts,flops_per_participant=sum(p['flops_per_participant'] for p in parts),
        convention='dense multiply/add=1 each; padded MAC=2; two eyes; mask amortized batch16; bridge only; excludes memory/indexing')

def structure(stages,mode,rank=16):
    return dict(id=mode+'_stages'+''.join(map(str,stages))+f'_r{rank}',stages=list(stages),
                mode=mode,M=32,S=64,k=3,r=rank,h=32*rank,
                total_mixer_parameters=len(stages)*3*(64*rank)**2,
                family='radon',role='factorial' if rank==16 else 'parameter_control',cost=arithmetic(stages,rank,mode))

def matrices(parents,bases):
    rows=[]
    structures=[structure(s,m) for s in SUBSETS for m in MODES]
    structures += [structure((3,),m,r) for r in (23,28) for m in MODES]
    for s in structures:
        for seed in SEEDS:
            for lr in LRS:
                cfg=configuration(dict(family='none'),seed,lr,'branch',parents[seed],{})
                for stage in s['stages']:
                    keys=[f'{b}_stage{stage}' for b in ('cfp','oct')]
                    cfg['bridges'].append(dict(nodes=keys,M=32,S=64,kernel_size=3,r=s['r'],h=s['h'],
                        rho=s['r']/(64*2**(stage-1)),mode=s['mode'],compression='fixed_svd_channel',
                        basis_files={k:copy.deepcopy(bases[seed][k]) for k in keys}))
                if len(s['stages'])>1:cfg['communication_clipping']='per_bridge'
                rows.append(row(f'depth_{s["id"]}_lr{lr:g}_seed{seed}',cfg,copy.deepcopy(s),'depth'))
    assert len(rows)==108 and len({r['fingerprint'] for r in rows})==108
    # Pair R/O and learning-rate/seed replicates within each fixed structure block.
    rows.sort(key=lambda r:(len(r['structure']['stages']),r['structure']['stages'],r['structure']['r'],r['seed'],r['backbone_lr'],r['structure']['mode']))
    return rows

def definitions(rows,baseline):
    """Weights on model branch means; no pooling predictions or survivor reweighting."""
    def group(stages,mode,rank=16):
        matches=baseline if not stages else [r for r in rows if r['structure']['stages']==list(stages)
                  and r['structure']['mode']==mode and r['structure']['r']==rank]
        assert len(matches)==6
        assert {(r['seed'],r['backbone_lr']) for r in matches}==set(itertools.product(SEEDS,LRS))
        return {r['id']:1/6 for r in matches}
    def combine(*terms):
        result={}
        for a,weights in terms:
            for k,v in weights.items():result[k]=result.get(k,0.)+a*v
        return {k:v for k,v in result.items() if abs(v)>1e-14}
    def diff(a,b):return combine((1,a),(-1,b))
    def mean(gs):return combine(*[(1/len(gs),g) for g in gs])
    def size(n,m):return mean([group(s,m) for s in SUBSETS if len(s)==n])
    defs=[]
    def add(name,w,primary=True,question=None):
        assert abs(sum(w.values()))<1e-10
        defs.append(dict(id=name,family='depth22' if primary else 'depth_secondary',primary=primary,
                         outcome='branch_mean_macro_f1',weights=w,question=question))
    presence={}
    for m in MODES:
        for stage in STAGES:
            others=[s for s in STAGES if s!=stage]
            backgrounds=[s for n in range(3) for s in itertools.combinations(others,n)]
            changes=[diff(group(sorted((*s,stage)),m),group(s,m)) for s in backgrounds]
            presence[stage,m]=mean(changes)
            add(f'{m}_stage{stage}_presence',presence[stage,m],question='Q1 placement conditional on four equally weighted backgrounds')
            for s,w in zip(backgrounds,changes):add(f'{m}_add{stage}_to'+(''.join(map(str,s)) or 'none'),w,False,'Q1 conditional cell')
        for n in (2,3):add(f'{m}_count{n}_minus_count{n-1}',diff(size(n,m),size(n-1,m)),question='Q2 equal weight over all placements')
        for n,rank in ((2,23),(3,28)):
            add(f'{m}_count{n}_minus_single_rank{rank}',diff(size(n,m),group((3,),m,rank)),question='Q3 approximate bridge parameter matching; not compute matching')
        for pair in itertools.combinations(STAGES,2):
            other=next(s for s in STAGES if s not in pair)
            for background in ((),(other,)):
                a,b=pair
                w=combine((1,group(sorted((*background,a,b)),m)),(-1,group(sorted((*background,a)),m)),
                          (-1,group(sorted((*background,b)),m)),(1,group(background,m)))
                add(f'{m}_interaction{a}{b}_background'+(''.join(map(str,background)) or 'none'),w,False,'Q4 descriptive training interaction on F1 scale')
        w=combine(*[((-1)**(3-len(s)),group(s,m)) for n in range(4) for s in itertools.combinations(STAGES,n)])
        add(m+'_three_way_interaction',w,False,'Q4 descriptive training interaction; not clinical causality')
    gains={n:diff(size(n,'radon'),size(n,'linear_resample')) for n in (1,2,3)}
    for n,w in gains.items():add(f'geometry_count{n}',w,question='Q5 Radon minus ordinary communication')
    for n in (2,3):add(f'geometry_count{n}_minus_count{n-1}',diff(gains[n],gains[n-1]),question='Q5 depth by geometry interaction')
    for stage in STAGES:add(f'geometry_stage{stage}_presence',diff(presence[stage,'radon'],presence[stage,'linear_resample']),question='Q5 position by geometry interaction')
    assert sum(d['primary'] for d in defs)==22 and len(defs)==60
    return defs

def apply_clipping(g,cfg):
    policy=cfg.get('communication_clipping')
    if policy is None:return
    if policy!='per_bridge' or len(cfg['bridges'])<2:raise ValueError('Explicit per-bridge clipping requires multiple bridges')
    g.separate_communication_clipping=True
