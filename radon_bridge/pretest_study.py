"""Outcome-independent completion matrix and the seventeen new contrasts."""
import copy
from .geometry_study import configuration, fingerprint, SEEDS, LRS

RHOS=(1/16,1/8,1/4)
COMPRESSIONS={'svd':'fixed_svd_channel','qr':'fixed_random_orthogonal_channel'}


def matrices(parents,svd,qr):
    geometry=[];frozen=[]
    for M,r in ((16,16),(64,32)):
        for mode in ('radon','linear_resample'):
            for seed in SEEDS:
                for lr in LRS:
                    s=dict(id=f'{mode}_M{M}_S64_k3_r{r}',M=M,S=64,k=3,r=r,h=M*r,rho=r/256,mode=mode)
                    cfg=configuration(s,seed,lr,'branch',parents[seed],svd[seed])
                    geometry.append(row(f'completion_{s["id"]}_lr{lr:g}_seed{seed}',cfg,s,'geometry_completion'))
    for basis in ('svd','qr'):
        for rho in RHOS:
            r=int(256*rho)
            for mode in ('radon','linear_resample','self'):
                for seed in SEEDS:
                    s=dict(id=f'frozen_{basis}_{mode}_r{r}',basis=basis,M=32,S=64,k=3,r=r,h=32*r,rho=rho,mode=mode)
                    cfg=configuration(s,seed,None,'branch',parents[seed],(svd if basis=='svd' else qr)[seed])
                    cfg.update(optimization_mode='bridge_only',head_lr=None)
                    cfg['bridges'][0]['compression']=COMPRESSIONS[basis]
                    frozen.append(row(f'{s["id"]}_seed{seed}',cfg,s,'frozen_completion'))
    assert len(geometry)==24 and len(frozen)==54
    assert len({r['fingerprint'] for r in geometry+frozen})==78
    return geometry,frozen


def row(identifier,cfg,structure,category):
    return dict(id=identifier,configuration=cfg,fingerprint=fingerprint(cfg),category=category,
                protocol='branch',seed=cfg['seed'],backbone_lr=cfg['backbone_lr'],structure=structure,
                state='pending',directory=None,attempts=[])


def contrast_definitions(geometry,frozen,full):
    """Sparse weights on model branch-mean F1. Parent references have stable ids."""
    defs=[]
    def g(M,r,mode):
        found=[x for x in geometry if x['structure'].get('M')==M and x['structure'].get('r')==r
               and x['structure'].get('S')==64 and x['structure'].get('k')==3 and x['structure'].get('mode')==mode]
        assert len(found)==6,(M,r,mode,len(found))
        return {x['id']:1/6 for x in found}
    def f(basis,mode):
        found=[x for x in frozen if x['structure']['basis']==basis and x['structure']['mode']==mode]
        assert len(found)==9
        return {x['id']:1/9 for x in found}
    def full_group(basis,mode):
        arm=basis+('_radon' if mode=='radon' else '_resample')
        found=[x for x in full if x.get('arm')==arm]
        assert len(found)==9
        return {x['id']:1/9 for x in found}
    def combine(*terms):
        out={}
        for scale,weights in terms:
            for k,v in weights.items():out[k]=out.get(k,0)+scale*v
        return {k:v for k,v in out.items() if abs(v)>1e-14}
    def diff(a,b):return combine((1,a),(-1,b))
    def geom(M,r):return diff(g(M,r,'radon'),g(M,r,'linear_resample'))
    def add(name,family,w):
        assert abs(sum(w.values()))<1e-10
        defs.append(dict(id=name,family=family,weights=w,outcome='branch_mean_macro_f1'))
    for r in (16,32):
        for M in (32,64):add(f'fixed_r{r}_geometry_M{M}_minus_M16','allocation8',diff(geom(M,r),geom(16,r)))
    channel={}
    for mode in ('radon','linear_resample'):
        channel[mode]=combine(*[(1/3,diff(g(M,32,mode),g(M,16,mode))) for M in (16,32,64)])
        add(mode+'_rank32_minus_rank16','allocation8',channel[mode])
    add('geometry_rank_interaction','allocation8',diff(channel['radon'],channel['linear_resample']))
    add('M64_M16_by_rank_interaction','allocation8',diff(diff(geom(64,32),geom(16,32)),diff(geom(64,16),geom(16,16))))
    parent={f'parent_seed{s}':1/3 for s in SEEDS};fg={}
    for basis in ('svd','qr'):
        for mode in ('linear_resample','self','parent'):
            add(f'frozen_{basis}_radon_minus_{mode}','frozen9',diff(f(basis,'radon'),parent if mode=='parent' else f(basis,mode)))
        fg[basis]=diff(f(basis,'radon'),f(basis,'linear_resample'))
    add('frozen_basis_geometry_interaction','frozen9',diff(fg['svd'],fg['qr']))
    for basis in ('svd','qr'):
        add(basis+'_full_minus_frozen_geometry','frozen9',diff(diff(full_group(basis,'radon'),full_group(basis,'linear_resample')),fg[basis]))
    assert len(defs)==17
    return defs
