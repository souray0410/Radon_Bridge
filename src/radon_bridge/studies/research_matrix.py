"""Finite, outcome-independent study arms; expanded discovery never rewrites a round."""
import copy
from radon_bridge.runtime.state import stable_hash


def arm(name, **kw):
    return dict(id=name,family='radon',compression='fixed_svd_channel',mode='radon',stages=[3],r=32,M=32,S=64,k=3,frozen=False,**kw)


def changed(name,**kw):
    a=arm(name);a.update(kw);return a


def arms(disease,architecture='resnet50'):
    result=[changed('continue',family='none'),arm('svd_radon'),changed('svd_resample',mode='linear_resample'),
        changed('svd_self',mode='self'),changed('mmtm256',family='mmtm'),changed('attention256',family='cross_attention')]
    if disease!='glaucoma' or architecture!='resnet50':return result
    for mode in ('radon','linear_resample','self'):
        result.append(changed('qr_'+mode,compression='fixed_random_orthogonal_channel',mode=mode))
    for compression,label in [('fixed_centered_svd_channel','centered'),('learned_channel','learned_channel')]:
        for mode in ('radon','linear_resample'):
            result.append(changed(label+'_'+mode,compression=compression,mode=mode))
    for compression,label in [('fixed_svd_channel','svd'),('fixed_random_orthogonal_channel','qr')]:
        result += [changed(label+'_spatial_scramble',compression=compression,mode='scrambled'),
                   changed(label+'_s_axis_scramble',compression=compression,s_axis_scramble=True)]
        for source,destination in [('oct','cfp'),('cfp','oct')]:
            result.append(changed(label+'_'+source+'_to_'+destination,compression=compression,direction=[source,destination]))
        for mode in ('radon','linear_resample','self'):
            result.append(changed(label+'_frozen_'+mode,compression=compression,mode=mode,frozen=True))
    for field,values in [('r',(16,64)),('M',(16,64)),('S',(32,128)),('k',(1,5))]:
        for value in values:
            for mode in ('radon','linear_resample'):
                result.append(changed(f'{field}{value}_{mode}',mode=mode,**{field:value}))
    for stages in ([2,3],[2,3,4]):
        # Finite depth test with matched ordinary/self controls. Total capacities
        # are recorded; no claim of equality to the single bridge is made.
        for mode in ('radon','linear_resample','self'):
            result.append(changed('depth'+''.join(map(str,stages))+'_'+mode,stages=stages,r=16,mode=mode))
    for host in ('mmtm256','attention256'):
        for extra in ('continue','radon','linear_resample'):
            result.append(changed(host+'_plus_'+extra,host=host,addition=extra))
    if len({a['id'] for a in result})!=len(result):raise ValueError('Duplicate research arm')
    return result


def comparisons(disease,architecture='resnet50'):
    result=[dict(id='primary_radon_minus_'+other,left='svd_radon',right=other,question='direct',branch='mean')
            for other in ('continue','svd_resample','svd_self','mmtm256','attention256')]
    if disease!='glaucoma' or architecture!='resnet50':return result
    for a in arms(disease,architecture)[6:]:
        name=a['id'];reference='svd_radon';question='robustness'
        if name.startswith('qr_'):question='basis_and_geometry'
        if 'frozen' in name:question='coadaptation'
        if 'scramble' in name:question='spatial_or_s_adjacency'
        if '_to_' in name:question='direction'
        if name.startswith('depth'):question='bridge_depth'
        if '_plus_' in name:
            question='host_augmentation'
            if name.endswith('_continue'):continue
            reference=a['host']+'_plus_continue'
        result.append(dict(id=name+'_minus_'+reference,left=name,right=reference,question=question,branch=('oct' if 'oct_to_cfp' in name else 'cfp') if '_to_' in name else 'mean'))
    # Explicit matched geometric contrasts and interactions, in addition to the
    # reference display. Model conditions, not favorable rankings, define these.
    for a in arms(disease,architecture):
        if a['mode']=='linear_resample' and not a.get('host'):
            candidates=[b for b in arms(disease,architecture) if {k:v for k,v in b.items() if k not in ('id','mode')}=={k:v for k,v in a.items() if k not in ('id','mode')} and b['mode']=='radon']
            if candidates:result.append(dict(id='geometry_'+a['id'],left=candidates[0]['id'],right=a['id'],question='matched_geometry',branch='mean'))
    return result


def coverage():
    return {'schema':'radon_research_coverage_v1','first_pair':'resnet50_2d__resnet50_3d',
        'diseases':['cataract','glaucoma','macular_degeneration'],'seeds':[3416,3417,3418],
        'arms':{d:arms(d) for d in ('cataract','glaucoma','macular_degeneration')},
        'comparisons':{d:comparisons(d) for d in ('cataract','glaucoma','macular_degeneration')},
        'test_access':False,'future_architectures':'new_locked_round_requires_real_2d_3d_parent_adapters',
        'selection':'development_mean_branch_macro_f1','final_fusion_head':False}
