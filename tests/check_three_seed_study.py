"""Matrix and report acceptance tests; generated fixtures are never study results."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from scripts.run_three_seed_study import SEEDS,RHOS,METHODS,matrix,key,config,identifier,OLD,SVD
from scripts.report_three_seed_study import build
from scripts.report_five_seed_study import read,sha,paired_bootstrap
from radonbridge.experiment import write_json


def check_matrix():
    xs=matrix();assert len(xs)==57 and len(set(xs))==57 and {x[0] for x in xs}=={3416,3417,3418}
    for s in SEEDS:
        assert sum(x[0]==s and x[1]=='no_bridge' for x in xs)==1
        for a in METHODS:assert {x[2] for x in xs if x[0]==s and x[1]==a}==set(RHOS)
    for s,a,r in xs:
        cfg=config(s,{'cfp':{},'oct':{}},{},{},a,r)
        assert cfg['backbone_lr']==6e-5 and cfg['microbatch']==cfg['effective_batch']==16
        for b in cfg['bridges']:
            assert b['rho']==r and b['M']==32 and b['S']==64
            assert int(256*r)*32==int(8192*r)
    assert not {3419,3420}&set(SEEDS)


def report_fixture():
    root=Path(tempfile.mkdtemp(prefix='three_report_QA_ONLY_'));rows=[];jobs=[]
    for s,a,r in matrix():
        source_seed=3416 if s==3418 else s
        if a=='no_bridge':source=OLD/f'seed{source_seed}_bb6e-05_independent'
        elif a=='learned':source=OLD/f'seed{source_seed}_bb6e-05_rho1_{round(1/r)}'
        else:source=SVD/f'seed{source_seed}_bb6e-05_rho1_{round(1/r)}_svd'
        name=identifier(s,a,r);path=root/name;path.mkdir();summary=read(source/'summary.json');summary['configuration']['seed']=s
        write_json(path/'summary.json',summary);write_json(path/'configuration.json',summary['configuration']);write_json(path/'model.json',read(source/'model.json'))
        for file in ['selected_predictions.npz','selected.pt']:(path/file).symlink_to(source/file)
        rows.append({'id':name,'seed':s,'arm':a,'rho':r,'directory':str(path),'reused':False,'configuration':summary['configuration']})
        diag=root/('diagnostic_'+name);diag.mkdir()
        phase={'probe_participants':128,'energy_participants':1264,'probe_ids_sha256':'QA_ONLY',
          'gradient_groups':{b+'_stage3':{'cosine':None if a=='no_bridge' else .01} for b in ['cfp','oct']},
          'energy':{b+'_stage3':{'retained_energy_ratio':.75,'delta_over_input_l2':.1} for b in ['cfp','oct']}}
        write_json(diag/'summary.json',{'passed':True,'preflight':False,'selected_sha256':sha(source/'selected.pt'),'phases':{'initial':phase,'selected':phase},'QA_ONLY':True})
        jobs += [{'id':n,'gpu_seconds':1.,'sampled_peak_process_mib':6000} for n in [name,'diagnostic_'+name]]
    write_json(root/'manifest.json',{'seeds':SEEDS,'rhos':RHOS,'rows':rows,'QA_ONLY':True})
    for name,obj in [('protocol',{'source_commit':'QA_ONLY'}),('ledger',{'jobs':jobs}),('gpu_acceptance',{'ledger':{'jobs':[]}}),('study_summary',{'state':'QA_ONLY'})]:write_json(root/(name+'.json'),obj)
    build(root,resamples=100);result=read(root/'report/results.json')
    assert len(result['rows'])==57 and sum(len(arms) for rhos in result['bootstrap']['comparisons'].values() for arms in rhos.values())==72
    # With identical models repeated, shared-index intervals must not narrow.
    path=OLD/'seed3416_bb6e-05_independent';main=SVD/'seed3416_bb6e-05_rho1_8_svd'
    assert paired_bootstrap([(path,main)],100)==paired_bootstrap([(path,main)]*3,100)
    # Strict completeness gate rejects an incomplete trial before any final report.
    f=root/rows[0]['id']/'summary.json';bad=read(f);bad['converged_by_policy']=False;write_json(f,bad)
    try:build(root,resamples=10,make_figures=False)
    except AssertionError:pass
    else:raise AssertionError('Accepted an incomplete trial')
    print(json.dumps({'passed':True,'QA_ONLY':True,'root':str(root),'matrix_count':57,'comparisons':72}))

if __name__=='__main__':
    check_matrix();report_fixture()
