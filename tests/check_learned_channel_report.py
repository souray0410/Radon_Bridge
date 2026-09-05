"""Synthetic report acceptance; all figures are watermarked QA_ONLY."""
import argparse,copy,json,tempfile
from pathlib import Path
from radonbridge.experiment import write_json
from scripts.report_five_seed_study import read,sha
from scripts.report_learned_channel_study import build,contrasts
from scripts.run_centered_study import SVD_ARMS,FILES
from scripts.run_three_seed_study import SEEDS,RHOS,identifier

def check(source):
    original=read(Path(source)/'report/results.json')['rows'];assert len(original)==57
    root=Path(tempfile.mkdtemp(prefix='channel_report_QA_ONLY_'));previous=root/'previous';(previous/'report').mkdir(parents=True)
    old=copy.deepcopy(original)
    for r in original:
        if r['arm'] in SVD_ARMS:
            c=copy.deepcopy(r);c['arm']='centered_'+r['arm'];c['id']=identifier(r['seed'],c['arm'],r['rho']);old.append(c)
    rows=[];jobs=[]
    for r in old:
        r.update(reused=True,accepted_hashes=r['result_hashes']);rows.append(copy.deepcopy(r))
    write_json(previous/'report/results.json',{'rows':old,'QA_ONLY':True})
    for r in original:
        if r['arm'] not in SVD_ARMS:continue
        n=copy.deepcopy(r);n.update(arm='channel_'+r['arm'],reused=False);n['id']=identifier(r['seed'],n['arm'],r['rho']);path=root/n['id'];path.mkdir();n['directory']=str(path)
        d=copy.deepcopy(r['summary']);d['configuration']['bridges'][0]['compression']='learned_channel';d['configuration']['bridges'][0].pop('basis_files')
        write_json(path/'summary.json',d);write_json(path/'configuration.json',d['configuration']);info=copy.deepcopy(r['model']);info['groups'][0].update(stored_bridge_parameters=r['stored_bridge_parameters']+4*256*int(256*r['rho']),effective_bridge_parameters=r['effective_bridge_parameters']+4*256*int(256*r['rho']));write_json(path/'model.json',info)
        for f in ['selected.pt','selected_predictions.npz']:(path/f).symlink_to(Path(r['directory'])/f)
        diag=copy.deepcopy(r['diagnostic']);diag['selected_sha256']=sha(path/'selected.pt')
        for phase in diag['phases'].values():
            phase['state_parameters_bn_gradients_rng_preserved']=True
            for e in phase['energy'].values():e.update(encoded_energy_ratio=.7,retained_variance_ratio=.6);e['projection_basis']['numerical_rank']=int(256*r['rho'])
        dp=root/('diagnostic_'+n['id']);dp.mkdir();write_json(dp/'summary.json',diag);rows.append(n)
        jobs.extend([{'id':n['id'],'gpu_seconds':1.,'sampled_peak_process_mib':6000},{'id':'diagnostic_'+n['id'],'gpu_seconds':1.,'sampled_peak_process_mib':6000}])
    write_json(root/'manifest.json',{'rows':rows,'seeds':SEEDS,'rhos':RHOS,'QA_ONLY':True});write_json(root/'protocol.json',{'predecessor':str(previous),'source_commit':'QA_ONLY'})
    write_json(root/'ledger.json',{'jobs':jobs});write_json(root/'gpu_acceptance.json',{});write_json(root/'study_summary.json',{})
    build(root,resamples=50);result=read(root/'report/results.json');assert len(result['rows'])==129
    assert sum(len(pairs) for rs in result['bootstrap']['comparisons'].values() for pairs in rs.values())==len(contrasts())*12==168
    for rs in result['bootstrap']['comparisons'].values():
        for pairs in rs.values():
            for arm in SVD_ARMS:
                for v in pairs['channel_'+arm+'__minus__'+arm].values():assert v['delta_pp']==0 and v['ci95_pp']==[0.,0.]
    f=Path(rows[-1]['directory'])/'summary.json';d=read(f);d['converged_by_policy']=False;write_json(f,d)
    try:build(root,resamples=5,make_figures=False)
    except AssertionError:pass
    else:raise AssertionError('Incomplete result accepted')
    print(json.dumps({'passed':True,'QA_ONLY':True,'root':str(root),'results':129,'comparisons':168,'identity_ci_exact':True,'incomplete_rejected':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();check(a.source)
