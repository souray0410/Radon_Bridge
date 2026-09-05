"""93-result report fixture. Uses synthetic duplication, never research evidence."""
import argparse,copy,json,tempfile
from pathlib import Path
from radonbridge.experiment import write_json
from scripts.report_five_seed_study import read,sha
from scripts.run_centered_study import SVD_ARMS,FILES
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.report_centered_study import build


def check(source):
    base=read(Path(source)/'report/results.json');root=Path(tempfile.mkdtemp(prefix='centered_report_QA_ONLY_'));previous=root/'previous';previous.mkdir();(previous/'report').mkdir()
    rows=[];oldrows=[];jobs=[]
    def diagnostic(r):
        phase={'probe_participants':128,'energy_participants':1264,'probe_ids_sha256':'QA_ONLY',
            'gradient_groups':{b+'_stage3':{'cosine':.01} for b in ['cfp','oct']},
            'energy':{b+'_stage3':{'retained_energy_ratio':.7,'retained_variance_ratio':.6,'delta_over_input_l2':0.,'input_energy':100.} for b in ['cfp','oct']}}
        path=root/('diagnostic_'+r['id']);path.mkdir();write_json(path/'summary.json',{'passed':True,'preflight':False,'selected_sha256':sha(Path(r['directory'])/'selected.pt'),'phases':{'initial':phase,'selected':phase}})
        jobs.append({'id':'diagnostic_'+r['id'],'gpu_seconds':1.,'sampled_peak_process_mib':6000})
    for original in base['rows']:
        r=copy.deepcopy(original);path=previous/r['id'];path.mkdir();r['directory']=str(path)
        write_json(path/'summary.json',r['summary']);write_json(path/'configuration.json',r['summary']['configuration']);write_json(path/'model.json',r['model'])
        for name in ['selected.pt','selected_predictions.npz']:(path/name).symlink_to(Path(original['directory'])/name)
        r['result_hashes']={n:sha(path/n) for n in FILES};r['reused']=True;r['accepted_hashes']=r['result_hashes'];rows.append(r);oldrows.append(r)
        if r['arm'] in SVD_ARMS:diagnostic(r)
    write_json(previous/'report/results.json',{'rows':oldrows,'QA_ONLY':True})
    for r in oldrows:
        if r['arm'] not in SVD_ARMS:continue
        new=copy.deepcopy(r);new['arm']='centered_'+r['arm'];new['paired_arm']=r['arm'];new['id']=identifier(r['seed'],new['arm'],r['rho']);new['reused']=False
        path=root/new['id'];path.mkdir();new['directory']=str(path);d=copy.deepcopy(r['summary']);d['configuration']['bridges'][0]['compression']='fixed_centered_svd_channel'
        info=copy.deepcopy(r['model']);info['groups'][0].update(stored_bridge_parameters=r['stored_bridge_parameters'],effective_bridge_parameters=r['effective_bridge_parameters'])
        write_json(path/'summary.json',d);write_json(path/'configuration.json',d['configuration']);write_json(path/'model.json',info)
        for name in ['selected.pt','selected_predictions.npz']:(path/name).symlink_to(Path(r['directory'])/name)
        rows.append(new);jobs.append({'id':new['id'],'gpu_seconds':1.,'sampled_peak_process_mib':6000});diagnostic(new)
    write_json(root/'manifest.json',{'rows':rows,'seeds':SEEDS,'rhos':RHOS,'QA_ONLY':True});write_json(root/'protocol.json',{'predecessor':str(previous),'source_commit':'QA_ONLY'})
    write_json(root/'ledger.json',{'jobs':jobs});write_json(root/'gpu_acceptance.json',{});write_json(root/'study_summary.json',{})
    build(root,resamples=100);out=read(root/'report/results.json');assert len(out['rows'])==93
    assert sum(len(a) for rs in out['bootstrap']['comparisons'].values() for a in rs.values())==48
    assert all(v['delta_pp']==0 and v['ci95_pp']==[0.,0.] for rs in out['bootstrap']['comparisons'].values() for arms in rs.values() for bs in arms.values() for v in bs.values())
    # No incomplete result is accepted.
    path=Path(rows[-1]['directory'])/'summary.json';d=read(path);d['converged_by_policy']=False;write_json(path,d)
    try:build(root,resamples=10,make_figures=False)
    except AssertionError:pass
    else:raise AssertionError('Accepted incomplete trial')
    print(json.dumps({'passed':True,'QA_ONLY':True,'root':str(root),'results':93,'paired_comparisons':48,'zero_difference_ci_exact':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();check(a.source)
