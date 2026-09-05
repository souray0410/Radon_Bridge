"""Synthetic-only full 213-training / 231-view report and counting acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from scripts.run_three_seed_study import SEEDS,RHOS,identifier
from scripts.run_mechanism_benchmark import matrix as old_matrix,BASELINES
from scripts.run_qr_nested_supplement import matrix,NESTED_ARMS
from scripts.report_qr_nested_supplement import build,make_weights
from scripts.queue_fixed_svd_study import sha


def fixture(root,plots=False):
    root=Path(root);root.mkdir(exist_ok=False,parents=True)
    oldarms=['learned','svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']+['centered_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]+['channel_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]
    configs=[(s,a,r) for s in SEEDS for a,r in [('no_bridge',None)]+[(a,r) for r in RHOS for a in oldarms]]+old_matrix()+matrix()
    assert len(configs)==len(set(configs))==213
    rng=np.random.default_rng(94);labels=np.tile([0,1],148);records=[];ledger=[]
    for index,(seed,arm,rho) in enumerate(configs):
        name=identifier(seed,arm,rho);trial=root/name;trial.mkdir();nested=arm in NESTED_ARMS
        compression='learned_channel' if arm.startswith('channel') or arm=='nested_learned_channel' else 'fixed_random_orthogonal_channel' if 'qr' in arm else 'learned_projected' if arm=='learned' else 'fixed_svd_channel'
        cfg={'seed':seed,'bridges':[] if arm=='no_bridge' else [{'compression':compression,'rho':rho,'M':32,'S':64,'mode':'radon'}]}
        if nested:cfg['bridges'][0]['nested_rhos']=RHOS
        if arm in BASELINES:cfg['bridges']=[{'family':'mmtm' if arm.startswith('mmtm') else 'cross_attention'}]
        write_json(trial/'configuration.json',cfg);(trial/'selected.pt').write_text('SYNTHETIC')
        views=[];per={}
        for r in RHOS if nested else [rho]:
            path=trial/'width_exports'/('rho1_'+str(round(1/r))) if nested else trial;path.mkdir(exist_ok=True,parents=True)
            viewcfg=json.loads(json.dumps(cfg))
            if nested:viewcfg['bridges'][0].pop('nested_rhos');viewcfg['bridges'][0]['rho']=r
            write_json(path/'configuration.json',viewcfg);(path/'selected.pt').write_text('SYNTHETIC')
            u=rng.uniform(.05,.95,(2,296));p=np.stack([1-u,u],-1)
            np.savez(path/'selected_predictions.npz',ids=np.asarray(['PRIVATE_SYNTHETIC_'+str(i) for i in range(296)]),y=labels,cfp=p[0],oct=p[1])
            tasks={b:classification_metrics(labels,p[i]) for i,b in enumerate(['cfp','oct'])};metrics={'tasks':tasks,'mean_task_macro_f1':sum(v['macro_f1'] for v in tasks.values())/2}
            if nested:per['rho1_'+str(round(1/r))]=metrics;views.append({'directory':str(path),'rho':r})
            if index>=186:
                dn='diagnostic_'+name+('_'+str(round(1/r)) if nested else '');dp=root/dn;dp.mkdir()
                phase={'probe_participants':128,'energy_participants':1264,'state_parameters_bn_gradients_rng_preserved':True,'probe_ids':['FORBIDDEN_PRIVATE_PROBE']}
                write_json(dp/'summary.json',{'passed':True,'selected_sha256':sha(path/'selected.pt'),'phases':{'initial':phase,'selected':phase}})
        selected={'per_width':per} if nested else metrics
        summary={'state':'complete','converged_by_policy':True,'stop_reason':'validation_plateau','test_used':False,'selected':selected,'selection':{'joint':{'best_epoch':2}},'epochs_ran':8,'parent_checkpoints':{},'parameters':1000,'peak_reserved_mib':100}
        write_json(trial/'summary.json',summary)
        if nested:write_json(trial/'width_exports.json',views)
        record={'id':name,'seed':seed,'arm':arm,'rho':rho,'directory':str(trial),'configuration':cfg,'reused':index<186}
        if index<186:record['accepted_hashes']={n:sha(trial/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']};record['reference_cost']={'id':name,'gpu_seconds':1}
        else:ledger.append({'id':name,'gpu_seconds':1})
        records.append(record)
    write_json(root/'manifest.json',{'rows':records});write_json(root/'ledger.json',{'jobs':ledger});write_json(root/'protocol.json',{'fixture':True});write_json(root/'study_summary.json',{'fixture':True})
    build(root,resamples=101,plots=plots)
    out=root/'report';d=json.loads((out/'results.json').read_text());assert len(d['rows'])==231 and len({r['training_id'] for r in d['rows']})==213
    assert 'FORBIDDEN_PRIVATE_PROBE' not in (out/'results.json').read_text() and 'PRIVATE_SYNTHETIC' not in (out/'results.json').read_text()
    for g in ['A','B']:
        defs=make_weights(d['rows'],g);assert len(defs)==3 and all(abs(v['weights'].sum())<1e-10 for v in defs)
    for filename,n in [('A_single_width_204_results.csv',204),('B_joint_width_27_views.csv',27)]:assert len((out/filename).read_text().splitlines())==n+1
    a=json.loads((out/'A_QR_mechanism_statistics.json').read_text());b=json.loads((out/'B_joint_width_statistics.json').read_text());assert a['shared_indices_sha256']==b['shared_indices_sha256']
    print(json.dumps({'passed':True,'fixture_only':True,'trainings':213,'views':231,'separate_primary_families':[3,3],'shared_participant_indices':True,'public_identity_exclusion':True,'figures':len(json.loads((out/'figures.json').read_text()))}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plots',action='store_true');a=p.parse_args();fixture(a.root,a.plots)
