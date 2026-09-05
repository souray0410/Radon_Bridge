"""186-row synthetic report fixture. NOT participant data or experiment evidence."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from scripts.run_three_seed_study import SEEDS,RHOS,key,identifier
from scripts.run_mechanism_benchmark import matrix,BASELINES
from scripts.run_centered_study import FILES
from scripts.queue_fixed_svd_study import sha
from scripts.report_mechanism_benchmark import build

def fixture(root,plots=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);oldroot=root/'previous';oldroot.mkdir();(oldroot/'report').mkdir();cache=root/'data';cache.mkdir()
    frame=[{'participant_id':'SYNTHETIC_'+str(i),'order':str(i),'split':split,'label_id':i%2,'age':40+i%30,'sex':i%2,'reference_standard_type':'SYNTHETIC FIXTURE'} for split,n in [('train',1264),('validation',296)] for i in range(n)]
    pd.DataFrame(frame).to_csv(cache/'selected.csv',index=False)
    oldarms=['learned','svd_radon','svd_self','svd_scrambled','svd_resample','qr_radon']+['centered_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]+['channel_'+a for a in ['svd_radon','svd_self','svd_scrambled','svd_resample']]
    configs=[(s,a,r) for s in SEEDS for a,r in [('no_bridge',None)]+[(a,r) for r in RHOS for a in oldarms]]+matrix();rows=[];oldrows=[];ledger=[];y=np.tile([0,1],148);rng=np.random.default_rng(176)
    def cost(name):return {'id':name,'gpu_seconds':1.,'sampled_peak_process_mib':1}
    for i,(s,a,rho) in enumerate(configs):
        name=identifier(s,a,rho);path=root/name;path.mkdir();parents={b:{'path':'SYNTHETIC','sha256':'0'*64} for b in ['cfp','oct']};cfg={'seed':s,'parent_checkpoints':parents,'bridges':[] if a=='no_bridge' else [{'M':32,'S':64,'rho':rho,'mode':'linear_resample' if a.endswith('resample') else 'radon'}]}
        if a in BASELINES:cfg['bridges']=[{'family':'mmtm' if a.startswith('mmtm') else 'cross_attention'}]
        p=rng.uniform(.01,.99,(2,296));probs=np.stack([1-p,p],-1);tasks={b:classification_metrics(y,probs[j]) for j,b in enumerate(['cfp','oct'])};scores={b:100*tasks[b]['macro_f1'] for b in tasks};scores['mean']=sum(scores.values())/2
        summary={'state':'complete','converged_by_policy':True,'stop_reason':'validation_plateau','test_used':False,'selected':{'tasks':tasks},'selection':{'joint':{'best_epoch':2}},'epochs_ran':8,'trainable_parameters':1234,'configuration':cfg}
        write_json(path/'configuration.json',cfg);write_json(path/'summary.json',summary);(path/'selected.pt').write_text('SYNTHETIC');np.savez(path/'selected_predictions.npz',ids=np.asarray(['SYNTHETIC_'+str(j) for j in range(296)]),y=y,cfp=probs[0],oct=probs[1]);hashes={n:sha(path/n) for n in FILES}
        energy={b+'_stage3':{'retained_energy_ratio':None if a in BASELINES else .5,'delta_over_input_l2':.1,'projection_basis':None} for b in ['cfp','oct']}
        phase={'probe_participants':128,'energy_participants':1264,'state_parameters_bn_gradients_rng_preserved':True,'energy':energy,'gradient_groups':{b+'_stage3':{'cosine':.01} for b in ['cfp','oct']},'probe_ids':['MUST_NOT_APPEAR_PUBLICLY']}
        diag={'passed':True,'preflight':False,'selected_sha256':hashes['selected.pt'],'phases':{'initial':phase,'selected':phase}}
        info={'groups':[] if a=='no_bridge' else [{'stored_bridge_parameters':100,'effective_bridge_parameters':100}]};write_json(path/'model.json',info)
        r={'id':name,'seed':s,'arm':a,'rho':rho,'directory':str(path),'reused':i<129,'configuration':cfg,'accepted_hashes':hashes};rows.append(r)
        if i<129:oldrows.append(dict(r,summary=summary,scores=scores,model=info,result_hashes=hashes,cost=cost(name),diagnostic=diag,stored_bridge_parameters=100,effective_bridge_parameters=100))
        else:
            dpath=root/('diagnostic_'+name);dpath.mkdir();write_json(dpath/'summary.json',diag);ledger.extend([cost(name),cost(dpath.name)])
        if a in ['svd_radon','qr_radon','svd_oct_to_cfp','svd_cfp_to_oct','qr_oct_to_cfp','qr_cfp_to_oct']:
            directory=root/('pairing_'+name);directory.mkdir();result={'conditions':[{'condition':'paired','repeat':None,'branches':{b:{'metrics':tasks[b]} for b in tasks}}],'permutation_aggregates':{condition:{b:{'macro_f1':{'mean':tasks[b]['macro_f1']-.01,'sample_sd':.01,'min':.4,'max':.6}} for b in tasks} for condition in ['shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both']}}
            write_json(directory/'summary.json',{'passed':True,'test_used':False,'selected_sha256':hashes['selected.pt'],'result':result});ledger.append(cost(directory.name))
        if a in BASELINES+['no_bridge','svd_radon','qr_radon']:
            directory=root/('latency_'+name);directory.mkdir();write_json(directory/'summary.json',{'passed':True,'test_used':False,'selected_sha256':hashes['selected.pt'],'result':{'median_ms':float(rng.uniform(10,30)),'q25_ms':10.,'q75_ms':30.,'timing_interfered':True}});ledger.append(cost(directory.name))
    write_json(oldroot/'report/results.json',{'rows':oldrows});write_json(root/'manifest.json',{'rows':rows});write_json(root/'protocol.json',{'predecessor':str(oldroot),'data':str(cache),'source_commit':'SYNTHETIC_FIXTURE'});write_json(root/'ledger.json',{'jobs':ledger});write_json(root/'study_summary.json',{'cumulative_gpu_minutes':0});write_json(root/'gpu_acceptance.json',{'passed':True})
    build(root,resamples=101,make_plots=plots)
    data=json.loads((root/'report/results.json').read_text());assert len(data['rows'])==186 and 'MUST_NOT_APPEAR_PUBLICLY' not in (root/'report/results.json').read_text();assert len(pd.read_csv(root/'report/results.csv'))==186
    assert sum(d['primary'] for d in data['bootstrap']['contrasts'])==31
    print(json.dumps({'passed':True,'fixture_only':True,'rows':186,'primary_contrasts':31,'public_identifier_exclusion':True,'plot_count':len(data['figures'])}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plots',action='store_true');a=p.parse_args();fixture(a.root,a.plots)
