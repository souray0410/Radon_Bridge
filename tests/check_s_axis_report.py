"""Full synthetic report fixture: counts, paired signs, privacy and zero variance."""
import tempfile,json,copy
from pathlib import Path
import numpy as np
from radonbridge.experiment import write_json
from radonbridge.metrics import classification_metrics
from scripts.queue_fixed_svd_study import sha,read
from scripts.run_s_axis_supplement import SEEDS,RHOS,BASES,new_rows
from radonbridge.s_axis import fixed_permutation
import scripts.report_s_axis_supplement as reporter

def run():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);prev=root/'previous';prev.mkdir();fusion=root/'fusion';fusion.mkdir();out=root/'new';out.mkdir()
        rng=np.random.default_rng(59);labels=np.tile([0,1],148);identity=np.array(['FORBIDDEN_ID_'+str(i) for i in range(296)])
        records=[]
        def materialize(r):
            trial=root/r['id'];trial.mkdir();r['directory']=str(trial)
            prob=rng.random((2,296,2));prob/=prob.sum(-1,keepdims=True)
            np.savez(trial/'selected_predictions.npz',ids=identity,y=labels,cfp=prob[0],oct=prob[1],fusion=prob.mean(0))
            (trial/'selected.pt').write_text('SYNTHETIC');write_json(trial/'configuration.json',r['configuration'])
            tasks={b:classification_metrics(labels,prob[i]) for i,b in enumerate(['cfp','oct'])}
            write_json(trial/'summary.json',dict(state='complete',converged_by_policy=True,stop_reason='validation_plateau',test_used=False,selected=dict(tasks=tasks),selection=dict(joint=dict(best_epoch=2)),epochs_ran=8,parameters=100,peak_reserved_mib=100))
            r['accepted_hashes']={n:sha(trial/n) for n in ['summary.json','selected.pt','selected_predictions.npz','configuration.json']}
            return r
        for seed in SEEDS:
            for arm,rho in [('no_bridge',None)]+[(b+'_radon',r) for b in BASES for r in RHOS]:
                cfg=dict(seed=seed,bridges=[] if rho is None else [dict(M=32,S=64,rho=rho,mode='radon')])
                records.append(materialize(dict(id=f'old_{seed}_{arm}_{rho}',seed=seed,arm=arm,rho=rho,configuration=cfg)))
        new=new_rows(records)
        for r in new:
            materialize(r);diag=root/('diagnostic_'+r['id']);diag.mkdir();r['diagnostic']=str(diag)
            phase=dict(probe_participants=128,energy_participants=1264,state_parameters_bn_gradients_rng_preserved=True,probe_ids=['FORBIDDEN_PROBE'])
            write_json(diag/'summary.json',dict(passed=True,selected_sha256=r['accepted_hashes']['selected.pt'],phases=dict(initial=phase,selected=phase)))
            old=next(a for a in records if a['id']==r['reference_id'])
            expected=copy.deepcopy(old['configuration']);expected['bridges'][0]['s_axis_permutation']=fixed_permutation(64)['permutation'];assert r['configuration']==expected
        fs=[]
        for seed in SEEDS:
            for lr in [3e-5,6e-5]:
                for arm in ['concat_mlp']+['baseline'+str(i) for i in range(8)]:fs.append(materialize(dict(id=f'fusion_{seed}_{lr}_{arm}',seed=seed,backbone_lr=lr,arm=arm,configuration={})))
        write_json(prev/'manifest.json',dict(rows=records));write_json(fusion/'manifest.json',dict(rows=fs));(fusion/'report').mkdir();write_json(fusion/'report/statistics.json',dict(fixture=True))
        write_json(out/'manifest.json',dict(rows=new,references=records));write_json(out/'s_axis_permutation.json',fixed_permutation(64))
        write_json(out/'study_summary.json',{});write_json(out/'source_acceptance.json',{});write_json(out/'protocol.json',dict(synthetic_fixture=True))
        reporter.PREDECESSOR=fusion;reporter.PREVIOUS=prev
        reporter.build(out,resamples=101,make_plots=True)
        data=read(out/'report/results.json');assert len(data['rows'])==39 and len(data['aggregate'])==13
        assert len(read(out/'report/error_transfers_task_fusion.json'))==48
        assert len(read(out/'report/error_transfers_mechanism.json'))==18
        assert len(read(out/'report/paired_cells.json'))==54
        assert 'FORBIDDEN' not in ''.join(p.read_text() for p in (out/'report').glob('*.json'))
        stats=read(out/'report/statistics.json');assert stats['primary_count']==3
        from scripts.report_qr_nested_supplement import bootstrap_models
        rng=np.random.default_rng(1);prob=rng.random((39,2,296,2));prob/=prob.sum(-1,keepdims=True)
        point,samples,_=bootstrap_models(prob,labels,101);actual=reporter.primary(data['rows'],point,samples)
        lookup={(r['seed'],r['arm'],r['rho']):i for i,r in enumerate(data['rows'])}
        differences=[]
        for basis in BASES:
            differences.append(np.mean([point[lookup[s,basis+'_radon',rho]]-point[lookup[s,basis+'_s_permuted',rho]] for s in SEEDS for rho in RHOS]))
        assert np.allclose([c['difference_pp'] for c in actual['contrasts']],[*differences,differences[0]-differences[1]])
        identical=np.broadcast_to(point[:1],point.shape).copy();draws=np.broadcast_to(identical,(101,*identical.shape)).copy();z=reporter.primary(data['rows'],identical,draws)
        assert all(c['zero_variance'] and c['simultaneous_ci95_pp'] is None for c in z['contrasts'])
        e=reporter.error_transfers(np.array([0,1,0,1]),np.array([0,0,1,1]),np.array([1,1,1,1]))['all']
        assert e==dict(n=4,old_wrong_new_correct=1,old_correct_new_wrong=1,both_correct=1,both_wrong=1)
        import shutil
        target=Path('/tmp/radonbridge_s_axis_report_fixture');shutil.copytree(out/'report',target,dirs_exist_ok=True)
        print(json.dumps(dict(passed=True,synthetic_only=True,new_trainings=18,reference_results=21,figures=len(data['figures']),primary_count=3,privacy=True,zero_variance=True,paired_signs=True,fixture=str(target))))
if __name__=='__main__':run()
