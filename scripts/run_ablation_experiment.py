"""Bounded overnight M/S/rho and bridge-position ablations."""
import argparse
import hashlib
import json
from pathlib import Path

from radonbridge.experiment import bridge_configs, write_json
from scripts.run_integer_experiment import Controller, job, read_json


def make_config(protocol, spec, estimate_epochs):
    bridges=bridge_configs(spec['stages'], 'radon', M=spec['M'], S=spec['S'], rho=spec['rho'])
    return {'seed':protocol['seed'], 'backbone_lr':protocol['backbone_lr'],
            'bridges':bridges, 'convergence':protocol['convergence'],
            'microbatch':protocol['microbatch'], 'effective_batch':16,
            'training_stage':'communication',
            'parent_checkpoints':protocol['parent_checkpoints'],
            'budget_estimate_epochs':estimate_epochs}


def main(args):
    c=Controller(args)
    p=c.protocol
    preflight=read_json(c.root/'preflight.json')
    assert preflight['passed']
    # Reference trials are immutable links used only by the aggregate exporter.
    for identifier,item in p['reference_trials'].items():
        source=Path(item['directory'])
        if hashlib.sha256((source/'summary.json').read_bytes()).hexdigest()!=item['summary_sha256']:
            raise ValueError('Reference trial summary hash mismatch')
        target=c.root/identifier
        if not target.exists():target.symlink_to(source,target_is_directory=True)
    report={'study':'M S rho and position ablation','source_commit':c.commit,
            'reference_trials':p['reference_trials'],'trials':{},'test_used':False}
    seconds_per_epoch=p['initial_seconds_per_epoch']
    expected_epochs=p['initial_estimate_epochs']
    for pair in p['pairs']:
        jobs=[]
        for spec in pair:
            cfg=make_config(p,spec,expected_epochs)
            jobs.append(job(spec['id'],cfg))
        c.phase='ablation_'+'_'.join(x['id'] for x in pair)
        if not c.group_fits(jobs,seconds_per_epoch):
            write_json(c.root/'study_summary.json',report);return
        completed=c.run_jobs(jobs)
        report['trials'].update(completed);write_json(c.root/'study_summary.json',report)
        if not all(x['converged_by_policy'] for x in completed.values()):
            c.status('needs_attention',reason='Ablation reached epoch cap without plateau');return
        seconds_per_epoch=max(t for x in completed.values() for t in x['epoch_seconds'])
        expected_epochs=min(p['convergence']['max_epochs'],
                            int(max(x['epochs_ran'] for x in completed.values())*1.25+0.999999))
    if p.get('replicate_best'):
        # Select among R&B configurations only; the reference stage3 result is
        # eligible.  This is exploratory development-set selection and the
        # second seed is a stability check, not an untouched external test.
        candidates={spec['id']:spec for pair in p['pairs'] for spec in pair}
        reference=read_json(Path(p['reference_trials']['confirm_3416_radon']['directory'])/'summary.json')
        scores={'reference_stage3':reference['selected']['mean_task_macro_f1']}
        scores.update({name:value['selected']['mean_task_macro_f1'] for name,value in report['trials'].items()})
        best=max(scores,key=scores.get)
        best_spec={'id':'reference_stage3','stages':[3],'M':16,'S':64,'rho':.125} if best=='reference_stage3' else candidates[best]
        report['selected_for_seed_3417']={'configuration':best_spec,'seed_3416_mean_macro_f1':scores[best]}
        warm={'seed':3417,'backbone_lr':p['backbone_lr'],'bridges':[],
              'convergence':p['convergence'],'microbatch':p['microbatch'],'effective_batch':16,
              'training_stage':'independent','budget_estimate_epochs':14}
        warm_job=job('pretrain_3417',warm)
        c.phase='replication_pretrain_3417'
        if not c.group_fits([warm_job],seconds_per_epoch):
            write_json(c.root/'study_summary.json',report);return
        trained=c.run_jobs([warm_job])['pretrain_3417'];report['trials']['pretrain_3417']=trained
        if not trained['converged_by_policy']:
            c.status('needs_attention',reason='Replication pretraining did not reach plateau');return
        parents=trained['modality_checkpoints']
        baseline={'seed':3417,'backbone_lr':p['backbone_lr'],'bridges':[],
                  'convergence':p['convergence'],'microbatch':p['microbatch'],'effective_batch':16,
                  'training_stage':'communication','parent_checkpoints':parents,'budget_estimate_epochs':expected_epochs}
        chosen=dict(baseline)
        chosen['bridges']=bridge_configs(best_spec['stages'],'radon',M=best_spec['M'],S=best_spec['S'],rho=best_spec['rho'])
        replication=[job('replicate_3417_independent',baseline),job('replicate_3417_best',chosen)]
        c.phase='replication_best_3417'
        if not c.group_fits(replication,seconds_per_epoch):
            write_json(c.root/'study_summary.json',report);return
        completed=c.run_jobs(replication);report['trials'].update(completed)
        report['replication_complete']=all(x['converged_by_policy'] for x in completed.values())
    report['new_gpu_minutes']=c.used();write_json(c.root/'study_summary.json',report)
    c.status('complete')


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--phase',choices=['run'],default='run')
    q.add_argument('--output',required=True);q.add_argument('--protocol',required=True);q.add_argument('--data',required=True)
    main(q.parse_args())
