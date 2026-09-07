"""Queue six frozen-network trials and 24 four-state component diagnostics."""
import argparse,copy,fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from radonbridge.artifacts import SOURCE,STUDY,resolve,sha256
from radonbridge.geometry_study import configuration,fingerprint,SEEDS
from scripts.geometry_evidence import read,write,dependencies
from scripts.run_integer_experiment import source_hashes
from scripts.run_geometry_mechanism import Queue,scan_completed,interrupted
from scripts.run_task_fusion_benchmark import live_workers


def frozen_rows(parents,bases):
    rows=[]
    for mode in ('radon','linear_resample'):
        for seed in SEEDS:
            s=dict(id='frozen_'+mode,M=32,S=64,k=3,r=16,h=512,rho=1/16,mode=mode)
            cfg=configuration(s,seed,None,'branch',parents[seed],bases[seed])
            cfg.update(optimization_mode='bridge_only',head_lr=None)
            rows.append(dict(id=f'frozen_{mode}_seed{seed}',protocol='branch',category='frozen',
                             seed=seed,backbone_lr=None,structure=s,configuration=cfg,fingerprint=fingerprint(cfg),
                             state='pending',directory=None,attempts=[]))
    assert len({r['fingerprint'] for r in rows})==6
    return rows


def diagnostic_jobs(source_rows):
    selected=[r for r in source_rows if r['category']=='augmentation' and r['structure']['arm'] in ('radon','linear_resample')]
    assert len(selected)==24 and all(r['state']=='accepted' and not r['configuration'].get('task_fusion') for r in selected)
    jobs=[]
    for r in selected:
        directory=resolve(r['directory'])
        host=next(x for x in source_rows if x['id']==r['id'].rsplit('_augment_',1)[0])
        jobs.append(dict(id='switch_'+r['id'],component_ablation=True,source_row=r,
                         config=dict(host_best_epoch=host['best_epoch'],checkpoint=dict(path=str(directory/'selected.pt'),sha256=r['accepted_hashes']['selected.pt']),
                         selected_predictions=dict(path=str(directory/'selected_predictions.npz'),sha256=r['accepted_hashes']['selected_predictions.npz']))))
    return jobs


class Supplement(Queue):
    def __init__(self,root,parent,commit):
        self.root=root;self.parent=parent;self.commit=commit;self.segment=None;self.rows=[]
        self.cat=dict(version='functional_dependence_and_frozen_native_v1',result_positions=6,
                      diagnostic_checkpoints=24,diagnostic_states_per_checkpoint=4,test_used=False)
        source=read(parent/'manifest.json');assert len(source['rows'])==378 and all(r['state']=='accepted' for r in source['rows'])
        self.jobs=diagnostic_jobs(source['rows']);self.diag=read(root/'diagnostics.json') if (root/'diagnostics.json').exists() else {}
        self.p=read(parent/'protocol.json');self.p.update(source_commit=commit,accepted_source_hashes=source_hashes(),study=self.cat,
              protocol_document_sha256=sha256('experiments/geometry_mechanism/DEPENDENCY_SUPPLEMENT.zh-CN.md'),
              parent_manifest_sha256=sha256(parent/'manifest.json'),scope='separate_frozen_and_component_supplement')
        path=root/'protocol.json'
        if path.exists():assert read(path)==self.p,'Supplement protocol/source changed'
        else:write(path,self.p)
        self.parents,self.bases=dependencies();expected=frozen_rows(self.parents,self.bases)
        if (root/'manifest.json').exists():
            self.rows=read(root/'manifest.json')['rows']
            assert [(r['id'],r['fingerprint']) for r in self.rows]==[(r['id'],r['fingerprint']) for r in expected]
        else:self.rows=expected
        self.save()

    def status(self,state,**extra):
        super().status(state,diagnostic_checkpoints_completed=len(getattr(self,'diag',{})),**extra)

    def preflight(self):
        path=self.root/'preflight.json';done=read(path) if path.exists() else {}
        jobs=[dict(id='profile_'+r['id'],config=dict(r['configuration'],profile=True)) for r in self.rows if r['seed']==3416]
        # Verify all two host families × two added paths on training batch16.
        for host in ('mmtm_hidden256','attention_d256'):
            for arm in ('radon','linear_resample'):
                job=next(j for j in self.jobs if j['source_row']['structure']['host']==host and j['source_row']['structure']['arm']==arm)
                jobs.append(dict(id='profile_'+job['id'],component_ablation=True,config=dict(job['config'],preflight=True)))
        for i in range(0,len(jobs),2):
            pending=[j for j in jobs[i:i+2] if j['id'] not in done]
            if not pending:continue
            segment,results=self.execute(pending,'preflight','supplement_preflight')
            for job in pending:
                result=results[job['id']];assert result['passed'] and result['peak_reserved_mib']<=10240
                done[job['id']]=dict(summary=result,summary_sha256=sha256(segment/job['id']/'summary.json'))
            write(path,done);self.archive(segment)
        assert len(done)==6

    def run(self):
        self.preflight()
        self.train_rows(self.rows,'frozen_bridge_training',diagnostics=False)
        for row in self.rows:
            s=read(resolve(row['directory'])/'summary.json')
            assert s['native_parameters_and_buffers_unchanged'] and s['initial_native_sha256']==s['selected_native_sha256']
            assert s['optimizer_parameter_groups']==['bridge_0_exchange']
        for i in range(0,len(self.jobs),2):
            pending=[j for j in self.jobs[i:i+2] if j['id'] not in self.diag]
            if not pending:continue
            segment,results=self.execute(pending,'run','component_diagnostics')
            for job in pending:
                s=results[job['id']];assert s['passed'] and s['parameters_buffers_gradients_rng_preserved']
                self.diag[job['id']]=dict(directory=str(segment/job['id']),summary_sha256=sha256(segment/job['id']/'summary.json'),
                                         source_id=job['source_row']['id'],host=job['source_row']['structure']['host'],
                                         mode=job['source_row']['structure']['arm'],seed=job['source_row']['seed'],backbone_lr=job['source_row']['backbone_lr'],host_best_epoch=job['config']['host_best_epoch'])
            write(self.root/'diagnostics.json',self.diag);self.archive(segment)
        assert len(self.diag)==24
        self.status('reporting')
        from scripts.report_dependency_supplement import build
        build(self.root,self.parent)
        self.status('complete',report_complete=True)


def main(root,parent):
    root=Path(root);parent=Path(parent);root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(SOURCE/'.active.lock').open('a') as project:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        while True:
            if (root/'drain.request').exists():raise InterruptedError('User paused supplement')
            q=read(parent/'queue_status.json')
            if q['state'] in ('needs_attention','interrupted','withdrawn'):
                write(root/'queue_status.json',dict(state='blocked_parent',parent_state=q['state'],pid=os.getpid(),updated_at=time.time()))
                return
            if q['state']=='complete':
                try:fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            write(root/'queue_status.json',dict(state='waiting_parent',parent=str(parent),parent_state=q['state'],pid=os.getpid(),
                                               source_commit=commit,total_positions=6,completed=0,updated_at=time.time()))
            time.sleep(15)
        assert not live_workers(SOURCE)
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        q=Supplement(root,parent,commit)
        try:q.run()
        except BaseException as e:
            q.status('interrupted' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',default=str(SOURCE/'runs'/STUDY/'branch_only'))
    p.add_argument('--root',default=str(SOURCE/'runs'/STUDY/'dependency_supplement'));a=p.parse_args();main(a.root,a.parent)
