"""Append the bounded depth study after completion72 releases the project lock."""
import argparse,copy,fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from radonbridge.artifacts import SOURCE,STUDY,resolve,relocate,sha256
from radonbridge.depth_study import matrices,definitions,VERSION
from scripts.geometry_evidence import read,write,dependencies
from scripts.run_geometry_mechanism import Queue,scan_completed,interrupted
from scripts.run_integer_experiment import source_hashes
from scripts.run_task_fusion_benchmark import live_workers

class Depth(Queue):
    def __init__(self,root,commit,predecessor):
        self.root=root;self.commit=commit;self.segment=None;self.rows=[];self.predecessor=predecessor
        self.cat=dict(version=VERSION,result_positions=96,reference_views=18,
                      planned_new_training=96,test_used=False)
        self.p=read(predecessor/'protocol.json')
        self.p.update(source_commit=commit,accepted_source_hashes=source_hashes(),study=self.cat,
            scope='authorized_multidepth96',predecessor_manifest_sha256=sha256(predecessor/'manifest.json'),
            protocol_document_sha256=sha256('experiments/geometry_mechanism/MULTIDEPTH.zh-CN.md'))
        self.locked_json('protocol.json',self.p)
        self.parents,self.bases=dependencies()
        p=root/'manifest.json'
        if p.exists():self.rows=read(p)['rows']

    def locked_json(self,name,value):
        p=self.root/name
        if p.exists():assert read(p)==value,'Immutable depth evidence changed: '+name
        else:write(p,value)

    def prepare_bases(self):
        p=self.root/'new_bases.json';done=read(p) if p.exists() else {}
        jobs=[]
        for seed in self.parents:
            if str(seed) in done:continue
            cfg=dict(seed=seed,parent_checkpoints=self.parents[seed],nodes=[f'{b}_stage{s}' for s in (2,4) for b in ('cfp','oct')],
                microbatch=16,source_commit=self.commit,fit_domain='unbridged independently pretrained native stage2/stage4 TRAIN features; all eyes and spatial positions equally weighted')
            jobs.append(dict(id=f'basis_seed{seed}',basis_fit=True,config=cfg))
        for i in range(0,len(jobs),2):
            seg,res=self.execute(jobs[i:i+2],'run','depth_basis_fit')
            for job in jobs[i:i+2]:
                d=res[job['id']];assert d['passed'] and d['test_used'] is False
                done[str(job['config']['seed'])]=dict(bases=d['bases'],summary_sha256=sha256(seg/job['id']/'summary.json'))
            write(p,done);self.archive(seg)
        for seed in self.bases:
            self.bases[seed].update(relocate(done[str(seed)]['bases']))
            for ref in self.bases[seed].values():assert sha256(ref['path'])==ref['sha256']

    def initialize_rows(self):
        expected=matrices(self.parents,self.bases)
        old=read(SOURCE/'runs'/STUDY/'branch_only/manifest.json')['rows']
        references=[];pending=[]
        for r in expected:
            found=[x for x in old if x['fingerprint']==r['fingerprint']]
            if found:
                assert len(found)==1;v=copy.deepcopy(found[0]);scan_completed(v)
                v.update(id=r['id'],structure=r['structure'],category='depth',reused=True)
                references.append(v)
            else:pending.append(r)
        assert len(references)==12 and len(pending)==96
        baseline=[copy.deepcopy(x) for x in old if x['structure']['id']=='no_communication']
        assert len(baseline)==6
        for r in baseline:scan_completed(r)
        self.references=references
        self.locked_json('references.json',dict(single_stage3=references,no_communication=baseline))
        self.locked_json('comparisons.json',definitions(expected,baseline))
        if self.rows:assert [(r['id'],r['fingerprint']) for r in self.rows]==[(r['id'],r['fingerprint']) for r in pending]
        else:self.rows=pending
        self.save()

    def preflight(self):
        p=self.root/'preflight.json';done=read(p) if p.exists() else {}
        reps={r['structure']['id']:r for r in self.references+self.rows}
        assert len(reps)==18
        # Every structure, including all subset/width controls, must be checked.
        for key,r in sorted(reps.items()):
            if key in done:continue
            job=dict(id='profile',config=dict(r['configuration'],profile=True,save_profile_checkpoint=True,measure_latency=True))
            seg,res=self.execute([job],'preflight','depth_preflight',True);s=res['profile']
            entry=dict(training=s,feasible=bool(s.get('passed')) and s['peak_reserved_mib']<=10240,source_commit=self.commit)
            if entry['feasible']:
                cfg=dict(checkpoint=dict(path=str(seg/'profile/profile_selected.pt'),sha256=sha256(seg/'profile/profile_selected.pt')),preflight=True)
                ds,dr=self.execute([dict(id='diagnostic',depth_diagnostic=True,config=cfg)],'preflight','depth_diagnostic_preflight',True)
                d=dr['diagnostic'];entry.update(diagnostic=d,feasible=bool(d.get('passed')) and d['peak_reserved_mib']<=10240)
                self.archive(ds)
            elif s.get('state')!='oom':raise RuntimeError('Depth preflight failed: '+key)
            done[key]=entry;write(p,done);self.archive(seg)
        for r in self.rows:
            if not done[r['structure']['id']]['feasible']:
                r.update(state='infeasible',infeasible_reason='batch16 resource or diagnostic preflight; no automatic configuration change')
        self.save()
        self.locked_json('candidate_lock.json',dict(protocol_sha256=sha256(self.root/'protocol.json'),
            comparison_sha256=sha256(self.root/'comparisons.json'),preflight_sha256=sha256(p),
            new_training_planned=96,new_training_feasible=sum(r['state']!='infeasible' for r in self.rows),
            reused_single_stage3=12,reused_no_communication=6,test_used=False))

    def run(self):
        self.prepare_bases();self.initialize_rows();self.preflight()
        self.train_rows(self.rows,'multidepth_training',diagnostics=False)
        for r in self.rows:
            if r['state']!='accepted' or r.get('depth_diagnostic'):continue
            directory=resolve(r['directory'])
            cfg=dict(checkpoint=dict(path=str(directory/'selected.pt'),sha256=r['accepted_hashes']['selected.pt']),
                selected_predictions=dict(path=str(directory/'selected_predictions.npz'),sha256=r['accepted_hashes']['selected_predictions.npz']))
            seg,res=self.execute([dict(id='diagnostic',depth_diagnostic=True,config=cfg)],'run','depth_diagnostics')
            assert res['diagnostic']['passed']
            r['depth_diagnostic']=dict(directory=str(seg/'diagnostic'),summary_sha256=sha256(seg/'diagnostic/summary.json'))
            self.save();self.archive(seg)
        from scripts.report_depth_study import build
        build(self.root)
        complete=all(r['state']=='accepted' and r.get('depth_diagnostic') for r in self.rows)
        self.status('training_complete_test_sealed' if complete else 'needs_attention',test_used=False,
                    awaiting='unified model registry, access audit, development replay and full 3D test data acceptance')
        write(self.root/'training_summary.json',dict(state='complete' if complete else 'needs_attention',
            accepted=sum(r['state']=='accepted' for r in self.rows),infeasible=sum(r['state']=='infeasible' for r in self.rows),test_used=False))

def main(root,predecessor):
    root=Path(root);predecessor=Path(predecessor);root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            while True:
                if (root/'drain.request').exists():raise InterruptedError('Depth queue drain requested')
                status=read(predecessor/'queue_status.json')
                if status['state']=='training_complete_test_sealed':break
                write(root/'queue_status.json',dict(state='waiting_predecessor',pid=os.getpid(),source_commit=commit,
                    updated_at=time.time(),planned_new_training=96,predecessor=str(predecessor),predecessor_state=status['state'],
                    predecessor_completed=status.get('completed'),gpu_preflight='pending after project lock release',test_used=False))
                time.sleep(30)
            with (SOURCE/'.active.lock').open('a') as project:
                while True:
                    try:fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(15)
                assert not live_workers(SOURCE)
                rows=read(predecessor/'manifest.json')['rows'];assert len(rows)==72
                for r in rows:assert r['state']=='accepted' and scan_completed(r)
                assert read(predecessor/'training_summary.json')['state']=='complete'
                q=Depth(root,commit,predecessor);q.run()
        except BaseException as e:
            write(root/'queue_status.json',dict(state='interrupted' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
                pid=os.getpid(),source_commit=commit,updated_at=time.time(),error=repr(e),traceback=traceback.format_exc(),test_used=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(SOURCE/'runs'/STUDY/'multidepth96'))
    p.add_argument('--predecessor',default=str(SOURCE/'runs'/STUDY/'pretest_completion_v2'));a=p.parse_args();main(a.root,a.predecessor)
