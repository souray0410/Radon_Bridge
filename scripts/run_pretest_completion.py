"""The authorized 72-training completion; never opens test data."""
import argparse,copy,fcntl,os,signal,subprocess,time,traceback
from pathlib import Path
from radonbridge.artifacts import SOURCE,STUDY,ARCHIVE,resolve,relocate,sha256
from radonbridge.pretest_study import matrices,contrast_definitions
from radonbridge.geometry_study import fingerprint
from scripts.geometry_evidence import read,write,dependencies
from scripts.run_geometry_mechanism import Queue,scan_completed,interrupted
from scripts.run_integer_experiment import source_hashes
from scripts.run_task_fusion_benchmark import live_workers


class Completion(Queue):
    def __init__(self,root,commit):
        self.root=root;self.commit=commit;self.segment=None;self.rows=[]
        self.cat=dict(version='pretest_completion72_v1',result_positions=72,geometry_new=24,frozen_new=48,test_used=False)
        study=SOURCE/'runs'/STUDY
        self.parents,self.bases=dependencies()
        historical=read(ARCHIVE/'history/runs/2026_09_05_22_42_53/manifest.json')['rows']
        qr={}
        for seed in self.parents:
            r=next(x for x in historical if x['seed']==seed and x['arm']=='qr_radon' and x['rho']==1/16)
            assert {k:v['sha256'] for k,v in r['configuration']['parent_checkpoints'].items()}=={k:v['sha256'] for k,v in self.parents[seed].items()}
            qr[seed]=relocate(r['configuration']['bridges'][0]['basis_files'])
            for ref in qr[seed].values():assert sha256(ref['path'])==ref['sha256']
        geometry,frozen=matrices(self.parents,self.bases,qr)
        prior=read(study/'dependency_supplement/manifest.json')['rows']
        references=[];new=[]
        for r in frozen:
            found=[x for x in prior if x['fingerprint']==r['fingerprint']]
            if found:
                assert len(found)==1
                copyrow=copy.deepcopy(found[0]);scan_completed(copyrow)
                copyrow.update(id=r['id'],structure=r['structure'],category=r['category'],reused=True)
                references.append(copyrow)
            else:new.append(r)
        assert len(references)==6 and len(new)==48
        expected=geometry+new
        self.references=references;self.historical=historical
        self.direct=read(study/'branch_only/manifest.json')['rows']
        for r in self.direct:assert r['state']=='accepted'
        contrasts=contrast_definitions([r for r in self.direct if r['category']=='direct']+geometry,frozen,historical)
        self.p=read(study/'dependency_supplement/protocol.json')
        self.p.update(source_commit=commit,accepted_source_hashes=source_hashes(),study=self.cat,scope='authorized_pretest_completion72',
                      protocol_document_sha256=sha256('experiments/geometry_mechanism/UNIFIED_TEST.zh-CN.md'),
                      parent_manifests={n:sha256(study/n/'manifest.json') for n in ('branch_only','dependency_supplement')})
        for name,value in [('protocol.json',self.p),('references.json',references),('new_contrasts.json',contrasts)]:
            path=root/name
            if path.exists():assert read(path)==value,'Immutable completion protocol/reference changed: '+name
            else:write(path,value)
        path=root/'manifest.json';self.rows=read(path)['rows'] if path.exists() else expected
        assert [(r['id'],r['fingerprint']) for r in self.rows]==[(r['id'],r['fingerprint']) for r in expected]
        self.save()

    def preflight(self):
        path=self.root/'preflight.json';done=read(path) if path.exists() else {}
        reps={r['structure']['id']:r for r in self.references+self.rows}
        assert len(reps)==22
        pending=[dict(id='profile_'+key,config=dict(r['configuration'],profile=True)) for key,r in sorted(reps.items()) if key not in done]
        for i in range(0,len(pending),2):
            jobs=pending[i:i+2];seg,res=self.execute(jobs,'preflight','completion_preflight')
            for job in jobs:
                s=res[job['id']];assert s['passed'] and s['peak_reserved_mib']<=10240
                done[job['id'].removeprefix('profile_')]=dict(summary=s,summary_sha256=sha256(seg/job['id']/'summary.json'))
            write(path,done);self.archive(seg)
        assert len(done)==22
        lock=dict(new_training_jobs=72,frozen_reused=6,preflight_sha256=sha256(path),
                  protocol_sha256=sha256(self.root/'protocol.json'),contrasts_sha256=sha256(self.root/'new_contrasts.json'),test_used=False)
        f=self.root/'candidate_lock.json'
        if f.exists():assert read(f)==lock
        else:write(f,lock)

    def run(self):
        self.preflight()
        self.train_rows([r for r in self.rows if r['category']=='geometry_completion'],'geometry_completion',diagnostics=True)
        self.train_rows([r for r in self.rows if r['category']=='frozen_completion'],'frozen_completion',diagnostics=False)
        for r in self.rows:
            assert r['state']=='accepted'
            if r['category']=='frozen_completion':
                s=read(resolve(r['directory'])/'summary.json')
                assert s['native_parameters_and_buffers_unchanged'] and s['initial_native_sha256']==s['selected_native_sha256']
                assert s['optimizer_parameter_groups']==['bridge_0_exchange']
        self.status('training_complete_test_sealed',test_used=False,awaiting='raw data, historical access audit, development replay and unified evaluation lock')
        write(self.root/'training_summary.json',dict(state='complete',accepted=72,reused_frozen=6,test_used=False))


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(SOURCE/'.active.lock').open('a') as project:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'drain.request').exists() and not live_workers(SOURCE)
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        q=Completion(root,commit)
        try:q.run()
        except BaseException as e:
            q.status('interrupted' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(SOURCE/'runs'/STUDY/'pretest_completion'));main(p.parse_args().root)
