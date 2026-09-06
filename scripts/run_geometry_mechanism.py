"""Immutable, restartable dual-GPU mechanism queue; no winner selection."""
import argparse,atexit,copy,fcntl,json,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
from radonbridge.artifacts import SOURCE,ARCHIVE,STUDY,resolve,sha256
from radonbridge.geometry_study import (mechanism_catalog,direct_rows,augmentation_configuration,
    fingerprint,stable_hash,FIXED_HOSTS)
from scripts.geometry_evidence import read,write,dependencies,historical_index,accept,storage_check,archive_completed
from scripts.run_integer_experiment import Controller,source_hashes
from scripts.run_task_fusion_benchmark import live_workers

DATA=str(SOURCE/'cache/full1264_296')

def profile_key(row):
    return row['protocol']+'_'+row['structure']['id']

def scan_completed(row):
    """Accept any completed attempt in creation order, never compare their scores."""
    if row.get('accepted_hashes'):
        row.update(accept(resolve(row['directory']),row['configuration'],row['protocol'],row['accepted_hashes']))
        row['state']='accepted';return True
    for original in row.get('attempts',[]):
        path=resolve(original)
        if (path/'summary.json').exists():
            row.update(accept(path,row['configuration'],row['protocol']));row['state']='accepted';return True
        if (path/'failure.json').exists():
            error=read(path/'failure.json')
            # Explicit worker failures are not automatically retried as interruptions.
            if error.get('state') not in ('interrupted',):
                raise RuntimeError(f'needs_attention: {path}: {error.get("state")}')
    return False

class Queue:
    def __init__(self,root,commit,resume_from=None):
        self.root=root;self.commit=commit;self.cat=mechanism_catalog();self.rows=[];self.segment=None
        self.p=dict(schema='two_stage_ratio_convergence_v3',review_status='approved',source_commit=commit,
                    accepted_source_hashes=source_hashes(),gpu_time_policy='unlimited_until_convergence',
                    max_gpu_minutes=None,prior_gpu_minutes=0,gpu_indices=[1,0],min_free_gpu_mib=12288,
                    skip_legacy_summary=True,project_memory_limit_mib=10240,study=self.cat,
                    protocol_document_sha256=sha256('experiments/geometry_mechanism/EXECUTION.zh-CN.md'),test_used=False)
        self.p=json.loads(json.dumps(self.p))  # Compare the persisted JSON representation on every restart.
        p=root/'protocol.json'
        if p.exists():
            old=read(p)
            if old!=self.p:
                assert resume_from==old['source_commit'],'Locked source/protocol differs; explicit versioned repair required'
                assert not (root/'candidate_lock.json').exists(),'Repair migration is limited to unfinished resource preflight'
                omit=('source_commit','accepted_source_hashes')
                assert {k:v for k,v in old.items() if k not in omit}=={k:v for k,v in self.p.items() if k not in omit}
                changed={k for k,v in old['accepted_source_hashes'].items() if self.p['accepted_source_hashes'].get(k)!=v}
                assert changed<= {'scripts/run_geometry_mechanism.py','tests/check_geometry_queue.py'},changed
                revision=root/'protocol_revisions'/('preflight_'+resume_from+'.json')
                write(revision,old)
                write(root/'preflight_repair.json',dict(previous_commit=resume_from,current_commit=commit,
                    previous_protocol_sha256=sha256(p),changed_files=sorted(changed),reason='Unregister retired segment exit callbacks; preserve interruption between batches; supply existing source bases to the no-communication diagnostic',training_algorithm_changed=False))
                write(p,self.p)
        else:write(p,self.p)
        self.status('initializing')

    def save(self):
        orders={r['participant_order_sha256'] for r in self.rows if r.get('participant_order_sha256')}
        assert len(orders)<=1,'Accepted participant identity/order mismatch'
        write(self.root/'manifest.json',dict(rows=self.rows,catalog=self.cat,source_commit=self.commit,
              total_positions=self.cat['result_positions'],test_used=False))

    def status(self,state,**extra):
        write(self.root/'queue_status.json',dict(state=state,pid=os.getpid(),source_commit=self.commit,
            updated_at=time.time(),total_positions=self.cat['result_positions'],
            completed=sum(r['state']=='accepted' for r in self.rows),
            reused=sum(bool(r.get('reused')) for r in self.rows),
            infeasible=sum(r['state']=='infeasible' for r in self.rows),
            active_segment=str(self.segment) if self.segment else None,**extra))

    def execute(self,jobs,phase,stage,allow_oom=False):
        # Small batches bound simultaneous checkpoint/optimizer/diagnostic storage.
        storage_check(self.root,32*1024**3)
        segments=self.root/'batches';segments.mkdir(exist_ok=True)
        segment=segments/f'{stage}_{time.time_ns()}';segment.mkdir();self.segment=segment
        write(segment/'protocol.json',self.p)
        write(segment/'jobs.json',jobs)
        self.status(stage,queued_ids=[j['id'] for j in jobs])
        c=Controller(SimpleNamespace(output=str(segment),protocol=str(segment/'protocol.json'),phase=phase,data=DATA));c.commit=self.commit
        try:
            result=c.run_jobs(jobs,allow_oom=allow_oom);c.status('complete')
        finally:
            c.shutdown();atexit.unregister(c.shutdown)
            signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        return segment,result

    def archive(self,segment):
        target,marker=archive_completed(segment)
        # Entries retain original immutable config paths, resolved via artifacts.resolve.
        for r in self.rows:
            if r.get('directory') and Path(r['directory']).is_relative_to(segment):
                r['directory']=str(target/Path(r['directory']).relative_to(segment))
        self.save()
        shutil.rmtree(segment)  # Only after full copy/hash verification and index persistence.
        return target

    def initialize(self):
        infrastructure=read(self.root/'infrastructure_gpu_acceptance.json')
        assert infrastructure['passed']
        core=lambda values:{k:v for k,v in values.items() if (k.startswith('radonbridge/') and k!='radonbridge/geometry_study.py') or k=='scripts/run_integer_experiment.py'}
        assert core(infrastructure['source_hashes'])==core(source_hashes()),'Training core changed since CPU/GPU acceptance'
        for name in ('geometry_cpu.json','network_interface_cpu.json','regression_cpu.json'):
            assert read(Path('experiments/geometry_mechanism/acceptance')/name)['passed']
        parents,bases=dependencies();self.parents=parents;self.bases=bases
        expected=direct_rows(parents,bases)
        manifest=self.root/'manifest.json'
        if manifest.exists():
            self.rows=read(manifest)['rows']
            actual=[r for r in self.rows if r['category']=='direct']
            assert [(r['id'],r['fingerprint']) for r in actual]==[(r['id'],r['fingerprint']) for r in expected]
        else:
            self.rows=expected
            found,excluded=historical_index({r['fingerprint']:r for r in expected})
            for r in self.rows:
                if r['fingerprint'] in found:r.update(found[r['fingerprint']],reused=True,state='accepted')
            write(self.root/'historical_acceptance.json',dict(passed=True,accepted=found,excluded=excluded,
                  archive_manifest_sha256=sha256(ARCHIVE/'archive_manifest.json'),test_used=False))
        for r in self.rows:
            if r.get('configuration'):scan_completed(r)
        self.save()

    def preflight(self):
        store=self.root/'structure_preflight.json';checked=read(store) if store.exists() else {}
        # Every structure/protocol; seed and LR do not change allocated tensor shapes.
        reps={profile_key(r):r for r in reversed(self.rows) if r['category']=='direct'}
        for key,row in reps.items():
            if key in checked:continue
            cfg=dict(row['configuration'],profile=True,measure_latency=True,
                     save_profile_checkpoint='M' not in row['structure'])
            # Pair consecutive jobs for independent devices, flush in batches of two.
            checked.setdefault('_pending',[]).append(dict(id=key,config=cfg))
        pending=checked.pop('_pending',[])
        for pos in range(0,len(pending),2):
            jobs=pending[pos:pos+2];segment,results=self.execute(jobs,'preflight','resource_preflight',True)
            for job in jobs:
                key=job['id'];s=results[key]
                if s.get('passed'):
                    assert s['peak_reserved_mib']<=10240
                    # New nonlinear sizes also undergo the complete read-only diagnostic.
                    if job['config']['save_profile_checkpoint']:
                        d=dict(trial_directory=str(segment/key),basis_files=self.bases[job['config']['seed']],selected_file='profile_selected.pt',
                               selected_sha256=sha256(segment/key/'profile_selected.pt'),preflight=True,probe_count=128,energy_limit=1264)
                        ds,dr=self.execute([dict(id='diagnostic',diagnostic=True,config=d)],'preflight','diagnostic_preflight',True)
                        if not dr['diagnostic'].get('passed'):s=dict(state='oom',diagnostic=dr['diagnostic'])
                        else:s=dict(s,diagnostic_passed=True)
                        self.archive(ds)
                elif s.get('state')!='oom':raise RuntimeError('Resource acceptance failed: '+key)
                checked[key]=dict(summary=s,feasible=bool(s.get('passed')),directory=str(segment/key))
            write(store,checked);self.archive(segment)
        assert set(checked)==set(reps)
        for row in self.rows:
            if row['category']=='direct' and not checked[profile_key(row)]['feasible']:
                if row.get('reused'):raise RuntimeError('Historical acceptance conflicts with current resource preflight')
                row['state']='infeasible';row['infeasible_reason']='batch16 resource preflight'
        self.save()
        lock=dict(locked_before_new_performance=True,source_commit=self.commit,
            protocol_sha256=sha256(self.root/'protocol.json'),preflight_sha256=sha256(store),
            direct_positions=len(reps)*6,reused=sum(bool(r.get('reused')) for r in self.rows),
            infeasible=sum(r['state']=='infeasible' for r in self.rows),
            new_direct_training=sum(r['category']=='direct' and not r.get('reused') and r['state']!='infeasible' for r in self.rows),
            prespecified_augmentation=72,host_feasibility_checked_after_fitting=True,test_used=False)
        path=self.root/'candidate_lock.json'
        if path.exists():assert read(path)==lock,'Candidate lock must be immutable'
        else:write(path,lock)

    def train_rows(self,rows,stage):
        for r in rows:
            if r['state']!='infeasible':scan_completed(r)
        self.save()
        pending=[r for r in rows if r['state']=='pending']
        for pos in range(0,len(pending),2):
            batch=pending[pos:pos+2]
            # Persist attempt paths BEFORE dispatch. execute reserves this unique batch here.
            storage_check(self.root,32*1024**3)
            segment=self.root/'batches'/f'{stage}_{time.time_ns()}';segment.mkdir(parents=True);self.segment=segment
            write(segment/'protocol.json',self.p)
            for r in batch:
                r['directory']=str(segment/r['id']);r['attempts'].append(r['directory'])
            self.save();self.status(stage,active_ids=[r['id'] for r in batch],new_performance_training_started=True)
            c=Controller(SimpleNamespace(output=str(segment),protocol=str(segment/'protocol.json'),phase='run',data=DATA))
            try:
                c.run_jobs([dict(id=r['id'],config=r['configuration']) for r in batch]);c.status('complete')
            finally:
                c.shutdown();atexit.unregister(c.shutdown)
                signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
            for r in batch:assert scan_completed(r)
            self.save();self.archive(segment)
            self.status(stage,new_performance_training_started=True)
        # Complete read-only initial/selected diagnostics, including resumed accepted trials.
        for r in rows:
            if r['state']!='accepted' or r.get('reused') or r.get('diagnostic'):continue
            directory=resolve(r['directory'])
            cfg=dict(trial_directory=str(directory),selected_sha256=r['accepted_hashes']['selected.pt'],
                     basis_files=self.bases[r['seed']],probe_count=128,energy_limit=1264)
            if r['category']=='augmentation' and len(r['configuration']['bridges'])>1:
                cfg['basis_files']=r['configuration']['bridges'][1]['basis_files']
            segment,result=self.execute([dict(id='diagnostic',diagnostic=True,config=cfg)],'run','diagnostics')
            assert result['diagnostic']['passed']
            r['diagnostic']=dict(directory=str(segment/'diagnostic'),sha256=sha256(segment/'diagnostic/summary.json'))
            self.save();self.archive(segment)

    def augment(self):
        hosts=[r for r in self.rows if r['category']=='direct' and r['structure']['id'] in FIXED_HOSTS]
        assert len(hosts)==24
        for host in hosts:
            if host['state']!='accepted':raise RuntimeError('Fixed host unavailable; no replacement selected: '+host['id'])
            ref=dict(path=str(resolve(host['directory'])/'selected.pt'),sha256=host['accepted_hashes']['selected.pt'])
            basis=host.get('augmentation_basis')
            if basis is None:
                segment,result=self.execute([dict(id='basis',host_basis=True,config=dict(host_checkpoint=ref))],'run','host_basis')
                assert result['basis']['passed'];basis=result['basis']['basis_files']
                host['augmentation_basis']=basis;self.save();self.archive(segment)
            for item in basis.values():assert sha256(resolve(item['path']))==item['sha256']
            group=[]
            for arm in ('continue','radon','linear_resample'):
                cfg=augmentation_configuration(host,ref,basis,arm);identifier=host['id']+'_augment_'+arm
                r=next((x for x in self.rows if x['id']==identifier),None)
                if r is None:
                    r=dict(id=identifier,configuration=cfg,fingerprint=fingerprint(cfg),category='augmentation',
                           protocol=host['protocol'],seed=host['seed'],backbone_lr=host['backbone_lr'],
                           structure=dict(id=host['structure']['id']+'_plus_'+arm,host=host['structure']['id'],arm=arm),
                           state='pending',directory=None,attempts=[]);self.rows.append(r)
                else:assert r['fingerprint']==fingerprint(cfg)
                group.append(r)
            self.save()
            for r in group:
                if r.get('preflight'):continue
                cfg=dict(r['configuration'],profile=True,save_profile_checkpoint=True,measure_latency=True)
                segment,result=self.execute([dict(id='profile',config=cfg)],'preflight','augmentation_preflight')
                assert result['profile']['passed']
                diagnostic=dict(trial_directory=str(segment/'profile'),basis_files=basis,
                    selected_file='profile_selected.pt',selected_sha256=sha256(segment/'profile/profile_selected.pt'),
                    preflight=True,probe_count=128,energy_limit=1264)
                ds,dr=self.execute([dict(id='diagnostic',diagnostic=True,config=diagnostic)],'preflight','augmentation_diagnostic')
                assert dr['diagnostic']['passed']
                r['preflight']=dict(profile=result['profile'],diagnostic_sha256=sha256(ds/'diagnostic/summary.json'))
                self.save();self.archive(ds);self.archive(segment)
            self.train_rows(group,'host_augmentation')

    def run(self):
        self.initialize();self.preflight()
        for protocol in ('branch','fusion'):
            direct=[r for r in self.rows if r['category']=='direct' and r['protocol']==protocol]
            self.train_rows(direct,protocol+'_training')
        self.augment()
        assert len(self.rows)==self.cat['result_positions']
        self.status('reporting')
        from scripts.report_geometry_mechanism import build
        build(self.root)
        self.status('complete',report_complete=True)


def interrupted(*_):
    raise InterruptedError("controller_signal")

def main(root,resume_from=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as own,(SOURCE/'.active.lock').open('a') as project:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not live_workers(SOURCE),'Existing project workers must finish or be reconciled'
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        q=Queue(root,commit,resume_from)
        try:q.run()
        except BaseException as e:
            q.status('interrupted' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
                     error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(SOURCE/'runs'/STUDY));p.add_argument('--resume-from-commit');a=p.parse_args();main(a.root,a.resume_from_commit)
