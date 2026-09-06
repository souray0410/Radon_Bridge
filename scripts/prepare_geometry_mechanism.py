"""Infrastructure preflight only. The revised performance matrix is not launched."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
from scripts.run_integer_experiment import Controller,source_hashes
from scripts.geometry_evidence import read,write,dependencies,storage_check
from radonbridge.artifacts import SOURCE,ARCHIVE,STUDY,sha256
from radonbridge.geometry_study import configuration,cost

def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'preparation.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        existing=root/'infrastructure_gpu_acceptance.json'
        if existing.exists():
            accepted=read(existing)
            core=lambda values:{k:v for k,v in values.items() if k.startswith('radonbridge/') or k=='scripts/run_integer_experiment.py'}
            if accepted['passed'] and core(accepted['source_hashes'])==core(source_hashes()):
                return  # Completed matching acceptance is immutable and never rerun.
            raise RuntimeError('Existing failed or changed-core acceptance requires explicit review')
        status=dict(state='preparing',source_commit=commit,new_performance_training_started=False,
                    revised_matrix_status='pending scope confirmation; no best-configuration selection',pid=os.getpid())
        write(root/'preparation_status.json',status)
        acceptance=read(ARCHIVE/'restore_acceptance.json')
        assert acceptance['passed'] and acceptance['archive_manifest_sha256']==sha256(ARCHIVE/'archive_manifest.json')
        parents,bases=dependencies()
        write(root/'dependencies.json',dict(parents=parents,bases=bases,archive_manifest_sha256=acceptance['archive_manifest_sha256'],test_used=False))
        for name in ('geometry_cpu.json','network_interface_cpu.json','regression_cpu.json'):
            assert read(Path('experiments/geometry_mechanism/acceptance')/name)['passed']
        jobs=[]
        for protocol in ('branch','fusion'):
            for M,S,k,h in ((16,32,1,512),(64,128,5,1024)):
                for mode in ('radon','linear_resample'):
                    r=h//M;structure=dict(M=M,S=S,k=k,r=r,h=h,rho=r/256,mode=mode)
                    cfg=configuration(structure,3416,6e-5,protocol,parents[3416],bases[3416])
                    cfg.update(profile=True,save_profile_checkpoint=True,measure_latency=True)
                    jobs.append(dict(id=f'{protocol}_{mode}_M{M}_S{S}_k{k}_h{h}',config=cfg))
        with (SOURCE/'.active.lock').open('a') as project:
            fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Surviving workers must be inspected, not duplicated after a crash.
            from scripts.run_task_fusion_benchmark import live_workers
            assert not live_workers(root),'Surviving preparation workers require inspection'
            segment=root/f'preflight_{1+len(list(root.glob("preflight_*"))):03d}';segment.mkdir()
            protocol=dict(schema='two_stage_ratio_convergence_v3',review_status='approved',
                          gpu_time_policy='unlimited_until_convergence',max_gpu_minutes=None,prior_gpu_minutes=0,
                          gpu_indices=[1,0],min_free_gpu_mib=12288,skip_legacy_summary=True,source_commit=commit,
                          scope='eight representative infrastructure profiles; not a performance matrix')
            write(segment/'protocol.json',protocol)
            storage_check(root,12*1024**3)
            c=Controller(SimpleNamespace(output=str(segment),protocol=str(segment/'protocol.json'),phase='preflight',data=str(SOURCE/'cache/full1264_296')))
            c.commit=commit
            try:
                profiles=c.run_jobs(jobs,allow_oom=True)
                feasible={k:v for k,v in profiles.items() if v.get('passed')}
                diagnostics=[]
                for identifier in feasible:
                    path=segment/identifier
                    diagnostics.append(dict(id='diagnostic_'+identifier,diagnostic=True,config=dict(
                        trial_directory=str(path),basis_files=bases[3416],selected_file='profile_selected.pt',
                        selected_sha256=sha256(path/'profile_selected.pt'),preflight=True,
                        probe_count=128,energy_limit=1264)))
                checked=c.run_jobs(diagnostics,allow_oom=True)
                good=len(feasible)==len(jobs) and all(v.get('passed') for v in checked.values())
                write(root/'infrastructure_gpu_acceptance.json',dict(passed=good,source_commit=commit,source_hashes=source_hashes(),
                      profiles=profiles,diagnostics=checked,ledger=c.ledger,scope=protocol['scope'],test_used=False))
                status.update(state='ready_for_revised_protocol' if good else 'needs_attention',
                              infrastructure_gpu_passed=good,profile_count=len(jobs),profile_passed=len(feasible),
                              gpu_minutes=c.used(),new_performance_training_started=False)
                c.status('complete' if good else 'needs_attention')
            finally:c.shutdown()
        write(root/'preparation_status.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(SOURCE/'runs'/STUDY));a=p.parse_args()
    try:main(a.root)
    except Exception as e:
        root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
        write(root/'preparation_status.json',dict(state='needs_attention',error=repr(e),traceback=traceback.format_exc(),new_performance_training_started=False))
        raise
