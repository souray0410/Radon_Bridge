"""Persistent, project-locked development replay queue; never launches test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from radonbridge.artifacts import SOURCE, ARCHIVE, resolve, sha256
from scripts.geometry_evidence import read, write
from scripts.gpu_allocation import Allocation, DEFAULT_PATH, memory_snapshot
from scripts.run_integer_experiment import devices


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root/'queue.lock').open('a') as own, (SOURCE/'.active.lock').open('a') as project:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(project, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Replay must use a committed immutable source tree')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        inv = read(args.inventory/'status.json')
        if inv['issues'] or inv['state'] != 'candidate_artifacts_verified_test_still_sealed':
            raise ValueError('Candidate inventory is not accepted')
        model_file = args.inventory/'candidate_models.json'
        if sha256(model_file) != inv['models_sha256']:
            raise ValueError('Candidate inventory SHA mismatch')
        rows = read(model_file)
        if len(rows) != inv['unique_model_views'] or len({r['model_view_id'] for r in rows}) != len(rows):
            raise ValueError('Candidate inventory count/identity mismatch')
        registration = dict(source_commit=commit, candidate_sha256=sha256(model_file),
            data_selected_sha256=sha256(resolve(args.data)/'selected.csv'), total=len(rows),
            split='validation', batch=16, probability_atol=1e-6, probability_rtol=1e-5,
            class_decisions_exact=True, min_free_gpu_mib=10240, max_project_gpu_mib=10240,
            test_inference_permitted=False)
        if (root/'registration.json').exists():
            if read(root/'registration.json') != registration:
                raise ValueError('Replay registration changed; use a new output directory')
        else:
            write(root/'registration.json', registration)
        allocation = Allocation(args.allocation, [], required=True)
        active = {}
        accepted = {}
        failed = []
        pending = []
        stop = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        ledger = read(root/'ledger.json') if (root/'ledger.json').exists() else []
        for r in rows:
            folder = root/'models'/r['model_view_id']
            done = folder/'acceptance.json'
            if done.exists():
                entry = read(done)
                summary = read(entry['summary_path'])
                if sha256(entry['summary_path']) != entry['summary_sha256'] or summary['state'] != 'accepted':
                    raise ValueError('Replay acceptance changed')
                if summary['model_view_id'] != r['model_view_id'] or summary['source_commit'] != commit:
                    raise ValueError('Replay identity/source mismatch')
                if sha256(Path(entry['summary_path']).parent/'predictions.npz') != summary['replay_prediction_sha256']:
                    raise ValueError('Accepted replay predictions changed')
                accepted[r['model_view_id']] = entry
            elif (list(folder.glob('attempt_*/failure.json')) or list(folder.glob('attempt_*/controller_failure.json'))) and not args.retry_failed:
                failed.append(r['model_view_id'])
            else:
                pending.append(r)
        start = time.monotonic()
        last_status = 0
        value = {}

        def status(state):
            write(root/'status.json', dict(state=state, controller_pid=os.getpid(), source_commit=commit,
                total=len(rows), accepted=len(accepted), failed=len(failed), pending=len(pending),
                active=[dict(model_view_id=k, pid=v['process'].pid, gpu=v['gpu']) for k, v in active.items()],
                failed_model_view_ids=failed, allocation=value, test_used=False, test_inference_permitted=False,
                elapsed_seconds=time.monotonic()-start, updated_at=time.time()))

        try:
            while pending or active:
                info = devices()
                value, _ = allocation.refresh(info)
                raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                               '--format=csv,noheader,nounits'], text=True)
                _, project_mem, project_pids = memory_snapshot(raw, active, info)
                for key, worker in list(active.items()):
                    proc = worker['process']
                    if project_mem.get(worker['gpu'], 0) > 10240:
                        worker['resource_error'] = 'Project GPU memory exceeded 10 GiB'
                        if proc.poll() is None and worker.get('stop_at') is None:
                            proc.terminate(); worker['stop_at'] = time.monotonic()
                    if stop[0] and proc.poll() is None and worker.get('stop_at') is None:
                        proc.terminate(); worker['stop_at'] = time.monotonic()
                    if proc.poll() is None and worker.get('stop_at') and time.monotonic()-worker['stop_at'] > 60:
                        os.killpg(proc.pid, signal.SIGKILL)
                    if proc.poll() is None:
                        continue
                    worker['log'].close()
                    path = worker['directory']/'summary.json'
                    good = proc.returncode == 0 and path.exists() and not worker.get('resource_error')
                    if good:
                        s = read(path)
                        good = (s['state'] == 'accepted' and s['model_view_id'] == key and
                                s['source_commit'] == commit and s['peak_reserved_mib'] <= 10240 and
                                s['test_used'] is False and s['strict_model_load'] and s['development_forward_replay'])
                    if good:
                        entry = dict(summary_path=str(path), summary_sha256=sha256(path))
                        write(worker['directory'].parent/'acceptance.json', entry)
                        accepted[key] = entry
                    else:
                        failed.append(key)
                        write(worker['directory']/'controller_failure.json', dict(exit_code=proc.returncode,
                              error=worker.get('resource_error', 'Worker did not pass'), test_used=False))
                    ledger.append(dict(model_view_id=key, gpu=worker['gpu'], seconds=time.monotonic()-worker['start'],
                                       exit_code=proc.returncode, accepted=good, directory=str(worker['directory'])))
                    write(root/'ledger.json', ledger)
                    del active[key]
                if not failed and not stop[0]:
                    occupied = {w['gpu'] for w in active.values()}
                    for gpu in value['selected_gpu_indices']:
                        if not pending or gpu in occupied or project_pids.get(gpu) or info[gpu]['free'] < 10240:
                            continue
                        if min(shutil.disk_usage(SOURCE).free, shutil.disk_usage(ARCHIVE).free) < 100*1024**3:
                            value['storage_error'] = 'Less than 100 GiB safety margin; dispatch paused'
                            break
                        r = pending.pop(0)
                        parent = root/'models'/r['model_view_id']
                        parent.mkdir(parents=True, exist_ok=True)
                        directory = parent/f'attempt_{time.time_ns()}'
                        directory.mkdir()
                        write(directory/'record.json', r)
                        log = (directory/'worker.log').open('w')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RB_REPLAY_COMMIT=commit,
                                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
                        proc = subprocess.Popen([sys.executable, '-m', 'radonbridge.development_replay',
                            '--record', str(directory/'record.json'), '--data', str(args.data), '--output', str(directory)],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        active[r['model_view_id']] = dict(process=proc, gpu=gpu, log=log, directory=directory,
                                                        start=time.monotonic())
                if time.monotonic()-last_status >= 5:
                    status('needs_attention_draining' if failed else 'stopping' if stop[0] else
                           'replaying_development' if active else 'waiting_for_selected_resources')
                    last_status = time.monotonic()
                if (failed or stop[0]) and not active:
                    break
                time.sleep(2)
            write(root/'accepted_models.json', accepted)
            complete = len(accepted) == len(rows) and not failed
            status('development_replay_complete_test_still_sealed' if complete else 'needs_attention')
            return 0 if complete else 1
        finally:
            for worker in active.values():
                if worker['process'].poll() is None:
                    worker['process'].terminate()
            for worker in active.values():
                try:
                    worker['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker['process'].pid, signal.SIGKILL)
                    worker['process'].wait()
                worker['log'].close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--allocation', type=Path, default=DEFAULT_PATH)
    p.add_argument('--retry-failed', action='store_true', help='Explicitly retry failed views, preserving attempts')
    sys.exit(run(p.parse_args()))
