"""Resume reporting only, with separate code provenance and immutable training evidence."""
import argparse, fcntl, os, subprocess, time, traceback
from pathlib import Path
from radonbridge.artifacts import SOURCE, resolve, sha256
from scripts.geometry_evidence import read, write
from scripts.report_geometry_mechanism import build


def main(root):
    root = Path(root)
    with (root / 'queue.lock').open('a') as own, (SOURCE / '.active.lock').open('a') as project:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(project, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'drain.request').exists()
        assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        old = read(root / 'queue_status.json')
        assert old['state'] in ('needs_attention', 'reporting')
        manifest = read(root / 'manifest.json')
        rows = manifest['rows']
        assert len(rows) == 378 and all(r['state'] == 'accepted' for r in rows)
        assert all(r['protocol'] == 'branch' for r in rows)
        for row in rows:
            if not row.get('reused'):
                d = row['diagnostic']
                assert sha256(resolve(d['directory']) / 'summary.json') == d['sha256']
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        record = dict(previous_status=old, report_source_commit=commit,
                      manifest_sha256=sha256(root / 'manifest.json'),
                      training_protocol_sha256=sha256(root / 'protocol.json'),
                      reason='Legacy model cost metadata compatibility; no training or selection changes')
        write(root / f'report_recovery_{time.time_ns()}.json', record)
        def status(state, **extra):
            value = {k:v for k,v in old.items() if k not in ('error', 'traceback')}
            value.update(state=state, pid=os.getpid(), updated_at=time.time(),
                         report_source_commit=commit, **extra)
            write(root / 'queue_status.json', value)
        status('reporting')
        try:
            build(root)
            assert sha256(root / 'manifest.json') == record['manifest_sha256']
            assert sha256(root / 'protocol.json') == record['training_protocol_sha256']
            status('complete', report_complete=True)
        except BaseException as e:
            status('needs_attention', error=repr(e), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
