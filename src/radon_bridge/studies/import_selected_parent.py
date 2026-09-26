"""Explicit one-time import of accepted current LOOK assets into R&B.

This tool never participates in normal model loading. It preserves all tensor and
scientific-provenance bytes; the destination uses only R&B's current contract.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from radon_bridge.models.native_materialization import verify_selected, load_selected
from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash


def publish(source, destination, expected_manifest_sha256, verify_consumer):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    original = source / 'selected_artifact.json'
    if file_sha256(original) != expected_manifest_sha256:
        raise ValueError('Pinned source publication changed')
    manifest = json.loads(original.read_text())
    if (manifest.get('schema') != 'look_selected_native_v2'
            or manifest.get('test_access') is not False
            or manifest.get('complete_training_resume') is not False
            or manifest.get('complete_task_model') is not True
            or manifest.get('selected_prediction_replay') is not True):
        raise ValueError('Accepted current selected publication required')
    files = manifest['files']
    required = {'best.pt', 'spec.json', 'history.json', 'development_predictions.npz', 'source_accepted.json'}
    if not required.issubset(files):
        raise ValueError('Incomplete selected evidence')
    provenance_name = 'provenance/imported_selected_manifest.json'
    if provenance_name in files or 'selected_artifact.json' in files:
        raise ValueError('Reserved publication path')
    for name, digest in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or file_sha256(path) != digest:
            raise ValueError('Source evidence changed or escaped root')
    spec = json.loads((source / 'spec.json').read_text())
    if spec.get('framework', {}).get('api') != 'V5' or spec.get('test_used') is not False:
        raise ValueError('Only independently migrated V5 assets may be imported')
    identity = dict(tool='explicit_current_selected_import_v1',
                    source_manifest_sha256=expected_manifest_sha256,
                    source_spec_sha256=files['spec.json'], source_best_sha256=files['best.pt'])
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / stable_hash(identity)
    with (destination / (target.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            accepted, _, _ = verify_selected(target, spec)
            if accepted['source'] != identity:
                raise ValueError('Conflicting imported publication')
            verify_consumer(target)
            return target
        pending = Path(tempfile.mkdtemp(prefix=target.name + '.', suffix='.partial', dir=destination))
        # Failed attempts remain separate; they never become current publications.
        copied = {}
        for name, digest in files.items():
            dst = pending / name
            if not dst.resolve().is_relative_to(pending):
                raise ValueError('Unsafe destination path')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
            if file_sha256(dst) != digest:
                raise ValueError('Source changed during import')
            with dst.open('rb') as stream:
                os.fsync(stream.fileno())
            copied[name] = digest
        provenance = pending / provenance_name
        provenance.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, provenance)
        if file_sha256(provenance) != expected_manifest_sha256:
            raise ValueError('Source manifest changed during import')
        copied[provenance_name] = expected_manifest_sha256
        atomic_write_json(dict(schema='radon_bridge_selected_native_v2', source=identity,
                               files=copied, test_access=False, complete_task_model=True,
                               complete_training_resume=False, selected_prediction_replay=True,
                               source_acceptance_verification='immutable_current_asset_import'),
                          pending / 'selected_artifact.json')
        verify_selected(pending, spec)
        verify_consumer(pending)
        # Consumer loading must not mutate the publication being accepted.
        verify_selected(pending, spec)
        os.replace(pending, target)
        fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    from mhd_framework.models import create_model
    def verify(path):
        load_selected(path, create_model, device='cpu')
    path = publish(args.source, args.destination, args.manifest_sha256, verify)
    print(json.dumps(dict(state='current_publication_strict_load_accepted', path=str(path),
                          manifest_sha256=file_sha256(path / 'selected_artifact.json'),
                          training_updates=0, test_access=False)))


if __name__ == '__main__':
    main()
