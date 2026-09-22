import copy
import json
from pathlib import Path

import pytest
import torch

from radon_bridge.runtime.pilot_checkpoint import read, save
from radon_bridge.studies.v5_augmentation_migration import convert_host, mapped_host
from radon_bridge.studies.v5_cohort_revision import configuration_identity
from radon_bridge.studies.v5_pilot_migration import equal, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    source = tmp_path/'source';source.mkdir()
    model = {'host': {'weight': torch.arange(6).reshape(2, 3).float()}}
    old = dict(name='mmtm_seed3416', seed=3416, bridges=[dict(family='mmtm')], parents={})
    current = copy.deepcopy(old)
    for branch in ('cfp', 'oct'):
        old['parents'][branch] = dict(path='historical/'+branch, sha256='a'*64)
        path = tmp_path/(branch+'.pt');torch.save(model, path)
        current['parents'][branch] = dict(path=str(path), sha256=sha(path))
    torch.save(dict(schema='radon_cohort_augmentation_host_v1', configuration=old, model=model), source/'model.pt')
    torch.save(dict(configuration=old, model=model, epoch=0), source/'best.pt')
    (source/'selected_predictions.npz').write_bytes(b'finite synthetic prediction fixture')
    a = dict(state='complete', converged_by_policy=True, test_used=False, best_epoch=0,
             configuration=old, files={n:sha(source/n) for n in ('best.pt', 'selected_predictions.npz')})
    write(source/'accepted.json', a)
    manifest = dict(schema='radon_cohort_augmentation_host_v1', configuration=old, test_used=False,
                    selected_epoch=0, source_receipt_sha256=sha(source/'accepted.json'),
                    model=dict(path=str(source/'model.pt'), sha256=sha(source/'model.pt')),
                    predictions=dict(path=str(source/'selected_predictions.npz'), sha256=sha(source/'selected_predictions.npz')))
    write(source/'manifest.json', manifest)
    rev = tmp_path/'revision'
    save(dict(configuration=current, model=model, epoch=0), rev/'best.pt', kind='selected')
    write(rev/'configuration.json', current)
    write(rev/'execution_revision.json', dict(run_id='run_1'))
    import radon_bridge.studies.v5_augmentation_migration as module
    commit = json.loads((Path(module.__file__).resolve().parents[3]/'framework.lock.json').read_text())['upstream_commit']
    row = dict(package='core', case=old['name'], run_id='run_1', revision_sha256=sha(rev/'execution_revision.json'),
               configuration_identity=configuration_identity(current), prediction_sha256=sha(source/'selected_predictions.npz'),
               artifacts={n:sha(rev/n) for n in ('best.pt', 'configuration.json')})
    proof = tmp_path/'acceptance.json'
    write(proof, dict(state='current_parent_reference_replay_verified', normal_v5_build_checked=True,
                      test_access=False, framework_commit=commit, rows=[row]))
    kwargs = dict(manifest_sha256=sha(source/'manifest.json'), source_receipt=source/'accepted.json',
                  revision=rev, acceptance=proof, acceptance_sha256=sha(proof), package='core', case_id=old['name'])
    return source, old, current, commit, kwargs


def test_host_conversion_and_mapped_dependency_preserve_original_state(tmp_path):
    source, old, current, commit, kwargs = fixture(tmp_path)
    hashes = {p:sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    out = tmp_path/'host'
    record = convert_host(source/'manifest.json', out, **kwargs)
    result = read(out/'model.pt', kind='augmentation_host', configuration=current)
    original = torch.load(source/'model.pt', weights_only=False)
    assert equal(result['model'], original['model']) and record['dispatch_allowed'] is False
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['original_training_framework'] == 'V4' and manifest['selected_epoch'] == 0
    cfg = copy.deepcopy(old)
    cfg['augmentation_host'] = dict(path=str(source/'manifest.json'), sha256=sha(source/'manifest.json'))
    cfg['bridges'].append(dict(family='radon', parallel_to=0))
    ref, inputs = mapped_host(cfg, current['parents'], out/'conversion.json', sha(out/'conversion.json'), commit)
    assert ref == record['target_manifest'] and all(sha(p) == digest for p,digest in inputs.items())
    assert all(sha(p) == digest for p,digest in hashes.items())
    with pytest.raises(FileExistsError):convert_host(source/'manifest.json', out, **kwargs)


@pytest.mark.parametrize('change', ['proof', 'manifest', 'prediction', 'revision', 'receipt', 'source_model'])
def test_conversion_rejects_changed_evidence_before_publishing(tmp_path, change):
    source, old, current, commit, kwargs = fixture(tmp_path)
    path = {'proof': kwargs['acceptance'], 'manifest': source/'manifest.json', 'prediction': source/'selected_predictions.npz',
            'revision': kwargs['revision']/'best.pt', 'receipt': source/'accepted.json', 'source_model':source/'model.pt'}[change]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):convert_host(source/'manifest.json', tmp_path/'out', **kwargs)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change', ['parents', 'seed', 'parallel', 'model', 'mapping'])
def test_mapped_host_rejects_stale_or_unmatched_dependencies(tmp_path, change):
    source, old, current, commit, kwargs = fixture(tmp_path)
    out = tmp_path/'host';convert_host(source/'manifest.json', out, **kwargs)
    cfg = copy.deepcopy(old);cfg['augmentation_host'] = dict(path=str(source/'manifest.json'),sha256=sha(source/'manifest.json'))
    digest = sha(out/'conversion.json')
    if change == 'parents':current['parents']['cfp']['sha256'] = 'f'*64
    elif change == 'seed':cfg['seed'] = 99
    elif change == 'parallel':cfg['bridges'].append(dict(family='radon',parallel_to=9))
    elif change == 'model':(out/'model.pt').write_bytes(b'corrupt')
    else:(out/'conversion.json').write_text('{}')
    with pytest.raises(ValueError):mapped_host(cfg,current['parents'],out/'conversion.json',digest,commit)
