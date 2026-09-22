import copy
import json
from pathlib import Path

import pytest
import torch

from radon_bridge.runtime.pilot_checkpoint import read, save
from radon_bridge.studies.v5_cohort_revision import configuration_identity, revise_parent_references
from radon_bridge.studies.v5_pilot_migration import equal, sha


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture(tmp_path):
    cfg = dict(name='none_seed3416', seed=3416, bridges=[], parents={})
    parents = {}
    model = {'head': {'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)}}
    for branch, char in [('cfp', 'a'), ('oct', 'b')]:
        folder = tmp_path/'parents'/branch
        save(dict(model=model, seed=3416, branch=branch), folder/'checkpoint.pt', kind='native_parent')
        mapping = dict(source_sha256=char*64, target_sha256=sha(folder/'checkpoint.pt'), original_payload_exact=True)
        write(folder/'conversion.json', mapping)
        cfg['parents'][branch] = dict(path='original/'+branch+'.pt', sha256=char*64)
        parents[branch] = dict(mapping, path=str(folder/'checkpoint.pt'), conversion_sha256=sha(folder/'conversion.json'))
    converted = tmp_path/'converted'
    selected = save(dict(model=model, configuration=cfg, epoch=0), converted/'selected/checkpoint.pt', kind='selected')
    save(dict(model=model, configuration=cfg, identity=configuration_identity(cfg), epoch=1,
              monitor={'best_epoch': 0}, history=[{'epoch': 1}], selected=selected,
              optimizer={'state': {0: {'exp_avg': torch.ones(2, 3)}}},
              rng={'torch': torch.get_rng_state()}), converted/'resume/checkpoint.pt', kind='resume')
    for kind in ('selected', 'resume'):
        write(converted/kind/'conversion.json', dict(kind=kind, target_sha256=sha(converted/kind/'checkpoint.pt'), original_payload_exact=True))
    import radon_bridge.studies.v5_cohort_revision as module
    lock = json.loads((Path(module.__file__).resolve().parents[3]/'framework.lock.json').read_text())
    proof = tmp_path/'acceptance.json'
    write(proof, dict(state='bounded_parent_replay_verified', normal_v5_build_checked=True,
                      test_access=False, current_framework_commit=lock['upstream_commit'], parents=parents))
    kwargs = dict(run_id='2026_09_18_11_32_21_119628', case_id=cfg['name'], parent_acceptance=proof, acceptance_sha256=sha(proof))
    return converted, cfg, kwargs


def test_revision_updates_current_references_without_rewriting_training(tmp_path):
    converted, cfg, kwargs = fixture(tmp_path)
    old = read(converted/'resume/checkpoint.pt', kind='resume')
    hashes = {str(p): sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path/'revision'
    record = revise_parent_references(converted, output, **kwargs)
    target = json.loads((output/'configuration.json').read_text())
    current = read(output/'resume.pt', kind='resume', configuration=target)
    assert record['run_id'] == kwargs['run_id'] and record['dispatch_allowed'] is False
    assert current['identity'] == configuration_identity(target) != old['identity']
    assert target['seed'] == cfg['seed'] and target['bridges'] == cfg['bridges']
    assert target['parents'] != cfg['parents']
    for key in ('model', 'optimizer', 'rng', 'monitor', 'history', 'epoch'):
        assert equal(current[key], old[key])
    assert equal(current['selected'], read(output/'best.pt', kind='selected', configuration=target))
    assert all(sha(path) == digest for path, digest in hashes.items())
    with pytest.raises(FileExistsError):
        revise_parent_references(converted, output, **kwargs)


@pytest.mark.parametrize('mutation', ['acceptance', 'framework', 'source_mapping', 'parent_bytes', 'seed', 'resume_identity', 'selected_boundary', 'augmented'])
def test_revision_rejects_invalid_dependencies_and_keeps_output_absent(tmp_path, mutation):
    converted, cfg, kwargs = fixture(tmp_path)
    proof = kwargs['parent_acceptance']
    p = json.loads(proof.read_text())
    if mutation == 'acceptance': p['normal_v5_build_checked'] = False
    elif mutation == 'framework': p['current_framework_commit'] = '0'*40
    elif mutation == 'source_mapping': p['parents']['cfp']['source_sha256'] = 'f'*64
    elif mutation == 'parent_bytes': Path(p['parents']['cfp']['path']).write_bytes(b'changed')
    elif mutation == 'seed':
        file = Path(p['parents']['cfp']['path']);state = torch.load(file, weights_only=False);state['seed'] = 999;torch.save(state, file)
        p['parents']['cfp']['target_sha256'] = sha(file)
        mapping = json.loads((file.parent/'conversion.json').read_text());mapping['target_sha256'] = sha(file)
        write(file.parent/'conversion.json', mapping);p['parents']['cfp']['conversion_sha256'] = sha(file.parent/'conversion.json')
    else:
        file = converted/'resume/checkpoint.pt';state = torch.load(file, weights_only=False)
        if mutation == 'resume_identity': state['identity'] = 'wrong'
        elif mutation == 'selected_boundary': state['selected']['epoch'] = 9
        else:
            for kind in ('selected', 'resume'):
                path = converted/kind/'checkpoint.pt';s = torch.load(path, weights_only=False)
                new = copy.deepcopy(s['configuration']);new['augmentation_host'] = {'path': 'old-host'}
                s['configuration'] = new
                if kind == 'resume':
                    s['identity'] = configuration_identity(new);s['selected']['configuration'] = new
                torch.save(s, path);receipt = json.loads((path.parent/'conversion.json').read_text());receipt['target_sha256'] = sha(path);write(path.parent/'conversion.json', receipt)
            state = torch.load(file, weights_only=False)
        torch.save(state, file)
        receipt = json.loads((file.parent/'conversion.json').read_text());receipt['target_sha256'] = sha(file);write(file.parent/'conversion.json', receipt)
    write(proof, p);kwargs['acceptance_sha256'] = sha(proof)
    with pytest.raises(ValueError):
        revise_parent_references(converted, tmp_path/'revision', **kwargs)
    assert not (tmp_path/'revision').exists()
