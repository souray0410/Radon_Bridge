"""Offline host-envelope conversion from an accepted current selected revision."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from radon_bridge.runtime.pilot_checkpoint import read, save
from radon_bridge.studies.v5_pilot_migration import equal, sha


def convert_host(manifest, output, *, manifest_sha256, source_receipt,
                 revision, acceptance, acceptance_sha256, package, case_id):
    """Preserve historical selection; derive only the current host contract."""
    manifest, output, source_receipt, revision, acceptance = map(
        Path, (manifest, output, source_receipt, revision, acceptance))
    if sha(manifest) != manifest_sha256 or sha(acceptance) != acceptance_sha256:
        raise ValueError('Host manifest or numerical acceptance changed')
    old = json.loads(manifest.read_text())
    proof = json.loads(acceptance.read_text())
    lock = json.loads((Path(__file__).resolve().parents[3]/'framework.lock.json').read_text())
    if (old.get('schema') != 'radon_cohort_augmentation_host_v1'
            or old.get('test_used') is not False
            or proof.get('state') != 'current_parent_reference_replay_verified'
            or proof.get('normal_v5_build_checked') is not True
            or proof.get('test_access') is not False
            or proof.get('framework_commit') != lock['upstream_commit']):
        raise ValueError('Original host and accepted current-framework revision required')
    rows = [r for r in proof['rows'] if r['package'] == package and r['case'] == case_id]
    if len(rows) != 1:
        raise ValueError('Exactly one accepted host revision required')
    row = rows[0]
    inputs = {manifest: manifest_sha256, acceptance: acceptance_sha256,
              source_receipt: old['source_receipt_sha256'],
              revision/'execution_revision.json': row['revision_sha256']}
    inputs.update({revision/name: digest for name, digest in row['artifacts'].items()})
    inputs.update({Path(old[key]['path']): old[key]['sha256'] for key in ('model', 'predictions')})
    original_acceptance = json.loads(source_receipt.read_text())
    if (original_acceptance.get('state') != 'complete'
            or original_acceptance.get('converged_by_policy') is not True
            or original_acceptance.get('test_used') is not False
            or original_acceptance['configuration'] != old['configuration']
            or original_acceptance['best_epoch'] != old['selected_epoch']
            or old['predictions']['sha256'] != row['prediction_sha256']
            or original_acceptance['files']['selected_predictions.npz'] != row['prediction_sha256']):
        raise ValueError('Historical selection or verified predictions changed')
    for name, digest in original_acceptance['files'].items():
        if Path(name).name != name:
            raise ValueError('Original receipt must contain local artifact filenames')
        inputs[source_receipt.parent/name] = digest
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('Host source or accepted revision bytes changed')
    selected = read(revision/'best.pt', kind='selected')
    old_state = torch.load(old['model']['path'], map_location='cpu', weights_only=False)
    if (old_state.get('schema') != old['schema']
            or old_state.get('configuration') != old['configuration']
            or not equal(old_state.get('model'), selected['model'])
            or selected['epoch'] != old['selected_epoch']):
        raise ValueError('Host model differs from accepted selected revision')
    target = copy.deepcopy(selected['configuration'])
    restored = copy.deepcopy(target)
    restored['parents'] = old['configuration']['parents']
    if (restored != old['configuration'] or target['name'] != case_id
            or len(target['bridges']) != 1
            or target['bridges'][0].get('family') not in ('mmtm', 'cross_attention', 'cmx_frm')):
        raise ValueError('Host scientific configuration changed')
    for parent in target['parents'].values():
        inputs[Path(parent['path'])] = parent['sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(output)
        stage = Path(tempfile.mkdtemp(prefix=output.name+'.', dir=output.parent))
        try:
            save(dict(schema='radon_cohort_augmentation_host_v2', configuration=target,
                      model=selected['model']), stage/'model.pt', kind='augmentation_host')
            if not equal(read(stage/'model.pt', kind='augmentation_host')['model'], old_state['model']):
                raise ValueError('Written host model changed')
            shutil.copyfile(old['predictions']['path'], stage/'selected_predictions.npz')
            current = dict(schema='radon_cohort_augmentation_host_v2', framework_api='V5',
                           configuration=target, source_receipt_sha256=old['source_receipt_sha256'],
                           model=dict(path=str(output/'model.pt'), sha256=sha(stage/'model.pt')),
                           predictions=dict(path=str(output/'selected_predictions.npz'), sha256=sha(stage/'selected_predictions.npz')),
                           selected_epoch=old['selected_epoch'], test_used=False,
                           original_training_framework='V4', source_manifest_sha256=manifest_sha256,
                           source_run_id=row['run_id'])
            (stage/'manifest.json').write_text(json.dumps(current, indent=2)+'\n')
            record = dict(schema='radon_augmentation_host_conversion_v1',
                          state='host_reference_converted_pending_downstream_replay',
                          source_manifest_sha256=manifest_sha256, source_configuration=old['configuration'],
                          target_manifest=dict(path=str(output/'manifest.json'), sha256=sha(stage/'manifest.json')),
                          target_configuration=target, model_tensor_values_preserved=True,
                          framework_commit=proof['framework_commit'], case_id=case_id, run_id=row['run_id'],
                          numerical_acceptance_sha256=acceptance_sha256, selected_revision_sha256=row['revision_sha256'],
                          source_receipt_sha256=old['source_receipt_sha256'],
                          inputs={str(path): digest for path, digest in inputs.items()},
                          converter_sha256=sha(__file__), test_access=False, dispatch_allowed=False)
            if any(sha(path) != digest for path, digest in inputs.items()):
                raise ValueError('Host dependencies changed during conversion')
            (stage/'conversion.json').write_text(json.dumps(record, indent=2)+'\n')
            os.rename(stage, output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return record


def mapped_host(configuration, parents, mapping_path, mapping_sha256, framework_commit):
    """Validate one explicit conversion mapping for a pending execution revision."""
    path = Path(mapping_path)
    if sha(path) != mapping_sha256:
        raise ValueError('Host conversion mapping changed')
    mapping = json.loads(path.read_text())
    if (mapping.get('schema') != 'radon_augmentation_host_conversion_v1'
            or mapping.get('model_tensor_values_preserved') is not True
            or mapping.get('test_access') is not False
            or mapping.get('framework_commit') != framework_commit
            or mapping.get('source_manifest_sha256') != configuration['augmentation_host']['sha256']):
        raise ValueError('Matching current host conversion required')
    old = mapping['source_configuration']
    current = mapping['target_configuration']
    unchanged = copy.deepcopy(current)
    unchanged['parents'] = old['parents']
    if (unchanged != old or old['parents'] != configuration['parents']
            or current['parents'] != parents or old['seed'] != configuration['seed']
            or old['bridges'] != configuration['bridges'][:1]
            or len(configuration['bridges']) not in (1, 2)
            or (len(configuration['bridges']) == 2 and configuration['bridges'][1].get('parallel_to') != 0)):
        raise ValueError('Host and augmented run scientific dependencies differ')
    ref = mapping['target_manifest']
    manifest = json.loads(Path(ref['path']).read_text())
    if (sha(ref['path']) != ref['sha256'] or manifest.get('framework_api') != 'V5'
            or manifest.get('schema') != 'radon_cohort_augmentation_host_v2'
            or manifest.get('test_used') is not False or manifest['configuration'] != current):
        raise ValueError('Current host manifest changed')
    inputs = {path: mapping_sha256, Path(ref['path']): ref['sha256']}
    original_ref = configuration['augmentation_host']
    inputs[Path(original_ref['path'])] = original_ref['sha256']
    for key in ('model', 'predictions'):
        item = manifest[key]
        inputs[Path(item['path'])] = item['sha256']
    for filename, digest in mapping['inputs'].items():
        inputs[Path(filename)] = digest
    if any(sha(filename) != digest for filename, digest in inputs.items()):
        raise ValueError('Host conversion input or output changed')
    read(manifest['model']['path'], kind='augmentation_host', configuration=current)
    return copy.deepcopy(ref), inputs
