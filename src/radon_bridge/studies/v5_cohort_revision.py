"""Offline revision of converted cohort states to verified current parents.

This is a migration tool, never a fallback in a training/checkpoint reader.
The research run ID stays unchanged; current configuration content gets a new
identity. A derived revision still requires downstream replay before dispatch.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from radon_bridge.runtime.pilot_checkpoint import read, save
from radon_bridge.studies.v5_pilot_migration import equal, sha


def configuration_identity(configuration):
    # Exactly the identity used by the current cohort trainer.
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def revise_parent_references(converted, output, *, run_id, case_id,
                             parent_acceptance, acceptance_sha256,
                             host_conversion=None, host_conversion_sha256=None):
    """Derive best/resume/configuration together; never overwrite source assets."""
    for identity in (run_id, case_id):
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identity):
            raise ValueError('Explicit unchanged run and case identities required')
    converted, output, acceptance = map(Path, (converted, output, parent_acceptance))
    if sha(acceptance) != acceptance_sha256:
        raise ValueError('Parent acceptance changed')
    proof = json.loads(acceptance.read_text())
    lock = json.loads((Path(__file__).resolve().parents[3]/'framework.lock.json').read_text())
    if (proof.get('state') != 'bounded_parent_replay_verified'
            or proof.get('normal_v5_build_checked') is not True
            or proof.get('test_access') is not False
            or proof.get('current_framework_commit') != lock['upstream_commit']
            or set(proof.get('parents', {})) != {'cfp', 'oct'}):
        raise ValueError('Accepted current-framework native parent replay required')
    states, inputs = {}, {}
    for kind in ('selected', 'resume'):
        folder = converted/kind
        receipt = json.loads((folder/'conversion.json').read_text())
        digest = sha(folder/'checkpoint.pt')
        if (receipt.get('target_sha256') != digest or receipt.get('kind') != kind
                or receipt.get('original_payload_exact') is not True):
            raise ValueError('Converted source or conversion receipt changed')
        states[kind] = read(folder/'checkpoint.pt', kind=kind)
        inputs[kind] = dict(checkpoint_sha256=digest,
                            conversion_sha256=sha(folder/'conversion.json'))
    configuration = states['resume']['configuration']
    old_identity = configuration_identity(configuration)
    if (configuration != states['selected']['configuration']
            or states['resume']['identity'] != old_identity
            or not equal(states['resume']['selected'], states['selected'])):
        raise ValueError('Selected/resume boundary or configuration mismatch')
    if 'augmentation_host' in configuration and (host_conversion is None or host_conversion_sha256 is None):
        raise ValueError('Augmented runs require host-reference migration as well')
    if 'augmentation_host' not in configuration and (host_conversion is not None or host_conversion_sha256 is not None):
        raise ValueError('Unexpected host mapping for a non-augmented run')
    if configuration.get('name') != case_id or set(configuration.get('parents', {})) != {'cfp', 'oct'}:
        raise ValueError('Case or parent branches changed')
    target = copy.deepcopy(configuration)
    parent_inputs = {}
    for branch, original in configuration['parents'].items():
        parent = proof['parents'][branch]
        path = Path(parent['path'])
        conversion = path.parent/'conversion.json'
        if (parent['source_sha256'] != original['sha256']
                or sha(path) != parent['target_sha256']
                or sha(conversion) != parent['conversion_sha256']):
            raise ValueError('Parent mapping or verified artifact changed')
        mapping = json.loads(conversion.read_text())
        if (mapping.get('source_sha256') != parent['source_sha256']
                or mapping.get('target_sha256') != parent['target_sha256']
                or mapping.get('original_payload_exact') is not True):
            raise ValueError('Parent conversion mapping changed')
        state = read(path, kind='native_parent')
        if state.get('branch') != branch or state.get('seed') != configuration['seed']:
            raise ValueError('Parent branch or seed changed')
        parent_inputs[path] = parent['target_sha256']
        parent_inputs[conversion] = parent['conversion_sha256']
        target['parents'][branch].update(path=str(path), sha256=parent['target_sha256'])
    if 'augmentation_host' in configuration:
        from radon_bridge.studies.v5_augmentation_migration import mapped_host
        ref, host_inputs = mapped_host(configuration, target['parents'], host_conversion,
                                       host_conversion_sha256, proof['current_framework_commit'])
        target['augmentation_host'] = ref
        parent_inputs.update(host_inputs)
    target_identity = configuration_identity(target)
    if target_identity == old_identity:
        raise ValueError('No parent reference migration requested')
    derived = copy.deepcopy(states)
    derived['selected']['configuration'] = target
    derived['resume']['configuration'] = target
    derived['resume']['identity'] = target_identity
    derived['resume']['selected'] = copy.deepcopy(derived['selected'])
    # Check the complete state after undoing only the documented metadata edits.
    restored = copy.deepcopy(derived)
    restored['selected']['configuration'] = configuration
    restored['resume']['configuration'] = configuration
    restored['resume']['identity'] = old_identity
    restored['resume']['selected'] = copy.deepcopy(states['selected'])
    if not equal(restored, states):
        raise ValueError('Unexpected training-state change')
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(output)
        stage = Path(tempfile.mkdtemp(prefix=output.name+'.', dir=output.parent))
        try:
            for kind, filename in [('selected', 'best.pt'), ('resume', 'resume.pt')]:
                payload = {k: v for k, v in derived[kind].items()
                           if k not in ('format', 'framework_api', 'kind')}
                save(payload, stage/filename, kind=kind)
                if not equal(read(stage/filename, kind=kind, configuration=target), derived[kind]):
                    raise ValueError('Written revision changed state')
            (stage/'configuration.json').write_text(json.dumps(target, indent=2)+'\n')
            record = dict(schema='radon_execution_revision_v1', run_id=run_id,
                          case_id=case_id, original_configuration_identity=old_identity,
                          configuration_identity=target_identity, framework_api='V5',
                          framework_commit=proof['current_framework_commit'],
                          original_training_framework='V4', inputs=inputs,
                          parent_acceptance_sha256=acceptance_sha256,
                          converter_sha256=sha(__file__), training_state_unchanged=True,
                          changed_fields=['configuration.parents', 'resume.identity',
                                          'resume.selected.configuration'],
                          state='awaiting_current_reference_replay', dispatch_allowed=False,
                          test_access=False, artifacts={n: sha(stage/n) for n in
                              ('best.pt', 'resume.pt', 'configuration.json')})
            if 'augmentation_host' in configuration:
                record['changed_fields'].append('configuration.augmentation_host')
                record['host_conversion_sha256'] = host_conversion_sha256
            # Recheck sources before publishing this one complete revision.
            if sha(acceptance) != acceptance_sha256:
                raise ValueError('Acceptance changed during revision')
            for kind, item in inputs.items():
                if (sha(converted/kind/'checkpoint.pt') != item['checkpoint_sha256']
                        or sha(converted/kind/'conversion.json') != item['conversion_sha256']):
                    raise ValueError('Source changed during revision')
            if any(sha(path) != digest for path, digest in parent_inputs.items()):
                raise ValueError('Parent changed during revision')
            (stage/'execution_revision.json').write_text(json.dumps(record, indent=2)+'\n')
            os.rename(stage, output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return record
