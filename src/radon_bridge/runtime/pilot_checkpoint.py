"""Current V5 contract for PilotGraph's named-module states.

The native-host state_dict contract is separate: PilotGraph saves one state_dict
per named module. Historical files are converted offline, never in this reader.
"""
import torch
from radon_bridge.runtime.host_checkpoint import atomic_save

FORMAT = 'radon_pilot_state_v2'
KINDS = frozenset({'selected', 'native_parent', 'resume', 'profile_resume',
                   'stopping', 'interrupted', 'augmentation_host'})


def validate(state, *, kind, configuration=None):
    kinds = (kind,) if isinstance(kind, str) else tuple(kind)
    if not kinds or any(k not in KINDS for k in kinds):
        raise ValueError('Unknown PilotGraph checkpoint role')
    if (not isinstance(state, dict) or state.get('format') != FORMAT
            or state.get('framework_api') != 'V5' or state.get('kind') not in kinds):
        raise ValueError('Current V5 PilotGraph checkpoint required; explicitly migrate old artifacts')
    model = state.get('model')
    if (not isinstance(model, dict) or not model
            or any(not isinstance(k, str) or not isinstance(v, dict) for k, v in model.items())):
        raise ValueError('PilotGraph named-module state required')
    if configuration is not None and state.get('configuration') != configuration:
        raise ValueError('PilotGraph checkpoint configuration changed')
    if state['kind'] == 'resume':
        required = {'identity', 'configuration', 'epoch', 'monitor', 'history', 'optimizer', 'rng', 'selected'}
        if not required <= state.keys():
            raise ValueError('Incomplete PilotGraph resume state')
        validate(state['selected'], kind='selected', configuration=state['configuration'])
    return state


def save(state, path, *, kind):
    if {'format', 'framework_api', 'kind'} & state.keys():
        raise ValueError('Checkpoint metadata is owned by the V5 writer')
    result = dict(state, format=FORMAT, framework_api='V5', kind=kind)
    validate(result, kind=kind)
    atomic_save(path, result)
    return result


def read(path, *, kind, configuration=None):
    return validate(torch.load(path, map_location='cpu', weights_only=False),
                    kind=kind, configuration=configuration)
