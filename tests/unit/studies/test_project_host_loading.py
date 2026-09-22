import json

import pytest
import torch

from radon_bridge.runtime.host_checkpoint import save_selected
from radon_bridge.runtime.state import file_sha256
from radon_bridge.studies import project_case, project_units


class Host(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 2)

    def node_identity(self):
        return [('input', 0), ('output', 1)]


def accepted_host(tmp_path, monkeypatch):
    spec = {'arms': [{'id': 'host'}], 'seed': 3416}
    bases = {}
    path = tmp_path / 'bases' / 'accepted.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'bases': bases}))
    identity = project_units.arm_identity(spec, tmp_path, spec['arms'][0])
    root = tmp_path / 'arms' / 'host'
    original = Host()
    save_selected(root / 'best.pt', model=original, identity=identity,
                  epoch=0, node_ids=original.node_identity())
    for name in ('last.pt', 'development_predictions.npz', 'history.json', 'replay_predictions.npz'):
        (root / name).write_bytes(b'accepted evidence fixture')
    receipt = dict(state='accepted', identity=identity, plateau=True, test_access=False,
                   files={p.name: file_sha256(p) for p in root.iterdir()})
    (root / 'accepted.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(project_case, 'build', lambda *args: Host())
    return spec, bases, root, original, receipt


def test_current_accepted_host_loads_exact_values(tmp_path, monkeypatch):
    spec, bases, root, original, _ = accepted_host(tmp_path, monkeypatch)
    actual, state = project_case.load_host(spec, tmp_path, 'host', {}, {}, bases)
    x = torch.randn(4, 3)
    torch.testing.assert_close(actual(x), original(x), rtol=0, atol=0)
    assert state['framework_api'] == 'V5' and state['epoch'] == 0


@pytest.mark.parametrize('mutation', ['header', 'nodes', 'identity', 'shape', 'unaccepted', 'hash'])
def test_host_rejects_unconverted_wrong_or_unaccepted_state(tmp_path, monkeypatch, mutation):
    spec, bases, root, original, receipt = accepted_host(tmp_path, monkeypatch)
    state = torch.load(root / 'best.pt', weights_only=False)
    if mutation == 'header': state.pop('framework_api')
    elif mutation == 'nodes': state['node_ids'] = [('wrong', 0)]
    elif mutation == 'identity': state['identity'] = 'another host'
    elif mutation == 'shape': state['model']['weight'] = torch.zeros(4, 3)
    elif mutation == 'unaccepted': receipt['plateau'] = False
    else: state['epoch'] = 1
    torch.save(state, root / 'best.pt')
    # Even a correctly rehashed receipt must not admit a malformed current state.
    if mutation != 'hash': receipt['files']['best.pt'] = file_sha256(root / 'best.pt')
    (root / 'accepted.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError, RuntimeError)):
        project_case.load_host(spec, tmp_path, 'host', {}, {}, bases)
