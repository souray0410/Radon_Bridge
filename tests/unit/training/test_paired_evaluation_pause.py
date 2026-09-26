"""Exercise clean lease signals before selection and during final acceptance replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from radon_bridge.models.modern_communication import ModernMMTMHost
from radon_bridge.training import paired_native as training
from tests.unit.models.test_modern_communication import parent


class SmallPair(Dataset):
    def __init__(self, split):
        self.split = split
        self.participant_ids = [split + str(i) for i in range(4)]

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, i):
        generator = torch.Generator().manual_seed(200 + i)
        return dict(cfp=torch.randn(1, 1, 4, 4, generator=generator),
                    oct=torch.randn(1, 1, 3, 4, 4, generator=generator),
                    label=i % 2, participant_id=self.participant_ids[i])


def make_model():
    torch.manual_seed(3416)
    return ModernMMTMHost({'cfp': parent(2), 'oct': parent(3)},
                          {'cfp': (1, 4, 4), 'oct': (1, 3, 4, 4)}, frozen=True)


def fake_evaluator(out, interrupt_at=None):
    calls = []
    before_interrupt = {}

    def evaluate(model, loader, device, output, should_pause):
        calls.append(Path(output).name)
        if len(calls) == interrupt_at:
            if (out / 'last.pt').exists():
                before_interrupt['sha'] = hashlib.sha256((out / 'last.pt').read_bytes()).hexdigest()
            raise InterruptedError('Synthetic lease signal during dev evaluation')
        # Fixed scores make initialization the selected A, distinct from last.
        with Path(output).open('wb') as handle:
            np.savez(handle, participant_ids=np.array(['dev0', 'dev1']),
                     labels=np.array([0, 1]), joint=np.array([[.7, .3], [.2, .8]]))
        return {'joint': {'macro_f1': .9 if Path(output).name != 'candidate_predictions.npz' else .8}}

    return evaluate, calls, before_interrupt


def run(model, out):
    cfg = dict(training.DEFAULTS, microbatch=1, effective_batch=2,
               minimum_epochs=1, maximum_epochs=4, patience=2, lr_patience=1)
    return training.train(model, SmallPair('train'), SmallPair('development'),
                          cfg, 3416, out, 'lease-boundary-fixture', torch.device('cpu'))


@pytest.mark.parametrize('interrupt_at', [1, 4])
def test_initial_and_final_evaluation_pause_reaches_accepted_after_reload(tmp_path, monkeypatch, interrupt_at):
    torch.set_num_threads(2)
    evaluator, calls, before = fake_evaluator(tmp_path, interrupt_at)
    monkeypatch.setattr(training, 'evaluate', evaluator)
    model = make_model()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = run(model, tmp_path)
    assert result['state'] == 'paused'
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'paused'
    assert not (tmp_path / 'accepted.json').exists()
    last = torch.load(tmp_path / 'last.pt', weights_only=False)
    if interrupt_at == 1:
        assert not (tmp_path / 'best.pt').exists()
        assert last['progress']['updates'] == 0
        assert all(torch.equal(last['model'][k], v) for k, v in original.items())
    else:
        # The loaded selected model must never overwrite last's optimizer state.
        assert hashlib.sha256((tmp_path / 'last.pt').read_bytes()).hexdigest() == before['sha']
        selected = torch.load(tmp_path / 'best.pt', weights_only=False)
        assert any(not torch.equal(last['model'][k], v) for k, v in selected['model'].items())
        assert last['progress']['updates'] == 4
    monkeypatch.setattr(training, 'evaluate', fake_evaluator(tmp_path)[0])
    result = run(make_model(), tmp_path)
    assert result['state'] == 'accepted'
    assert result['best_epoch'] == 0 and result['stop_epoch'] == 2
    assert torch.load(tmp_path / 'last.pt', weights_only=False)['progress']['updates'] == 4


def test_unknown_evaluation_failure_is_not_retried_or_called_clean_pause(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('corrupt input')
    monkeypatch.setattr(training, 'evaluate', fail)
    with pytest.raises(ValueError, match='corrupt input'):
        run(make_model(), tmp_path)
    assert not (tmp_path / 'accepted.json').exists()
