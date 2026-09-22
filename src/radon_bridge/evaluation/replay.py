"""Strict read-only replay on the fixed development cohort. No test entry point."""
from radon_bridge.runtime.pilot_checkpoint import read as read_state
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from radon_bridge.runtime.artifacts import resolve, relocate, sha256
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.training.trainer import evaluate, write_json
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.models.model import PilotGraph


def compare_arrays(actual, expected):
    for key in ('ids', 'y'):
        if not np.array_equal(actual[key], expected[key]):
            raise ValueError(f'Development {key} order/value mismatch')
    if len(actual['ids']) != 296 or len(set(actual['ids'])) != 296:
        raise ValueError('Expected 296 unique development participants')
    result = {}
    for key in ('cfp', 'oct'):
        a, b = actual[key], expected[key]
        if a.shape != (296, 2) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid development probabilities')
        if not np.allclose(a.sum(1), 1., atol=1e-5):
            raise ValueError('Probabilities are not normalized')
        if not np.allclose(a, b, atol=1e-6, rtol=1e-5):
            raise ValueError(f'{key} probability replay mismatch: {np.max(np.abs(a-b))}')
        if not np.array_equal(a.argmax(1), b.argmax(1)):
            raise ValueError(f'{key} class decisions changed')
        f1 = classification_metrics(actual['y'], a)['macro_f1']
        if f1 != classification_metrics(expected['y'], b)['macro_f1']:
            raise ValueError(f'{key} F1 replay mismatch')
        result[key] = dict(max_absolute_probability_error=float(np.max(np.abs(a-b))), macro_f1=f1)
    return result


def state_hash(g):
    h = hashlib.sha256()
    for name, module in sorted(g.modules_by_name().items()):
        for key, value in sorted(module.state_dict().items()):
            t = value.detach().cpu().contiguous()
            h.update(str((name, key, str(t.dtype), tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def run(record, data_root, out):
    start = time.monotonic()
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)
    data = PairedDataset(str(resolve(data_root)), 'validation', 224)
    if len(data) != 296 or any(r['split'] != 'validation' for r in data.rows):
        raise ValueError('Only the fixed development set is permitted')
    if record.get('factory') == 'PilotGraph_independent_parent_pair':
        seed = record['seed']
        g = PilotGraph(seed=seed, bridge_configs=[], device='cuda')
        pred_dirs = set()
        for branch, ref in record['parent_checkpoints'].items():
            if sha256(ref['path']) != ref['sha256']:
                raise ValueError('Parent checkpoint SHA changed')
            saved = read_state(resolve(ref['path']),kind='native_parent')
            if saved['branch'] != branch or saved['seed'] != seed or saved['training_stage'] != 'independent':
                raise ValueError('Unexpected independent parent metadata')
            g.load_native_state(saved['model'], branch=branch)
            pred_dirs.add(resolve(ref['path']).parent)
            del saved
        if len(pred_dirs) != 1:
            raise ValueError('Parent pair predictions require a shared pretraining run')
        parent_dir = next(iter(pred_dirs))
        summary = json.loads((parent_dir/'summary.json').read_text())
        if summary['test_used'] is not False or summary['state'] != 'complete':
            raise ValueError('Parent summary did not pass')
        expected_path = parent_dir/'selected_predictions.npz'
        expected_sha = sha256(expected_path)
    else:
        cfg = relocate(record['configuration'])
        if cfg.get('task_fusion'):
            raise ValueError('Withdrawn task fusion is outside this registry')
        seed = cfg['seed']
        if sha256(record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('Selected checkpoint SHA changed')
        g = PilotGraph(seed=seed, bridge_configs=cfg['bridges'], device='cuda')
        saved = read_state(resolve(record['checkpoint_path']),kind='selected')
        g.load_complete_state(saved['model'])
        del saved
        expected_path = resolve(record['development_prediction_path'])
        expected_sha = record['development_prediction_sha256']
        if sha256(expected_path) != expected_sha:
            raise ValueError('Saved development predictions SHA changed')
    g.graph.eval()
    before = state_hash(g)
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    np_rng = np.random.get_state()
    py_rng = random.getstate()
    if any(p.grad is not None for m in g.modules_by_name().values() for p in m.parameters()):
        raise ValueError('Unexpected pre-existing gradients')
    metrics = evaluate(g, data, 16, seed, out/'predictions.npz', stop=lambda: stop[0])
    if state_hash(g) != before or any(m.training for m in g.modules_by_name().values()):
        raise ValueError('Parameters or BN state changed during inference')
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
        raise ValueError('Torch RNG changed during inference')
    now_np = np.random.get_state()
    if np_rng[0] != now_np[0] or not np.array_equal(np_rng[1], now_np[1]) or np_rng[2:] != now_np[2:] or py_rng != random.getstate():
        raise ValueError('Python/NumPy RNG changed during inference')
    if any(p.grad is not None for m in g.modules_by_name().values() for p in m.parameters()):
        raise ValueError('Inference created parameter gradients')
    with np.load(out/'predictions.npz', allow_pickle=False) as actual, np.load(expected_path, allow_pickle=False) as expected:
        comparison = compare_arrays(actual, expected)
    write_json(out/'summary.json', dict(state='accepted', model_view_id=record['model_view_id'],
        strict_model_load=True, development_forward_replay=True, parameters_BN_gradients_RNG_preserved=True,
        participants=296, batch=16, comparison=comparison, metrics=metrics,
        expected_prediction_sha256=expected_sha, replay_prediction_sha256=sha256(out/'predictions.npz'),
        record_sha256=sha256(out/'record.json'), state_sha256=before,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        source_commit=os.environ.get('RB_REPLAY_COMMIT'), seconds=time.monotonic()-start,
        test_used=False, test_inference_permitted=False, updated_at=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        if (a.output/'summary.json').exists():
            raise RuntimeError('Refuse to overwrite accepted replay')
        run(json.loads(a.record.read_text()), a.data, a.output)
    except Exception:
        write_json(a.output/'failure.json', dict(state='needs_attention', error=traceback.format_exc(), test_used=False, updated_at=time.time()))
        raise
