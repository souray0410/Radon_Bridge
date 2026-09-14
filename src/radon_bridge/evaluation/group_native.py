"""Distinct task outputs; probabilities are fused only within the same disease."""
from pathlib import Path
import numpy as np
import torch
from radon_bridge.evaluation.paired_native import move


@torch.no_grad()
def evaluate(model, loader, device, output=None, should_pause=lambda: False):
    from radon_bridge.evaluation.metrics import binary_metrics as metrics
    mode = model.training; model.eval(); p = {k: [] for k in model.source_keys}; y = {k: [] for k in p}; ids = []
    try:
        for batch in loader:
            if should_pause(): raise InterruptedError('Pause read-only evaluation')
            logits, _ = model(move(batch, device))
            for key in p:
                p[key].append(logits[key].softmax(1).cpu().numpy()); y[key].extend(batch['labels'][key].tolist())
            ids.extend(batch['participant_id'])
    finally:
        model.train(mode)
    if not ids or len(set(ids)) != len(ids): raise ValueError('Empty/duplicate evaluation coverage')
    p = {k: np.concatenate(v) for k, v in p.items()}; y = {k: np.asarray(v) for k, v in y.items()}
    result = {k: metrics(y[k], p[k]) for k in p}
    result['mean_macro_f1'] = sum(result[k]['macro_f1'] for k in p)/len(p)
    result['diseases'] = {}
    for disease in sorted({s['disease'] for s in model.sources}):
        keys = [s['key'] for s in model.sources if s['disease'] == disease]
        if any(not np.array_equal(y[keys[0]], y[k]) for k in keys): raise ValueError('Same task label mismatch')
        result['diseases'][disease] = dict(mean_branch_macro_f1=sum(result[k]['macro_f1'] for k in keys)/len(keys),
            fixed_probability_fusion=metrics(y[keys[0]], sum(p[k] for k in keys)/len(keys)))
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.partial')
        with tmp.open('wb') as stream:
            np.savez(stream, participant_ids=np.asarray(ids, dtype=str), **p, **{'labels__'+k: v for k, v in y.items()})
        tmp.replace(path)
    return result
