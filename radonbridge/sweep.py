"""Bounded, adaptive screening with fixed within-trial optimization and seed replication."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedDataset
from .metrics import classification_metrics
from .model import PilotGraph
from .optimization import configure_optimizer, clip_task_gradients
from .protocol import validate_formal_protocol


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def macro_f1_batch(y, prediction):
    values = []
    for label in [0, 1]:
        tp = ((y == label) & (prediction == label)).sum(1)
        denom = (y == label).sum(1) + (prediction == label).sum(1)
        values.append(np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0))
    return np.mean(values, axis=0)


def paired_interval(baseline_path, other_path):
    result = {}
    with np.load(baseline_path) as a, np.load(other_path) as b:
        assert np.array_equal(a['ids'], b['ids']) and np.array_equal(a['y'], b['y'])
        ix = np.random.default_rng(910).integers(0, len(a['y']), (1000, len(a['y'])))
        y = a['y'][ix]
        for task in ['cfp', 'oct']:
            d = macro_f1_batch(y, b[task].argmax(1)[ix]) - macro_f1_batch(y, a[task].argmax(1)[ix])
            result[task] = {'paired_bootstrap_95pct': np.quantile(d, [.025, .975]).tolist()}
    return result


def main(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise RuntimeError('Existing sweep cannot be overwritten or implicitly resumed')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Commit source before running')
    protocol = json.loads(Path(args.protocol).read_text())
    validate_formal_protocol(protocol)
    if protocol.get('phase_mode') == 'formal':
        evidence = Path(protocol['qualification_summary_path']).read_bytes()
        if hashlib.sha256(evidence).hexdigest() != protocol['qualification_summary_sha256']:
            raise ValueError('Qualification evidence changed')
        qualification = json.loads(evidence)
        selected = qualification['trials'][qualification['selections']['baseline']]['configuration']['recipe']
        if selected != protocol['recipes'][0] or protocol['prior_gpu_minutes'] < qualification['elapsed_minutes']:
            raise ValueError('Recipe or prior budget does not match qualification')
    assert protocol['loss_reduction'] == 'sum' and protocol['clip_policy'] == 'per_task'
    started = time.monotonic()
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    train = PairedDataset(args.data, 'train', 224)
    val = PairedDataset(args.data, 'validation', 224)
    assert not ({r['participant_id'] for r in train.rows} & {r['participant_id'] for r in val.rows})
    weight = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/resnet18-f37072fd.pth'
    total_trials = (len(protocol['formal_arms']) * len(protocol['confirmation_seeds']) if protocol.get('phase_mode') == 'formal'
                    else len(protocol['recipes']) + len(protocol['bridges']) + 5 * len(protocol['confirmation_seeds']))
    source = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'mhd_commit': subprocess.check_output(['git', '-C', 'third_party/MHD_Project', 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol_sha256': hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
              'weight_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(), 'protocol': protocol,
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'weight_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in weight.parent.glob('resnet*.pth')},
              'train_count': len(train), 'validation_count': len(val), 'test_used': False,
              'data_audit': json.loads((Path(args.data) / 'audit.json').read_text()),
              'cfp_audit': json.loads((Path(args.data) / 'cfp224/audit.json').read_text())}
    write_json(out / 'config.json', source)
    report = {'source_commit': source['commit'], 'phase': protocol.get('phase_mode', 'baseline_screen'), 'total_trials': total_trials,
              'trials': {}, 'selections': {}, 'comparisons': {}, 'test_used': False,
              'limitations': protocol['limitations']}
    own_peak = 0

    def status(**extra):
        write_json(out / 'status.json', {'state': 'running', 'phase': report['phase'],
                   'completed_trials': len(report['trials']), 'total_trials': total_trials,
                   'elapsed_minutes': (time.monotonic() - started) / 60, 'pid': os.getpid(),
                   'sampled_peak_process_mib': own_peak, **extra})

    def budget():
        if time.monotonic() - started > protocol['max_minutes'] * 60:
            raise RuntimeError('Sweep wall-clock budget exceeded')

    def memory():
        nonlocal own_peak
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        used = max([int(row.split(',')[1]) for row in rows.splitlines() if row.split(',')[0].strip() == str(os.getpid())] or [0])
        own_peak = max(used, own_peak)
        if used > 9728:
            raise RuntimeError('Own process exceeds 9.5 GiB')

    def loader(data, seed, epoch=None):
        return DataLoader(data, batch_size=protocol['batch_size'], shuffle=epoch is not None,
                          num_workers=2, pin_memory=True,
                          generator=torch.Generator().manual_seed(seed + (epoch or 0)))

    @torch.no_grad()
    def evaluate(g, data, seed, prediction_path=None):
        g.graph.eval(); values = {k: [] for k in g.branches}; labels = []; ids = []; ratios = {}
        for c, o, y, keys in loader(data, seed):
            logits, _ = g.forward(c.cuda(), o.cuda(), y.cuda())
            for k in values:
                values[k].append(logits[k].softmax(1).cpu().numpy())
            labels.extend(y.numpy()); ids.extend(keys)
            for group in g.communication_groups:
                stage = group['stage']
                for task in g.branches:
                    x = g.by_name[f'{task}_stage{stage}'].feature_message.current_state
                    dx = g.by_name[f'bridge_s{stage}_{task}_delta'].feature_message.current_state
                    row = ratios.setdefault(f's{stage}_{task}', [0., 0.])
                    row[0] += float(dx.square().sum()); row[1] += float(x.square().sum())
        arrays = {k: np.concatenate(v) for k, v in values.items()}
        if prediction_path:
            np.savez(prediction_path, ids=np.asarray(ids), y=np.asarray(labels), **arrays)
        tasks = {k: classification_metrics(labels, v) for k, v in arrays.items()}
        return {'tasks': tasks, 'mean_task_macro_f1': float(np.mean([v['macro_f1'] for v in tasks.values()])),
                'delta_over_feature_l2': {k: (a / b)**.5 if b else 0. for k, (a, b) in ratios.items()}}

    def trial(identifier, recipe, arm, seed):
        budget(); status(current_trial=identifier, epoch=0)
        directory = out / identifier; directory.mkdir()
        torch.manual_seed(seed); np.random.seed(seed)
        g = PilotGraph(arm['mode'], seed=seed, device='cuda', backbone=recipe.get('backbone', 'resnet18'), cfp_size=224,
                       bridge_stages=tuple(arm['stages']), upsilon=tuple(arm['upsilon']),
                       mesh_references=arm['mesh_reference'], mixer_kernel_size=arm['kernel'], loss_reduction='sum')
        opt = configure_optimizer(g, recipe)
        info = {'recipe': recipe, 'arm': arm, 'seed': seed, 'groups': g.communication_groups,
                'parameters': sum(p.numel() for p in g.graph.parameters()),
                'trainable_parameters': sum(p.numel() for p in g.graph.parameters() if p.requires_grad)}
        write_json(directory / 'model.json', info)
        initial = evaluate(g, val, seed); best = initial; best_epoch = 0
        torch.cuda.reset_peak_memory_stats(); trial_start = time.monotonic()
        for epoch in range(protocol['epochs']):
            g.graph.train()
            for m in g.graph.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    m.eval()
            losses = {k: 0. for k in g.branches}; gradients = {}; clipped = {}; count = 0; steps = 0
            for batch, (c, o, y, _) in enumerate(loader(train, seed, epoch)):
                opt.zero_grad(set_to_none=True)
                g.forward(c.cuda(), o.cuda(), y.cuda()); g.backward()
                norms = clip_task_gradients(g, protocol['clip_max_norm']); opt.step()
                for k, value in norms.items():
                    value = float(value); gradients[k] = gradients.get(k, 0.) + value
                    clipped[k] = clipped.get(k, 0) + int(value > protocol['clip_max_norm'])
                for k in losses:
                    losses[k] += float(g.by_name[k + '_loss'].feature_message.current_state.detach()) * len(y)
                count += len(y); steps += 1; budget()
                if batch % 16 == 0:
                    memory()
            current = evaluate(g, val, seed); memory()
            row = {'trial': identifier, 'phase': report['phase'], 'epoch': epoch + 1,
                   'validation': current, 'train_ce': {k: v / count for k, v in losses.items()},
                   'mean_preclip_gradient_norm': {k: v / steps for k, v in gradients.items()},
                   'clip_fraction': {k: v / steps for k, v in clipped.items()}}
            with (out / 'history.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            if current['mean_task_macro_f1'] > best['mean_task_macro_f1']:
                best = current; best_epoch = epoch + 1
            status(current_trial=identifier, epoch=epoch + 1)
        last = evaluate(g, val, seed, directory / 'last_predictions.npz')
        train_metrics = evaluate(g, train, seed)
        torch.save({'model': g.save_state(), 'epoch': protocol['epochs'], 'source_commit': source['commit'], 'configuration': info}, directory / 'last.pt')
        result = {'configuration': info, 'initial': initial, 'fixed_last': last, 'train_last': train_metrics,
                  'selected': best, 'selected_epoch': best_epoch,
                  'train_validation_f1_gap': {k: train_metrics['tasks'][k]['macro_f1'] - last['tasks'][k]['macro_f1'] for k in g.branches},
                  'seconds': time.monotonic() - trial_start,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2}
        write_json(directory / 'summary.json', result)
        report['trials'][identifier] = result
        write_json(out / 'partial_summary.json', report)
        print(json.dumps({'completed': identifier, 'f1': {k: v['macro_f1'] for k, v in last['tasks'].items()}, 'completed_count': len(report['trials']), 'total': total_trials}), flush=True)
        del opt, g; gc.collect(); torch.cuda.empty_cache()
        return result

    no_bridge = {'id': 'independent', 'mode': 'baseline', 'stages': [], 'upsilon': [1., 1., 1/32],
                 'mesh_reference': {'cfp': [8], 'oct': [4, 4]}, 'kernel': 3}
    def finish():
        report['phase'] = 'complete'
        report['elapsed_minutes'] = (time.monotonic() - started) / 60
        report['sampled_peak_process_mib'] = own_peak
        report['cumulative_008_gpu_minutes'] = protocol.get('prior_gpu_minutes', 0.) + report['elapsed_minutes']
        assert len(report['trials']) == total_trials
        write_json(out / 'summary.json', report); status(state='complete')

    try:
        status()
        if protocol.get('phase_mode') == 'formal':
            report['phase'] = 'seed_confirmation'
            recipe = protocol['recipes'][0]
            for seed in protocol['confirmation_seeds']:
                for arm in protocol['formal_arms']:
                    identifier = f'confirm_{seed}_{arm["id"]}'
                    trial(identifier, recipe, arm, seed)
                    if arm['mode'] != 'baseline':
                        baseline_id = f'confirm_{seed}_independent'
                        intervals = paired_interval(out / baseline_id / 'last_predictions.npz', out / identifier / 'last_predictions.npz')
                        for task in intervals:
                            intervals[task]['delta_macro_f1'] = report['trials'][identifier]['fixed_last']['tasks'][task]['macro_f1'] - report['trials'][baseline_id]['fixed_last']['tasks'][task]['macro_f1']
                        report['comparisons'][identifier] = intervals
            finish()
            return
        for recipe in protocol['recipes']:
            trial('baseline_' + recipe['id'], recipe, no_bridge, protocol['screen_seed'])
        chosen_id = max(report['trials'], key=lambda k: (report['trials'][k]['fixed_last']['mean_task_macro_f1'], -report['trials'][k]['configuration']['trainable_parameters']))
        recipe = report['trials'][chosen_id]['configuration']['recipe']
        baseline = report['trials'][chosen_id]['fixed_last']['tasks']
        report['selections']['baseline'] = chosen_id
        if protocol.get('phase_mode') == 'qualification':
            report['phase'] = 'complete'
            report['elapsed_minutes'] = (time.monotonic() - started) / 60
            report['sampled_peak_process_mib'] = own_peak
            assert len(report['trials']) == total_trials
            write_json(out / 'summary.json', report); status(state='complete')
            return
        report['phase'] = 'bridge_screen'; status(selected_baseline=chosen_id)
        for arm in protocol['bridges']:
            trial('screen_' + arm['id'], recipe, arm, protocol['screen_seed'])
        candidates = [k for k, v in report['trials'].items() if k.startswith('screen_') and v['configuration']['arm']['mode'] == 'radon']
        def candidate_rank(k):
            tasks = report['trials'][k]['fixed_last']['tasks']
            return (min(tasks[t]['macro_f1'] - baseline[t]['macro_f1'] for t in ['cfp', 'oct']), report['trials'][k]['fixed_last']['mean_task_macro_f1'])
        candidates = sorted(candidates, key=candidate_rank, reverse=True)[:2]
        report['selections']['candidates'] = candidates
        top = report['trials'][candidates[0]]['configuration']['arm']
        confirm_arms = [no_bridge] + [report['trials'][k]['configuration']['arm'] for k in candidates]
        confirm_arms += [dict(top, id='matched_' + mode, mode=mode) for mode in ['self', 'scrambled']]
        report['phase'] = 'seed_confirmation'
        for seed in protocol['confirmation_seeds']:
            for arm in confirm_arms:
                identifier = f'confirm_{seed}_{arm["id"]}'
                trial(identifier, recipe, arm, seed)
                if arm['mode'] != 'baseline':
                    baseline_id = f'confirm_{seed}_independent'
                    intervals = paired_interval(out / baseline_id / 'last_predictions.npz', out / identifier / 'last_predictions.npz')
                    for task in intervals:
                        intervals[task]['delta_macro_f1'] = report['trials'][identifier]['fixed_last']['tasks'][task]['macro_f1'] - report['trials'][baseline_id]['fixed_last']['tasks'][task]['macro_f1']
                    report['comparisons'][identifier] = intervals
        report['phase'] = 'complete'
        report['elapsed_minutes'] = (time.monotonic() - started) / 60
        report['sampled_peak_process_mib'] = own_peak
        assert len(report['trials']) == total_trials
        write_json(out / 'summary.json', report); status(state='complete')
    except Exception:
        status(state='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['data', 'output', 'protocol', 'lock']:
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args()
    with open(args.lock, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args)
