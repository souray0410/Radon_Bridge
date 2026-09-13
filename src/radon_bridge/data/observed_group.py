"""Explicit common-cohort views with source-specific disease labels; never test."""
import hashlib
import json
import torch
from torch.utils.data import Dataset


def common_cohort(datasets, sources, split, expected_count=None):
    if split not in ('train', 'development'):
        raise ValueError('Training cohort builder refuses test')
    keys = [s['key'] for s in sources]
    if set(keys) != set(datasets) or len(set(keys)) != len(keys):
        raise ValueError('Source/dataset identity mismatch')
    indices = {}
    for key in keys:
        rows = datasets[key].rows
        index = {str(r['id']): i for i, r in enumerate(rows)}
        if len(index) != len(rows): raise ValueError('Duplicate participant')
        indices[key] = index
    ids = sorted(set.intersection(*(set(v) for v in indices.values())))
    if not ids or expected_count is not None and len(ids) != expected_count:
        raise ValueError('Common cohort count does not match locked protocol')
    for pid in ids:
        eyes = None; labels = {}; visits = set()
        for s in sources:
            row = datasets[s['key']].rows[indices[s['key']][pid]]
            if eyes is None: eyes = row['eyes']
            if row['eyes'] != eyes or len(eyes) not in (1, 2) or len(set(eyes)) != len(eyes):
                raise ValueError('Eye membership/order differs across sources')
            d = s['disease']; label = int(row['label'])
            if label not in (0, 1): raise ValueError('Binary task label required')
            if d in labels and labels[d] != label: raise ValueError('Same disease has conflicting labels')
            labels[d] = label
            if 'visit' in row: visits.add(str(row['visit']))
        if len(visits) > 1: raise ValueError('Visit mismatch')
    return ids, indices


class ObservedGroup(Dataset):
    @staticmethod
    def collate_fn(rows):
        return collate_group(rows)

    def __init__(self, datasets, sources, split, participant_ids, *, augment=False):
        ids, indices = common_cohort(datasets, sources, split)
        locked = list(participant_ids)
        if not locked or len(set(locked)) != len(locked) or not set(locked) <= set(ids):
            raise ValueError('Locked common participant manifest cannot be replayed')
        if split != 'train' and augment: raise ValueError('Evaluation augmentation prohibited')
        if any(bool(d.augment) != augment for d in datasets.values()): raise ValueError('Augmentation contract mismatch')
        self.datasets = datasets; self.sources = sources; self.split = split; self.augment = augment
        self.participant_ids = locked; self.indices = indices
        first = sources[0]['key']
        self.counts = [len(datasets[first].rows[indices[first][p]]['eyes']) for p in locked]
        self.manifest_sha256 = hashlib.sha256(json.dumps(dict(split=split, ids=locked), sort_keys=True).encode()).hexdigest()

    def __len__(self): return len(self.participant_ids)

    def __getitem__(self, index):
        pid = self.participant_ids[index]; inputs = {}; labels = {}
        for s in self.sources:
            key = s['key']; row_index = self.indices[key][pid]
            x, y, actual_index = self.datasets[key][row_index]
            row = self.datasets[key].rows[row_index]
            if actual_index != row_index or int(y) != int(row['label']) or len(x) != self.counts[index]:
                raise ValueError('Source sample changed after locking')
            inputs[key] = x; labels[key] = int(y)
        return dict(inputs=inputs, labels=labels, participant_id=pid)

    def set_epoch(self, epoch):
        for d in self.datasets.values(): d.epoch = epoch


def collate_group(rows):
    keys = tuple(rows[0]['inputs'])
    if any(set(r['inputs']) != set(keys) or set(r['labels']) != set(keys) for r in rows):
        raise ValueError('Missing source during collation')
    counts = [len(r['inputs'][keys[0]]) for r in rows]
    if any(len(r['inputs'][key]) != count for r, count in zip(rows, counts) for key in keys):
        raise ValueError('Eye counts disagree')
    return dict(inputs={k: torch.cat([r['inputs'][k] for r in rows]) for k in keys},
                labels={k: torch.tensor([r['labels'][k] for r in rows], dtype=torch.long) for k in keys},
                counts=counts, participant_id=[r['participant_id'] for r in rows])
