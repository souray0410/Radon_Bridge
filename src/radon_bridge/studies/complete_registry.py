"""Scientific reference registry, separate from the shared GPU claim mechanism."""
import json
import math
import sqlite3
from pathlib import Path

from radon_bridge.runtime.state import stable_hash
from radon_bridge.studies.complete_matrix import VERSION, all_positions, candidates, groups


class Registry:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS positions (id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                                                  execution_id TEXT REFERENCES executions(id));
            CREATE TABLE IF NOT EXISTS executions (id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                                                   state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS selections (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        ''')

    def register(self):
        with self.db:
            for p in all_positions():
                value = json.dumps(p, sort_keys=True)
                old = self.db.execute('SELECT payload FROM positions WHERE id=?', (p['id'],)).fetchone()
                if old and old[0] != value:
                    raise ValueError('Logical protocol position changed')
                self.db.execute('INSERT OR IGNORE INTO positions(id,payload) VALUES (?,?)', (p['id'], value))

    def bind(self, position, specification):
        """Bind provenance only; production eligibility still requires the existing runtime gate."""
        required = {'protocol', 'sources', 'parents', 'cohort', 'basis', 'training', 'method',
                    'runtime', 'resource_acceptance', 'test_access'}
        if not required <= set(specification) or specification['protocol'] != VERSION or specification['test_access'] is not False:
            raise ValueError('Incomplete or sealed execution specification')
        for key in ('parents', 'cohort', 'basis', 'runtime', 'resource_acceptance'):
            if not specification[key]:
                raise ValueError('Missing acceptance provenance: '+key)
        # Labels, paths and model source revisions are part of this immutable spec.
        identity = stable_hash(specification)
        payload = json.dumps(specification, sort_keys=True)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT payload,execution_id FROM positions WHERE id=?', (position,)).fetchone()
            if row is None: raise ValueError('Unknown research position')
            if row[1] not in (None, identity): raise ValueError('Cannot replace a bound experiment')
            p = json.loads(row[0])
            for dependency in p['dependencies']:
                if dependency.startswith('public_config_'):
                    accepted = self.db.execute('SELECT id FROM selections WHERE id=?', (dependency,)).fetchone()
                else:
                    accepted = self.db.execute('''SELECT e.id FROM positions p JOIN executions e ON p.execution_id=e.id
                                                  WHERE p.id=? AND e.state='accepted' ''', (dependency,)).fetchone()
                if not accepted: raise ValueError('Dependency not accepted: '+dependency)
            self.db.execute("INSERT OR IGNORE INTO executions VALUES (?,?,'waiting_execution_validation')", (identity, payload))
            self.db.execute('UPDATE positions SET execution_id=? WHERE id=?', (identity, position))
        return identity

    def counts(self):
        return dict(positions=self.db.execute('SELECT COUNT(*) FROM positions').fetchone()[0],
                    bound_positions=self.db.execute('SELECT COUNT(*) FROM positions WHERE execution_id IS NOT NULL').fetchone()[0],
                    executions=dict(self.db.execute('SELECT state,COUNT(*) FROM executions GROUP BY state')))


def select_public(family, source_count, records):
    """All predeclared reference evaluations required; missing scores never disappear."""
    table = candidates(family)
    reference = {g['id'] for g in groups() if g['tuning_reference'] and len(g['sources']) == source_count}
    if not reference: raise ValueError('Unsupported network count')
    by_key = {}
    for row in records:
        key = (row['group'], row['candidate'])
        if key in by_key or key[0] not in reference or not 0 <= key[1] < 32:
            raise ValueError('Unexpected/duplicate candidate evidence')
        if row.get('seed') != 3416 or row.get('test_access') is not False or row.get('family') != family:
            raise ValueError('Selection evidence has wrong role')
        by_key[key] = row
    if set(by_key) != {(g, c) for g in reference for c in range(32)}:
        raise ValueError('Reference search not complete')
    ranked = []; excluded = []
    for index, configuration in enumerate(table):
        rows = [by_key[(g, index)] for g in sorted(reference)]
        if any(r.get('state') != 'accepted' for r in rows):
            # Only predeclared resource infeasibility excludes a candidate. Training
            # failures or unfinished plateaus require resolution before selection.
            if any(r.get('state') not in ('accepted', 'infeasible_before_performance') for r in rows):
                raise ValueError('Unresolved candidate prevents public selection')
            if any(not r.get('resource_receipt_sha256') for r in rows if r['state'] != 'accepted'):
                raise ValueError('Unproven resource exclusion')
            excluded.append(index); continue
        if any(not r.get('plateau') or not r.get('receipt_sha256') for r in rows):
            raise ValueError('Unaccepted training evidence')
        fields = ('mean_macro_f1', 'arithmetic_cost', 'parameters')
        if any(not math.isfinite(r[k]) for r in rows for k in fields):
            raise ValueError('Nonfinite ranking evidence')
        means = {k: sum(r[k] for r in rows)/len(rows) for k in fields}
        ranked.append(((-means['mean_macro_f1'], means['arithmetic_cost'], means['parameters'], stable_hash(configuration)), index, means))
    if not ranked: raise ValueError('No common feasible candidate')
    _, index, means = min(ranked)
    return dict(id=f'public_config_{source_count}_{family}', protocol=VERSION, family=family,
                source_count=source_count, candidate=index, configuration=table[index], means=means,
                excluded_candidates=excluded, evidence_sha256=stable_hash(records), test_access=False)
