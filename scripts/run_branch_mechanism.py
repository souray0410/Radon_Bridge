"""Versioned branch-only successor; never dispatch a learned terminal fusion head."""
import argparse,copy,fcntl,json,signal,subprocess,traceback
from pathlib import Path
from radonbridge.artifacts import SOURCE,STUDY,sha256
from radonbridge.geometry_study import mechanism_catalog,direct_rows,fingerprint
from scripts.geometry_evidence import read,write,dependencies
from scripts.run_integer_experiment import source_hashes
from scripts.run_geometry_mechanism import Queue,scan_completed,interrupted
from scripts.run_task_fusion_benchmark import live_workers


def branch_catalog():
    cat=copy.deepcopy(mechanism_catalog())
    for key in ('equal_parameter_positions','geometry_positions','baseline_positions',
                'augmentation_positions','direct_positions','result_positions'):
        assert cat[key]%2==0
        cat[key]//=2
    cat.update(version='geometry_mechanism_branch_only_v1',protocols=['branch'],
               scope_revision='User withdrew learned terminal fusion after partial outcomes were observed',
               direct_comparisons=28,augmentation_comparisons=4)
    assert cat['direct_positions']==342 and cat['augmentation_positions']==36 and cat['result_positions']==378
    return json.loads(json.dumps(cat))


def assert_branch_rows(rows):
    for row in rows:
        cfg=row['configuration']
        assert row['protocol']=='branch' and not cfg.get('task_fusion'),'Terminal fusion is withdrawn'
        assert cfg.get('selection_metric') in (None,'mean_task_macro_f1'),'Branch selection only'
        assert row['fingerprint']==fingerprint(cfg),'Configuration changed during scope filtering'


class BranchQueue(Queue):
    def __init__(self,root,commit):
        self.root=root;self.commit=commit;self.rows=[];self.segment=None;self.cat=branch_catalog()
        self.snapshot=root/'source_snapshot'
        proof=read(self.snapshot/'snapshot_hashes.json')
        for name,digest in proof.items():assert sha256(self.snapshot/name)==digest
        self.p=read(self.snapshot/'protocol.json')
        self.p.update(source_commit=commit,accepted_source_hashes=source_hashes(),study=self.cat,
                      scope='branch_only',source_snapshot_hashes=proof,
                      protocol_document_sha256=sha256('experiments/geometry_mechanism/BRANCH_ONLY.zh-CN.md'))
        path=root/'protocol.json'
        if path.exists():assert read(path)==self.p,'Branch-only source/protocol is immutable'
        else:write(path,self.p)
        self.status('initializing')

    def initialize(self):
        infrastructure=read(self.snapshot/'infrastructure_gpu_acceptance.json')
        core=lambda values:{k:v for k,v in values.items() if (k.startswith('radonbridge/') and k!='radonbridge/geometry_study.py') or k=='scripts/run_integer_experiment.py'}
        assert infrastructure['passed'] and core(infrastructure['source_hashes'])==core(source_hashes())
        self.parents,self.bases=dependencies()
        expected=[r for r in direct_rows(self.parents,self.bases) if r['protocol']=='branch']
        source=read(self.snapshot/'manifest.json')['rows']
        imported=[r for r in source if r['protocol']=='branch' and r['category']=='direct']
        assert len(imported)==342 and all(r['state']=='accepted' for r in imported)
        assert [(r['id'],r['fingerprint']) for r in imported]==[(r['id'],r['fingerprint']) for r in expected]
        preflight=read(self.snapshot/'structure_preflight.json')
        assert all(preflight['branch_'+r['structure']['id']]['feasible'] for r in imported)
        path=self.root/'manifest.json'
        self.rows=read(path)['rows'] if path.exists() else copy.deepcopy(imported)
        assert_branch_rows(self.rows)
        actual=[r for r in self.rows if r['category']=='direct']
        assert [(r['id'],r['fingerprint']) for r in actual]==[(r['id'],r['fingerprint']) for r in imported]
        for row in self.rows:scan_completed(row)
        self.save()
        lock=dict(source_commit=self.commit,protocol_sha256=sha256(self.root/'protocol.json'),
                  source_manifest_sha256=sha256(self.snapshot/'manifest.json'),
                  direct_positions=342,imported_accepted=342,prespecified_augmentation=36,
                  total_positions=378,terminal_fusion_allowed=False,test_used=False,
                  scope_changed_after_partial_results=True)
        path=self.root/'candidate_lock.json'
        if path.exists():assert read(path)==lock
        else:write(path,lock)

    def save(self):
        assert_branch_rows(self.rows)
        super().save()

    def execute(self,jobs,phase,stage,allow_oom=False):
        assert not (self.root/'drain.request').exists(),'Explicit branch-only drain request'
        for job in jobs:
            assert not job['config'].get('task_fusion'),'Cannot dispatch withdrawn fusion training'
        return super().execute(jobs,phase,stage,allow_oom)

    def run(self):
        self.initialize()
        self.train_rows([r for r in self.rows if r['category']=='direct'],'branch_training')
        self.augment()
        assert len(self.rows)==378
        self.status('reporting')
        from scripts.report_geometry_mechanism import build
        build(self.root)
        self.status('complete',report_complete=True)


def main(root):
    root=Path(root)
    assert root.name=='branch_only' and (root/'source_snapshot/snapshot_hashes.json').exists()
    assert not (root/'drain.request').exists(),'Explicit pause: do not restart'
    with (root/'queue.lock').open('a') as own,(SOURCE/'.active.lock').open('a') as project:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(project,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not live_workers(SOURCE),'Existing project workers must finish first'
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        q=BranchQueue(root,commit)
        try:q.run()
        except BaseException as e:
            q.status('interrupted' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
                     error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(SOURCE/'runs'/STUDY/'branch_only'))
    main(p.parse_args().root)
