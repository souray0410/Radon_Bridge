"""Prepare the full existing train/development cohort, without opening test images."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from radonbridge.data import prepare
from radonbridge.resolution import main as prepare_resolution


def main():
    root = Path('/data/mengh/RadonBridge/runs/exp008_data')
    root.mkdir(parents=True, exist_ok=True)
    old = Path('/data/mengh/RadonBridge/cache/pilot256_128')
    new = Path('/data/mengh/RadonBridge/cache/full1264_296')
    labels = Path('/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv')
    image_root = '/data/mengh/LOOK/2026_09_03_08_30_00/dataset'
    source_root = '/mnt/ukbiobank/UKB'
    start = time.monotonic()
    def status(state, phase, **extra):
        value = {'state': state, 'phase': phase, 'pid': os.getpid(), 'cache': str(new),
                 'elapsed_minutes': (time.monotonic()-start)/60, 'test_images_opened': False, **extra}
        temp=root/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(root/'status.json')
    if new.exists() or (root/'status.json').exists():
        raise RuntimeError('Existing preparation must be inspected, never overwritten')
    try:
        digest = hashlib.sha256(labels.read_bytes()).hexdigest()
        assert digest == json.loads((old/'audit.json').read_text())['labels_sha256']
        frame = pd.read_csv(labels,dtype={'participant_id':str})
        assert not frame.participant_id.duplicated().any()
        counts = frame.groupby(['split','label_id']).size()
        assert counts['train'].to_dict() == {0:632,1:632}
        assert counts['validation'].to_dict() == {0:148,1:148}
        provenance = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'labels_sha256':digest,'train':1264,'validation':296,'old_train':256,'old_validation':128,
                      'workers':2,'test_images_opened':False,'validation_role':'development; used in previous task selection',
                      'runtime_source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['radonbridge/data.py','radonbridge/resolution.py']}}
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
        new.mkdir();(new/'cfp224').mkdir()
        # These functions only validate existing files, never mutate them.
        for path in old.glob('*.npz'):os.link(path,new/path.name)
        for path in (old/'cfp224').glob('*.npy'):os.link(path,new/'cfp224'/path.name)
        status('running','oct_and_cfp96',train=1264,validation=296)
        prepare(SimpleNamespace(output=str(new),labels=str(labels),train=1264,validation=296,
                                workers=2,image_root=image_root,source_root=source_root))
        status('running','cfp224',train=1264,validation=296)
        prepare_resolution(SimpleNamespace(data=str(new),image_root=image_root,workers=2))
        selected=pd.read_csv(new/'selected.csv',dtype={'participant_id':str})
        previous=pd.read_csv(old/'selected.csv',dtype={'participant_id':str})
        assert len(selected)==1560 and not selected.participant_id.duplicated().any()
        assert set(selected.split)=={'train','validation'}
        overlap=previous.merge(selected,on='participant_id',suffixes=('_old','_new'),validate='one_to_one')
        assert len(overlap)==384 and (overlap.split_old==overlap.split_new).all() and (overlap.label_id_old==overlap.label_id_new).all()
        unchanged=hashlib.sha256((old/'selected.csv').read_bytes()).hexdigest()==json.loads((old/'audit.json').read_text())['selected_sha256']
        assert unchanged
        status('running','cache_fingerprint',train=1264,validation=296)
        fingerprint=hashlib.sha256()
        for key in sorted(selected['order']):
            for path in [new/(key+'.npz'),new/'cfp224'/(key+'.npy')]:
                fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
        status('complete','complete',cache_content_sha256=fingerprint.hexdigest(),train=1264,validation=296,new_train=1008,new_validation=168,
               all_previous_labels_and_splits_preserved=True,original_small_manifest_unchanged=hashlib.sha256((old/'selected.csv').read_bytes()).hexdigest()==json.loads((old/'audit.json').read_text())['selected_sha256'])
    except Exception:
        status('failed','preparation',error=traceback.format_exc())
        raise

if __name__=='__main__':
    with open('/data/mengh/RadonBridge/.data_prepare.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
