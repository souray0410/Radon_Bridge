import hashlib,json
from pathlib import Path
import pandas as pd
labels=Path('/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv')
old=Path('/data/mengh/RadonBridge/cache/pilot256_128')
audit=json.loads((old/'audit.json').read_text());digest=hashlib.sha256(labels.read_bytes()).hexdigest()
assert digest==audit['labels_sha256']
f=pd.read_csv(labels,dtype={'participant_id':str});previous=pd.read_csv(old/'selected.csv',dtype={'participant_id':str})
assert not f.participant_id.duplicated().any()
joined=previous.merge(f,on='participant_id',suffixes=('_old','_source'),validate='one_to_one')
assert len(joined)==len(previous) and (joined.split_old==joined.split_source).all() and (joined.label_id_old==joined.label_id_source).all()
row=f[f.split=='train'].iloc[0]
export=Path('/data/mengh/LOOK/2026_09_03_08_30_00/dataset');raw=Path('/mnt/ukbiobank/UKB')
paths_ok=all((export/row[e+'_fundus_path']).is_file() and (raw/Path(row[e+'_oct_path']).parent.with_suffix('.zip')).is_file() for e in ['left','right'])
assert paths_ok
report={'labels':str(labels),'labels_sha256':digest,'image_root':str(export),'source_root':str(raw),'counts':{str(k):int(v) for k,v in f.groupby(['split','label_id']).size().items()},'old_manifest_all_rows_labels_and_splits_match':True,'new_to_rb_train':int(sum((f.split=='train')&~f.participant_id.isin(previous.participant_id))),'new_to_rb_validation':int(sum((f.split=='validation')&~f.participant_id.isin(previous.participant_id))),'first_training_pair_sources_exist':paths_ok,'test_images_opened':False,'caveat':'Full validation cohort was used by prior LOOK task scouting; new-to-RB validation is not globally untouched validation.'}
print(json.dumps(report,indent=2))
