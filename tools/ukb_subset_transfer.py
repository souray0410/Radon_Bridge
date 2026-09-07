"""Copy exactly the locked R&B cohort's source images; never evaluate models."""
import argparse,collections,csv,fcntl,hashlib,json,os,shutil,time,traceback,zipfile
from pathlib import Path
EXPECTED='eae604091a1094ed70ff4edbcf6e59d00b57b5be124fb4ac5f76ece1beac0d8d'
LABELS=Path('/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv')
CACHE=Path('/data/mengh/RadonBridge/cache/full1264_296')
CFP=Path('/data/mengh/LOOK/2026_09_03_08_30_00/dataset')
OCT=Path('/media/mengh/My Book/UKB')
DEST=Path('/backup/mengh/UKB/RadonBridge_2026_09_06_14_05_08')
RESERVE=(100+64+32)*2**30 # safety + ongoing experiment outputs + preprocessing/cache

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(path,data):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)
def relsafe(s):
 p=Path(s)
 assert not p.is_absolute() and '..' not in p.parts
 return p

def main(dest,copy):
 os.umask(0o077);dest.mkdir(parents=True,exist_ok=True)
 with (dest/'transfer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   assert digest(LABELS)==EXPECTED
   assert json.loads((CACHE/'audit.json').read_text())['labels_sha256']==EXPECTED
   rows=list(csv.DictReader(LABELS.open()));counts=collections.Counter(r['split'] for r in rows)
   assert counts=={'train':1264,'validation':296,'test':290}
   assert len({r['participant_id'] for r in rows})==1850
   selected=list(csv.DictReader((CACHE/'selected.csv').open()))
   chosen={r['participant_id']:r for r in selected}
   for row in rows:
    if row['split']=='test':assert row['participant_id'] not in chosen
    else:
     old=chosen[row['participant_id']]
     assert all(old[k]==v for k,v in row.items())
   jobs=[];missing=[]
   for row in rows:
    for eye in ('left','right'):
     for kind,base,relative in [('cfp',CFP,relsafe(row[eye+'_fundus_path'])),('oct',OCT,relsafe(row[eye+'_oct_path']).parent.with_suffix('.zip'))]:
      src=base/relative
      if not src.is_file():missing.append(dict(path=str(src),kind=kind,split=row['split']));continue
      assert not src.is_symlink(),src
      jobs.append(dict(source=str(src),relative=str(relative),kind=kind,split=row['split'],bytes=src.stat().st_size,source_mtime_ns=src.stat().st_mtime_ns))
   assert len({j['relative'] for j in jobs})==len(jobs),'Unexpected shared source; inspect before deduplication'
   total=sum(j['bytes'] for j in jobs);free=shutil.disk_usage(dest).free
   plan=dict(scope='locked1850_source_subset_not_complete_UKB',labels_sha256=EXPECTED,participants=dict(counts),files=len(jobs),bytes=total,GiB=total/2**30,
       by_kind={k:dict(files=sum(j['kind']==k for j in jobs),bytes=sum(j['bytes'] for j in jobs if j['kind']==k)) for k in ('cfp','oct')},
       source_roots=dict(cfp=str(CFP),oct=str(OCT)),destination=str(dest/'data'),reserve_bytes=RESERVE,available_bytes=free,
       missing=missing,model_evaluation=False,test_image_bytes_transport_included=True,jobs=jobs)
   planpath=dest/'transfer_plan.json'
   if planpath.exists():
    old=json.loads(planpath.read_text());assert old['jobs']==jobs and old['labels_sha256']==EXPECTED,'Source plan changed'
   else:write(planpath,plan)
   summary={k:v for k,v in plan.items() if k not in ('jobs','missing')};summary['missing_count']=len(missing)
   write(dest/'inventory_summary.json',summary);print(json.dumps(summary),flush=True)
   assert not missing,'Missing required images; inspect restricted transfer_plan.json'
   assert len(jobs)==7400
   if not copy:return
   needed=sum(j['bytes'] for j in jobs if not (dest/'data'/j['relative']).exists())
   assert free>=needed+RESERVE,'Insufficient space for subset and declared reserve'
   start=time.time();done=0;done_bytes=0
   ledger=dest/'verified_files.jsonl';verified={}
   if ledger.exists():
    for line in ledger.read_text().splitlines():
     if line.strip():
      r=json.loads(line);verified[r['relative']]=r
   for j in jobs:
    src=Path(j['source']);out=dest/'data'/j['relative'];out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(dest).free>=RESERVE+(0 if out.exists() else j['bytes']),'Space guard reached'
    if j['relative'] in verified:
     prev=verified[j['relative']]
     assert out.is_file() and out.stat().st_size==j['bytes']
     assert digest(out)==prev['sha256'],'Previously verified destination changed'
     assert src.stat().st_size==j['bytes'] and src.stat().st_mtime_ns==j['source_mtime_ns']
    else:
     partial=out.with_name(out.name+'.partial')
     if not out.exists():
      # Resume only a byte-identical prefix; hash the entire source during copy.
      offset=partial.stat().st_size if partial.exists() else 0
      assert offset<=j['bytes']
      h=hashlib.sha256()
      with src.open('rb') as f:
       if offset:
        with partial.open('rb') as old:
         remaining=offset
         while remaining:
          n=min(8*2**20,remaining);a=f.read(n);b=old.read(n);assert len(a)==n and a==b,'Partial prefix mismatch';h.update(a);remaining-=n
       with partial.open('ab') as target:
        for block in iter(lambda:f.read(8*2**20),b''):
         target.write(block);h.update(block)
        target.flush();os.fsync(target.fileno())
      source_sha=h.hexdigest();assert partial.stat().st_size==j['bytes'];assert digest(partial)==source_sha
      partial.replace(out)
     else:
      source_sha=digest(src);assert out.stat().st_size==j['bytes'] and digest(out)==source_sha,'Existing destination mismatch'
     assert src.stat().st_size==j['bytes'] and src.stat().st_mtime_ns==j['source_mtime_ns']
     record=dict(relative=j['relative'],bytes=j['bytes'],sha256=source_sha,kind=j['kind'],split=j['split'],verified_at=time.time())
     with ledger.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
     verified[j['relative']]=record
    done+=1;done_bytes+=j['bytes']
    if done%20==0 or done==len(jobs):
     write(dest/'transfer_status.json',dict(state='copying',pid=os.getpid(),verified_files=done,total_files=len(jobs),verified_bytes=done_bytes,total_bytes=total,updated_at=time.time(),elapsed_seconds=time.time()-start,model_evaluation=False))
   target=dest/'reference_labels.csv'
   if not target.exists():shutil.copyfile(LABELS,target)
   assert digest(target)==EXPECTED
   assert len(list((dest/'data').rglob('*.*')))==len(jobs),'Unexpected destination file inventory'
   result=dict(state='verified',scope=plan['scope'],participants=dict(counts),files=done,bytes=done_bytes,manifest_sha256=digest(ledger),plan_sha256=digest(planpath),labels_sha256=EXPECTED,
     original_sources_unchanged=True,source_and_destination_sha256_matched=True,model_evaluation=False,test_evaluation_authorized_by_this_copy=False,elapsed_seconds=time.time()-start,updated_at=time.time())
   write(dest/'transfer_status.json',result);print(json.dumps(result),flush=True)
  except BaseException as e:
   write(dest/'transfer_status.json',dict(state='needs_attention',error=repr(e),traceback=traceback.format_exc(),updated_at=time.time(),model_evaluation=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=DEST);p.add_argument('--copy',action='store_true');a=p.parse_args();main(a.destination,a.copy)
