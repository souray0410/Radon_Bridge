"""Reproduce existing train/dev inputs and prepare sealed test inputs, with no model inference."""
import argparse,collections,csv,fcntl,hashlib,json,os,re,time,traceback,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from radonbridge.data import read_cfp,CACHE_SCHEMA
from tools.ukb_subset_transfer import DEST,CACHE,EXPECTED,digest,write

def decode(base,row):
 cfps=[];high=[];octs=[];indices=[];counts=[];sizes=[]
 for eye in ('left','right'):
  cfp=base/row[eye+'_fundus_path'];cfps.append(read_cfp(cfp,96));high.append(read_cfp(cfp,224))
  archive=base/Path(row[eye+'_oct_path']).parent.with_suffix('.zip')
  with zipfile.ZipFile(archive) as z:
   members=[v for v in z.infolist() if Path(v.filename).suffix.lower()=='.png']
   ordered=sorted((int(re.search(r'_(\d+)\.[^.]+$',v.filename).group(1)),v) for v in members)
   numbers=np.asarray([v[0] for v in ordered]);assert len(numbers)>=32 and np.all(np.diff(numbers)==1)
   pick=np.round(np.linspace(0,len(ordered)-1,32)).astype(int);planes=[];native=None
   for k in pick:
    with z.open(ordered[k][1]) as f:
     with Image.open(f) as im:
      if native is None:native=im.size
      assert min(im.size)>=32 and im.size==native
      planes.append(np.asarray(im.convert('L').resize((96,96),Image.Resampling.BILINEAR)))
   octs.append(np.stack(planes)[None]);indices.append(numbers[pick]);counts.append(len(numbers));sizes.append(native)
 return dict(cfp=np.stack(cfps),oct=np.stack(octs),schema=np.asarray(CACHE_SCHEMA),slice_numbers=np.stack(indices),native_slice_counts=np.asarray(counts),native_sizes=np.asarray(sizes)),np.stack(high)

def main(root,workers):
 os.umask(0o077);out=root/'prepared_test32';out.mkdir(exist_ok=True)
 with (out/'prepare.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   accepted=json.loads((root/'transfer_status.json').read_text());assert accepted['state']=='verified'
   assert digest(root/'verified_files.jsonl')==accepted['manifest_sha256']
   labels=root/'reference_labels.csv';assert digest(labels)==EXPECTED
   rows=list(csv.DictReader(labels.open()));rows.sort(key=lambda r:(r['split']=='test',r['split'],r['participant_id']))
   test=[r for r in rows if r['split']=='test'];assert len(test)==290
   for r in rows:r['order']=hashlib.sha256(('rb-pilot-3407:'+r['participant_id']).encode()).hexdigest()
   (out/'cfp224').mkdir(exist_ok=True);start=time.time();counts=collections.Counter();native=collections.Counter()
   def one(row):
    arrays,high=decode(root/'data',row);key=row['order']
    if row['split']!='test':
     with np.load(CACHE/(key+'.npz'),allow_pickle=False) as old:
      for k in ('cfp','oct','schema','slice_numbers','native_slice_counts'):assert np.array_equal(arrays[k],old[k]),'Native preprocessing mismatch: '+k
      if 'native_sizes' in old:assert np.array_equal(arrays['native_sizes'],old['native_sizes'])
     assert np.array_equal(high,np.load(CACHE/'cfp224'/(key+'.npy'),allow_pickle=False))
    else:
     target=out/(key+'.npz')
     if target.exists():
      with np.load(target,allow_pickle=False) as old:
       for k,v in arrays.items():assert np.array_equal(v,old[k])
     else:
      temp=target.with_suffix('.partial')
      with temp.open('wb') as f:np.savez(f,**arrays)
      temp.replace(target)
     target=out/'cfp224'/(key+'.npy')
     if target.exists():assert np.array_equal(np.load(target,allow_pickle=False),high)
     else:
      temp=target.with_suffix('.partial')
      with temp.open('wb') as f:np.save(f,high,allow_pickle=False)
      temp.replace(target)
    return row['split'],arrays['native_sizes'].tolist()
   # Exhaust train/development verification before submitting any test decode.
   for group in ([r for r in rows if r['split']!='test'],test):
    with ThreadPoolExecutor(max_workers=workers) as pool:
     for split,sizes in pool.map(one,group):
      counts[split]+=1
      for v in sizes:native['x'.join(map(str,v))]+=1
      if sum(counts.values())%16==0:
       write(out/'preparation_status.json',dict(state='preparing_inputs',counts=dict(counts),total=1850,updated_at=time.time(),elapsed_seconds=time.time()-start,model_inference=False,performance_metrics_computed=False))
   assert counts=={'train':1264,'validation':296,'test':290}
   columns=list(test[0])
   manifest=out/'selected.csv'
   import io
   f=io.StringIO();w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(sorted(test,key=lambda r:r['order']))
   if manifest.exists():assert manifest.read_text()==f.getvalue()
   else:manifest.write_text(f.getvalue())
   hashes=[]
   for r in sorted(test,key=lambda r:r['order']):
    for p in (out/(r['order']+'.npz'),out/'cfp224'/(r['order']+'.npy')):hashes.append(dict(relative=str(p.relative_to(out)),sha256=digest(p)))
   write(out/'cache_files.json',hashes)
   result=dict(state='verified_inputs_only_test_evaluation_still_sealed',counts=dict(counts),all_1560_train_development_reconstructions_pixel_identical=True,
      test_participants=290,test_oct_shape=[2,1,32,96,96],test_cfp_shape=[2,3,224,224],sampling='existing full-span 32 ordered B-scans; same read_cfp crop/pad/resize',
      native_shapes=dict(native),labels_sha256=EXPECTED,transfer_manifest_sha256=accepted['manifest_sha256'],cache_manifest_sha256=digest(out/'cache_files.json'),
      selected_sha256=digest(manifest),implementation_sha256=digest(Path(__file__)),data_implementation_sha256=digest(Path(__import__('radonbridge.data',fromlist=['']).__file__)),
      model_inference=False,performance_metrics_computed=False,model_selection=False,training=False,updated_at=time.time(),elapsed_seconds=time.time()-start)
   write(out/'preparation_status.json',result);print(json.dumps(result),flush=True)
  except BaseException as e:
   write(out/'preparation_status.json',dict(state='needs_attention',error=repr(e),traceback=traceback.format_exc(),updated_at=time.time(),model_inference=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEST);p.add_argument('--workers',type=int,default=2);a=p.parse_args();assert 1<=a.workers<=4;main(a.root,a.workers)
