"""Copy and verify the authorized R&B history; retire only after restore acceptance."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

STUDY='2026_09_06_14_05_08'
SOURCE=Path('/data/mengh/RadonBridge')
ARCHIVE=Path('/backup/mengh/RadonBridge/archive')/STUDY

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def inventory():
    entries=[];tops=[]
    for group in ('runs','cache','weights','accounting','reports','retired_runs'):
        root=SOURCE/group
        if not root.exists():continue
        if group=='runs':
            roots=[p for p in sorted(root.iterdir()) if p.name!=STUDY]
            tops=[str(p.relative_to(SOURCE)) for p in roots]
        else:roots=[root]
        for p in roots:
            pending=[p]
            while pending:
                q=pending.pop();s=q.lstat();rel=str(q.relative_to(SOURCE))
                if q.is_symlink():entries.append(dict(path=rel,kind='symlink',target=os.readlink(q)))
                elif q.is_dir():
                    entries.append(dict(path=rel,kind='directory'));pending.extend(sorted(q.iterdir(),reverse=True))
                elif q.is_file():entries.append(dict(path=rel,kind='file',size=s.st_size,mtime_ns=s.st_mtime_ns))
                else:raise RuntimeError('Unsupported file type: '+rel)
    return entries,tops

def run(retire=False):
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'.active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A controller's lock alone is insufficient after its crash.
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=proc.read_bytes().split(b'\0')
            except OSError:continue
            if b'radonbridge.experiment' in args or b'radonbridge.diagnostics' in args:
                raise RuntimeError('R&B worker survives; archive blocked')
        manifest=ARCHIVE/'archive_manifest.json'
        if retire:
            d=json.loads(manifest.read_text());a=json.loads((ARCHIVE/'restore_acceptance.json').read_text())
            assert d['state']=='verified' and a['passed'] and a['archive_manifest_sha256']==sha(manifest)
            assert a['all_runtime_dependencies_resolved'] and a['strict_load_and_predictions']
            if (ARCHIVE/'retirement.json').exists():
                assert json.loads((ARCHIVE/'retirement.json').read_text())['state']=='complete';return
            progress_path=ARCHIVE/'retirement_progress.json'
            retired=json.loads(progress_path.read_text()).get('retired',[]) if progress_path.exists() else []
            for rel in d['retire_candidates']:
                if rel in retired:continue
                p=SOURCE/rel
                assert p.parent==SOURCE/'runs' and p.name!=STUDY
                # Refuse any additions or changed links since the accepted inventory.
                known={e['path']:e for e in d['entries']}
                pending=[p] if p.exists() or p.is_symlink() else []
                while pending:
                    current=pending.pop();relative=str(current.relative_to(SOURCE))
                    assert relative in known,'Unarchived new path: '+relative
                    expected=known[relative]
                    if current.is_symlink():
                        assert expected['kind']=='symlink' and os.readlink(current)==expected['target']
                    elif current.is_dir():
                        assert expected['kind']=='directory';pending.extend(current.iterdir())
                    else:assert expected['kind']=='file'
                # Source metadata has not changed since its full hash verification.
                for entry in d['entries']:
                    if entry['kind']=='file' and (entry['path']==rel or entry['path'].startswith(rel+'/')):
                        source_file=SOURCE/entry['path']
                        # An interrupted rmtree may already have removed part of this verified tree.
                        if not source_file.exists():
                            assert sha(ARCHIVE/'history'/entry['path'])==entry['sha256']
                            continue
                        s=source_file.stat()
                        assert s.st_size==entry['size'] and s.st_mtime_ns==entry['mtime_ns']
                if p.is_symlink() or p.is_file():p.unlink()
                elif p.exists():shutil.rmtree(p)
                retired.append(rel)
                write(progress_path,dict(retired=retired,last_retired=rel))
            write(ARCHIVE/'retirement.json',dict(state='complete',retired=d['retire_candidates'],source_cache_preserved=True,LOOK_untouched=True))
            return
        if manifest.exists() and json.loads(manifest.read_text()).get('state')=='verified':return
        entries,tops=inventory();needed=sum(x.get('size',0) for x in entries)
        dest=ARCHIVE/'history';dest.mkdir(exist_ok=True)
        # Existing partial copies may be resumed; maintain a separate safety floor.
        if shutil.disk_usage(ARCHIVE).free<100*1024**3:raise RuntimeError('Archive free space below safety floor')
        write(ARCHIVE/'copy_inventory.json',dict(entries=entries,retire_candidates=tops,logical_bytes=needed))
        roots=tops+[g for g in ('cache','weights','accounting','reports','retired_runs') if (SOURCE/g).exists()]
        for rel in roots:
            p=SOURCE/rel;q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True)
            write(ARCHIVE/'progress.json',dict(state='copying',path=rel))
            subprocess.run(['rsync','-aH','--partial','--',str(p),str(q.parent)+'/'],check=True)
        verified=[];start=time.time()
        for i,e in enumerate(entries):
            p=SOURCE/e['path'];q=dest/e['path'];e=dict(e)
            if e['kind']=='file':
                s=p.stat();assert s.st_size==e['size'] and s.st_mtime_ns==e['mtime_ns']
                e['sha256']=sha(p);assert q.stat().st_size==e['size'] and sha(q)==e['sha256']
            elif e['kind']=='directory':assert q.is_dir()
            else:
                assert q.is_symlink()
                target=(p.parent/e['target']).resolve() if not os.path.isabs(e['target']) else Path(e['target'])
                if target.is_relative_to(SOURCE):
                    mapped=dest/target.relative_to(SOURCE)
                    assert mapped.exists(),str(mapped)
                    rewritten=os.path.relpath(mapped,q.parent)
                    q.unlink();q.symlink_to(rewritten);e['archived_target']=rewritten
                else:e['external_dependency']=str(target)
            verified.append(e)
            if i%50==0:write(ARCHIVE/'progress.json',dict(state='verifying',verified=i,total=len(entries),elapsed_seconds=time.time()-start))
        code=ARCHIVE/'source';code.mkdir(exist_ok=True)
        bundle=code/'RadonBridge.bundle'
        if not bundle.exists():subprocess.run(['git','-C','/home/mengh/RadonBridge','bundle','create',str(bundle),'--all'],check=True)
        write(manifest,dict(state='verified',source=str(SOURCE),archive=str(dest),entries=verified,retire_candidates=tops,
             code_bundle_sha256=sha(bundle),LOOK_untouched=True,raw_UKB_included=False,test_images_opened=False))
        write(ARCHIVE/'progress.json',dict(state='verified_waiting_restore',files=len(entries),manifest_sha256=sha(manifest)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retire',action='store_true');args=p.parse_args();run(args.retire)
