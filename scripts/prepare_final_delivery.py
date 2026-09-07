"""Verify completed evidence and prepare a lossless aggregate-only delivery."""
import argparse,csv,gzip,json,shutil,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from radonbridge.artifacts import sha256
from scripts.geometry_evidence import read,write


def pdf_check(path):
    raw=subprocess.check_output(['pdftotext','-bbox',str(path),'-'],stderr=subprocess.DEVNULL)
    tree=ET.fromstring(raw);pages=[e for e in tree.iter() if e.tag.endswith('page')];outside=[];texts=[]
    for page in pages:
        width=float(page.attrib['width']);height=float(page.attrib['height'])
        for word in page.iter():
            if not word.tag.endswith('word'):continue
            texts.append(word.text or '')
            if float(word.attrib['xMin'])<-.5 or float(word.attrib['yMin'])<-.5 or float(word.attrib['xMax'])>width+.5 or float(word.attrib['yMax'])>height+.5:outside.append(word.text)
    if len(pages)!=1 or outside:raise ValueError('PDF text outside page or wrong page count: '+str(path))
    png=path.with_suffix('.png')
    with Image.open(png) as im:
        im.verify()
    with Image.open(png) as im:
        a=np.asarray(im.convert('RGB').resize((300,300)))
        if a.std()<5 or min(im.size)<500:raise ValueError('Blank or unexpectedly small graphic: '+str(path))
    return dict(pdf_sha256=sha256(path),png_sha256=sha256(png),page_count=1,text_within_page=True,raster_nonblank=True),texts


def run(root,out):
    if out.exists():raise ValueError('Use a new delivery directory')
    sources={};audit={};safe=[]
    for name in ('unified_test','sinogram_atlas','fourier_diagnostics'):
        a=read(root/('active_'+name+'.json'));d=Path(a['directory']);state=read(d/'status.json')
        if state['state']!='complete':raise ValueError('Source not complete: '+name)
        report=Path(a.get('report_directory',state.get('report',str(d/'report'))));sources[name]=report
        for file,digest in read(report/'artifact_manifest.json').items():
            if sha256(report/file)!=digest:raise ValueError('Report hash mismatch')
        audit[name]=dict(directory=str(d),status_sha256=sha256(d/'status.json'),original_report_manifest_sha256=sha256(report/'artifact_manifest.json'))
    test=sources['unified_test'].parent;atlas=sources['sinogram_atlas'].parent
    counts={}
    for name,d,queue,expected in [('unified_test',test,'test',855),('sinogram_atlas',atlas,'all',21)]:
        accepted=read(d/queue/'accepted_jobs.json')
        if len(accepted)!=expected:raise ValueError('Acceptance count changed')
        total=0
        for entry in accepted.values():
            p=Path(entry['summary_path']);s=read(p)
            if sha256(p)!=entry['summary_sha256'] or s['state']!='accepted':raise ValueError('Acceptance changed')
            for file,digest in s['prediction_files'].items():
                f=p.parent/file
                if sha256(f)!=digest:raise ValueError('Accepted output hash mismatch')
                total+=1
                if f.name=='artifact_manifest.json':
                    for item,h in read(f).items():
                        if sha256(p.parent/item)!=h:raise ValueError('Restricted atlas array hash mismatch')
                        total+=1
        counts[name]=dict(accepted=expected,verified_output_files=total)
    private=[]
    for p in sorted((atlas/'private_figures').glob('*.pdf')):
        v,texts=pdf_check(p);text=' '.join(texts)
        if 'RESTRICTED' not in text:raise ValueError('Private figure lacks restriction label')
        if '_packets' in p.stem and any(t not in text for t in ('eye0','eye1','channel0','channel1','channel2')):raise ValueError('Private panel labels incomplete')
        private.append(dict(file=p.name,**v))
    if len(private)!=48:raise ValueError('Expected all 48 private case figures')
    out.mkdir(parents=True,mode=0o700);conversion=[];figures=[];pdfs=[]
    forbidden={'participant_id','participant_ids','ids','y','per_participant','bootstrap_indices','permutations'}
    def scan(value):
        if isinstance(value,dict):
            blocked=forbidden&set(value)
            # This published aggregate field is only the count of diagnostic repeats.
            if type(value.get('permutations')) is int and value['permutations']==20:blocked.discard('permutations')
            if blocked:raise ValueError('Restricted field found in public aggregate')
            for v in value.values():scan(v)
        elif isinstance(value,list):
            for v in value:scan(v)
    for name,source in sources.items():
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(source);target=out/name/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.json':scan(read(p))
            if p.suffix=='.csv':
                with p.open() as f:
                    if forbidden&set(next(csv.reader(f))):raise ValueError('Restricted CSV field')
            if p.name=='aggregates.json':
                data=read(p);dest=target.with_suffix('.json.gz')
                with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as f:f.write(p.read_bytes())
                if gzip.decompress(dest.read_bytes())!=p.read_bytes():raise ValueError('Lossy aggregate JSON compression')
                arrays={}
                for phase,groups in data.items():
                    for key,g in groups.items():
                        for field,value in g['arrays'].items():arrays[phase+'/'+key+'/'+field]=np.asarray(value,dtype=np.float64)
                npz=target.with_suffix('.npz');np.savez_compressed(npz,**arrays)
                with np.load(npz,allow_pickle=False) as z:
                    if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):raise ValueError('NPZ roundtrip changed aggregate values')
                conversion.append(dict(original=str(p),original_sha256=sha256(p),gzip=str(dest.relative_to(out)),npz=str(npz.relative_to(out)),arrays=len(arrays),rule='NPZ float64 parsed JSON arrays; gzip decompresses byte-exact original JSON'))
            elif p.name=='artifact_manifest.json':shutil.copy2(p,target.with_name('source_artifact_manifest.json'))
            else:shutil.copy2(p,target)
            if p.suffix=='.pdf':v,_=pdf_check(p);pdfs.append(dict(source=name,file=str(rel),**v))
            if p.suffix=='.png':figures.append((name,p))
    sheets=out/'visual_review';sheets.mkdir();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for start in range(0,len(figures),4):
        sheet=Image.new('RGB',(2400,1900),'#dddddd');draw=ImageDraw.Draw(sheet)
        for i,(name,p) in enumerate(figures[start:start+4]):
            im=Image.open(p).convert('RGB');im.thumbnail((1180,870));x=(i%2)*1200;y=(i//2)*950
            sheet.paste(im,(x+(1200-im.width)//2,y+50));draw.text((x+12,y+12),name+' / '+p.name,fill='black',font=font)
        sheet.save(sheets/f'public_{start//4+1:02d}.png')
    write(out/'acceptance.json',dict(source_audits=audit,accepted_counts=counts,public_figures=pdfs,
        private_figures=dict(count=48,files=private,review='Automated PDF text-bounds, required panel labels and raster integrity on ws02; private pixels not exported'),
        public_visual_review='contact sheets prepared; model review pending',restricted_files_copied=False,aggregate_conversion=conversion))
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps(dict(delivery=str(out),public_pdf_figures=len(pdfs),private_figures=48,compressed_aggregate_files=len(conversion),MiB=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/1024**2)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root,a.output)
