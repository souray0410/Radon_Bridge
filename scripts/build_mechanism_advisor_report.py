"""English 16:9 advisor report from accepted 186-result aggregate evidence.

Only stdlib + ReportLab are needed locally. Run the PDF skill artifact marker
before authoring; render every page and inspect before final delivery.
"""
import argparse,csv,json,math
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

W,H=1152,648
NAVY='#183C4A';TEAL='#007E87';GRAY='#596873'

def build(directory,output):
    root=Path(directory);data=json.loads((root/'results.json').read_text());status=json.loads((root/'report_status.json').read_text())
    assert status['complete_trials']==status['total_trials']==186 and not data['test_used']
    rows=data['rows'];assert len(rows)==186 and all(r['summary']['converged_by_policy'] for r in rows)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Do not overwrite a prior report; choose a new reviewed version')
    c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('R&B (Radon Bridge) - Structured Cross-source Communication Benchmark');c.setAuthor('Souray Meng');number=0
    def para(text,x,y,width=1056,size=16,color=NAVY):
        st=ParagraphStyle('body',fontName='Helvetica',fontSize=size,leading=size*1.35,textColor=HexColor(color));p=Paragraph(text,st);_,height=p.wrap(width,H)
        if y+height>596:raise ValueError('Report paragraph exceeds content area')
        p.drawOn(c,x,H-y-height);return height
    def page(title,subtitle=''):
        nonlocal number
        if number:c.showPage()
        number+=1;c.setFillColor(HexColor('#F8FAFB'));c.rect(0,0,W,H,fill=1,stroke=0);c.setFillColor(HexColor(TEAL));c.rect(0,H-10,W,10,fill=1,stroke=0)
        para(title,48,34,size=27);para(subtitle,48,79,size=12,color=GRAY)
        c.setStrokeColor(HexColor('#D8E0E5'));c.line(48,40,W-48,40);c.setFillColor(HexColor(GRAY));c.setFont('Helvetica',9);c.drawString(48,24,'R&B (Radon Bridge) | Development-set controlled benchmark | Not independent clinical validation');c.drawRightString(W-48,24,str(number))
    def bullets(items,y=135):
        for text in items:y+=para(text,58,y,width=1036,size=17)+22
    def table(values,widths,y=135,font=12):
        styles=ParagraphStyle('cell',fontName='Helvetica',fontSize=font,leading=font*1.22)
        items=[[Paragraph(escape(str(v)),styles) for v in row] for row in values]
        t=Table(items,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#DAECEE')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#FFFFFF'),HexColor('#EFF3F5')]),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));_,height=t.wrap(W-96,H)
        if y+height>594:raise ValueError(f'Table overflow on page{number}: {height}')
        t.drawOn(c,48,H-y-height)
    page('Structured cross-source communication','Souray Meng | R&B (Radon Bridge) | Three-seed, three-width controlled benchmark')
    bullets(['<b>186 accepted second-stage results:</b> 129 historical references +57 prespecified supplemental trainings. Existing evidence remains intact.',
        '<b>Four questions:</b> additional value of Radon geometry; dependence on the fixed subspace; direction and correct pairing; comparison with common communication methods.',
        '<b>Research boundary:</b> same glaucoma endpoint, paired CFP/OCT, existing task-adapted native networks. No extra pretraining or basis fitting in this supplement.',
        '<b>Interpretation:</b> controlled mechanism evidence on a historically selected development set. This is one component of a future paper, not complete NBE submission evidence.'])
    page('Networks, communication and shared training','Original MHD V4 stages, task heads, CE losses and Node IDs are preserved.')
    bullets(['CFP input [B,2,3,224,224]; OCT input [B,2,1,32,96,96]. Both complete ResNet18 task networks were separately trained to plateau.',
        'Restore the same per-seed CFP/OCT best checkpoints, insert one stage3 communication adapter, then train all native and adapter parameters. BatchNorm continues updating.',
        'Native learning rate6e-5; heads/communication1e-4. Batch16, AdamW, weight decay0.01; separate gradient clipping at5.',
        'At least8 epochs; plateau after6 epochs without improvement greater than0.001. LR x0.3 after3 stagnant epochs. Sixty epochs is a guard, not convergence.',
        'A common second-stage checkpoint is selected by the mean of the two branch macro-F1 scores. No test data are read.'])
    page('Prespecified experiment matrix','Seeds3416/3417/3418; rho is a dimension ratio. Common baselines do not have a rho axis.')
    table([['Supplement','Configurations','Training tasks'],['Random QR / linear resampling','3 seeds x3 widths','9'],['SVD and QR / directional Radon','2 bases x2 directions x3 seeds x3 widths','36'],['MMTM single-stage adaptation','Reduction4/8; hidden256/128; 3 seeds','6'],['Bidirectional cross-attention','Dimension128/256; 4 heads; 3 seeds','6'],['Reused historical results','Same parents, bases, predictions and converged summaries; hashes verified','129']], [260,680,116],font=13)
    page('What is controlled, and what is not','Matching training conditions does not imply identical capacity or computational cost.')
    bullets(['Radon paths: stage3, M32, S64; rho1/16,1/8,1/4 gives channel rank16/32/64 and per-source mixer width512/1024/2048.',
        'Ordinary communication uses fixed linear resampling with matched forward/return operator row L2 norms. Packing factor32 is not an angle count; rank and singular spectrum are not matched.',
        'Directional training retains both self blocks and masks one cross block. Stored and effective parameters are reported separately; effective capacity is not fully matched.',
        'MMTM and attention retain their required nonlinearities. They are single-stage communication adaptations, not reproductions of complete published task systems. Both predeclared configurations are reported.',
        'Historical search budgets differ. Existing results are not relabeled as new prespecified independent validation.'])
    page('Statistical analysis and decision rules','31 fixed primary comparisons; branch, width and seed strata remain visible.')
    bullets(['Seven mechanism contrasts average each model F1 equally across three seeds and three widths. Twenty-four common-baseline contrasts average across seeds for each fixed basis/width/configuration.',
        '10,000 paired bootstrap draws resample the same296 participants for all models. Calculate model F1 first, then average; do not ensemble across seeds or widths.',
        'Ordinary95% percentile intervals and approximate31-contrast max-|t| simultaneous intervals are both reported. Zero bootstrap variance is marked undefined.',
        'Research reference band +/-1 percentage point: the simultaneous interval supports substantive improvement, substantive decline, practical similarity, or remains unresolved.',
        'Intervals condition on fitted models and selected checkpoints. Three seeds and20 derangements do not increase the independent participant count.'])
    for figure in data['figures']:
        page(figure['title'],'Full vector figure and source data are provided separately; this page preserves the shared plotting convention.')
        image=ImageReader(str(root/figure['file']));iw,ih=image.getSize();width=1080;height=width*ih/iw
        if height>465:height=465;width=height*iw/ih
        c.drawImage(image,(W-width)/2,H-110-height,width,height,mask='auto')
    primary=[d for d in data['bootstrap']['contrasts'] if d['primary']]
    for start in range(0,31,6):
        page('Claim-by-claim evidence and uncertainty',f'Prespecified primary comparisons {start+1}-{min(start+6,31)}; differences are percentage points.')
        values=[['Contrast','Difference','Simultaneous95% interval','Decision']]
        for d in primary[start:start+6]:
            ci=d['simultaneous_ci95_pp'];values.append([d['id'].replace('_',' '),f"{d['difference_pp']:+.2f}",'undefined' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]',d['classification'].replace('_',' ')])
        table(values,[480,100,210,266],font=12)
    page('Direction and correctly paired information','An inference perturbation is a mechanism diagnostic; it is not a randomized clinical trial.')
    bullets(['54 selected Radon models: two fixed bases x three directions x three widths x three seeds.',
        'Compare original pairing, each cross direction disabled, both disabled, each direction mismatched, and both mismatched. Self terms, recipient native features and recipient labels remain unchanged.',
        'Twenty label-independent whole-development derangements are shared across models. Both eyes move together; simultaneous mismatch uses a permutation and its inverse.',
        'Cached pre-bridge features are checked against complete MHD probabilities. Show F1 and probability changes, permutation mean/SD/range, and every seed. Diagnostics never affect training or selection.'])
    page('Data card and biomedical interpretation','UKB paired CFP/OCT; two predictors of the same record-derived glaucoma phenotype.')
    table([['Data role','Participants','Cases / controls'],['Training','1264','632 /632'],['Development','296','148 /148'],['Test','Not accessed','Not applicable'],['Reference standard','UKB record-derived clinical phenotype','Not a new expert-image adjudication'],['Generalizability','Balanced case-control sample','Predictive values/calibration do not imply population performance']], [220,440,396],font=13)
    page('Reporting completeness and outstanding author information','TRIPOD-related mapping is an applicability checklist, not a claim of full clinical validation compliance.')
    table([['Reporting domain','Evidence / location','Remaining need'],['Participants and source','data_card.json; cache audit; counts and available covariates','Source cohort denominator; complete exclusions'],['Outcome and predictors','UKB record phenotype; CFP/OCT; documented feature processing','Verify label timing and clinical reference provenance'],['Model and analysis','Versioned code/configs; initialization; convergence; statistics','No hidden test evaluation claimed'],['Results and uncertainty','186 rows; all seeds; auxiliary metrics;31 simultaneous intervals','Independent external validation'],['Ethics and availability','Restricted participant-level evidence retained in authorized environment','Confirm ethics, UKB application/license and sharing terms']], [220,480,356],font=12)
    page('What would strengthen a future NBE submission?','External evidence should answer new questions; more variants on this development set cannot replace it.')
    bullets(['Independent data: prespecify the model and evaluation, then validate on an appropriate external or independently held-out cohort.',
        'Generalization: extend to additional clinically meaningful tasks, modalities and backbone families under separate protocols.',
        'Biomedical contribution: establish the practical setting in which connecting existing task networks improves a useful clinical or scientific outcome.',
        'Strong task-specific comparators and realistic prevalence: assess reliability, subgroup effects, missing modalities and actual deployment constraints where relevant.',
        'If a primary interval is unresolved, say so. A more complex bridge or a larger number of in-sample ablations is not an automatic remedy.'])
    page('Sources, reproducibility and evidence boundaries','Full code, protocol, environment, aggregate data and hashes accompany this report.')
    bullets(['MMTM: Joze et al., CVPR2020. Paper2*sigmoid excitation; this adapter adds zero output initialization. Public author code differs in its excitation scale.',
        'Attention: Vaswani et al., NeurIPS2017. Standard scaled dot-product multi-head attention adapted for bidirectional stage3 communication; no complete Transformer claim.',
        'NBE: aims and scope, reporting standards, and clinical research policies. Relevant reporting domains are mapped; unverified ethics/license details remain explicit.',
        'Participant-level predictions, features and permutations stay in the authorized environment. GitHub contains aggregate evidence and reproducible source.',
        'Source commit: '+escape(data['source_commit'])])
    for start in range(0,len(rows),12):
        page('Complete second-stage results',f'Accepted results {start+1}-{min(start+12,len(rows))} of186. F1 in percent; fusion is a separate output.')
        values=[['Seed','Method / rho','CFP','OCT','Mean','Fusion','Best/stop']]
        for r in rows[start:start+12]:
            rho='N/A' if r['rho'] is None else '1/'+str(round(1/r['rho']));d=r['summary'];values.append([r['seed'],r['method_name']+' / '+rho,f"{r['scores']['cfp']:.2f}",f"{r['scores']['oct']:.2f}",f"{r['scores']['mean']:.2f}",f"{100*r['late_fusion']['macro_f1']:.2f}",str(d['selection']['joint']['best_epoch'])+'/'+str(d['epochs_ran'])])
        table(values,[60,496,85,85,85,85,160],y=119,font=10)
    c.save();return number

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps({'pages':build(a.directory,a.output),'visual_review':'required'}))
