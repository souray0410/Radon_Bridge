from pathlib import Path
import csv,json,io,statistics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader,PdfWriter
from xml.sax.saxutils import escape
W,H=1152,648
w=Path.cwd();d=w/'output/radon_bridge_qr_nested_supplement';tmp=w/'.build/rb_final_213';out=w/'output/pdf/Radon_Bridge_Integrated_213_Report.pdf'
A=json.loads((d/'A_QR_mechanism_statistics.json').read_text());B=json.loads((d/'B_joint_width_statistics.json').read_text())
a=list(csv.DictReader((d/'A_single_width_204_results.csv').open()));b=list(csv.DictReader((d/'B_joint_width_27_views.csv').open()))
assert len(a)==204 and len(b)==27 and len(set(x['training_id'] for x in b))==9
index=[]
class Deck:
 def __init__(self,prefix):self.buf=io.BytesIO();self.c=canvas.Canvas(self.buf,pagesize=(W,H));self.n=0;self.prefix=prefix
 def para(self,t,y,size=18,width=1056,x=48):
  p=Paragraph(t,ParagraphStyle('p',fontName='Helvetica',fontSize=size,leading=size*1.32,textColor=HexColor('#183C4A')));_,h=p.wrap(width,500)
  assert y+h<601,(t,y+h)
  p.drawOn(self.c,x,H-y-h);return h
 def page(self,title,sub=''):
  if self.n:self.c.showPage()
  self.n+=1;index.append({'section':self.prefix,'local_page':self.n,'title':title})
  self.c.setFillColor(HexColor('#F8FAFB'));self.c.rect(0,0,W,H,fill=1,stroke=0);self.c.setFillColor(HexColor('#007E87'));self.c.rect(0,H-10,W,10,fill=1,stroke=0)
  self.para(title,30,27);self.para(sub,77,12)
  self.c.setFillColor(HexColor('#596873'));self.c.setFont('Helvetica',9);self.c.drawString(48,24,'R&B (Radon Bridge) | Development-set evidence | '+self.prefix)
 def bullets(self,ls):
  y=130
  for t in ls:y+=self.para(t,y)+21
 def table(self,rows,widths,font=12):
  style=ParagraphStyle('c',fontName='Helvetica',fontSize=font,leading=font*1.2)
  t=Table([[Paragraph(escape(str(x)),style) for x in row] for row in rows],colWidths=widths)
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#DAECEE')),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#FFFFFF'),HexColor('#EFF3F5')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));_,h=t.wrap(W-96,H);assert h<450,h;t.drawOn(self.c,48,H-128-h)
 def save(self):self.c.save();self.buf.seek(0);return PdfReader(self.buf)
f=Deck('Integrated overview')
f.page('R&B: integrated mechanism benchmark','213 training configurations | 231 evaluation views | 27 new trainings completed | 6 September 2026')
f.bullets(['<b>Complete:</b> 186 prior second-stage results +18 QR control trainings +9 joint-width trainings. The latter yield27 views of9 shared checkpoints.', '<b>Study A:</b> QR Radon exceeds self-only by2.63 pp. Its mean advantage over spatial scrambling is only0.18 pp; the interval does not resolve the difference.', '<b>Study B:</b> all9 joint-width trainings select epoch0 after reaching the prespecified plateau. Flat width curves here mean no recovered improvement, not successful robust communication.', '<b>Working conclusion:</b> independently trained noncentered-SVD Radon remains the most balanced configuration in this benchmark. Correct spatial structure has stronger support under SVD than QR.', '<b>Scope:</b> historical development data;296 participants, not231 independent datasets. This is controlled mechanism evidence, not external clinical validation.'])
f.page('How to read this integrated report','The preserved 186-result study and two follow-ups retain separate comparison families.')
f.bullets(['Pages4-54 preserve the186-result stage report: common methods, geometry, direction, pairing, diagnostics,31 primary contrasts and the complete186-row table.', 'Pages55 onward contain the new QR mechanism and joint-width studies, including every new training/view result. All231 evaluation views are also supplied asCSV.', 'Study A and Study B each use their own three-contrast max-|t| family. The original31-comparison family is unchanged.10,000 draws share participant indices.', 'The +/-1 pp research band is a prespecified interpretation convention. An interval excluding zero need not clear the +1 pp substantive-improvement boundary.', 'Original23-page report and the short project-introduction deck remain separate. No historical result, failed attempt or completed study is overwritten.'])
f.page('What the completed follow-ups establish','All effects below average model F1 equally over the same three seeds, three widths and two branches.')
vals=[['Contrast','Effect (pp)','Simultaneous95% interval','Interpretation']]
labels={'qr_radon_minus_self':'QR Radon - self','qr_radon_minus_scrambled':'QR Radon - scrambled','basis_spatial_interaction':'SVD - QR spatial gain','nested_learned_channel_joint_minus_independent':'Joint - independent: learned channel','nested_svd_joint_minus_independent':'Joint - independent: SVD','nested_qr_joint_minus_independent':'Joint - independent: QR'}
for z in A['contrasts']+B['contrasts']:
 ci=z['simultaneous_ci95_pp'];vals.append([labels[z['id']],f"{z['difference_pp']:+.2f}",f'[{ci[0]:+.2f}, {ci[1]:+.2f}]',z['classification'].replace('_',' ')])
f.table(vals,[445,110,235,266],11)
front=f.save()
s=Deck('Supplement A / B')
s.page('Two follow-ups, separate research questions','18 QR controls +9 joint-width trainings; same parents, bases, seeds and development participants.')
s.bullets(['A tests whether cross-source and spatial-structure explanations extend to random QR. Self-only and scrambled Radon use the same per-seed QR basis as standard Radon.', 'B tests a new training regime: three nested widths on the same16 participants, equal mean losses, one optimizer update; the six branch/width F1 values choose one checkpoint.', 'Shared BN starts each width from the same pre-batch buffers; running-stat updates are averaged and the batch counter advances once. This is not pooled-mixture variance.', 'B changes parameter sharing, gradient sources, BN updates, checkpoint selection and work per epoch. A failure of this regime does not prove that every multi-width training algorithm fails.', 'No new pretraining, basis fitting, data, task, backbone or width search. Native parameters and BN remain trainable. Original convergence and memory requirements are preserved.'])
s.page('Study A: communication survives; geometry is basis-dependent','QR Radon, QR self-only and QR scrambling are complete across all nine seed/width cells.')
s.bullets(['QR Radon - self-only: <b>+2.63 pp</b>; simultaneous95% CI[+0.24,+5.02]. All9 QR self controls select epoch0. This supports a positive cross-source contribution under the current protocol.', 'QR Radon - scrambled: <b>+0.18 pp</b>; simultaneous95% CI[-1.12,+1.48]. Current evidence resolves neither a useful geometric advantage nor practical equivalence.', 'Noncentered-SVD Radon exceeds its scrambled counterpart in9/9 cells; average gap1.98 pp. The SVD-minus-QR spatial-gain interaction is+1.79 pp, simultaneous CI[+0.03,+3.56].', 'A stronger SVD spatial gain does not imply an independent semantic alignment or a universal advantage of Radon geometry. The basis and spatial operator act together.', 'The positive QR-self and interaction intervals do not lie entirely above+1 pp; they remain unresolved under the stricter substantive-effect classification.'])
s.page('Study B: flat width curves are a negative result','All nine shared models select epoch0; all27 exported views recover the native baseline predictions.')
s.bullets(['Every joint-width training ran8 epochs and met the unchanged plateau rule. It did not improve the common development selection score above initialization.', 'Mean branch F1 is65.02% across all three compression families. Relative to independent widths: learned channel -2.90 pp; SVD -4.19 pp; QR -2.63 pp.', 'For SVD, the simultaneous interval[-7.17,-1.22] pp supports a substantive decline under the prespecified1 pp reference margin.', 'A zero spread across widths at the selected zero bridge is not evidence that the bridge learned a stable representation. Stability must be interpreted together with useful performance.', 'Do not select a later per-width checkpoint or change the training rule to rescue this result. The negative evidence is retained; the independent-width regime remains the working reference.'])
for fig in json.loads((d/'figures.json').read_text()):
 s.page(fig['title'],fig.get('note','Thin: simultaneous interval; thick: ordinary95% interval. Reference band: +/-1 pp.'))
 im=ImageReader(str(d/fig['file']));iw,ih=im.getSize();ww=min(1080,465*iw/ih);hh=ww*ih/iw;s.c.drawImage(im,(W-ww)/2,H-117-hh,ww,hh,mask='auto')
newa=[x for x in a if x['arm'] in ['qr_self','qr_scrambled']]
assert len(newa)==18
for kind,rows in [('A: all18 new independent-width trainings',newa),('B: all27 views of nine joint-width models',b)]:
 for start in range(0,len(rows),9):
  s.page(kind,f'Rows{start+1}-{min(start+9,len(rows))}; F1 in percent. Best/stop epochs are preserved.')
  vals=[['Seed','Arm / rho','CFP','OCT','Mean','Best / stop']]
  for z in rows[start:start+9]:vals.append([z['seed'],z['arm']+' / 1:'+str(round(1/float(z['rho']))),*[f"{float(z[k]):.2f}" for k in ['cfp_f1_percent','oct_f1_percent','mean_f1_percent']],z['best_epoch']+' / '+z['stop_epoch']])
  s.table(vals,[65,480,115,115,115,166],11)
s.page('Audit trail, limitations and the next decision','Complete aggregate data and figure sources accompany this report; restricted records remain remote.')
s.bullets(['Accepted additions:27 trainings and45 initial/selected diagnostic views, with all training summaries converged by policy.5 structures and11 diagnostic views passed GPU preflight.', 'Total accounted GPU time1560.62 minutes, including the preserved failed preflight attempt. Each project process remained subject to the original10 GiB/card limit; batch16 was unchanged.', 'Training source6ada3a35dc7ea7a740e040c3cbd0623f7248218a. The failed preflight was repaired before production training without widening the numerical tolerance.', 'The overall independent-width ranking favors noncentered SVD on this development benchmark; configuration selection remains exploratory. No new method or parameter search is justified merely to obtain a positive result.', 'Next external questions require separately prespecified cohorts, tasks or models. Ethics, license, source exclusions and label timing remain author-verification items; this report does not assert clinical deployment validity.'])
supp=s.save();old=PdfReader(w/'.build/rb_186_stage_20260906.pdf');writer=PdfWriter();parts=[front,old,supp];total=sum(len(x.pages) for x in parts)
for part in parts:
 for page in part.pages:
  buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(W,H));c.setFont('Helvetica',8);c.setFillColor(HexColor('#596873'));c.drawCentredString(W/2,10,f'Integrated report | {len(writer.pages)+1} / {total}');c.save();buf.seek(0);page.merge_page(PdfReader(buf).pages[0]);writer.add_page(page)
writer.add_metadata({'/Title':'R&B: Integrated213-configuration Mechanism Benchmark','/Author':'Souray Meng'})
with out.open('wb') as f:writer.write(f)
(tmp/'index.json').write_text(json.dumps({'pages':total,'front_pages':len(front.pages),'prior_pages':len(old.pages),'supplement_pages':len(supp.pages),'index':index},indent=2))
print(out,total)
