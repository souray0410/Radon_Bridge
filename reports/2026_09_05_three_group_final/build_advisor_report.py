"""Vector, English advisor report from accepted research outputs only."""
import json,math,statistics,hashlib,csv,argparse
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,Color,white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

parser=argparse.ArgumentParser();parser.add_argument('--workspace-root',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
ROOT=args.workspace_root.resolve();DIR=ROOT/'output/radon_bridge_learned_channel_final';DATA=json.loads((DIR/'results.json').read_text());CENTER=json.loads((ROOT/'output/radon_bridge_centered_final/results.json').read_text());OLD=json.loads((ROOT/'output/radon_bridge_three_seed_final/results.json').read_text())
ROWS=DATA['rows'];SEEDS=[3416,3417,3418];RHOS=[.0625,.125,.25];BRANCHES=['cfp','oct','mean'];MAIN=['svd_radon','centered_svd_radon','channel_svd_radon'];COLORS={'svd_radon':'#007E87','centered_svd_radon':'#7159A5','channel_svd_radon':'#C33E67','learned':'#B56D37','qr_radon':'#A38B4F','no_bridge':'#647380'}
NAMES={'svd_radon':'Uncentered SVD','centered_svd_radon':'Centered-fit SVD','channel_svd_radon':'Learned channel','learned':'Learned CM','qr_radon':'Random QR','no_bridge':'No bridge'}
INK='#183442';MUTED='#59707C';LIGHT='#E6EEF1';BG='#F7FAFB'
lookup={(r['seed'],r['arm'],r['rho']):r for r in ROWS}
assert len(ROWS)==len(lookup)==129
for r in ROWS:
 assert r['summary']['converged_by_policy'] and r['summary']['stop_reason']=='validation_plateau' and not r['summary']['test_used'] and r['diagnostic']['passed']
 assert abs(r['scores']['mean']-(r['scores']['cfp']+r['scores']['oct'])/2)<1e-9
pdfmetrics.registerFont(TTFont('Arial','/System/Library/Fonts/Supplemental/Arial.ttf'));pdfmetrics.registerFont(TTFont('Arial-Bold','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
W,H=1152,648;OUT=ROOT/'output/pdf/Radon_Bridge_Three_Group_Advisor_Report.pdf';c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('R&B (Radon Bridge) - Three-group paired study');c.setAuthor('Souray Meng');PAGE=0

def text(x,y,s,size=13,color=INK,bold=False):
 c.setFillColor(HexColor(color));c.setFont('Arial-Bold' if bold else 'Arial',size);c.drawString(x,H-y-size*.8,str(s))
def para(x,y,s,width=1056,size=15,color=INK):
 style=ParagraphStyle('body',fontName='Arial',fontSize=size,leading=size*1.4,textColor=HexColor(color))
 p=Paragraph(s.replace('R&B', 'R&amp;B'),style);_,height=p.wrap(width,H);p.drawOn(c,x,H-y-height);return height

def line(x1,y1,x2,y2,color=LIGHT,width=1):
 c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x1,H-y1,x2,H-y2)
def box(x,y,w,h,fill=BG):
 c.setFillColor(HexColor(fill));c.roundRect(x,H-y-h,w,h,8,stroke=0,fill=1)
def page(title,subtitle=''):
 global PAGE
 if PAGE:c.showPage()
 PAGE+=1;c.setFillColor(white);c.rect(0,0,W,H,stroke=0,fill=1)
 text(48,24,'R&B  /  RADON BRIDGE',10,MUTED,True);text(48,53,title,min(25,25*1056/max(1056,pdfmetrics.stringWidth(title,'Arial-Bold',25))),INK,True)
 if subtitle:para(48,91,subtitle,size=12,color=MUTED)
 line(48,610,1104,610);text(48,621,'Development-set exploration  |  1,264 train / 296 development participants  |  No test data',9,MUTED)
 text(1050,621,f'{PAGE:02d}',10,MUTED)

def table(headers,rows,x=48,y=145,widths=None,rowh=28,size=12,colors=None):
 if widths is None:widths=[1056/len(headers)]*len(headers)
 xends=[x]
 for w in widths:xends.append(xends[-1]+w)
 c.setFillColor(HexColor(INK));c.rect(x,H-y-rowh,sum(widths),rowh,fill=1,stroke=0)
 for j,h in enumerate(headers):text(xends[j]+8,y+8,h,size,'#FFFFFF',True)
 for i,row in enumerate(rows):
  yy=y+(i+1)*rowh
  if i%2==0:c.setFillColor(HexColor(BG));c.rect(x,H-yy-rowh,sum(widths),rowh,fill=1,stroke=0)
  for j,v in enumerate(row):
   col=colors[i] if colors and j==0 else INK
   assert pdfmetrics.stringWidth(str(v),'Arial',size)<=widths[j]-13,(PAGE,j,str(v),widths[j])
   text(xends[j]+8,yy+(rowh-size)/2,v,size,col)
  line(x,yy+rowh,xends[-1],yy+rowh,LIGHT,.5)
 return y+(len(rows)+1)*rowh

def rho(r):return '1/'+str(round(1/r))
def group(arm):
 for prefix,label in [('channel_','L'),('centered_','C')]:
  if arm.startswith(prefix):return label
 return 'U' if arm.startswith('svd_') else '-'
def mechanism(arm):
 if arm.endswith('svd_radon'):return 'Radon'
 if arm.endswith('svd_self'):return 'Self only'
 if arm.endswith('svd_scrambled'):return 'Scrambled'
 if arm.endswith('svd_resample'):return 'Resampling'
 return NAMES.get(arm,arm)
def rows_for(arm,r):return [lookup[s,arm,None if arm=='no_bridge' else r] for s in SEEDS]
def stat(arm,r,b='mean'):
 v=[x['scores'][b] for x in rows_for(arm,r)];return statistics.mean(v),statistics.stdev(v)
def fmtstat(arm,r,b='mean'):
 m,sd=stat(arm,r,b);return f'{m:.2f} +/- {sd:.2f}'
def scatter(x,y,color,shape='o',size=4):
 c.setFillColor(HexColor(color));c.setStrokeColor(HexColor(color))
 if shape=='s':c.rect(x-size,H-y-size,2*size,2*size,fill=1,stroke=0)
 elif shape=='t':
  p=c.beginPath();p.moveTo(x,H-y+size);p.lineTo(x-size,H-y-size);p.lineTo(x+size,H-y-size);p.close();c.drawPath(p,fill=1,stroke=0)
 else:c.circle(x,H-y,size,fill=1,stroke=0)

def plot(x,y,w,h,series,title,ylim=(60,77),baseline=None,sd=True):
 # series: label, color, values, uncertainties
 text(x+35,y,title,15,INK,True);left=x+38;top=y+29;pw=w-53;ph=h-90;lo,hi=ylim
 X=lambda i:left+pw*(.08+.84*i/2);Y=lambda z:top+ph*(hi-z)/(hi-lo)
 ticks=range(math.ceil(lo),math.floor(hi)+1,2)
 for t in ticks:line(left,Y(t),left+pw,Y(t),LIGHT,.6);text(x,Y(t)-5,str(t),10,MUTED)
 line(left,top,left,top+ph,MUTED,.8);line(left,top+ph,left+pw,top+ph,MUTED,.8)
 if baseline is not None:
  c.setDash(2,3);line(left,Y(baseline),left+pw,Y(baseline),'#647380',1);c.setDash()
 for si,(label,color,vals,errs) in enumerate(series):
  for i,v in enumerate(vals):
   if i:line(X(i-1),Y(vals[i-1]),X(i),Y(v),color,2)
   if sd:
    e=errs[i];line(X(i),Y(v-e),X(i),Y(v+e),color,1);line(X(i)-4,Y(v-e),X(i)+4,Y(v-e),color,1);line(X(i)-4,Y(v+e),X(i)+4,Y(v+e),color,1)
   scatter(X(i),Y(v),color,['o','s','t'][si%3])
 for i,r in enumerate(RHOS):
  text(X(i)-32,top+ph+10,'rho '+rho(r),11,MUTED);text(X(i)-34,top+ph+29,f'r{int(256*r)} / h{int(8192*r)}',9,MUTED)

def legend(items,x=58,y=550,step=330):
 for i,(label,color) in enumerate(items):line(x+i*step,y+6,x+i*step+22,y+6,color,2);scatter(x+i*step+11,y+6,color,['o','s','t'][i%3]);text(x+i*step+30,y,label,12)

def forest(x,y,w,h,entries,title):
 # entries: label, observed, lower, upper, color
 text(x,y,title,15,INK,True);labelw=148;left=x+labelw;pw=w-labelw-10;top=y+35;dy=(h-65)/len(entries)
 limit=6;X=lambda z:left+pw*(z+limit)/(2*limit)
 for t in range(-limit,limit+1,2):line(X(t),top-8,X(t),top+len(entries)*dy,LIGHT,.6);text(X(t)-5,top+len(entries)*dy+10,str(t),9,MUTED)
 line(X(0),top-8,X(0),top+len(entries)*dy,MUTED,1)
 for i,(label,d,lo,hi,color) in enumerate(entries):
  yy=top+i*dy+dy/2;text(x,yy-5,label,11);line(X(lo),yy,X(hi),yy,color,1.8);scatter(X(d),yy,color,size=3)

def heat(x,y,w,h,values,title,labels,vmin=0,vmax=1):
 text(x,y,title,14,INK,True);lw=102;cw=(w-lw)/2;rh=(h-50)/len(labels)
 for j,b in enumerate(['CFP','OCT']):text(x+lw+j*cw+cw/2-13,y+25,b,11,MUTED)
 for i,label in enumerate(labels):
  yy=y+46+i*rh;text(x,yy+rh/2-5,label,10)
  for j in range(2):
   v=values[i][j]
   if v is None:fill='#E6EEF1';labelv='N/A'
   else:
    t=max(0,min(1,(v-vmin)/(vmax-vmin)));fill=Color(.92-.80*t,.96-.47*t,.97-.44*t);labelv=f'{v:.3f}'
   c.setFillColor(HexColor(fill) if isinstance(fill,str) else fill);c.rect(x+lw+j*cw,H-yy-rh+1,cw-2,rh-2,fill=1,stroke=0)
   text(x+lw+j*cw+10,yy+rh/2-5,labelv,11,'#102F3B' if v is None or v<.6*vmax else '#FFFFFF')

# 1
page('Three-group paired study: complete exploratory results','Souray Meng  |  05 September 2026  |  Native CFP and OCT task networks connected through R&B')
for i,(big,small) in enumerate([('129','accepted stage-two results'),('3','training seeds'),('3 x 3 x 4','groups x widths x mechanisms')]):
 box(48+360*i,146,336,103);text(66+360*i,163,big,31,MAIN and COLORS[MAIN[i]],True);text(66+360*i,211,small,13)
para(52,283,'<b>1. Fixed uncentered SVD remains a strong reference.</b> It reaches 68.83-69.47% mean F1 across the three widths; the smallest width has the lowest observed seed SD among its configurations.',size=19)
para(52,372,'<b>2. Centered fitting does not reproduce the first-seed gain consistently.</b> Wider configurations have slightly higher means, but substantially larger seed variation.',size=19)
para(52,461,'<b>3. Learning the channel encoder and decoder is feasible, but not a demonstrated improvement.</b> Mean F1 rises with width and remains below uncentered SVD at every tested width.',size=19)
#2
page('One comparison standard, three explicit groups','108 main results plus 21 reference results; shared baselines are counted once.')
table(['Group','Channel transform','Mechanisms','Seeds x widths','Results'],[
 ['U: uncentered SVD','Fixed Q from E[xxT]','Radon / self / scrambled / resample','3 x 3','36'],
 ['C: centered-fit SVD','Fixed Q from E[xxT] - mean meanT','Same four mechanisms','3 x 3','36'],
 ['L: learned channel','Learned encoder + decoder','Same four mechanisms','3 x 3','36'],
 ['References','No bridge / learned CM / random QR','Shared baseline / standard Radon','3 + 9 + 9','21']],widths=[185,285,285,191,110],rowh=38,size=12)
para(48,366,'<b>Fixed:</b> stage3, M32, S64; rho = 1/16, 1/8, 1/4 gives r = 16, 32, 64 and per-source h = 512, 1024, 2048. Backbone LR 6e-5; head/bridge LR 1e-4. Batch 16, AdamW, weight decay 0.01; separate branch/bridge gradient clipping at 5.',size=15)
para(48,456,'<b>Training:</b> full native-network updates and BN updates. Shared independently trained parent checkpoints within each seed; same data order. At least 8 epochs; plateau after 6 epochs without improvement &gt;0.001; LR x0.3 after 3 stagnant epochs. The 60-epoch cap is not a completion criterion.',size=15)
para(48,554,'<b>Selection:</b> one common stage-two checkpoint by mean of CFP/OCT macro-F1. All 129 results reached the prespecified plateau.',size=13)
#3
page('A shared geometric bridge, different channel maps','All bridge paths are linear at fixed parameters; the native task networks remain nonlinear and trainable.')
steps=[('Native feature','C = 256'),('Channel encoder','C -> r'),('Radon geometry','r x M x S'),('Linear mixer','kernel 3, no bias'),('Ordinary return','direct backprojection'),('Channel decoder','r -> C')]
for i,(a,b) in enumerate(steps):
 x=48+i*181;box(x,149,166,91);text(x+12,166,a,13,INK,True);text(x+12,198,b,12,MUTED)
 if i<5:text(x+166,182,'>',16,MUTED)
para(48,259,'Decoder output is a residual, added to the original native feature at its original MHD Node ID. Each branch retains its own backbone, head, cross-entropy loss and predictions.',size=15)
table(['Group','Encoder / decoder','Trainable bridge modules'],[
 ['U','Q^T / Q; fixed uncentered basis','Only the middle bridge convolution'],
 ['C','Q^T / Q; fixed centered-fit basis','Only the middle bridge convolution'],
 ['L','E / D; initialize Q^T / Q from QR','Encoder, middle convolution and decoder']],y=330,widths=[125,460,471],rowh=35,size=13)
para(48,493,'<b>Important distinction:</b> C subtracts the training channel mean only when fitting Q. Its runtime never subtracts or adds that mean. L starts from the same QR directions as the random reference, but E and D train independently; they need not remain orthogonal or transposes.',size=15)
para(48,566,'Learned CM is a separate legacy reference: it compresses the flattened C x M channels after Radon.',size=12,color=MUTED)
#4
page('Main comparison: performance across communication widths','Macro-F1 (%): three-seed mean +/- sample SD. Error bars show seed variation; dotted lines show no bridge.')
for j,b in enumerate(BRANCHES):
 series=[(NAMES[a],COLORS[a],[stat(a,r,b)[0] for r in RHOS],[stat(a,r,b)[1] for r in RHOS]) for a in MAIN]
 plot(48+j*358,144,341,365,series,b.upper(),ylim=(62,76),baseline=stat('no_bridge',.125,b)[0])
legend([(NAMES[a],COLORS[a]) for a in MAIN],y=529)
para(48,570,'A narrow peak is not required for validity. Uncentered SVD is comparatively flat across rho; learned-channel performance increases over the tested range. Neither pattern establishes a theoretical ceiling or a universal monotonic law.',size=12,color=MUTED)
#5
page('Main numerical results: branches and baseline gains','F1 in percent; +/- is the sample SD across three seeds. Baseline mean F1: 65.02%.')
rr=[];cc=[]
for r in RHOS:
 for a in MAIN:
  rr.append([rho(r),NAMES[a],fmtstat(a,r,'cfp'),fmtstat(a,r,'oct'),fmtstat(a,r),f'{stat(a,r)[0]-stat("no_bridge",r)[0]:+.2f}'])
  cc.append(COLORS[a])
table(['rho','Group','CFP','OCT','Mean','Gain (pp)'],rr,widths=[76,220,210,210,210,130],rowh=36,size=13)
para(48,543,'Centered fitting has the highest descriptive mean at rho 1/8 and 1/4. That does not establish superiority: the advantage is small relative to seed variation and the paired intervals cross zero.',size=14)
#6
page('Replication: show each training seed separately','Mean macro-F1 (%); dotted lines show no bridge. Seeds 3416/3417 informed selection; 3418 is a training repeat, not independent test data.')
for j,s in enumerate(SEEDS):
 series=[(NAMES[a],COLORS[a],[lookup[s,a,r]['scores']['mean'] for r in RHOS],[0]*3) for a in MAIN]
 plot(48+j*358,151,341,362,series,f'Seed {s}',ylim=(63,75),baseline=lookup[s,'no_bridge',None]['scores']['mean'],sd=False)
legend([(NAMES[a],COLORS[a]) for a in MAIN],y=532)
para(48,572,'The early centered-fit peak in seed 3416 does not recur at the same magnitude in 3417 or 3418. Seed-wise directions are retained in all paired tables.',size=12,color=MUTED)
#7-9
for prefix,title in [('', 'Uncentered SVD'),('centered_','Centered-fit SVD'),('channel_','Learned channel')]:
 page(title+': what does the communication mechanism add?','Macro-F1 (%): mean +/- seed SD; dotted lines show no bridge. Matched channel family, rho, parents and training.')
 arms=[prefix+'svd_'+x for x in ['radon','self','scrambled','resample']];cols=[COLORS[prefix+'svd_radon'],'#9674A6','#71828D','#5E879A']
 for j,b in enumerate(BRANCHES):
  series=[(mechanism(a),co,[stat(a,r,b)[0] for r in RHOS],[stat(a,r,b)[1] for r in RHOS]) for a,co in zip(arms,cols)]
  plot(48+j*358,144,341,356,series,b.upper(),ylim=(60,77),baseline=stat('no_bridge',.125,b)[0])
 legend([(mechanism(a),co) for a,co in zip(arms,cols)],y=517,step=266)
 para(48,560,'Self-only uses a block-diagonal mixer mask and has fewer effective parameters. Scrambling preserves channel maps but changes spatial organization. Resampling matches width and operator row norms, not geometry or singular spectrum.',size=13,color=MUTED)
#10
page('Matched group differences: uncertainty stays visible','F1 difference (percentage points), 95% paired intervals; 10,000 shared participant resamples across seeds. No multiplicity correction.')
for j,b in enumerate(BRANCHES):
 entries=[]
 for r in RHOS:
  rs=rho(r)
  v=CENTER['bootstrap']['comparisons']['all3'][rs]['svd_radon'][b];entries.append((f'C - U | rho {rs}',v['delta_pp'],*v['ci95_pp'],COLORS['centered_svd_radon']))
  for control,label in [('svd_radon','L - U'),('centered_svd_radon','L - C')]:
   v=DATA['bootstrap']['comparisons']['all3'][rs]['channel_svd_radon__minus__'+control][b];entries.append((label+' | rho '+rs,v['delta_pp'],*v['ci95_pp'],COLORS['channel_svd_radon']))
 forest(48+j*357,153,342,354,entries,b.upper())
para(48,553,'Intervals condition on the trained models and their development-selected checkpoints. They do not include uncertainty from retraining, and do not turn the three copies of each participant into independent observations.',size=14)
#11
page('Additional references: width, performance and parameter cost','The three primary groups remain separate from learned CM, random QR and the shared no-bridge baseline.')
rr=[]
for r in RHOS:
 for a in ['learned','qr_radon']+MAIN:
  rs=rows_for(a,r);rr.append([rho(r),NAMES[a],fmtstat(a,r),f"{rs[0]['stored_bridge_parameters']/1e6:.3f}",f"{rs[0]['summary']['trainable_parameters']/1e6:.3f}"])
table(['rho','Method','Mean F1 +/- SD','Bridge params (M)','Total params (M)'],rr,widths=[72,256,256,236,236],rowh=27,size=12)
para(48,587,'At matched rho, fixed QR and fixed SVD have the same runtime structure and parameter count; QR avoids data-dependent basis fitting.',size=10,color=MUTED)
#12
page('Computation: distinguish model cost from stopping time','Recorded training time is affected by stopping epochs and shared-GPU load; it is not a controlled single-step speed benchmark.')
rr=[]
for r in RHOS:
 for a in MAIN:
  rs=rows_for(a,r);rr.append([rho(r),NAMES[a],f"{statistics.mean(v['cost']['gpu_seconds']/60 for v in rs):.2f}",f"{min(v['summary']['epochs_ran'] for v in rs)}-{max(v['summary']['epochs_ran'] for v in rs)}",f"{max(v['cost']['sampled_peak_process_mib'] for v in rs):.0f}"])
table(['rho','Group','Mean train GPU-min','Stop epoch range','Peak process MiB'],rr,widths=[80,260,250,226,240],rowh=34,size=13)
traincost=sum(r['cost']['gpu_seconds'] for r in ROWS)/60
para(48,523,f'Accepted 129 stage-two trainings sum to <b>{traincost:.2f} GPU-min</b>. Inherited project accounting totals <b>{DATA["study_summary"]["cumulative_gpu_minutes"]:.2f} GPU-min</b>, including earlier studies and separately recorded preparation, diagnostics and failures.',size=14)
para(48,576,'All accepted stage-two process peaks are below 10 GiB. The learned-channel four-mode preflight peak was 8572 MiB. No GPU-time budget was imposed.',size=11,color=MUTED)
#13-14
for phase,title in [('initial','At initialization'),('selected','After joint training')]:
 page(title+': subspace retention and residual magnitude','Standard Radon, three-seed mean; all 1,264 training participants. Evaluated at bridge input before residual write-back.')
 labels=[g+' '+rho(r) for r in RHOS for g in ['U','C','L']]
 for j,(metric,tt) in enumerate([('retained_energy_ratio','Raw energy retention'),('retained_variance_ratio','Centered variance retention'),('delta_over_input_l2','Residual / input L2')]):
  vals=[]
  for r in RHOS:
   for a in MAIN:
    vals.append([statistics.mean(v['diagnostic']['phases'][phase]['energy'][b+'_stage3'][metric] for v in rows_for(a,r)) for b in ['cfp','oct']])
  vmax=1 if j<2 else max(1,max(max(row) for row in vals))
  heat(48+j*358,146,337,374,vals,tt,labels,vmax=vmax)
 para(48,562,'U/C use the actual fixed orthogonal subspace. L uses an orthogonal diagnostic projector onto the current encoder row space; learned encoder amplification is recorded separately. Current-model features differ after training. Energy retention is not task information.',size=12,color=MUTED)
#15
page('Gradient diagnostics: local structural checks, not global guarantees','Each CE gradient is accumulated with participant weighting over the same 128 training participants in eval mode, then cosine is computed. Table: mean over defined seeds (count/3).')
rr=[]
for r in RHOS:
 for a in MAIN:
  rs=rows_for(a,r);values=[]
  for module in ['cfp_stage3','oct_stage3','bridge_0_exchange.mixer.conv']:
   vs=[v['diagnostic']['phases']['selected']['gradient_groups'][module]['cosine'] for v in rs];defined=[v for v in vs if v is not None]
   values.append('N/A' if not defined else f'{statistics.mean(defined):+.3f} ({len(defined)}/3)')
  rr.append([rho(r),NAMES[a]]+values)
table(['rho','Group','CFP stage3 cosine','OCT stage3 cosine','Mixer cosine'],rr,widths=[70,260,250,250,226],rowh=34,size=13)
para(48,521,'The single-bridge mixer has disjoint destination-row gradient support for the two tasks; zero cosine verifies that structure. It does not prove that native networks or learned channel encoders are free of gradient conflict.',size=14)
para(48,575,'Zero-norm gradients are undefined, never assigned cosine 0. Encoder/decoder gradient norms, cosines and learned spectra are retained in the machine-readable evidence.',size=11,color=MUTED)
#16
page('Interpretation and what the study does not establish','Use a robust working reference; keep the candidate comparison open rather than selecting the largest single score.')
para(48,149,'<b>Supported descriptive findings.</b> Uncentered SVD performs well over all tested widths. Centered fitting has larger seed variation at the wider settings. Learned channel compression is trainable end-to-end, but its average standard-Radon performance does not exceed uncentered SVD at any tested width.',size=18)
para(48,261,'<b>Mechanistic evidence is conditional.</b> Several Radon-versus-control contrasts favor the geometric bridge, while other intervals cross zero and some seed-wise controls win. Self-only is not effectively parameter-matched. Fixed-versus-learned comparisons also change constraints and, for SVD, initialization.',size=18)
para(48,374,'<b>No universal best rho, monotonicity or ceiling is established.</b> A larger nested subspace retains at least as much energy for the same fixed features. That fact does not guarantee improved development F1 after separate end-to-end training.',size=18)
para(48,478,'<b>Evidence boundary.</b> The development set informed task/configuration selection. Three training seeds do not create an external validation cohort. The sealed test set was not read. This report supports research decisions, not clinical or independent-test claims.',size=18)
#17
page('Provenance and acceptance','Immutable experiment directories and source commits; full configuration, prediction and summary hashes are retained locally.')
table(['Stage','Run directory suffix','Training source','Accepted results'],[
 ['Three-seed mechanisms','2026_09_05_09_29_16','359306572437','57'],
 ['Centered-fit supplement','2026_09_05_10_09_52','e0c49f289d6b','+36 = 93'],
 ['Learned-channel supplement','2026_09_05_13_32_31','35675f444cf7','+36 = 129']],widths=[285,280,281,210],rowh=38,size=13)
para(48,338,'<b>Audits:</b> all 129 results reached validation plateau; all referenced diagnostics passed. The 36 three-way configuration comparisons share seed, parent checkpoints, training controls, M/S and rho. New-mode GPU forward/backward/optimizer and full diagnostic preflights passed before deployment.',size=16)
para(48,438,'<b>Evidence files:</b> results.csv lists all 129 scores, gains, parameters, costs, epochs and hashes; paired_differences.csv lists the learned-channel contrasts. The prior centered and uncentered comparison files remain separate and are included in the evidence package. Full results.json preserves diagnostic and checkpoint provenance.',size=16)
para(48,550,'Next pages list every accepted stage-two result. U = uncentered SVD; C = centered-fit SVD; L = learned channel. F1 values are percentages. Best/stop are epoch numbers; best 0 denotes the initial parent predictions.',size=12,color=MUTED)
#18-23
order=['no_bridge','learned','qr_radon']+[prefix+'svd_'+suffix for prefix in ['', 'centered_','channel_'] for suffix in ['radon','self','scrambled','resample']]
for seed in SEEDS:
 rs=sorted([r for r in ROWS if r['seed']==seed],key=lambda r:(0 if r['rho'] is None else r['rho'],order.index(r['arm'])))
 assert len(rs)==43
 for part in range(2):
  chunk=rs[part*22:(part+1)*22];page(f'All results: seed {seed} / part {part+1} of 2','U / C / L identify the three main channel families. Reference rows are explicitly named. No failed or capped trial is counted.')
  rr=[]
  for r in chunk:
   s=r['summary'];a=r['arm'];label=(group(a)+' / '+mechanism(a)) if group(a)!='-' else mechanism(a);cost=r['cost']
   rr.append(['-' if r['rho'] is None else rho(r['rho']),label]+[f"{r['scores'][b]:.2f}" for b in BRANCHES]+[f"{s['selection']['joint']['best_epoch']} / {s['epochs_ran']}",f"{r['stored_bridge_parameters']/1e6:.3f}",f"{r['effective_bridge_parameters']/1e6:.3f}",f"{cost['gpu_seconds']/60:.2f}",str(cost['sampled_peak_process_mib'])])
  table(['rho','Method / mechanism','CFP','OCT','Mean','Best / stop','Bridge M','Effective M','GPU-min','Peak MiB'],rr,y=135,widths=[65,239,75,75,75,110,101,106,102,108],rowh=19.5,size=10.5)
  para(48,594,'Exact result IDs, full hashes, total native-plus-bridge parameters and baseline gains are available in the accompanying CSV.',size=9,color=MUTED)
c.save()
# Sanitized summary for GitHub: aggregate/seed metrics and provenance only; no participant IDs or predictions.
public={'project':DATA['project'],'results':129,'seeds':SEEDS,'rhos':RHOS,'report_pdf':OUT.name,'pages':PAGE,'source_commit':DATA['protocol']['source_commit'],
 'statistics':DATA['statistics'],'paired_learned_channel':DATA['bootstrap'],'paired_centering':CENTER['bootstrap'],'paired_uncentered':OLD['bootstrap'],
 'rows':[{k:r[k] for k in ['id','seed','arm','rho','scores','stored_bridge_parameters','effective_bridge_parameters','result_hashes']}|{'best_epoch':r['summary']['selection']['joint']['best_epoch'],'stop_epoch':r['summary']['epochs_ran'],'cost':r['cost']} for r in ROWS],
 'all 129_plateau':True,'test_used':False,'cumulative_project_gpu_minutes':DATA['study_summary']['cumulative_gpu_minutes']}
(DIR/'ADVISOR_SUMMARY.json').write_text(json.dumps(public,indent=2))
print(json.dumps({'pdf':str(OUT),'pages':PAGE,'results':len(ROWS),'all_plateau':True,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}))
