"""Read-only synthesis of the accepted aggregate evidence; no statistical refit."""
from pathlib import Path
import argparse,csv,json,hashlib,statistics as st,collections,io,re
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter,Transformation

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cjk-font',type=Path,default=Path('/Library/Fonts/Arial Unicode.ttf'));a=p.parse_args()
E=a.evidence;O=a.output;O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((E/'artifact_manifest.json').read_text())
for name,h in manifest.items():assert sha(E/name)==h,name
R=list(csv.DictReader((E/'unified_test/all_references_development_test.csv').open()))
T=json.loads((E/'unified_test/test_statistics.json').read_text());D=json.loads((E/'unified_test/development_statistics.json').read_text())
C={alias['id']:r for r in T['contrasts'] for alias in r['aliases']};DC={alias['id']:r for r in D['contrasts'] for alias in r['aliases']}
assert len(R)==804 and len({r['model_view_id'] for r in R})==777
assert len([r for r in T['contrasts'] if r['primary']])==121
G=collections.defaultdict(list)
for r in R:
 if r['source_group']=='current_direct_and_host' and r['category']=='direct':G[json.loads(r['structure'])['id']].append(r)
mean=lambda rs,key:st.mean(float(r[key]) for r in rs)
fmt=lambda x:f'{x:.2f}'
def value(rs,key):return fmt(100*mean(rs,key))
def contrast(key,label=None):
 r=C[key];ci=r.get('global_simultaneous_ci95_pp');return [label or key,fmt(r['difference_pp']) if r.get('difference_pp') is not None else 'N/E',f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else 'N/E']

pdfmetrics.registerFont(TTFont('Chinese',str(a.cjk_font)))
W,H=960,540;NAVY='#102c43';TEAL='#087f8c';GOLD='#b7791f';GRAY='#516372';BG='#f6f8fb'
out=io.BytesIO();c=canvas.Canvas(out,pagesize=(W,H));pages=[]
figs=[]
for folder in ['unified_test','sinogram_atlas','fourier_diagnostics']:
 figs += [(folder,f) for f in sorted((E/folder).rglob('*.pdf'))]
assert len(figs)==31
TOTAL=16+len(figs)
def paragraph(text,x,y,w,size=14,cn=False,color=NAVY,leading=None):
 style=ParagraphStyle('p',fontName='Chinese' if cn else 'Helvetica',fontSize=size,leading=leading or size*1.35,textColor=colors.HexColor(color),wordWrap='CJK' if cn else None)
 q=Paragraph(re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;)', '&amp;',text),style);_,h=q.wrap(w,1000);assert y-h>=(10 if y==27 else 38),(text[:80],y,h);q.drawOn(c,x,y-h);return y-h
def start(title,section,sub='',cn=False):
 c.setFillColor(colors.HexColor(BG));c.rect(0,0,W,H,fill=1,stroke=0)
 c.setFillColor(colors.HexColor(TEAL));c.rect(0,H-8,W,8,fill=1,stroke=0)
 paragraph('R&B / RADON BRIDGE',36,514,500,10,color=TEAL)
 c.setFont('Helvetica',10);c.setFillColor(colors.HexColor(GRAY));c.drawRightString(924,503,section.upper())
 paragraph(escape(title),36,479,890,26,cn=cn)
 if sub:paragraph(sub,36,436,890,11,cn=cn,color=GRAY)
 pages.append(title)
def end(note='Locked UKB cohort | train 1264 / development 296 / test 290 | 3 training seeds'):
 c.setStrokeColor(colors.HexColor('#d8e0e8'));c.line(36,35,924,35)
 paragraph(escape(note),36,27,820,8,color=GRAY)
 c.setFont('Helvetica',9);c.setFillColor(colors.HexColor(GRAY));c.drawRightString(924,17,f'{len(pages)} / {TOTAL}');c.showPage()
def table(headers,rows,widths,y=395,cn=False,size=13,rowpad=7):
 style=ParagraphStyle('t',fontName='Chinese' if cn else 'Helvetica',fontSize=size,leading=size*1.22,wordWrap='CJK' if cn else None,textColor=colors.HexColor(NAVY))
 def cell(s,head=False):
  stl=ParagraphStyle('th',parent=style,textColor=colors.white if head else colors.HexColor(NAVY))
  return Paragraph(escape(str(s)),stl)
 data=[[cell(h,True) for h in headers]]+[[cell(v) for v in row] for row in rows]
 t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf2f6')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),rowpad),('BOTTOMPADDING',(0,0),(-1,-1),rowpad),('LINEBELOW',(0,0),(-1,0),1,colors.HexColor(TEAL))]))
 _,h=t.wrap(888,1000);assert y-h>=55,(pages[-1],h,y);t.drawOn(c,36,y-h);return y-h
def callout(text,y=95,cn=False):
 paragraph(text,40,y,874,13,cn=cn,color=TEAL)
def bullets(items,y=390,cn=False,w=865,size=16,gap=15):
 for text in items:y=paragraph(text,48,y,w,size,cn=cn)-gap
 return y

start('R&B：统一test与机制研究总报告','Research synthesis','2026_09_06_14_05_08 研究批次 | 完成记录实时核对：2026-09-08',cn=True)
paragraph('有正向性能信号，有可核验的通信机制；<br/>普遍几何优势和临床推广仍未得到确定证据。',40,383,870,25,cn=True,leading=37)
for x,num,label in [(40,'855 / 855','test评价作业接受'),(340,'21 / 21','投影图谱检查点'),(640,'121','锁定的主要比较')]:
 c.setFillColor(colors.white);c.roundRect(x,168,278,94,9,fill=1,stroke=0)
 paragraph(num,x+18,245,240,28,color=TEAL);paragraph(label,x+18,198,240,13,cn=True)
paragraph('855项是评价及诊断作业，不是855次新增训练。包含777个模型视图、24项宿主诊断和54项配对诊断。全量原始汇总CSV与逐图源数据另附；本PDF覆盖关键结论和全部31幅已验收公开图。',40,139,875,14,cn=True)
end('Evidence snapshot 2f9c6c6 | Descriptive synthesis; no new model selection, training or bootstrap comparisons')

start('先看结论：哪些成立，哪些还不能确定','Chinese summary',cn=True)
table(['问题','综合判断'],[
 ['R&B是否有用？','参考配置test分支均值60.73% / 61.73%，无通信58.13%；两分支均有正向均值变化，但不能据此宣称普遍显著。'],
 ['是否胜过普通通信？','SVD/QR匹配组几何收益约+1.46 / +1.43 pp；全局同时区间均跨0，收益大小仍不确定。'],
 ['空间结构是否必要？','空间打乱有时更高；S轴邻接对照也不确定。尚未证明完整Radon几何是收益不可替代的来源。'],
 ['SVD或更大宽度一定更好？','未成立。test中中心化SVD也有较高均值；QR三档均值递增，但不构成一般单调性保证。'],
 ['多桥、冻结、宿主增强呢？','多桥没有明确额外收益；冻结组仍有正向信号；宿主增强训练约+2 pp，但区间仍宽。'],
 ['最明确的功能证据？','选中宿主+R&B模型关闭R&B后显著受损，说明功能依赖。约14 pp删除效应不能写成新增训练收益。'],
 ['现在应做什么？','承接固定机制假设扩大数据和骨干；审计标签与数据角色。继续扫描这批test不能建立独立证据。']
],[225,663],y=409,cn=True,size=14,rowpad=6)
end()

start('Completed evidence and scope','01 / inventory','Reference rows, unique model views and inference jobs are different counting units.')
table(['Evidence source','Reference rows / jobs','Role'],[
 ['Current direct + host + original frozen','384 rows','342 direct + 36 augmentation + 6 frozen'],
 ['Historical final213','231 rows','9 joint-width trainings expand to 27 evaluation views'],
 ['Historical S-axis adjacency','18 rows','Ordered versus permuted S-axis'],
 ['Mechanism completion','72 rows','24 fixed-r geometry + 48 frozen completion'],
 ['Multi-depth bridge study','96 rows','Single / double / triple bridge controls'],
 ['Independent parent checkpoints','3 rows','Original unimodal models'],
 ['After model-view deduplication','804 rows -> 777 views','Same checkpoint + inference configuration reused'],
 ['Unified test queue','777 + 24 + 54 = 855 jobs','All accepted; zero failed / active / pending'],
 ['Post-test descriptive diagnostics','21 atlas; 22 synthetic geometries','21 synthetic feasible; 1 dense-operator guard infeasible']
],[300,210,378],size=12,rowpad=6)
end('The synthetic infeasible geometry is a technical limit, not a poor performance result. Historical evidence is retained.')

start('Two task networks, one structured residual bridge','02 / architecture','Fixed-subspace formulation shown. Stage3 reference: M32 / S64 / k3; native networks remain nonlinear.')
def box(x,y,w,h,title,sub,color=TEAL):
 c.setFillColor(colors.white);c.setStrokeColor(colors.HexColor(color));c.roundRect(x,y,w,h,8,fill=1,stroke=1)
 paragraph(title,x+12,y+h-12,w-24,17,color=color);paragraph(sub,x+12,y+h-43,w-24,12)
box(36,303,235,90,'CFP ResNet18','2D stage3: 256 x 14 x 14')
box(689,303,235,90,'OCT ResNet18','3D stage3: 256 x 8 x 6 x 6')
box(326,290,308,110,'R&B / shared packet format','Channel basis -> Radon -> linear Conv1d -> ordinary return')
for x1,x2 in [(271,326),(634,689)]:
 c.setStrokeColor(colors.HexColor(TEAL));c.setLineWidth(2);c.line(x1,348,x2,348)
paragraph('z_i = A_i Q_i^T X_i<br/>u_i = sum_j K_ij * z_j<br/>Y_i = X_i + Q_i B_i u_i',350,271,300,20,leading=27)
bullets(['Two heads retain CE_CFP + CE_OCT and two predictions. Main metric = mean of their macro-F1 values.','Fixed Q, geometry and permutations are buffers. Only the central bridge convolution is learned in fixed-basis arms.','Full adaptation updates both networks and BN. Frozen arms hold backbone, head and BN fixed; only the bridge is trained.'],y=169,size=13,gap=9)
end('A common packet format does not establish anatomical, semantic or physical-frequency alignment.')

start('How strong is the statistical evidence?','03 / inference','Use the locked comparison weights. Report ordinary and simultaneous intervals, not selected test winners.')
table(['Global decision over 121 primary comparisons','Count','Meaning'],[
 ['Supports substantive improvement','3','All three are within-checkpoint functional switches'],
 ['Supports practical similarity','3','Entire interval lies inside the research band +/-1 pp'],
 ['Unresolved','114','Precision does not distinguish the prespecified categories'],
 ['Not estimable','1','No estimable matched terms; retained explicitly']
],[415,70,403],y=397,size=14)
bullets(['10,000 participant-level paired bootstrap resamples; 290 test participants. Models, seeds, widths and permutation repeats do not increase this sample size.','Global max-|t| approximate simultaneous intervals cover all primary comparisons; family and ordinary intervals are also retained.','The +/-1 pp band is a research convention, not a clinical threshold. Failure to reject is not proof of equivalence.','Development selected checkpoints and configurations. Earlier LOOK use of this cohort is recorded: this is not a pristine, never-touched historical test or external clinical validation.'],y=235,size=13,gap=10)
end()

DIRECT=[('no_communication','No communication'),('linear_resample_M32_S64_k3_r16','Ordinary r16 / h512'),('radon_M32_S64_k3_r16','R&B r16 / h512'),('linear_resample_M32_S64_k3_r32','Ordinary r32 / h1024'),('radon_M32_S64_k3_r32','R&B r32 / h1024')]+[(f'mmtm_hidden{n}',f'MMTM hidden{n}') for n in (128,256,512,1024)]+[(f'attention_d{n}',f'Attention d{n}') for n in (128,256,512,1024)]
summary=[]
for k,label in DIRECT:
 rs=G[k];assert len(rs)==6
 summary.append(dict(method=label,configuration=k,n=6,dev_mean=100*mean(rs,'development_branch_mean_macro_f1'),cfp=100*mean(rs,'test_cfp_macro_f1'),oct=100*mean(rs,'test_oct_macro_f1'),mean=100*mean(rs,'test_branch_mean_macro_f1'),fusion=100*mean(rs,'test_fixed_probability_fusion_macro_f1'),parameters=int(rs[0]['parameters'])))
start('Absolute performance: development is not test','04 / direct comparison','Three seeds x two backbone learning rates; equal weight per model. All values are percentages.')
table(['Configuration','Dev mean F1','Test CFP','Test OCT','Test mean F1','Fixed prob. fuse'],[[s['method'],fmt(s['dev_mean']),fmt(s['cfp']),fmt(s['oct']),fmt(s['mean']),fmt(s['fusion'])] for s in summary],[265,120,115,115,138,135],y=405,size=12,rowpad=3)
callout('Reference R&B improves both branch means. Absolute test performance is lower; it should not be reported as the earlier ~70% development result.',y=85)
end('M32 / S64 / k3 for linear references. Fixed probability fusion is a secondary output; it has no learned fusion head.')

start('Cost and baseline interpretation','05 / fairness','Identical training rules do not imply equal parameter counts, computation or historical search budgets.')
cost=[]
for k,label in [DIRECT[i] for i in [0,2,4,6,10,12]]:
 rs=G[k];cost.append([label,fmt(int(rs[0]['parameters'])/1e6),fmt((int(rs[0]['parameters'])-44338564)/1e6),fmt(max(float(x['peak_training_reserved_mib']) for x in rs)/1024),fmt(mean(rs,'training_seconds')/60)])
table(['Configuration','Total params M','Added params M','Peak reserved GiB','Mean train min'],cost,[288,150,150,150,150],y=399,size=13,rowpad=4)
bullets(['All 24 current MMTM runs (4 widths x 6 models) selected epoch 0. Their equal test scores reflect rollback to the initial model, not proof that channel width is intrinsically irrelevant.','R&B is linear inside the fixed bridge, but the h512/h1024 references add more parameters than these MMTM and attention configurations. Do not claim that linear automatically means cheaper.','Training wall time includes differing plateau lengths and shared-machine load. This table is accounting, not a controlled latency benchmark.'],y=207,size=14,gap=11)
end('Peak reserved memory is a per-process training measurement; project caps and concurrent processes were monitored separately.')

HIST=[('learned','Learned CM projection'),('channel_svd_radon','Learned channel projection'),('svd_radon','Uncentered SVD + Radon'),('centered_svd_radon','Centered SVD + Radon'),('qr_radon','Random QR + Radon'),('svd_resample','SVD + ordinary'),('svd_self','SVD + self-only'),('svd_scrambled','SVD + scrambled'),('qr_scrambled','QR + scrambled')]
hist=[]
start('Compression and width: no universal winner','06 / historical matched group','Three seeds, backbone LR6e-5, M32 / S64 / k3. Test branch-mean F1: mean (sample SD over three seeds), in percentage points.')
for arm,label in HIST:
 vals=[]
 for rho in [.0625,.125,.25]:
  rs=[r for r in R if r['source_group']=='history_final213' and r['arm']==arm and float(r['rho'])==rho];assert len(rs)==3;vals.append(value(rs,'test_branch_mean_macro_f1')+' ('+fmt(st.stdev(float(x['test_branch_mean_macro_f1'])*100 for x in rs))+')')
 hist.append([label]+vals)
table(['Method','rho1/16: r16 h512','rho1/8: r32 h1024','rho1/4: r64 h2048'],hist,[336,184,184,184],y=399,size=13,rowpad=6)
callout('Centered SVD also has higher observed means; scrambled controls can exceed standard Radon. QR rises across these three means, but this is not a monotonicity guarantee.',y=100)
end('No test-based method or rho selection is made. Learned CM h matches here, but its projection structure differs.')

start('What remains uncertain about Radon geometry','07 / mechanism','Differences are percentage points; intervals below are global simultaneous 95% intervals.')
rows=[contrast(k,l) for k,l in [
 ('svd_geometry','SVD: Radon - ordinary'),('qr_geometry','QR: Radon - ordinary'),('basis_geometry_interaction','SVD geometry gain - QR geometry gain'),
 ('qr_radon_minus_self','QR: Radon - self-only'),('qr_radon_minus_scrambled','QR: Radon - spatially scrambled'),('svd_ordered_minus_permuted','SVD: ordered S - permuted S'),('qr_ordered_minus_permuted','QR: ordered S - permuted S'),('branch_equal_compute_budget512','Equal-compute geometry gain: budget512'),('branch_equal_compute_budget1024','Equal-compute geometry gain: budget1024')]]
table(['Locked comparison','Test difference','Global 95% interval'],rows,[548,150,190],size=13,rowpad=6)
callout('Positive geometry means are encouraging. Current evidence does not establish that intact Radon geometry or S-axis adjacency is uniquely necessary.',y=87)
end('Different rows use their own prespecified matched groups; do not average these rows into a new headline effect.')

start('Directions, rank and bridge count','08 / resource allocation','Fixed r varies h with M; fixed h trades channel rank against directions. These answer different questions.')
table(['Locked comparison','Test difference pp','Global 95% interval'],[contrast(k,l) for k,l in [
 ('fixed_r16_geometry_M32_minus_M16','r16: geometry gain M32 - M16'),('fixed_r16_geometry_M64_minus_M16','r16: geometry gain M64 - M16'),('fixed_r32_geometry_M32_minus_M16','r32: geometry gain M32 - M16'),('fixed_r32_geometry_M64_minus_M16','r32: geometry gain M64 - M16'),('radon_rank32_minus_rank16','Radon rank32 - rank16, averaged across M'),('geometry_count2_minus_count1','Geometry gain: 2 bridges - 1 bridge'),('geometry_count3_minus_count2','Geometry gain: 3 bridges - 2 bridges'),('radon_count2_minus_single_rank23','Radon 2 bridges - single wider r23'),('radon_count3_minus_single_rank28','Radon 3 bridges - single wider r28')]], [548,150,190],size=13,rowpad=6)
callout('No established rule that more angles, more channels or more bridges always helps. Wider single bridges remain an essential capacity control.',y=87)
end('The full M/S/k and feasibility evidence is preserved in the original tables and appendix comparisons.')

start('Does the bridge require backbone co-adaptation?','09 / frozen mechanism','Frozen backbone, head and BN; bridge convolution alone is trained. Three seeds x three rho levels.')
table(['Frozen comparison','Test difference pp','Global 95% interval'],[contrast(k,l) for k,l in [
 ('frozen_svd_radon_minus_linear_resample','SVD: Radon - ordinary'),('frozen_svd_radon_minus_self','SVD: Radon - self-only'),('frozen_svd_radon_minus_parent','SVD: Radon - original parent'),('frozen_qr_radon_minus_linear_resample','QR: Radon - ordinary'),('frozen_qr_radon_minus_self','QR: Radon - self-only'),('frozen_qr_radon_minus_parent','QR: Radon - original parent'),('svd_full_minus_frozen_geometry','SVD geometry gain: full - frozen'),('qr_full_minus_frozen_geometry','QR geometry gain: full - frozen')]], [548,150,190],size=13,rowpad=6)
callout('Frozen models retain positive average signals. The intervals do not establish either that co-adaptation is necessary or that frozen training is equally effective.',y=113)
end('The original 6-row frozen subset is retained separately; do not count its overlapping references as independent evidence.')

start('Training benefit is not the deletion effect','10 / host augmentation','All rows use selected checkpoints, but the first four compare training procedures and the last three toggle one model.')
table(['Comparison','Difference pp','Global 95% interval'],[contrast(k,l) for k,l in [
 ('branch_mmtm_hidden256_augmentation_radon_minus_continue','Train MMTM + R&B versus host continuation'),('branch_attention_d256_augmentation_radon_minus_continue','Train attention + R&B versus host continuation'),('branch_mmtm_hidden256_augmentation_radon_minus_linear_resample','MMTM: add R&B versus add ordinary'),('branch_attention_d256_augmentation_radon_minus_linear_resample','Attention: add R&B versus add ordinary'),('mmtm_hidden256_radon_both_on_minus_host_only','Selected MMTM model: both on - host only'),('attention_d256_radon_both_on_minus_host_only','Selected attention model: both on - host only'),('attention_d256_radon_both_on_minus_new_only','Selected attention model: both on - R&B only')]], [548,150,190],size=13,rowpad=7)
callout('The ~14 pp effect establishes functional dependence after joint adaptation. Actual augmentation training benefits are ~2 pp with unresolved global intervals.',y=111)
end('Hosts and configurations were selected on development; these intervals do not remove that selection history.')

start('Correct pairing and interpretable message flow','11 / descriptive diagnostics','54 standard Radon checkpoints: 2 bases x 3 directions x 3 rho x 3 seeds. Permutations move both eyes together.')
pair=json.loads((E/'unified_test/pairing_diagnostics.json').read_text());assert len(pair)==54
pairrows=[]
for cond,label in [('disable_oct_to_cfp','Disable OCT -> CFP'),('disable_cfp_to_oct','Disable CFP -> OCT'),('disable_both','Disable all cross-source blocks'),('shuffle_oct_to_cfp','Shuffle OCT -> CFP'),('shuffle_cfp_to_oct','Shuffle CFP -> OCT'),('shuffle_both','Shuffle both directions')]:
 vals=[]
 for m in pair:
  base=next(x['mean_branch_f1'] for x in m['conditions'] if x['condition']=='paired');v=[x['mean_branch_f1'] for x in m['conditions'] if x['condition']==cond];vals.append(100*(base-st.mean(v)))
 pairrows.append([label,fmt(st.mean(vals)),f'[{min(vals):.2f}, {max(vals):.2f}]'])
table(['Original pairing minus perturbation','Mean drop pp','Range across 54 models'],pairrows,[548,150,190],size=13,rowpad=4)
bullets(['These are secondary descriptive summaries: average 20 shuffles within each checkpoint, then average checkpoints. No new significance test is introduced.','Original pairing performs better on average, but individual checkpoints can reverse the pattern. Deletion is a stronger intervention than mismatching a sender.','Sinograms, source-block spectra and returned residuals show where the computation changes; they do not alone identify lesions or prove causality.'],y=202,size=13,gap=10)
end('Direction-disabled training arms include structural zeros; permutation repetitions are not additional participants.')

start('What the Fourier and sinogram analyses add','12 / mathematical diagnostics','21 atlas checkpoints; initial construction and selected state; complete split aggregates plus fixed limited probes.')
table(['Question','Observation / evidence','Boundary'],[
 ['Is projection mathematically interpretable?','Synthetic trajectories; continuous multilinear Fourier reference; moment errors.','Finite quadrature is approximate. 21/22 synthetic configurations feasible.'],
 ['Which frequencies are emphasized?','120 source kernel blocks; some k3 kernels emphasize low frequencies.','Ordinary communication can look similar; kernel gain is not task gain.'],
 ['Does a message remain a sinogram?','Input projection and mixed messages have different structure.','Arbitrary source/direction mixing need not satisfy Radon consistency.'],
 ['What does return do?','Ordinary BA has a spatially spread point response.','B is not a precise inverse; sharp reconstruction is not the training objective.'],
 ['Do wider bridges retain more signal?','Energy retention and residual size can rise with width.','Retained energy and larger residuals do not imply improved classification.']
],[243,328,317],y=399,size=14,rowpad=10)
end('3D hyperplane Radon -> 1D Fourier radial line. Feature-grid coordinates do not imply shared physical or semantic units.')

start('What to carry into the next dataset and backbone','13 / research decisions','The current cohort supports a controlled mechanism benchmark, not a complete clinical validation claim.')
bullets(['Retain the core method and controls: R&B, matched ordinary communication, self-only, spatial/S-axis controls, and a strong nonlinear baseline under the same adaptation scenario.','Audit the expanded cohort before training: label provenance, acquisition timing, pairing, exclusions, age/sex/center and split overlap. Current test cannot tune the next protocol.','Prespecify a small set of primary questions and separate confirmation from exploration. More configurations on 290 test people do not create more independent evidence.','Evaluate retrained cross-dataset applicability separately from external validation of a fixed model. Repeat the numerical and resource checks after changing feature shapes.','Keep branch-mean F1 and fixed probability-fusion F1 separate. Calibrated risk and predictive values from this balanced case-control cohort are not population estimates.'],y=397,size=16,gap=15)
end('Practical decision: expand the evidence base rather than continue searching this test cohort for a preferred winner.')

start('Technical atlas and complete comparison appendix','Appendix guide','All 31 accepted figures follow. Line/interval PDFs retain vectors; heatmaps use original high-resolution PNGs to avoid PDF seams.')
table(['Pages','Content','How to read'],[
 ['17 - 24','All primary test comparison plots (8)','Thick ordinary intervals; thin global simultaneous intervals.'],
 ['25 - 38','Sinogram atlas (14)','Synthetic geometry, full-cohort message/return maps, widths, bases and depths.'],
 ['39 - 47','Fourier and BA diagnostics (9)','Point returns, numerical slice discrepancy and matrix-valued kernel gain.']
],[140,300,448],y=397,size=15,rowpad=8)
bullets(['Complete numeric sources: all_references_development_test.csv (804 references), all_model_metrics.csv (777 unique model views per split), and the full contrast JSON/CSV files.','This synthesis adds only transparent aggregate means and descriptive tables. Original models, accepted predictions, primary comparison weights and all bootstrap intervals are unchanged.','Restricted participant images, predictions, features and permutation arrays remain on ws02. Public figures contain synthetic fields or cohort-level aggregates.'],y=235,size=14,gap=15)
end('Primary source: reports/2026_09_08_unified_evidence at GitHub commit 2f9c6c6; SHA manifest verified before synthesis.')

assert len(pages)==16
for i,(group,f) in enumerate(figs):
 start(f'Figure {i+1:02d} | '+f.stem.replace('_',' '),'Technical appendix',group+' / '+f.stem)
 if group=='sinogram_atlas' or f.stem.startswith('BA_'):
  from PIL import Image
  im=Image.open(f.with_suffix('.png'));fw,fh=im.size;sc=min(912/fw,355/fh);c.drawImage(str(f.with_suffix('.png')),(W-fw*sc)/2,52+(355-fh*sc)/2,width=fw*sc,height=fh*sc)
 end('Original accepted figure; source-specific normalization and limits apply. Full PNG / SVG / PDF source pairs are retained.')
c.save();writer=PdfWriter();reader=PdfReader(io.BytesIO(out.getvalue()))
for i,page in enumerate(reader.pages):
 if i>=16 and figs[i-16][0]!='sinogram_atlas' and not figs[i-16][1].stem.startswith('BA_'):
  fp=PdfReader(str(figs[i-16][1])).pages[0];box=fp.mediabox;fw,fh=float(box.width),float(box.height);scale=min(912/fw,355/fh)
  tr=Transformation().translate(-float(box.left),-float(box.bottom)).scale(scale).translate((W-fw*scale)/2,52+(355-fh*scale)/2)
  page.merge_transformed_page(fp,tr,over=True)
 writer.add_page(page)
for i,title in enumerate(pages):writer.add_outline_item(title,i)
writer.add_metadata({'/Title':'R&B - Unified Test and Mechanism Research Report','/Author':'Souray Meng | R&B research project','/Subject':'Locked test evidence, limitations and reusable Radon diagnostics','/Keywords':'Radon Bridge, UKB, CFP, OCT, mechanism, paired bootstrap'})
dest=O/'RB_Comprehensive_Report_2026_09_08.pdf'
with dest.open('wb') as f:writer.write(f)
assert len(PdfReader(dest).pages)==47
derived=dict(source_commit='2f9c6c67748fe166b784cd5f947536c1623c07e1',source_manifest_sha256=sha(E/'artifact_manifest.json'),report_sha256=sha(dest),pages=47,primary_comparisons_unchanged=True,figures=[str(f.relative_to(E)) for _,f in figs],direct_summary=summary,historical_summary=hist,pairing_summary=pairrows,methods='Read-only arithmetic means of existing metrics. No new training, model selection, bootstrap or hypothesis test. All source file hashes verified.')
(O/'report_provenance.json').write_text(json.dumps(derived,indent=2))
with (O/'direct_comparison_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(O/'README.zh-CN.md').write_text('本目录为47页整合PDF及其数据溯源。前16页综合解读，后31页为已验收的全部公开图。原始全量CSV/JSON与源图位于同一项目的 reports/2026_09_08_unified_evidence，提交2f9c6c6。所有表格均为原指标的明确加权汇总，没有新增模型选择或统计检验。\n')
print(json.dumps(dict(pdf=str(dest),pages=len(pages),figures=len(figs),MiB=dest.stat().st_size/1024**2)))
