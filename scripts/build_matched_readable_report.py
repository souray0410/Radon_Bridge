"""Matched development/test report from accepted aggregates only; no new inference.

Produces a guided main PDF and an exhaustive 804-reference paired appendix PDF.
"""
from pathlib import Path
import argparse,collections,csv,hashlib,io,json,math,re,statistics as st,shutil,textwrap
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter,Transformation
from pypdf.generic import RectangleObject
from fractions import Fraction

ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--font',type=Path,default=Path('/Library/Fonts/Arial Unicode.ttf'));a=ap.parse_args()
E=a.repo/'reports/2026_09_08_unified_evidence';A=a.repo/'reports/2026_09_08_expansion_audit';O=a.output;O.mkdir(parents=True,exist_ok=True);F=O/'figures';F.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in read(E/'artifact_manifest.json').items():assert sha(E/name)==h
R=list(csv.DictReader((E/'unified_test/all_references_development_test.csv').open()));assert len(R)==804 and len({r['model_view_id'] for r in R})==777
D=read(E/'unified_test/development_statistics.json');T=read(E/'unified_test/test_statistics.json');Change=read(E/'unified_test/test_minus_development_effect_change.json')
DD={r['contrast_id']:r for r in D['contrasts']};CC={r['contrast_id']:r for r in Change['contrasts']};primary=[r for r in T['contrasts'] if r['primary']];assert len(primary)==121
DEV='#c88322';TEST='#259681';NAVY='#102c43';GRAY='#536575';BG='#f7f9fc'
pdfmetrics.registerFont(TTFont('Chinese',str(a.font)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':True,'grid.alpha':.16})
GUIDES=[];COVERAGE=[]

class Book:
    def __init__(self,width=960,height=540):self.w=width;self.h=height;self.stream=io.BytesIO();self.c=canvas.Canvas(self.stream,pagesize=(width,height));self.titles=[];self.assets=[]
    def para(self,text,x,y,w,size=14,color=NAVY):
        p=Paragraph(escape(str(text)).replace('\n','<br/>'),ParagraphStyle('p',fontName='Chinese',fontSize=size,leading=size*1.4,textColor=colors.HexColor(color),wordWrap='CJK'))
        _,h=p.wrap(w,3000);assert y-h>=48,(self.titles[-1],text[:70],y,h);p.drawOn(self.c,x,y-h);return y-h
    def start(self,title,sub=''):
        self.titles.append(title);c=self.c;c.setFillColor(colors.HexColor(BG));c.rect(0,0,self.w,self.h,fill=1,stroke=0);c.setFillColor(colors.HexColor(TEST));c.rect(0,self.h-7,self.w,7,fill=1,stroke=0)
        self.para('R&B  /  RADON BRIDGE',32,self.h-22,self.w-64,10,TEST);self.para(title,32,self.h-56,self.w-64,24)
        if sub:self.para(sub,32,self.h-98,self.w-64,11,GRAY)
    def table(self,headers,rows,widths,y=None,size=12,pad=6):
        y=self.h-145 if y is None else y
        style=ParagraphStyle('t',fontName='Chinese',fontSize=size,leading=size*1.28,wordWrap='CJK',textColor=colors.HexColor(NAVY))
        def cell(v,head=False):return Paragraph(escape(str(v)).replace('\n','<br/>'),ParagraphStyle('c',parent=style,textColor=colors.white if head else colors.HexColor(NAVY)))
        tab=Table([[cell(v,True) for v in headers]]+[[cell(v) for v in row] for row in rows],colWidths=widths)
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf2f6')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad)]))
        _,h=tab.wrap(sum(widths),3000);assert y-h>=49,(self.titles[-1],y,h);tab.drawOn(self.c,32,y-h);return y-h
    def end(self,note='Development 296 / Test 290 | 同一选中模型，分别评价；两集合参与者不配对'):
        self.c.setStrokeColor(colors.HexColor('#d9e2ea'));self.c.line(32,35,self.w-32,35);self.c.setFont('Chinese',8);self.c.setFillColor(colors.HexColor(GRAY));self.c.drawString(32,20,note);self.c.showPage()
    def image(self,path,region=None):
        pdf=Path(path).with_suffix('.pdf')
        if pdf.exists():
            page=PdfReader(pdf).pages[0];w=float(page.mediabox.width);fullh=float(page.mediabox.height)
            lo,hi=(0,1) if region is None else region;bottom=fullh*(1-hi);top=fullh*(1-lo);h=top-bottom
            page.cropbox=RectangleObject([0,bottom,w,top])
            scale=min((self.w-50)/w,(self.h-165)/h);ww,hh=w*scale,h*scale
            x=(self.w-ww)/2;y=48+(self.h-165-hh)/2
            self.assets.append((len(self.titles)-1,page,Transformation().translate(0,-bottom).scale(scale).translate(x,y)))
        else:
            from PIL import Image
            im=Image.open(path);w,h=im.size;scale=min((self.w-50)/w,(self.h-165)/h);ww,hh=w*scale,h*scale
            self.c.drawImage(str(path),(self.w-ww)/2,48+(self.h-165-hh)/2,ww,hh,preserveAspectRatio=True,mask='auto')
    def finish(self,path):
        self.c.save();reader=PdfReader(io.BytesIO(self.stream.getvalue()));writer=PdfWriter()
        assert len(reader.pages)==len(self.titles)
        for i,page in enumerate(reader.pages):
            for index,asset,transform in self.assets:
                if index==i:page.merge_transformed_page(asset,transform)
            overlay=io.BytesIO();cv=canvas.Canvas(overlay,pagesize=(self.w,self.h));cv.setFont('Helvetica',9);cv.setFillColor(colors.HexColor(GRAY));cv.drawRightString(self.w-32,20,f'{i+1} / {len(reader.pages)}');cv.save();page.merge_page(PdfReader(io.BytesIO(overlay.getvalue())).pages[0]);writer.add_page(page)
            writer.add_outline_item(self.titles[i],i)
        writer.write(path)

B=Book()
def savefig(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(F/(name+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig);return F/(name+'.png')
def guide(title,fields,source):
    number=len(GUIDES)+1;B.start(f'读图 {number:02d}：'+title,'图后的固定六问：轴是什么、看什么、何为好、比较什么、发现什么、边界是什么')
    labels=['横纵轴与单位','点线、颜色与样本','怎样代表更好','具体比较','观察与结论','不能推出什么']
    B.table(['阅读问题','说明'],list(zip(labels,fields)),[158,738],size=14,pad=7)
    B.end('图表来源：'+source[:115]);GUIDES.append(dict(number=number,title=title,page=len(B.titles),source=source,explanations=dict(zip(labels,fields))))
def chart(title,path,fields,source):
    B.start(title,'下一页提供逐图解释；原始数值及匹配范围见附表和图源数据');B.image(path);B.end();guide(title,fields,source)
num=lambda x:f'{x:.2f}'
ci=lambda x:'不可估计/未定义' if not x else f'[{x[0]:+.2f}, {x[1]:+.2f}]'
mean=lambda rs,k:st.mean(float(r[k]) for r in rs)
G=collections.defaultdict(list)
for r in R:
    if r['source_group']=='current_direct_and_host' and r['category']=='direct':G[json.loads(r['structure'])['id']].append(r)
direct=[('no_communication','No communication'),('linear_resample_M32_S64_k3_r16','Ordinary r16 / h512'),('radon_M32_S64_k3_r16','R&B r16 / h512'),('linear_resample_M32_S64_k3_r32','Ordinary r32 / h1024'),('radon_M32_S64_k3_r32','R&B r32 / h1024')]+[(f'mmtm_hidden{d}',f'MMTM hidden{d}') for d in (128,256,512,1024)]+[(f'attention_d{d}',f'Attention d{d}') for d in (128,256,512,1024)]

B.start('R&B：开发与测试完整匹配解读','2026-09-08修订版 | 本轮统一评价 + 图谱/傅里叶 + 扩展前核查')
B.para('先明确每张图怎样读，再讨论证据支持什么。\n同一批模型的development与test并排报告。',40,378,880,27)
B.table(['完整性','本版交付'],[['804条研究引用','逐条提供开发/test的CFP、OCT、分支平均F1；另附完整辅助指标CSV。'],['121项主要比较','每项同时显示开发、test和增益变化；未定义和不可估计不删除。'],['23幅数学与图谱诊断','每图配独立中文说明页；训练/开发/test与合成算子实验严格区分。'],['两份PDF','本册为图表解读；另册为804项完整匹配附表，可用编号追溯。']],[190,706],y=270,size=15,pad=8)
B.end('旧版汇总PDF由本版替代；历史模型、原始统计和已接受图源不变')
B.start('阅读约定：development、validation、test是什么','本项目中的validation就是development，均为296名参与者的选优集合')
B.table(['对象','用法与边界'],[['Train：1264人','用于训练和固定基拟合；训练F1不能当独立泛化表现。'],['Development / validation：296人','选epoch、判平台和调学习率；历史还参与过任务/配置探索，存在选优乐观偏差。'],['Test：290人','读取开发已选中的固定模型；不选epoch、ρ、阈值，不重新训练。此前LOOK使用史仍需披露。'],['分支平均F1','先算CFP和OCT各自macro-F1，再取算术平均；有桥后分支输出可依赖另一模态。'],['固定概率平均融合','(p_CFP+p_OCT)/2再计算F1，是另一指标；不能与分支F1平均混称。'],['开发到test的变化','两组是不同的人：各集合内做方法配对，跨集合变化用独立重采样。']],[210,686],size=15,pad=8)
B.end()
B.start('符号与误差线：同一套读图规则','新性能图与既有图谱统一：development橙色、test绿色；图谱train蓝色')
B.table(['符号/读法','意义'],[['ρ、r、h','ρ是通道维数比例；当前stage3 C256，r=ρC，h=rM。历史M32下ρ1/16、1/8、1/4对应r16/32/64与h512/1024/2048。'],['M、S、k','M为Radon方向数，S为投影轴采样数，k为卷积核长度。普通重采样的M仅是打包数，不称角度。'],['F1 / AUROC','越大表示该指标越好；不是越大就一定显著或有临床价值。'],['NLL / Brier / 数值误差','通常越小越好，但数值误差更小不保证分类更好。'],['均值±种子SD','描述几个已训练模型的波动，不是参与者bootstrap置信区间。'],['差值与置信区间','A−B>0表示A在指定指标更高。灰带±1 pp是研究参考；区间跨0通常不能明确判断方向。']],[210,686],size=14,pad=7)
B.end()
B.start('实验范围和完整对应关系','所有已接受结果纳入；没有为了图形整齐删去负结果、epoch0或不可估计项')
B.table(['来源','引用数','说明'],[['本轮直接/增强/原冻结','384','342直接 + 36增强 + 6冻结'],['历史最终213项','231','9项多宽度训练展开为27个视图；不是额外训练18次'],['历史S轴邻接','18','保留相同模型条件下的轴邻接对照'],['补齐机制','72','固定r的M补齐24 + 冻结机制补齐48'],['多桥验证','96','位置/桥数与近似容量参照'],['独立父模型','3','种子3416–3418的各自最佳分支'],['总计','804 → 777','804引用去重为777模型视图；855评价作业包括宿主/配对诊断']],[285,125,486],size=14,pad=7)
B.end()

B.start('绝对性能：开发与test逐项并排','当前参考组；每格为3种子×2学习率平均。R&B/普通通信为非中心化SVD、M32/S64/k3')
rows=[]
for key,label in direct:
    rs=G[key];assert len(rs)==6
    rows.append([label]+[num(100*mean(rs,f'{split}_{metric}')) for split in ('development','test') for metric in ('cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1')])
B.table(['方法','Dev CFP','Dev OCT','Dev均值','Test CFP','Test OCT','Test均值'],rows,[278,103,103,103,103,103,103],size=11,pad=4)
B.end('单位：% | 本表是已指定配置的均值，不按test选赢家；全部配置见804项附表')
fig,axs=plt.subplots(1,3,figsize=(13,6),layout='constrained',sharey=True)
for ax,metric,title in zip(axs,['cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1'],['CFP','OCT','Mean of branch F1']):
    y=np.arange(len(direct))
    for split,col,offset in [('development',DEV,-.12),('test',TEST,.12)]:ax.plot([100*mean(G[k],f'{split}_{metric}') for k,_ in direct],y+offset,'o',color=col,label=split)
    ax.set(xlabel='Macro-F1 (%) - higher is better',title=title,xlim=(52,74),yticks=y,yticklabels=[v for _,v in direct]);ax.invert_yaxis()
axs[0].legend(loc='lower left',fontsize=9)
chart('同一参考模型：开发与test的绝对成绩',savefig(fig,'01_direct_matched'),['横轴为macro-F1百分数；纵轴为相同配置。三个面板分别为CFP、OCT、两分支平均。','橙点=development；绿点=test。每点先对每个模型算指标，再对3种子×2学习率平均，无误差线。','在同一面板中越向右，该指标越高；两点相距大仅表示数据集成绩差，不直接说明统计显著。','每行保持模型配置不变。R&B和普通通信使用相同r/h；MMTM和注意力另列自身规模，不称等宽。','R&B两个参考宽度在两集合都高于无通信均值；多种方法从开发到test下降，不能只归咎R&B。','均值高于某基线不代表所有种子都更好；test下降也不能单凭图归因于人口学或预处理。'],'all_references_development_test.csv / current direct references')

hist=[r for r in R if r['source_group']=='history_final213'];bases=[('learned','Learned CM'),('channel_svd_radon','Learned channel'),('svd_radon','SVD uncentered'),('centered_svd_radon','SVD centered fit'),('qr_radon','Random QR')]
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey=True)
palette=['#566a85','#9c6b9c','#23799b','#ce784a','#4b9561']
for row,split in enumerate(['development','test']):
    for col,metric in enumerate(['cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1']):
        ax=axs[row,col]
        for (arm,label),color in zip(bases,palette):
            vals=[[100*float(r[f'{split}_{metric}']) for r in hist if r['arm']==arm and float(r['rho'] or 0)==rho] for rho in (1/16,1/8,1/4)];assert all(len(v)==3 for v in vals)
            ax.errorbar(np.arange(3),[st.mean(v) for v in vals],yerr=[st.stdev(v) for v in vals],color=color,marker='o',capsize=2,label=label)
        ax.set(title=split+' | '+['CFP','OCT','Branch mean'][col],xticks=[0,1,2],xticklabels=['1/16\nh512','1/8\nh1024','1/4\nh2048'],xlabel='rho; M32 / S64 / k3',ylabel='Macro-F1 (%)')
axs[0,0].legend(fontsize=8)
chart('压缩方式与三档ρ：开发/test完整对应',savefig(fig,'02_compression_matched'),['横轴为ρ及实际h（r仅适用于通道压缩版本，CM压缩不另设r）；纵轴为F1百分数。上排开发、下排test；三列分别CFP、OCT、平均。','颜色区分五种压缩方法；折线连接三档预设ρ；误差线是3个种子的样本SD，统一历史LR6e-5。','同集合、同面板内F1越高越好；SD较小仅表示这三个模型波动较小，不是普遍稳定性保证。','每种方法都覆盖相同三档ρ。Learned CM压缩通道×方向，Learned channel压缩原生通道，结构有差异。','没有跨所有宽度/集合的统一赢家；QR当前三档test分支平均均值递增，SVD部分档位先升后降。','不能按test曲线重新挑ρ或基；ρ增大同时增加宽度/参数，不能解释成单纯多保留能量。'],'history_final213 / matched 3 seeds and LR6e-5')

B.start('概率指标也必须成对看','先算每模型每分支，再平均；Brier为二分类正类概率形式，越小越好')
rr=[]
for key,label in [direct[0],direct[2],direct[4],direct[6],direct[-1]]:
    for split in ('development','test'):
        rs=G[key];rr.append([label,split,num(100*mean(rs,f'{split}_branch_mean_macro_f1'))]+[f'{st.mean(mean(rs,f"{split}_{b}_{metric}") for b in ("cfp","oct")):.3f}' for metric in ('auroc','log_loss','brier_binary')])
B.table(['方法','集合','平均F1% ↑','AUROC ↑','NLL ↓','Brier ↓'],rr,[296,150,110,110,110,110],size=12,pad=5)
B.end('F1/AUROC与概率质量可能有取舍；当前test没有拟合温度、阈值或校准器')
B.start('扩展前核查：已排除与尚未确定','66个参考视图、1850人清单/缓存、24个MMTM停止点；新增训练0')
B.table(['核查','结论'],[['数据和划分','缓存标签与上游记录一致；三组无参与者交叉；3700眼CFP/OCT各无精确内容重复。'],['样本规模','77,118条候选记录中，925名符合既定规则的青光眼现患病例全部纳入，再配925对照。增加正常人不增加阳性病例。'],['MMTM是否更新','24项各632次更新，训练F1全部100%；初始开发65.02%，训练后最高轮均值63.47%，所以选回epoch0。'],['MMTM数值状态','128人训练探针无sigmoid饱和；调制0.9853–1.0155，接近恒等。4配置停止梯度非零，未见永久断梯度。'],['概率质量','R&B参考test的F1/AUROC改善但NLL更差；错误预测仍较自信。扩大研究必须同时评价和校准概率。'],['原因边界','现有核查不能定量分开选优偏差、抽样波动、标签异质性、初始化和优化适配等原因。']],[210,686],size=14,pad=7)
B.end('完整核查见2026_09_08_expansion_audit；探针只用train、GPU0，无新test前向')

# Primary forest plots include both splits; each is followed by exact values and explanations.
family_names={'direct28':'直接比较：等参数/等计算与非线性基线','augmentation4':'宿主增强：训练收益','mechanism31':'基与几何、通信方向及历史基线','qr_mechanism3':'QR机制：自身与空间打乱','joint_width3':'多宽度联合训练','s_axis3':'S轴邻接','allocation8':'固定通道秩的方向分配','frozen9':'冻结网络的完整机制对照','depth22':'桥数量与位置','frozen2':'原冻结参考组','component8':'宿主开关：同一模型的功能依赖'}
def label(alias):
    explicit={f'{b}_{direction}_contribution':f'{b.upper()}：双向−删除{arrow}（{recipient} F1）' for b in ('svd','qr') for direction,arrow,recipient in [('oct_to_cfp','OCT→CFP','CFP'),('cfp_to_oct','CFP→OCT','OCT')]}
    if alias in explicit:return explicit[alias]
    s=alias.replace('mmtm_r4','MMTM hidden256 (reduction4)').replace('mmtm_r8','MMTM hidden128 (reduction8)')
    s=s.replace('branch_','').replace('reference_','R&B参考_').replace('equal_parameters_','等参数几何增益_').replace('equal_compute_budget','等计算预算').replace('augmentation_','增强训练_')
    for old,new in [('basis_geometry_interaction','SVD与QR几何增益之差'),('basis_spatial_interaction','两种基的空间结构增益之差'),('basis_S_axis_interaction','两种基的S轴邻接增益之差'),('fixed_r','固定r'),('geometry','几何增益'),('linear_resample','普通通信'),('count1','单桥'),('count2','双桥'),('count3','三桥'),('single_rank23','单桥r23'),('single_rank28','单桥r28'),('rank32','r32'),('rank16','r16'),('frozen','冻结'),('full','全参数'),('radon','R&B'),('ordered','有序'),('permuted','打乱'),('self','自身'),('scrambled','空间打乱'),('parent','父模型'),('continue','宿主继续训练'),('both_on','两路开'),('host_only','仅宿主'),('new_only','仅新增'),('joint','联合宽度'),('independent','独立宽度'),('nested','多宽度'),('contribution','方向训练贡献'),('oct_to_cfp','OCT→CFP'),('cfp_to_oct','CFP→OCT'),('presence','出现的边际效应'),('stage2','stage2'),('svd','SVD'),('qr','QR')]:s=s.replace(old,new)
    s=re.sub(r'rho1_(16|8|4)',lambda m:'ρ=1/'+m[1],s)
    return s.replace('_minus_',' − ').replace('_by_',' 随 ').replace('_interaction','交互').replace('_',' ')
families=collections.OrderedDict()
for r in primary:
    al=next(z for z in r['aliases'] if z['primary']);families.setdefault(al['family'],[]).append(r)
paired_rows=[];plot_index=0
classification={'supports_substantive_improvement':'支持实质提升','supports_substantive_decline':'支持实质下降','supports_substantive_decrease':'支持实质下降','supports_practical_similarity':'支持实际接近','unresolved':'尚不能分辨','not_estimable':'不可估计','undefined_zero_variance':'零方差未定义'}
for fam,entries in families.items():
    for offset in range(0,len(entries),6):
        chunk=entries[offset:offset+6];plot_index+=1;ids=[next(z for z in x['aliases'] if z['primary'])['id'] for x in chunk]
        fig,ax=plt.subplots(figsize=(12.5,5.5),layout='constrained');ys=np.arange(len(chunk))
        for split,lookup,col,shift in [('development',DD,DEV,-.16),('test',{x['contrast_id']:x for x in chunk},TEST,.16)]:
            for k,r in enumerate(chunk):
                x=lookup[r['contrast_id']];v=x.get('difference_pp');interval=x.get('global_simultaneous_ci95_pp')
                if v is None:ax.text(.02,k+shift,'N/E',transform=ax.get_yaxis_transform(),color=col);continue
                ax.plot(v,k+shift,'o',color=col,label=split if k==0 else None)
                if interval:ax.plot(interval,[k+shift,k+shift],color=col,lw=2)
        ax.axvspan(-1,1,color='#dde7ee',alpha=.8);ax.axvline(0,color=NAVY,lw=1)
        ax.set(yticks=ys,yticklabels=[f'C{primary.index(r)+1:03d}' for r in chunk],xlabel='Prespecified difference (percentage points) | global simultaneous 95% interval',ylabel='Comparison ID: full definition on next page');ax.invert_yaxis()
        ax.legend(handles=[plt.Line2D([],[],marker='o',color=DEV,label='development'),plt.Line2D([],[],marker='o',color=TEST,label='test')],loc='best')
        path=savefig(fig,f'primary_matched_{plot_index:02d}');B.start(f'配对主要比较：{family_names.get(fam,fam)}',f'本页C{primary.index(chunk[0])+1:03d}起；横轴正值支持所定义差值左项，交互项不等于某方法绝对更优');B.image(path);B.end()
        guide_number=len(GUIDES)+1;B.start(f'读图 {guide_number:02d}：定义、两集合结果与增益变化','橙=开发，绿=test；区间为全局同时95%区间。灰带±1 pp不是临床阈值')
        tr=[]
        for r,alias in zip(chunk,ids):
            d=DD[r['contrast_id']];ch=CC[r['contrast_id']];idx=primary.index(r)+1
            def cell(x):return '不可估计' if x.get('difference_pp') is None else f'{x["difference_pp"]:+.2f}\n{ci(x.get("global_simultaneous_ci95_pp"))}'
            tr.append([f'C{idx:03d}\n'+label(alias),cell(d),cell(r),cell(ch)])
            paired_rows.append(dict(number=f'C{idx:03d}',contrast_id=r['contrast_id'],alias=alias,family=fam,definition=label(alias),development=d.get('difference_pp'),development_global_ci=d.get('global_simultaneous_ci95_pp'),test=r.get('difference_pp'),test_global_ci=r.get('global_simultaneous_ci95_pp'),test_minus_development=ch.get('difference_pp'),change_global_ci=ch.get('global_simultaneous_ci95_pp'),development_decision=d.get('classification'),test_decision=r.get('classification')))
        y=B.table(['编号 / 比较定义','开发差值 pp','Test差值 pp','Test−开发 pp'],tr,[350,170,170,206],size=11,pad=5)
        decisions=collections.Counter(r.get('classification') for r in chunk)
        outcome='本页test：'+'；'.join(f'{classification.get(k,k)}{v}项' for k,v in decisions.items())+'。'
        B.para(outcome+' 几何增益=R&B−同条件普通通信；方向项看指定接收分支，其他指标遵循锁定权重。区间整体>+1支持实质提升，在±1内支持实际接近；变化列用两集合独立重采样。',32,y-14,896,12)
        B.end('完整权重/别名来自已锁定比较；表中“不可估计”和“零方差未定义”没有强行置零')
        GUIDES.append(dict(number=guide_number,title=family_names.get(fam,fam),page=len(B.titles),source='locked primary contrast IDs '+','.join(ids),explanations={'横纵轴与单位':'横轴为已定义差值pp；纵轴C编号逐项对应本页定义。','点线颜色':'橙色development、绿色test；点为均值差，线为全局同时95%区间。','怎样更好':'正值表示左项指标更高；交互只表明增益差。','比较范围':'严格沿用每项锁定权重，历史LR6e-5组不混入本轮两LR均值。','结论':'读取本页开发、test、变化三列；区间跨0不等于无效。','限制':'两数据集参与者不同；模型/种子不是额外独立参与者。'}))
assert len(paired_rows)==121

# All 54 historical development pairing checkpoints matched to test by selected SHA.
old_pair=read(a.repo/'reports/2026_09_06_integrated_213/prior186/pairing_diagnostics.json');test_pair=read(E/'unified_test/pairing_diagnostics.json')
refs_by_view={r['model_view_id']:r for r in R};old_by_sha={v['selected_sha256']:v for v in old_pair.values()};assert len(old_by_sha)==54
perturb=[];conditions=['disable_oct_to_cfp','disable_cfp_to_oct','disable_both','shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both']
for t in test_pair:
    ref=refs_by_view[t['model_view_id']];d=old_by_sha[ref['checkpoint_sha256']];assert d['passed'] and not d['test_used']
    for split,cs in [('development',d['result']['conditions']),('test',t['conditions'])]:
        original=next(z for z in cs if z['condition']=='paired')
        for b in ('cfp','oct'):
            assert abs(original['branches'][b]['metrics']['macro_f1']-float(ref[f'{split}_{b}_macro_f1']))<1e-8
        for cond in conditions:
            values=[z for z in cs if z['condition']==cond];assert len(values)==(20 if cond.startswith('shuffle') else 1)
            for metric in ('cfp','oct','mean'):
                get=lambda x:st.mean(x['branches'][b]['metrics']['macro_f1'] for b in ('cfp','oct')) if metric=='mean' else x['branches'][metric]['metrics']['macro_f1']
                perturb.append(dict(model_view_id=t['model_view_id'],checkpoint_sha256=ref['checkpoint_sha256'],split=split,condition=cond,metric=metric,paired_minus_perturbed_pp=100*(get(original)-st.mean(get(v) for v in values))))
fig,axs=plt.subplots(1,3,figsize=(13,5),layout='constrained',sharey=True)
for ax,metric in zip(axs,['cfp','oct','mean']):
    for split,col,offset in [('development',DEV,-.12),('test',TEST,.12)]:
        vals=[[r['paired_minus_perturbed_pp'] for r in perturb if r['split']==split and r['condition']==cond and r['metric']==metric] for cond in conditions]
        assert all(len(v)==54 for v in vals);ax.plot([st.mean(v) for v in vals],np.arange(6)+offset,'o',color=col,label=split)
    ax.axvline(0,color=NAVY);ax.set(title=metric.upper(),xlabel='Correct - perturbed F1 (pp)',yticks=np.arange(6),yticklabels=['Off OCT->CFP','Off CFP->OCT','Both cross off','Shuffle OCT->CFP','Shuffle CFP->OCT','Shuffle both']);ax.invert_yaxis()
axs[0].legend()
chart('正确配对与关闭/打乱：54个模型开发/test对应',savefig(fig,'pairing_matched'),['横轴为正确配对F1减扰动F1，单位pp；纵轴依次为两方向关闭、全关闭、两方向打乱、全打乱。','橙=开发，绿=test。先在每个模型内平均20次打乱，再等权平均54个模型；没有把20次置换当20倍样本。','向右表示当前模型更依赖正确配对或对应通路；不是新增模块的训练收益。','开发与test通过同一选中检查点SHA匹配；原始配对F1与804项表逐项核对。三个面板保留两分支差异。','两集合均可直接比较平均功能依赖；原始数据保留逐模型结果，平均依赖不能排除反例和结构性零效应。','没有新显著性检验；全部54项包括单向模型，未启用通路出现零效应是结构事实，不能称无效训练。'],'historical prior186 pairing + unified test pairing; 54 checkpoint SHAs matched')

# Diagnostic figure notes: actual axes and interpretation, never impose higher=better.
def diagnostic_notes(file):
    n=file.stem;oct_='oct' in n;source='OCT（三维）' if oct_ else 'CFP（二维）'
    common='公开图为聚合特征统计，不是患者原图或病灶定位。没有新增训练或按test挑样本。'
    if n.startswith('01_synthetic'):
        return ['左/右图是原生特征网格坐标；中图横轴为有符号距离s。二维纵轴是法向角度，三维纵轴是EEM方向编号。','左为合成高斯点，中为实际离散投影，右为普通反投影。色条表示幅值；二维虚线为s=n·x0。','本图不评价分类。投影轨迹与理论位置相符是几何核查；返回更亮不代表分类更好。','观察同一合成输入经过投影和普通返回之后，位置/形状如何改变。','二维点呈正弦轨迹；三维是球面方向的投影表。返回点会扩散，说明普通反投影不是精确逆。','合成点结果不证明真实病变可定位；三维编号相邻不代表方向在球面相邻。']
    if n.startswith('02_projected_flow'):
        return ['每小图横轴是投影轴有符号距离；CFP纵轴是角度，OCT纵轴是方向编号。四行：输入投影、自身消息、跨来源消息、总消息。','三列依次train、validation、test。色条为RMS幅值；每行按训练图定色限，clipped是超出色限的比例。','没有“越亮越好”：亮表示该位置的聚合幅度更大。不同算子阶段的幅度不是同一分类指标。','同一来源、同一行横向比较三集合；纵向看桥如何把输入改成自身和跨来源消息。','可观察非零跨来源消息以及各集合模式；总消息可能相长或相消，RMS不能把自身与跨来源直接相加。',common+' 固定SVD r16/h512；跨种子按平方能量汇总，不声称通道语义一一对应。']
    if n.startswith('03_return_maps'):
        return ['坐标是原生特征网格索引。CFP为二维图；OCT将三个中央正交截面并排，间隔不代表连续空间。','三列train/validation/test；四行输入、自身返回、跨来源返回、总返回。色条为RMS，每行共享训练色限。','残差幅度大只是修改更强，不等于性能提升；幅度小也不等于无作用。','同一行横向看集合间返回模式，比较输入与总返回在何处具有较大能量。','能看到桥怎样把消息写回原生特征空间；它是残差修改的数值描述，而非图像重建质量评测。',common+' 后续主干仍会非线性处理；不能从亮区推出解剖病灶或因果机制。']
    if n.startswith('04_label_difference'):
        return ['横轴是有符号距离；CFP纵轴为角度，OCT为EEM方向编号。两行分别为输入投影和跨来源消息。','三列train/validation/test；色条为病例组RMS减对照组RMS，正负色表示差异方向，白色接近0。','更红/更蓝只表示组差更大，不代表诊断性能更好；色差没有对应的显著性检验。','在同一固定模型组中比较两类记录表型的聚合幅度，观察这种描述性模式跨集合是否相似。','显示特征与标签的聚合关联；即使模式相似，也不能单独证明桥提取了临床可解释的病理特征。',common+' 固定训练色限，不按test单独增强颜色；不从此图设阈值或筛选患者。']
    if n.startswith('05_oct'):
        return ['三维坐标轴是单位法向量的三个分量，不是患者空间位置；点的编号对应32个EEM方向。','三面板为train、validation、test。颜色是每方向投影RMS，统一训练色限。','没有优劣排序；颜色越亮仅表示沿该法向的投影幅度较大。','对照热图中的方向编号与球面上的真实方向，防止把三维方向误读成一条角度轴。','OCT方向分布在球面代表域；二维正弦图的单一角度解读不能直接套到三维。','颜色不能证明某个投影角度诊断最重要；共同格式也不保证CFP/OCT物理对齐。']
    if n.startswith(('06_width','07_basis')):
        x='横轴为三档ρ及r/h' if n.startswith('06') else '横轴为SVD Radon、QR Radon、SVD普通重采样（均r16/h512）'
        return [x+'；三列纵轴依次为通道能量保留率、残差/输入范数、自身与跨来源余弦；两行为CFP/OCT。','蓝=train，橙=validation，绿=test。点为种子值，连线为3种子均值，误差线为样本SD。','保留率高只表示能量保留多；残差比无统一好坏；余弦正/负表示同向/相消，不是F1。','在同一来源和同一指标列比较宽度或方法；不用不同列的数值大小互相排名。','可以分清“保留多少能量”和“修改多强”，也看到种子差异；这些都应与匹配性能图共同解释。','能量不是判别信息，残差大不证明有效。均值±SD可能超出指标合法范围，仅是对称波动描述。']
    if n.startswith('08_s_axis'):
        return ['横轴为每S采样点的周期数0–0.5；纵轴是该频点占投影域包总能量的比例。','两行CFP/OCT；四列投影、自身、跨来源、总消息；蓝/橙/绿依次train/validation/test，包含直流0频。','低频或高频能量更多都不自动代表更好；它是信号能量分布，不是分类曲线。','比较桥前后频谱，以及同阶段跨数据集合的频谱形状。','可观察能量是否集中于低频、消息是否改变频率分布；输入谱和卷积核响应是两个不同对象。','单位不是mm或Hz；两模态相同归一化频率不意味着相同物理尺度。不能据低频占优证明几何不可替代。']
    if n.startswith('09_multidepth'):
        return ['横轴为桥接stage；纵轴为先逐参与者求||ΔX||/||X||再平均的相对残差，无物理单位。','两行CFP/OCT；三列单桥、双桥、三桥；蓝/橙/绿为三集合，误差线为3种子SD。','残差越大表示该处改动越强，不等于多桥性能更好；请结合多桥差值区间。','每桥r16/h512，比较不同位置的修改强度；后续桥输入已包含前面桥的作用。','呈现深度不同导致的残差尺度差异；不能把所有桥看成作用在同一份未经修改的特征上。','本图不是严格容量匹配的性能比较，也不是把每桥贡献简单相加的证据。']
    if n.startswith('10_initial'):
        return ['横轴为距离/投影支持半径，归一化到[-1,1]；纵轴是按方向聚合的输入投影RMS。','两行CFP/OCT、三列train/validation/test；灰线为从同父模型构造的零桥初始，彩线为开发选中模型。','曲线变大不等于性能更好；本图看原生表示是否随联合训练改变。','同一集合内比较初始和选中状态，跨集合比较形状；初始不是额外训练出的另一基线。','初始桥消息为零，但原生输入并不为零；联合训练后输入表示可变，固定SVD基未必仍能量最优。','不能把表示变化直接解释为跨来源净收益，或把零桥输入幅度误认为桥的贡献。']
    if n.startswith('BA_'):
        return ['坐标为原生特征网格索引；OCT显示三个中央正交截面，中间空隙不是空间连续部分。','左列固定单位脉冲，右列同脉冲经过投影A和普通返回B；两行为中心/偏移位置。色条表示返回幅度。','若讨论重建，接近输入位置/形状更像逆；本图只展示算子性质，不给分类优劣评分。','在相同stage几何下，对比BA与单位映射；不同stage色条不能忽略后直接比亮度。','返回脉冲展宽且有位置/边界差异，BA并非恒等；R&B学习的是残差通信，不是直接无损重建。','这是合成算子实验，无train/val/test区分；不是患者诊断，也不是已学Conv1d后的完整模型。']
    if n=='fourier_synthetic_errors':
        return ['横轴为S采样数；纵轴为投影的一维傅里叶结果与插值场解析频谱参考的相对L2误差。','六面板为CFP/OCT的stage2/3/4，颜色为三种预定合成场；缺口是原算子容量限制不可行。','数值误差越小表示此离散几何更接近指定数学参考；不代表分类F1更高。','用相同定义比较S改变后的积分/边界离散误差；不是训练实验，也无开发/test性能。','量化有限采样误差与结构差异；误差不一定简单单调，不能把切片定理当成离散实现绝对精确。','21个可行、1个不可行配置均保留；切片定理是已有数学性质，不能当作R&B独有创新或效果证据。']
    if n=='fourier_feature_probes':
        return ['横轴为公共保守Nyquist上限的比例；纵轴为误差幅度除以全频带参考RMS。','两行CFP/OCT、三列train/validation/test；灰=构造初始，彩色=选中；线和带为3种子均值/SD。','较低仅意味着与指定插值场频谱的数值误差较小，不等价于诊断表现更好。','每集合固定16人、双眼、前3通道，对比初始和选中状态；不是全部1264/296/290人的频谱统计。','实际特征上的数值偏差可直接检查；分母使用全频带RMS，避免单个频点接近零时相对误差爆炸。','三维超平面Radon的一维傅里叶对应三维频谱径向直线；共同归一化频率不表示物理配准。']
    if n=='kernel_frequency_gains':
        return ['横轴为cycles/S sample；纵轴为每通道对的RMS矩阵增益。四面板分别为两个自身块和两个跨来源块。','三条颜色线对应r16/h512、r32/h1024、r64/h2048；线/带为3种子均值和SD，基于已选卷积权重。','增益越高只表示核更强地响应该频率；没有“越高越好”，它不等于实际信号功率。','在同一方向块内比较宽度下的频率响应；输出行=接收来源，输入列=发送来源。','部分核偏重低频；普通通信也可能出现相似形状，因此频率偏好不证明Radon几何独有优势。','它不是标量ramp滤波器，也不是完整BAK算子的奇异谱；核固定后无train/val/test三套不同权重。']
    raise ValueError(n)
for folder in ['sinogram_atlas','fourier_diagnostics']:
    for f in sorted((E/folder/'figures').glob('*.png')):
        title={'sinogram_atlas':'Radon图谱','fourier_diagnostics':'傅里叶与返回算子'}[folder]+'：'+f.stem
        notes=diagnostic_notes(f)
        if f.stem.startswith(('02_projected_flow','03_return_maps')):
            for region,part in [((0,.52),'上：输入与自身'),((.52,1),'下：跨来源与总返回')]:
                B.start(title+' / '+part,'固定SVD r16/h512、M32/S64/k3；三列依次train / validation / test；色限与已接受原图完全一致')
                B.image(f,region=region);B.end()
            notes[1]+=' 原四行拆成前两页上/下各两行，保留全部原坐标和色条。'
            guide(title,notes,str(f.relative_to(E)))
        else:chart(title,f,notes,str(f.relative_to(E)))
        for ext in ('png','pdf','svg'):
            original=f.with_suffix('.'+ext)
            if original.exists():shutil.copyfile(original,F/('diagnostic_'+original.name))

B.start('开发发现是否延续到test：逐项判断，不混同证据','下列是配对主比较的完整分类汇总；分类基于各集合自己的全局同时区间')
ct=collections.Counter((DD[r['contrast_id']].get('classification'),r.get('classification')) for r in primary)
B.table(['开发判断','Test判断','项数'],[[classification.get(d,d),classification.get(t,t),n] for (d,t),n in sorted(ct.items(),key=lambda x:-x[1])],[390,390,116],size=12,pad=5)
B.end('“均值同向”与“区间支持”不同；详细121项及独立重采样变化已在前文逐项列出')
B.start('带入扩展研究的结论与限制','结论不要求R&B获胜；本次test后没有再增加训练或改变模型')
B.table(['主张','匹配后的判断','下一步'],[['R&B有任务信号','参考组开发/test均有双分支均值提升；区间仍限制强度','新队列和骨干复核'],['Radon几何不可替代','空间/S轴打乱存在反例；目前不成立为确定主张','保留普通通信与结构对照'],['更多宽度/方向/桥必然更好','不支持普遍单调性；通道秩和计算量需共同解释','固定有限代表配置'],['宿主增强','训练收益约+2 pp且不确定；开关约14 pp是功能依赖','避免把删除效应当训练提升'],['可靠风险概率','判别改善与NLL退化并存，当前概率过于自信','独立开发校准，保持test独立'],['队列推广','925病例+925匹配对照；不等于自然分布或专家影像金标准','先扩有效病例、临床标签与外部证据']],[195,440,261],size=14,pad=7)
B.end('详细核查、完整图说明和配对数据与本册一起交付；不在现有test上挑新的赢家')
main_path=O/'RB_Matched_Development_Test_Guided_Report.pdf';B.finish(main_path)

# Exhaustive table, compact reproducible IDs connect the PDF and complete CSV.
P=Book(1440,810);P.start('R&B：804条研究引用的完整开发/test配对附表','每行是一个研究引用；相同模型可服务多个问题，去重模型视图为777。完整指标与SHA在同名CSV。')
P.table(['列','解释'],[['编号','R0001–R0804按原已锁定来源顺序编号，无按test排序。'],['来源/配置','来源保留历史/当前/冻结/多桥区别；M/S/k/r/h从结构或arm记录读取，不靠结果推断。'],['Seed/LR','训练种子和主干学习率；冻结组原生LR不适用，具体制度见regime列。'],['开发与test','各列均为该模型的CFP/OCT macro-F1与两分支平均，单位%。是同模型跨集合，不是同参与者配对。'],['最佳/停止','开发选中的epoch与实际停止epoch；0为父模型/身份初始化被保留，不能删掉。'],['全量CSV','另含precision/recall/AUROC/NLL/Brier/混淆矩阵、固定概率融合、参数/耗时/显存、完整配置与检查点SHA。'],['多宽度联合训练','一项训练展开多个宽度视图，不能把视图数叫作训练数；该制度单独标识。']],[270,1106],size=20,pad=12)
P.end('本附表与主报告构成完整交付；原生数据、个体预测和标识不导出')
sources={'current_direct_and_host':'Current','history_final213':'History213','s_axis18':'S-axis','completion72':'Complete72','current_frozen':'Frozen6','multidepth96':'Multidepth','independent_parents':'Parent'}
complete=[]
for i,r in enumerate(R):complete.append({'report_row':f'R{i+1:04d}',**r})
for off in range(0,len(complete),12):
    chunk=complete[off:off+12];P.start(f'完整配对附表：R{off+1:04d}–R{off+len(chunk):04d}','F1单位%：前三数为development CFP/OCT/平均，后三数为test CFP/OCT/平均；更高不自动等于显著')
    rows=[]
    for r in chunk:
        stc=json.loads(r['structure']) if r['structure'] else {};config=stc.get('id',r['arm'] or 'independent parents')
        if r['rho']:config+=' ρ='+str(Fraction(float(r['rho'])).limit_denominator(256))
        if stc.get('h'):config+=' h'+str(stc['h'])
        if stc.get('rho'):config+=' ρ='+str(Fraction(float(stc['rho'])).limit_denominator(256))
        if r['arm'] in ('mmtm_r4','mmtm_r8'):config={'mmtm_r4':'MMTM hidden256 / reduction4','mmtm_r8':'MMTM hidden128 / reduction8'}[r['arm']]
        if r['rho'] and r['arm']!='learned':config+=' r'+str(round(256*float(r['rho'])))+'/h'+str(round(8192*float(r['rho'])))
        if r['rho'] and r['arm']=='learned':config+=' h'+str(round(8192*float(r['rho'])))+' (CM)'
        regime=r['regime'];lr=r['backbone_lr'] or 'N/A'
        if 'frozen' in r['category'] or regime=='bridge_only':lr='N/A frozen'
        seedlr=r['seed']+' / '+lr
        rows.append([r['report_row'],sources[r['source_group']],config,seedlr]+[num(100*float(r[f'{s}_{k}'])) for s in ['development','test'] for k in ['cfp_macro_f1','oct_macro_f1','branch_mean_macro_f1']]+[(r['best_epoch'] or 'N/A')+' / '+(r['stop_epoch'] or 'N/A')])
    P.table(['编号','来源','配置 / ρ','Seed / LR','Dev CFP','Dev OCT','Dev平均','Test CFP','Test OCT','Test平均','最佳/停止'],rows,[60,95,315,130,100,100,100,100,100,100,130],size=13,pad=6)
    P.end('完整结构、制度、精确来源和SHA：all_804_matched_results.csv；不按test重新选择任何行')
appendix_path=O/'RB_All_804_Matched_Results.pdf';P.finish(appendix_path)
def csvwrite(path,rows):
    keys=list(rows[0]);f=path.open('w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]);f.close()
csvwrite(O/'all_804_matched_results.csv',complete);csvwrite(O/'all_121_primary_matched_comparisons.csv',paired_rows);csvwrite(O/'all_54_paired_perturbations.csv',perturb)
for name in ['development_statistics.json','test_statistics.json','test_minus_development_effect_change.json']:shutil.copyfile(E/'unified_test'/name,O/name)
guide_lines=['# R&B逐图中文阅读说明','\n与主报告页码、比较编号同步；不把幅度/能量/频率增益当性能。']
for g in GUIDES:
    guide_lines += [f'\n## 图解{g["number"]:02d}：{g["title"]}（主报告第{g["page"]}页）',f'\n来源：{g["source"]}']+[f'\n**{k}：** {v}' for k,v in g['explanations'].items()]
(O/'逐图阅读说明.zh-CN.md').write_text('\n'.join(guide_lines)+'\n')
(O/'page_index.json').write_text(json.dumps({'main':B.titles,'appendix':P.titles,'guides':GUIDES},ensure_ascii=False,indent=2))
verification=dict(source_manifest_sha256=sha(E/'artifact_manifest.json'),source_evidence_commit='2f9c6c67748fe166b784cd5f947536c1623c07e1',main_pages=len(B.titles),appendix_pages=len(P.titles),reference_rows=804,unique_model_views=777,primary_comparisons=121,paired_perturbation_checkpoints=54,diagnostic_figures=23,reading_guides=len(GUIDES),all_primary_development_test_and_change_present=True,all_804_have_both_splits=True,new_training=False,new_inference=False,new_bootstrap=False,private_data_exported=False,main_sha256=sha(main_path),appendix_sha256=sha(appendix_path))
(O/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification),flush=True)
