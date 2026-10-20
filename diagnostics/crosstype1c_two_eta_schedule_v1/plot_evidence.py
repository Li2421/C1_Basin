"""Standalone SVG scientific plots from sealed numerical evidence only.

No plotting package is installed in either available Python. Standard SVG is
generated directly, with plot arrays preserved in JSON; no package is added.
"""
from common import *
from deliver_tables import TYPE_NAMES
import html,xml.etree.ElementTree as ET
COLORS=['#2166ac','#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#a6761d','#008b8b','#333333']
KINDS=sorted(TYPE_NAMES);COLOR=dict(zip(KINDS,COLORS));LOW=[.5,-.5,0.];HIGH=[1.25,.5,.75];COMPONENTS=['目标方向权重','安全垂直方向权重','相对方向权重']
def text(x,y,s,size=12,fill='#111',anchor='start'):return f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def start(title,w=1280,h=650):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',text(25,30,title,19)]
def save(path,body):path.write_text('\n'.join([*body,'</svg>'])+'\n')
def line(x1,y1,x2,y2,color='#777',dash=''):return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-dasharray="{dash}"/>'
def axes(body,x,y,w,h,xmin,xmax,ymin,ymax,xlabel,ylabel):
    body.extend([line(x,y,x,y+h),line(x,y+h,x+w,y+h),text(x+w/2,y+h+39,xlabel,12,anchor='middle'),text(x,y-12,ylabel,12)])
    for value in np.linspace(xmin,xmax,5):
        px=x+(value-xmin)/(xmax-xmin)*w;body.extend([line(px,y+h,px,y+h+5),text(px,y+h+20,f'{value:.2f}',10,anchor='middle')])
    for value in np.linspace(ymin,ymax,5):
        py=y+h-(value-ymin)/(ymax-ymin)*h;body.extend([line(x-5,py,x,py),text(x-9,py+4,f'{value:.2f}',10,anchor='end')])
    return lambda a,b:(x+(a-xmin)/(xmax-xmin)*w,y+h-(b-ymin)/(ymax-ymin)*h)
def legend(body,y=500):
    for i,kind in enumerate(KINDS):
        x=35+(i%3)*410;yy=y+(i//3)*25;body.append(f'<rect x="{x}" y="{yy-10}" width="12" height="12" fill="{COLOR[kind]}"/>');body.append(text(x+19,yy,TYPE_NAMES[kind],11))
def footer(body,s='每控制器/新状态一个冻结选择；仅读取保存的数值，不改变控制、物理或成功标准。'):body.append(text(30,628,s,11))
def scatter(out,rows,refs):
    for k,name in enumerate(COMPONENTS):
        body=start('冻结主方法的双η选择：'+name);pad=(HIGH[k]-LOW[k])*.04;xy=axes(body,100,75,1020,345,LOW[k]-pad,HIGH[k]+pad,LOW[k]-pad,HIGH[k]+pad,'前期η分量','后期η分量');x1,y1=xy(LOW[k],LOW[k]);x2,y2=xy(HIGH[k],HIGH[k]);body.append(line(x1,y1,x2,y2,'#aaa','4 4'))
        for r in rows:
            x,y=xy(r['eta_early'][k],r['eta_late'][k]);body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="{COLOR[r["type"]]}" fill-opacity="0.43"/>')
        for key,label in [('eta13','η13固定参照'),('eta10','η10固定参照')]:
            value=refs[key][k];x,y=xy(value,value);body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="white" stroke="#111"/>');body.append(text(x-8,y-10,label,10,anchor='end'))
        x,y=xy(refs['eta13'][k],refs['eta10'][k]);body.extend([line(x-6,y,x+6,y,'#c00'),line(x,y-6,x,y+6,'#c00'),text(x-10,y-12,'η13→η10人工参照',10,'#c00',anchor='end')]);legend(body);footer(body,'每控制器/新状态一点，Flow实例较多；主统计另按类型等权。人工参照不在新候选中。');save(out/f'eta_pair_scatter_component{k}.svg',body)
def histogram(out,rows,refs):
    plot=[]
    for k,name in enumerate(COMPONENTS):
        body=start('冻结主方法的η分布：'+name);bins=np.linspace(LOW[k],HIGH[k],21);centers=(bins[:-1]+bins[1:])/2
        for j,(field,label) in enumerate([('eta_early','前期η'),('eta_late','后期η')]):
            curves={}
            for kind in KINDS:
                values=[r[field][k] for r in rows if r['type']==kind];counts,_=np.histogram(values,bins);curves[kind]=counts/max(len(values),1);plot.append(dict(component=k,phase=field,type=kind,bins=bins,counts=counts,within_type_fraction=curves[kind]))
            top=max(.1,max(float(v.max()) for v in curves.values())*1.1);xy=axes(body,80+j*625,85,510,320,LOW[k],HIGH[k],0,top,label,'类型内频率')
            for kind,values in curves.items():
                points=' '.join(f'{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}' for x,y in zip(centers,values));body.append(f'<polyline points="{points}" fill="none" stroke="{COLOR[kind]}" stroke-width="1.7"/>')
            for key,color in [('eta13','#444'),('eta10','#c00')]:
                x,y=xy(refs[key][k],0);body.append(line(x,85,x,405,color,'4 3'));body.append(text(x,68,f'{key}={refs[key][k]:.3f}',10,color,anchor='middle'))
        legend(body);footer(body,'每条曲线在各类型内部归一化；η13、η10仅标示历史参照，不进入新候选或critic监督。');save(out/f'eta_pair_hist_component{k}.svg',body)
    return plot
def cosine(out,rows):
    body=start('双η方向与历史η10−η13方向的夹角余弦');bins=np.linspace(-1,1,22);centers=(bins[:-1]+bins[1:])/2;curves={};data=[]
    for kind in KINDS:
        selected=[r for r in rows if r['type']==kind];values=[r['direction_cosine'] for r in selected if r['direction_cosine'] is not None];counts,_=np.histogram(values,bins);curves[kind]=counts/max(len(values),1);data.append(dict(type=kind,supported=len(values),undefined_diagonal=len(selected)-len(values),bins=bins,counts=counts))
    top=max(.1,max(float(v.max()) for v in curves.values())*1.1);xy=axes(body,100,80,1050,335,-1,1,0,top,'方向夹角余弦（原始η坐标）','有定义样本中的类型内频率')
    for kind,v in curves.items():body.append('<polyline points="'+' '.join(f'{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}' for x,y in zip(centers,v))+f'" fill="none" stroke="{COLOR[kind]}" stroke-width="1.7"/>')
    legend(body);footer(body,'对角元的方向未定义，单独计数并排除直方图，不把它们伪造为余弦0。');save(out/'direction_cosine.svg',body);return data
def times(out,distributions,method):
    colors=['#aaa','#2166ac','#4393c3','#1b9e77','#e6ab02','#d95f02'];labels=['无实际η切换','200步','300步','400步','500步','600步'];body=start(('自学时间表' if method=='self_schedule' else '固定第500步切换')+'：逐类型选中切换时刻分布',h=650)
    for i,kind in enumerate(KINDS):
        r=distributions[kind][method];counts=[r['diagonal_no_effective_switch'],*[r['non_diagonal_counts'][str(t)] for t in (200,300,400,500,600)]];y=80+i*43;body.append(text(295,y+17,TYPE_NAMES[kind],11,anchor='end'));x=315
        for count,color in zip(counts,colors):
            width=count/r['total']*900;body.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="25" fill="{color}"/>')
            if width>35:body.append(text(x+width/2,y+17,f'{count/r["total"]*100:.1f}%',10,'white',anchor='middle'))
            x+=width
    for i,(label,color) in enumerate(zip(labels,colors)):
        x=40+(i%3)*410;y=515+(i//3)*28;body.append(f'<rect x="{x}" y="{y-11}" width="14" height="14" fill="{color}"/>');body.append(text(x+22,y,label,12))
    footer(body,'对角元没有有意义的切换时刻；非对角元的所选时刻不等同于episode结束前已执行过切换。');save(out/f'switch_times_{method}.svg',body)
def critic_curves(out):
    models=read(HERE/'SELECTED_MODELS.json');folds=read(HERE/'DEVELOPMENT_OUTER_FOLDS.json');rank_groups=[('diagonal','对角元之间'),('non_diagonal','非对角对之间'),('cross','对角元与非对角对'),('same_pair_different_time','同对不同时刻')]
    for kind in KINDS:
        branch=models['outer'][kind]['full']['full'];folder=Path(branch['checkpoint_path']).parent;history=read(folder/'HISTORY.json');steps=np.asarray([r['step'] for r in history]);nll=np.asarray([r['validation'][kind]['nll'] for r in history]);body=start(TYPE_NAMES[kind]+'：完整分支外层曲线（只报告，不选参）');top=max(1.,float(nll.max())*1.15);xy=axes(body,80,90,480,320,100,2000,0,top,'训练步数','留出负对数似然');body.append('<polyline points="'+' '.join(f'{xy(s,n)[0]:.2f},{xy(s,n)[1]:.2f}' for s,n in zip(steps,nll))+'" fill="none" stroke="#2166ac" stroke-width="2"/>');x,_=xy(branch['step'],0);body.append(line(x,90,x,410,'#c00','5 4'));body.append(text(80,460,f'源内所选checkpoint：第{branch["step"]}步',12))
        xy=axes(body,700,90,480,320,100,2000,0,1,'训练步数','同状态可辨识候选排序准确率')
        for j,(group,label) in enumerate(rank_groups):
            points=[(r['step'],r['validation'][kind]['ranking'][group]['accuracy']) for r in history if r['validation'][kind]['ranking'][group]['accuracy'] is not None]
            if points:body.append('<polyline points="'+' '.join(f'{xy(s,v)[0]:.2f},{xy(s,v)[1]:.2f}' for s,v in points)+f'" fill="none" stroke="{COLORS[j]}" stroke-width="1.8"/>')
            body.append(text(35+(j%2)*610,515+(j//2)*27,label,12,COLORS[j]))
        footer(body,'三个seed集成；经验成功率差≥0.25的已知候选对；无可辨识比较不补成0，支持数见HISTORY。');save(out/f'critic_curves_{kind}.svg',body)
        row=next(r for r in folds if r['held_type']==kind)['source_only_selection'];curve=row['lambda_curve'];body=start(TYPE_NAMES[kind]+'：源类型内部的controller分支权重选择');values=np.asarray([r['source_mean_nll'] for r in curve]);low=float(values.min())-.005;high=float(values.max())+.005;xy=axes(body,100,90,1050,340,.1,1,low,high,'保留完整controller分支的概率权重','源内留一类型负对数似然');body.append('<polyline points="'+' '.join(f'{xy(r["full_weight"],r["source_mean_nll"])[0]:.2f},{xy(r["full_weight"],r["source_mean_nll"])[1]:.2f}' for r in curve)+'" fill="none" stroke="#2166ac" stroke-width="2"/>')
        for r in curve:
            x,y=xy(r['full_weight'],r['source_mean_nll']);chosen=r['full_weight']==row['selected']['full_weight'];body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{"#c00" if chosen else "#2166ac"}"/>')
        footer(body,'红点仅由源类型内部NLL决定，不读取该外层留出类型来挑权重。');save(out/f'source_weights_{kind}.svg',body)
def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists();out=HERE/'figures';out.mkdir(exist_ok=True);r=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json');rows=r['rows'];refs=r['human_reference_only'];scatter(out,rows,refs);hist=histogram(out,rows,refs);cos=cosine(out,rows);switches=read(HERE/'CONFIRMATORY_RESULTS.json')['endpoints']['endpoint2']['switch_distributions']
    for method in ('self_schedule','fixed500'):times(out,switches,method)
    critic_curves(out);write(HERE/'FIGURE_PLOT_DATA.json',dict(utc=utc(),eta_histograms=hist,direction_histograms=cos,switches=switches,selected_eta_rows_source=str(HERE/'SELECTED_ETA_DESCRIPTIONS.json')))
    paths=sorted(out.glob('*.svg'))
    for p in paths:ET.parse(p)
    (HERE/'FIGURES_README.md').write_text('# 图像证据\n\n所有SVG均为独立静态矢量文件，数值来自冻结选择和已保存的实验结果，没有额外rollout。既有venv和系统Python均无Matplotlib/Plotly；按不安装包约束用标准SVG输出，原始绘图数组保留在FIGURE_PLOT_DATA.json。\n\n三个η分量各给前期/后期散点与直方图，类型颜色固定，η13/η10只画历史参照。方向余弦对零差向量未定义，明确排除并保留计数。切换图把对角元单列为无实际切换；实际执行过后期η的比例另见PIPELINE_SWITCH_EXECUTION。外层NLL/排序曲线仅诊断，checkpoint及权重只按源内选择。\n')
    seal('FIGURES.receipt.json',[HERE/'FIGURE_PLOT_DATA.json',HERE/'FIGURES_README.md',HERE/'plot_evidence.py',*paths]);print('Wrote',len(paths),'standalone scientific SVG plots',flush=True)
if __name__=='__main__':main()
