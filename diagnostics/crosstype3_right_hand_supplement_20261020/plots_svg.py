"""Package-free scientific SVG fallback, using historical pure SVG primitives."""
from common import *
from svg_helpers import text,start,save,line,axes
from labels import type_name
import xml.etree.ElementTree as ET
COLORS=['#2166ac','#d95f02','#1b9e77','#7570b3','#e7298a','#333333']
OUT=HERE/'figures'
DATA=[]
def poly(body,xy,x,y,color,width=2):
    points=' '.join(f'{xy(a,b)[0]:.3f},{xy(a,b)[1]:.3f}' for a,b in zip(x,y) if np.isfinite(a) and np.isfinite(b))
    body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>')
def chart(name,title,series,xlabel,ylabel,xlimits=None,ylimits=None):
    body=start(title,1000,600);xx=np.concatenate([np.asarray(s[1]) for s in series]);yy=np.concatenate([np.asarray(s[2]) for s in series]);low=float(xx.min()) if len(xx) else 0;high=float(xx.max()) if len(xx) else 1;bottom=float(yy.min()) if len(yy) else 0;top=float(yy.max()) if len(yy) else 1
    if xlimits:low,high=xlimits
    if ylimits:bottom,top=ylimits
    if high<=low:high=low+1
    if top<=bottom:top=bottom+1
    xy=axes(body,100,80,800,400,low,high,bottom,top,xlabel,ylabel)
    for j,(label,x,y) in enumerate(series):poly(body,xy,x,y,COLORS[j%len(COLORS)]);body.append(text(80+(j%3)*300,545+(j//3)*20,label,11,COLORS[j%len(COLORS)]))
    save(OUT/(name+'.svg'),body);DATA.append(dict(name=name,series=[dict(label=s[0],x=s[1],y=s[2]) for s in series]))
def hist(name,title,series,xlabel,bins=None):
    allvalues=np.concatenate([np.asarray(v,float) for _,v in series]);allvalues=allvalues[np.isfinite(allvalues)]
    if bins is None:
        lo=float(allvalues.min()) if len(allvalues) else 0;hi=float(allvalues.max()) if len(allvalues) else 1
        if hi<=lo:lo-=.01;hi+=.01
        bins=np.linspace(lo,hi,21)
    curves=[]
    for label,values in series:
        counts,_=np.histogram(values,bins);curves.append((label,(bins[:-1]+bins[1:])/2,counts/max(len(values),1)))
    top=max(.1,max((float(np.max(s[2])) for s in curves),default=.1)*1.1);chart(name,title,curves,xlabel,'样本内频率',(float(bins[0]),float(bins[-1])),(0,top));DATA[-1]['bins']=bins
def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists();OUT.mkdir(exist_ok=True);r=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json');refs=r['human_reference_only'];rows=r['rows'];maps=read(HERE/'NEW_EVAL_METHOD_UIDS.json')
    for kind in ('transformer_bc','gradient_mpc','conflict_based_search'):
        selected=[x for x in rows if x['type']==kind]
        if not selected:continue
        early=np.asarray([x['eta_early'] for x in selected]);late=np.asarray([x['eta_late'] for x in selected])
        for k in range(3):
            low=min(early[:,k].min(),late[:,k].min(),refs['eta13'][k],refs['eta10'][k]);high=max(early[:,k].max(),late[:,k].max(),refs['eta13'][k],refs['eta10'][k]);pad=max((high-low)*.05,.01);body=start(type_name(kind)+f'：η分量{k+1}前后期散点',1000,600);xy=axes(body,100,80,800,400,low-pad,high+pad,low-pad,high+pad,'前期η','后期η')
            for a,b in zip(early[:,k],late[:,k]):
                x,y=xy(a,b);body.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.5" fill="#2166ac" fill-opacity="0.35"/>')
            for anchor,color in [('eta13','#d95f02'),('eta10','#111')]:
                x,y=xy(refs[anchor][k],refs[anchor][k]);body.extend([line(x-6,y,x+6,y,color),line(x,y-6,x,y+6,color),text(x-8,y-10,anchor,11,color,anchor='end')])
            body.append(text(80,550,'人工η13/η10只作参照；每个控制器/状态一个冻结选择。'));save(OUT/f'{kind}_eta_scatter_{k}.svg',body)
            bins=np.linspace(low-pad,high+pad,21);hist(f'{kind}_eta_hist_{k}',type_name(kind)+f'：η分量{k+1}直方图',[('前期',early[:,k]),('后期',late[:,k])],'η分量',bins)
            path=OUT/f'{kind}_eta_hist_{k}.svg';content=path.read_text();marks=[]
            for anchor,color in [('eta13','#d95f02'),('eta10','#111')]:
                x=100+(refs[anchor][k]-bins[0])/(bins[-1]-bins[0])*800;marks.extend([line(x,80,x,480,color,'4 3'),text(x,66,anchor,10,color,anchor='middle')])
            path.write_text(content.replace('</svg>','\n'.join(marks)+'\n</svg>'))
        hist(kind+'_eta_distances',type_name(kind)+'：到人工参照的距离',[(phase+' to '+anchor,[x[f'distance_{phase}_{anchor}'] for x in selected]) for phase in ('early','late') for anchor in ('eta13','eta10')],'原始η空间中的欧氏距离')
        values=[x['direction_cosine'] for x in selected if x['direction_cosine'] is not None];hist(kind+'_direction_cosine',type_name(kind)+f'：方向余弦（对角元未定义{len(selected)-len(values)}个）',[('方向余弦',values)],'后期−前期与η10−η13的方向余弦',np.linspace(-1,1,21))
        counts=[sum(x['diagonal'] for x in selected)]+[sum(x['switch_step']==t for x in selected) for t in (200,300,400,500,600)];body=start(type_name(kind)+'：选中切换时刻',1000,600);xy=axes(body,100,80,800,400,-.5,5.5,0,max(counts)*1.1,'对角、200、300、400、500、600步','控制器/状态数')
        for j,(label,count) in enumerate(zip(['对角','200','300','400','500','600'],counts)):
            x,y=xy(j,count);_,zero=xy(j,0);body.append(f'<rect x="{x-40:.3f}" y="{y:.3f}" width="80" height="{zero-y:.3f}" fill="#2166ac"/>');body.append(text(x,zero+55,label,12,anchor='middle'))
        save(OUT/f'{kind}_switch_times.svg',body)
        record=next(x for x in maps if x['type']==kind and x['state_index']==0);body=start(type_name(kind)+'：预定代表实例/状态0/seed0的保存轨迹',1280,530)
        for panel,(method,label) in enumerate([('all_safety_baseline','只加安全投影'),('all_S500','人工第500步切换'),('pipeline','冻结主方法')]):
            x0=30+panel*420;y0=90;size=370;xy=lambda a,b:(x0+(a+3.4)/6.8*size,y0+(3.4-b)/6.8*size);cx,cy=xy(0,0);body.extend([text(x0+size/2,65,label,14,anchor='middle'),f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{3.25/6.8*size:.3f}" fill="none" stroke="#777"/>',f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{.9/6.8*size:.3f}" fill="#ddd"/>']);rid=record['method_uids'][method][0];raw=read(HERE/'new_evaluation/raw'/f'{rid}.json');assert sha(raw['trace_path'])==raw['trace_sha256']
            with np.load(raw['trace_path'],allow_pickle=False) as z:p=z['positions'];goals=z['goals']
            for robot in range(4):
                poly(body,xy,p[:,robot,0],p[:,robot,1],COLORS[robot],1.4);x,y=xy(*p[0,robot]);body.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{COLORS[robot]}"/>');x,y=xy(*goals[robot]);body.extend([line(x-5,y,x+5,y,COLORS[robot]),line(x,y-5,x,y+5,COLORS[robot])])
        body.append(text(30,505,'首个合格实例；不按成功与否挑图。只读取真实保存的仿真轨迹。'));save(OUT/f'{kind}_representative_trajectories.svg',body)
    histories=[]
    for cfg in read(HERE/'TRANSFORMER_TRAINING_CONFIGS.json'):
        history=read(HERE/'policies'/cfg['name']/'TRAINING_HISTORY.json');histories.append((cfg['name'],[x['step'] for x in history],[x['dev_loss'] for x in history]))
    chart('transformer_training_curves','六个固定Transformer的开发集训练曲线',histories,'训练步数','固定开发集联合高斯NLL')
    features=read(HERE/'FEATURE_OUT_OF_RANGE.json')['rows'];valid=[x for x in features if x['out_of_range_fraction'] is not None];chart('unseen_feature_ranges','未见控制器特征越出训练范围的比例',[('24维中可观察分量',np.arange(len(valid)),[x['out_of_range_fraction'] for x in valid])],'实例顺序（FEATURE_OUT_OF_RANGE.json）','越界维度比例',ylimits=(0,1))
    selected=read(HERE/'SELECTED_MODELS.json');folds=read(HERE/'DEVELOPMENT_OUTER_FOLDS.json');records=[(x['held_type'],x['source_only_selection']) for x in folds]+[('all16',selected['all_types']['source_only_selection'])]
    for name,record in records:
        series=[]
        for mode,branch in record['branches'].items():series.append((mode,[x['step'] for x in branch['history']],[x['mean_nll'] for x in branch['history']]))
        chart('nested_source_curves_'+name,type_name(name)+'：源内checkpoint曲线',series,'训练步数','源内留一类型平均NLL')
        curve=record['lambda_curve'];chart('source_weights_'+name,type_name(name)+'：源内概率混合权重',[('源内NLL',[x['full_weight'] for x in curve],[x['source_mean_nll'] for x in curve])],'完整分支概率权重','源内平均NLL')
    files=sorted(OUT.glob('*.svg'))
    for p in files:ET.parse(p)
    write(HERE/'FIGURE_PLOT_DATA.json',dict(utc=utc(),curves_and_histograms=DATA,eta_scatter_source=str(HERE/'SELECTED_ETA_DESCRIPTIONS.json'),trajectory_source=str(HERE/'NEW_EVAL_METHOD_UIDS.json')))
    (HERE/'FIGURES_README.md').write_text('独立科学SVG图，采用上一轮已使用的标准SVG轴线与标注函数，不依赖Matplotlib，不安装任何包。所有数值和轨迹来自保存证据；三个人工参照轨迹固定使用第500步切换，代表实例固定为首个合格实例/state0/seed0。图表包括各未见类型的三分量散点和直方图、参照距离、方向余弦、切换时刻、轨迹、特征越界，以及六个策略与嵌套源内选择曲线。所有SVG通过XML解析核查，曲线/分箱数据保留FIGURE_PLOT_DATA.json。\n')
    seal('FIGURES.receipt.json',files+[HERE/'FIGURE_PLOT_DATA.json',HERE/'FIGURES_README.md',HERE/'plots_svg.py',HERE/'svg_helpers.py',HERE/'SVG_HELPERS_SOURCE.json',HERE/'CONFIRMATORY_RESULTS.receipt.json'],static_saved_evidence_only=True,matplotlib_unavailable_no_install=True)
    print('SVG figures generated and parsed:',len(files),flush=True)
if __name__=='__main__':main()
