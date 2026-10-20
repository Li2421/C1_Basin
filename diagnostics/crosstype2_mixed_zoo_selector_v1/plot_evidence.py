"""Standalone scientific SVGs from sealed arrays and actual simulator traces."""
from core import *
from deliver_tables import TYPE_NAMES
from html import escape
from baselines import repair_class,CLASSES
from zoo_design import LOCKED

COLORS=['#777777','#2166ac','#762a83'];ROBOTS=['#2166ac','#b2182b','#1b7837','#e08214']


def text(x,y,value,size=12):return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}">{escape(str(value))}</text>'
def start(title,w=1120,h=460):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',text(25,28,title,18)]
def save(p,body):p.write_text('\n'.join([*body,'</svg>'])+'\n')


def distribution(data,methods,out):
    names=['均值固定','均值进度',*[f'样本{i}{kind}' for i in range(1,5) for kind in ('固定','进度')],'安全投影','500步切换','进度切换','右手固定','右手短单一','右手短单二']
    for kind in sorted(set(data['types'])):
        selected=methods['pipeline'][data['types']==kind].reshape(-1);frequency=np.bincount(selected,minlength=16)/len(selected);body=start(TYPE_NAMES[str(kind)]+'：冻结选中动作分布')
        for a,f in enumerate(frequency):
            x=45+a*66;height=280*f;color=COLORS[int(repair_class(a))]
            body.extend([f'<rect x="{x}" y="{345-height:.2f}" width="44" height="{height:.2f}" fill="{color}"/>',text(x,336-height,f'{100*f:.1f}%',10),text(x-7,365,names[a],9)])
        body.extend([text(45,410,'灰：不修正；蓝：修正参数；紫：持续右手规则。各控制器与状态等权。'),text(45,433,'候选编号相同的生成器动作，其实际η仍随状态变化。',11)])
        save(out/f'action_distribution_{kind}.svg',body)


def branch(model):return model['full'] if model.get('kind')=='probability_shrinkage' else model


def critic_curves(out):
    models=read(HERE/'SELECTED_MODELS.json')['outer']
    for held,group in models.items():
        body=start(TYPE_NAMES[held]+'：完整分支外层留出曲线（只报告，不选参）');histories={init:read(Path(branch(group['full_'+init])['checkpoint_path']).parent/'HISTORY.json') for init in ('scratch','warm')}
        for panel,metric in enumerate(('nll','ranking_accuracy')):
            x0=65+panel*540;y0=80;w=450;h=270
            values=[r['validation'][held][metric] for hist in histories.values() for r in hist if r['validation'][held][metric] is not None];high=max(values+[1.]) if metric=='nll' else 1.
            body.extend([text(x0,60,'留出负对数似然' if metric=='nll' else '同状态内候选排序准确率'),f'<path d="M{x0},{y0} V{y0+h} H{x0+w}" fill="none" stroke="#555"/>'])
            for j,(init,hist) in enumerate(histories.items()):
                points=[(x0+(r['step']-100)/1900*w,y0+h-r['validation'][held][metric]/high*h) for r in hist if r['validation'][held][metric] is not None]
                body.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{ROBOTS[j]}" stroke-width="2"/>')
                body.append(text(x0+j*225,y0+h+45,('从头训练' if init=='scratch' else '原critic初始化')+f'；源内选步数{branch(group["full_"+init])["step"]}',11))
            body.extend([text(x0,373,'100'),text(x0+w-30,373,'2000'),text(x0-40,y0+8,f'{high:.2f}'),text(x0-20,y0+h,'0')])
        body.append(text(65,435,'曲线为三个seed集成；排序仅包含经验成功率差≥0.25的候选对，checkpoint由源内验证选择。',11));save(out/f'critic_curves_{held}.svg',body)


def shrinkage(out):
    for row in read(HERE/'DEVELOPMENT_OUTER_FOLDS.json'):
        held=row['held_type'];body=start(TYPE_NAMES[held]+'：源内选择controller分支权重')
        for panel,key in enumerate(('full','no_state')):
            selection=row['source_only_selections'][key];curve=selection['lambda_curve'];vv=[r['source_mean_nll'] for r in curve];pad=max((max(vv)-min(vv))*.1,.001);lo=min(vv)-pad;hi=max(vv)+pad;x0=75+panel*540;y0=80;w=420;h=270
            body.extend([text(x0,60,'完整critic' if key=='full' else '匹配去状态消融'),f'<path d="M{x0},{y0} V{y0+h} H{x0+w}" fill="none" stroke="#555"/>'])
            points=[(x0+(r['full_weight']-.1)/.9*w,y0+h-(r['source_mean_nll']-lo)/(hi-lo)*h) for r in curve]
            body.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="#2166ac" stroke-width="2"/>')
            for r,(x,y) in zip(curve,points):
                color='#b2182b' if r['full_weight']==selection['selected']['full_weight'] else '#2166ac';body.extend([f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>',text(x-10,375,r['full_weight'])])
            body.extend([text(x0-48,y0+8,f'{hi:.3f}'),text(x0-48,y0+h,f'{lo:.3f}')])
        body.append(text(75,420,'横轴：保留controller分支概率的权重；纵轴：源类型内部负对数似然；红点为源内最优。',11));save(out/f'source_weights_{held}.svg',body)


def scaling(out):
    rows=read(HERE/'SCALING_RESULTS.json')['rows'];data=[]
    for axis,values,label in [('m',[1,2,4,7,10,13],'源类型数'),('repair_class_count',[1,2,3],'源类型最佳修法类别覆盖数')]:
        points=[]
        for value in values:
            selected=[r for r in rows if r[axis]==value];types=sorted(set(r['held_type'] for r in selected));means=[float(np.mean([r['regret_pp'] for r in selected if r['held_type']==t])) for t in types]
            if not means:continue
            points.append(dict(x=value,mean=float(np.mean(means)),sd=float(np.std(means)),held_types=len(types),subsets=len(selected)))
        body=start('源多样性规模曲线：'+label);x0=80;y0=70;w=950;h=300;low=min([p['mean']-p['sd'] for p in points]+[0.]);high=max([p['mean']+p['sd'] for p in points]+[1.]);pad=max((high-low)*.1,.1);low-=pad;high+=pad
        xy=lambda p:(x0+(p['x']-min(values))/max(max(values)-min(values),1)*w,y0+h-(p['mean']-low)/(high-low)*h)
        body.append(f'<path d="M{x0},{y0} V{y0+h} H{x0+w}" fill="none" stroke="#555"/>')
        body.append(f'<polyline points="{" ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in points)}" fill="none" stroke="#2166ac" stroke-width="2"/>')
        for p in points:
            x,y=xy(p);dy=p['sd']/(high-low)*h
            body.extend([f'<path d="M{x},{y-dy} V{y+dy} M{x-5},{y-dy} H{x+5} M{x-5},{y+dy} H{x+5}" stroke="#2166ac"/>',f'<circle cx="{x}" cy="{y}" r="5" fill="#2166ac"/>',text(x-10,395,p['x']),text(x-20,y-10,f'{p["mean"]:.2f}',10)])
        body.extend([text(10,80,f'{high:.1f}pp'),text(10,370,f'{low:.1f}pp'),text(80,425,'纵轴：留出类型对共同seed0样本内oracle的遗憾；误差线为跨留出类型标准差，非置信区间。',11),text(80,447,'类别覆盖数与源类型数相关；该曲线描述关联，不单独识别因果效应。',11)])
        save(out/f'scaling_{axis}.svg',body);data.append(dict(axis=axis,points=points))
    write(HERE/'SCALING_PLOT_DATA.json',data)


def trajectories(data,methods,out):
    index=read(HERE/'NEW_EVAL_TRACE_INDEX.json');lookup={(r['controller'],r['state_index'],r['candidate_index'],r['seed']):r for r in index};cfg=read(HERE/'BASE_PROTOCOL.json')['native_manifest']['scenario_config'];examples=[];lo=(data['labels']==1).mean(-1)
    for kind in sorted(set(data['types'])&set(LOCKED)):
        pool=np.flatnonzero(data['types']==kind);options=[]
        for i in pool:
            for s in range(24):options.append((-(lo[i,s,int(methods['pipeline'][i,s])]-lo[i,s,10]),str(data['names'][i]),s,i))
        selected=None
        for _,name,s,i in sorted(options):
            aa=[10,12,13,int(methods['pipeline'][i,s])]
            seed_order=sorted((0,1),key=lambda q:(-(int(data['labels'][i,s,aa[-1],q]==1)-int(data['labels'][i,s,10,q]==1)),q))
            for q in seed_order:
                if any(data['labels'][i,s,a,q]<0 for a in aa):continue
                if all((name,s,a,q) in lookup for a in aa):selected=(name,s,i,q,aa);break
            if selected:break
        assert selected is not None
        name,s,i,q,aa=selected;body=start(TYPE_NAMES[str(kind)]+'：同状态、同seed四种执行轨迹',1280,460)
        for panel,(a,label) in enumerate(zip(aa,['只加安全投影','按进度切换','固定持续右手规则','冻结pipeline选择'])):
            row=lookup[(name,s,a,q)];assert sha(row['trace_path'])==row['trace_sha256']
            with np.load(row['trace_path'],allow_pickle=False) as z:p=z['positions'];g=z['goals']
            x0=160+panel*320;y0=215;scale=40
            for radius,fill in [(cfg['outer_radius'],'none'),(cfg['obstacle_radius'],'#ddd')]:body.append(f'<circle cx="{x0}" cy="{y0}" r="{radius*scale}" fill="{fill}" stroke="#555"/>')
            for robot in range(4):
                xy=' '.join(f'{x0+x*scale:.2f},{y0-y*scale:.2f}' for x,y in p[:,robot]);color=ROBOTS[robot];body.append(f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="1.8"/>');body.append(f'<circle cx="{x0+p[0,robot,0]*scale}" cy="{y0-p[0,robot,1]*scale}" r="3" fill="{color}"/>');body.append(f'<circle cx="{x0+g[robot,0]*scale}" cy="{y0-g[robot,1]*scale}" r="5" fill="none" stroke="{color}"/>')
            body.extend([text(x0-100,380,label,13),text(x0-100,402,f'候选{a}；事件{row["outcome"]}；{len(p)-1}步',11)])
        body.extend([text(30,430,f'{name}；新状态{s}；seed{q}。仅用四臂均有已知事件的示例，按相对安全基线增益排序，仅作描述。',11),text(30,450,'实心点为起点，空心圈为目标。轨迹来自保存的仿真证据；timeout不是死锁判定。',11)])
        path=out/f'locked_trajectory_{kind}.svg';save(path,body);examples.append(dict(type=str(kind),controller=name,state_index=s,seed=q,candidates=aa,path=str(path)))
    write(HERE/'TRAJECTORY_EXAMPLES.json',examples)


def main():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in ('names','types','labels')}
    with np.load(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    out=HERE/'figures';out.mkdir(exist_ok=True);distribution(data,methods,out);critic_curves(out);shrinkage(out);scaling(out);trajectories(data,methods,out)
    import xml.etree.ElementTree as ET
    paths=sorted(out.glob('*.svg'))
    for path in paths:ET.parse(path)
    write(HERE/'FIGURE_MANIFEST.json',dict(utc=utc(),figures=[dict(path=str(p),sha256=sha(p)) for p in paths],xml_valid=True,simulation_read_only=True))
    (HERE/'FIGURES_README.md').write_text('全部SVG为可独立分享的静态科学图。绘图仅读取封存概率、选择、训练曲线和仿真轨迹，没有运行物理或更改事件。当前venv不含matplotlib等绘图库且按指令不安装包，沿用前轮确定性SVG输出。动作分布按控制器和状态等权；三类修法颜色固定。规模曲线的误差线为跨留出类型标准差，并非95%CI；类别覆盖和源类型数相关。典型轨迹要求同状态同seed四臂均有已知事件；先按状态的相对安全基线平均增益排序，再在该状态内优先选择配对增益更大的seed，用于描述；未知事件完整保留在总体统计中。\n')
    seal('FIGURES.receipt.json',[HERE/'FIGURE_MANIFEST.json',HERE/'FIGURES_README.md',HERE/'TRAJECTORY_EXAMPLES.json',HERE/'SCALING_PLOT_DATA.json',HERE/'plot_evidence.py',*paths])
    print('Generated and XML-checked',len(paths),'standalone SVG figures')


if __name__=='__main__':main()
