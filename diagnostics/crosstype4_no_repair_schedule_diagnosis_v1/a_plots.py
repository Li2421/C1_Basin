"""Standalone scientific SVG figures from simulation outcomes only."""
from common import *
from html import escape
from actions import TIMES
NAMES={'Flow':'环形交换场景流匹配策略','transformer_bc':'Transformer 行为克隆策略','conflict_based_search':'冲突搜索多机规划','centralized_mppi':'集中式模型预测路径积分控制','distributed_cross_entropy':'分布式交叉熵方法模型预测控制'}
def text(x,y,s,size=16,anchor='start'):return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(s))}</text>'
def doc(body):return '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="650" viewBox="0 0 1100 650"><rect width="1100" height="650" fill="white"/><g font-family="Noto Sans CJK SC, sans-serif" fill="#172337">'+''.join(body)+'</g></svg>'
def make(full,cover,result):
    folder=HERE/'figures';folder.mkdir(exist_ok=True);paths=[]
    for typ in sorted({d['type'] for d in full.values()}):
        selected=[d for d in cover.values() if d['type']==typ];start=np.mean([d[0]['old_selected'][:,0].mean() for d in selected])*100;r=result['coverage'][typ];ds=[r[k]['point'] for k in ('切换方式','critic 选 η 对','生成器覆盖')];levels=np.cumsum([start,*ds]);body=[text(70,45,NAMES[typ]+'：提议覆盖子集差距拆解',24),text(70,76,'开发证据；交叉拟合提议对；单位：成功率百分点',15)]
        lo=min(0,float(levels.min())-5);hi=max(100,float(levels.max())+5);y=lambda v:520-(v-lo)/(hi-lo)*390
        for value in np.linspace(lo,hi,6):body.extend([f'<path d="M80 {y(value)} H1040" stroke="#e0e5ec" fill="none"/>',text(68,y(value)+5,f'{value:.0f}',14,'end')])
        xs=[130,330,530,730,930];labels=['学习双 η·开局选切换步','切换方式','critic 选 η 对','生成器覆盖','人定双 η·按进度切换'];bars=[(0,start),*[(levels[k],levels[k+1]) for k in range(3)],(0,levels[-1])]
        for k,(a,b) in enumerate(bars):
            color='#496b9c' if k in (0,4) else ('#287d68' if b>=a else '#b74945');top=min(y(a),y(b));height=max(abs(y(a)-y(b)),1);body.append(f'<rect x="{xs[k]-50}" y="{top}" width="100" height="{height}" fill="{color}"/>');body.append(text(xs[k],top-9,f'{(b if k in (0,4) else b-a):+.2f}',16,'middle'));body.append(text(xs[k],555,labels[k],13,'middle'))
        body.extend([text(80,600,'总差距 = 切换方式 + critic 选 η 对 + 生成器覆盖；95% CI 见 DIAGNOSIS.md。',15),text(80,625,'只使用仿真记录；不修改控制器、物理、事件或成功标准。',14)]);p=folder/f'A_waterfall_{typ}.svg';p.write_text(doc(body));paths.append(p)
        data=[d for d in full.values() if d['type']==typ];curves={}
        for branch in ('selected','human'):
            pts=[]
            for mode in [*map(str,TIMES),'progress']:
                values=[]
                for d in data:
                    mask=~d['diagonal_mask']
                    if mask.any():values.append(d[0][branch+'_'+mode][mask].mean())
                pts.append(np.mean(values)*100 if values else np.nan)
            curves[branch]=pts
        body=[text(70,45,NAMES[typ]+'：切换步敏感性',24),text(70,76,'仅保留选中非对角 η 对的 controller-state，两条曲线使用同一批状态；两个 seed 同簇。',15)];y=lambda v:510-v*3.9;xx=[150,310,470,630,790,950]
        for v in (0,20,40,60,80,100):body.extend([f'<path d="M80 {y(v)} H1040" stroke="#e0e5ec"/>',text(68,y(v)+5,str(v),14,'end')])
        for branch,color,label in [('selected','#496b9c','选中 η 对·第 t 步切换 / 选中 η 对·按进度切换'),('human','#287d68','人定双 η·第 t 步切换 / 人定双 η·按进度切换')]:
            pts=curves[branch];coords=' '.join(f'{x},{y(v)}' for x,v in zip(xx,pts) if np.isfinite(v));body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x,v in zip(xx,pts):
                if np.isfinite(v):body.extend([f'<circle cx="{x}" cy="{y(v)}" r="5" fill="{color}"/>',text(x,y(v)-10,f'{v:.1f}',14,'middle')])
            body.append(text(80,595 if branch=='selected' else 620,label,16));
        for x,label in zip(xx,[*map(str,TIMES),'按进度']):body.append(text(x,545,label,16,'middle'))
        p=folder/f'A_timing_{typ}.svg';p.write_text(doc(body));paths.append(p)
    return paths
