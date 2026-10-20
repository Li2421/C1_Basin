"""Static SVG evidence from stored predictions and simulator trajectories only."""
from core import *
from deliver_tables import TYPE_NAMES
from html import escape

COLORS=['#2166ac','#b2182b','#1b7837','#762a83']
def svg_start(title,width=960,height=440):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',f'<text x="24" y="27" font-family="sans-serif" font-size="18">{escape(title)}</text>']
def save(path,body):path.write_text('\n'.join(body+['</svg>'])+'\n')
def text(x,y,value,size=12):return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}">{escape(str(value))}</text>'


def distribution(data,fixed,out):
    picks=fixed['pipeline'];types=data['types']
    labels=['均值固定','均值进度','样本1固定','样本1进度','样本2固定','样本2进度','样本3固定','样本3进度','样本4固定','样本4进度','安全投影','500步切换','进度切换']
    for t in sorted(set(types)):
        freq=np.bincount(picks[types==t].reshape(-1),minlength=13)/picks[types==t].size
        body=svg_start(TYPE_NAMES[str(t)]+'：冻结模型在新state的选中动作分布')
        for a,f in enumerate(freq):
            x=48+a*68;h=280*f
            body.extend([f'<rect x="{x}" y="{335-h}" width="47" height="{h}" fill="{"#2166ac" if a<10 else "#d95f02"}"/>',
                text(x,327-h,f'{100*f:.1f}%'),text(x-8,359,labels[a],10)])
        body.append(text(45,402,'生成器：蓝色；锚点：橙色。每个controller和state等权；频率不读取成败标签。'))
        save(out/f'action_distribution_{t}.svg',body)


def curves(out):
    models=read(HERE/'SELECTED_MODELS.json')['outer']
    for held,group in models.items():
        histories={init:read(Path(group['full_'+init]['checkpoint_path']).parent/'HISTORY.json') for init in ('scratch','warm')}
        body=svg_start(TYPE_NAMES[held]+'：完整分支外层留出曲线（仅报告，不用于该折选参）')
        for panel,metric in enumerate(('nll','ranking_accuracy')):
            x0=55+panel*465;y0=70;w=390;h=270
            values=[r['validation'][held][metric] for hist in histories.values() for r in hist if r['validation'][held][metric] is not None]
            low=0.;high=max(values+[1.]) if metric=='nll' else 1.
            body.append(text(x0,y0-15,'留出负对数似然' if metric=='nll' else '同一state候选排序准确率'))
            body.append(f'<path d="M {x0} {y0} V {y0+h} H {x0+w}" fill="none" stroke="#555"/>')
            for j,(init,hist) in enumerate(histories.items()):
                points=[(x0+(r['step']-100)/1900*w,y0+h-(r['validation'][held][metric]-low)/(high-low)*h) for r in hist if r['validation'][held][metric] is not None]
                body.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{COLORS[j]}" stroke-width="2"/>')
                body.append(text(x0+j*180,y0+h+43,('从头训练' if init=='scratch' else '旧critic初始化')+f'；选中步数{group["full_"+init]["step"]}',11))
            body.extend([text(x0,y0+h+18,'100'),text(x0+w-35,y0+h+18,'2000'),text(x0-35,y0+8,f'{high:.2f}'),text(x0-20,y0+h,'0')])
        save(out/f'critic_curves_{held}.svg',body)


def shrinkage_curves(out):
    for fold in read(HERE/'DEVELOPMENT_OUTER_FOLDS_V2B.json'):
        held=fold['held_type'];body=svg_start(TYPE_NAMES[held]+'：正controller权重的源类型内选择')
        for panel,key in enumerate(('full','no_state')):
            record=fold['source_only_selections'][key];rows=record['lambda_curve'];values=[r['source_mean_nll'] for r in rows]
            low=min(values);high=max(values);pad=max((high-low)*.1,.001);low-=pad;high+=pad
            x0=70+panel*465;y0=70;w=360;h=270
            body.extend([text(x0,y0-15,'完整critic' if key=='full' else '匹配去state消融'),
                f'<path d="M {x0} {y0} V {y0+h} H {x0+w}" fill="none" stroke="#555"/>'])
            points=[(x0+(r['full_weight']-.1)/.9*w,y0+h-(r['source_mean_nll']-low)/(high-low)*h) for r in rows]
            body.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="#2166ac" stroke-width="2"/>')
            for row,(x,y) in zip(rows,points):
                body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{"#b2182b" if row["full_weight"]==record["selected"]["full_weight"] else "#2166ac"}"/>')
                body.append(text(x-8,y0+h+20,row['full_weight']))
            body.extend([text(x0-45,y0+8,f'{high:.3f}'),text(x0-45,y0+h,f'{low:.3f}'),
                text(x0,y0+h+48,'横轴：保留controller信息的分支权重；纵轴：源内NLL',11)])
        save(out/f'source_shrinkage_{held}.svg',body)


def trajectories(data,fixed,out):
    types=data['types'];lo=(data['labels']==1).mean(-1);cfg=read(HERE/'BASE_PROTOCOL.json')['native_manifest']['scenario_config'];log=[]
    for t in sorted(set(types)):
        pool=np.flatnonzero(types==t);selected=None
        # Descriptive example: prefer the largest observed pipeline gain over
        # projected safety, then lexicographic controller/state/seed. No reruns.
        choices=[]
        for i in pool:
            for s in range(24):choices.append((-(lo[i,s,fixed['pipeline'][i,s]]-lo[i,s,10]),int(i),s))
        for _,i,s in sorted(choices):
            aa=[10,12,int(fixed['pipeline'][i,s])]
            for q in range(2):
                if all(data['traces'][i,s,a,q] for a in aa):selected=(i,s,q,aa);break
            if selected:break
        if selected is None:log.append(dict(type=str(t),gap='No stored common-state trace triplet'));continue
        i,s,q,aa=selected;body=svg_start(TYPE_NAMES[str(t)]+'：同一state三种执行轨迹',960,420)
        for panel,(a,label) in enumerate(zip(aa,['只加安全投影','一律按进度切换','生成器与critic选中动作'])):
            path=Path(data['traces'][i,s,a,q]);assert path.resolve().is_relative_to(HERE)
            assert sha(path)==data['trace_hashes'][i,s,a,q]
            with np.load(path,allow_pickle=False) as z:positions=z['positions'];goals=z['goals']
            x0=160+panel*315;y0=210;scale=42
            for radius,fill in ((cfg['outer_radius'],'none'),(cfg['obstacle_radius'],'#ddd')):
                body.append(f'<circle cx="{x0}" cy="{y0}" r="{radius*scale}" fill="{fill}" stroke="#555"/>')
            for robot in range(positions.shape[1]):
                xy=positions[:,robot];points=' '.join(f'{x0+x*scale:.2f},{y0-y*scale:.2f}' for x,y in xy)
                color=COLORS[robot%len(COLORS)];gx,gy=goals[robot]
                body.extend([f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>',
                    f'<circle cx="{x0+xy[0,0]*scale}" cy="{y0-xy[0,1]*scale}" r="3" fill="{color}"/>',
                    f'<circle cx="{x0+gx*scale}" cy="{y0-gy*scale}" r="5" fill="none" stroke="{color}"/>'])
            body.append(text(x0-115,370,label+f'；候选{a}；标签{int(data["labels"][i,s,a,q])}',11))
        body.append(text(24,401,f'{data["names"][i]}；state索引{s}；seed {q}。展示例按相对安全投影增益排序，仅用于描述。',11))
        path=out/f'trajectories_{t}.svg';save(path,body)
        log.append(dict(type=str(t),controller=str(data['names'][i]),state_uid=str(data['state_uids'][s]),seed=q,
            actions=aa,file=str(path),selection='Largest observed pipeline gain over safety; lexical ties; descriptive only'))
    write(HERE/'TRAJECTORY_EXAMPLES.json',log)


def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_CHOICES.npz',allow_pickle=False) as z:fixed={k:z[k] for k in z.files if k not in ('names','types','state_uids')}
    distribution(data,fixed,out);curves(out);shrinkage_curves(out);trajectories(data,fixed,out)
    seal('FIGURES.receipt.json',[*sorted(out.glob('*.svg')),HERE/'TRAJECTORY_EXAMPLES.json',HERE/'plot_evidence.py',
        HERE/'NEW_EVAL_BUNDLE.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json'],simulation_evidence_only=True)


if __name__=='__main__':main()
