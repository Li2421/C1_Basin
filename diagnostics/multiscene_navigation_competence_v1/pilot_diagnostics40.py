"""Aggregate saved evidence only; route deviation is never a gate or selection rule."""
from common import *
from collect39 import POLICIES,checkpoint
from expert_results import NAMES
import csv

def summarize(raw):
    counts={key:sum(r['numerical_failure'] if key=='unknown' else r[key] for r in raw) for key in ['success','collision','timeout','unknown']}
    assert sum(counts.values())==len(raw)
    deviations=[];total=0;outside=0;clearances=[]
    for r in raw:
        trace=np.load(r['trace_path']);positions=trace['positions'][:,0];path=np.asarray(r['case']['state']['paths'][0]);delta=np.diff(path,axis=0)
        v=positions[:,None]-path[None,:-1];t=np.clip(np.sum(v*delta[None],-1)/np.maximum(np.sum(delta*delta,-1),1e-30),0,1)
        distance=np.linalg.norm(v-t[...,None]*delta[None],axis=-1).min(-1)
        deviations.append(bool(np.any(distance>.5)));outside+=int(np.sum(distance>.5));total+=len(distance)
        if len(trace['wall_clearance']):clearances.append(float(np.min(trace['wall_clearance'])))
    return dict(episodes=len(raw),**counts,rates={k:v/len(raw) for k,v in counts.items()} if raw else None,route_deviation_episode_rate=float(np.mean(deviations)) if deviations else None,route_deviation_state_rate=outside/total if total else None,route_deviation_definition='distance to original Geometry.path >0.5m; solely descriptive',minimum_wall_clearance_quantiles=dict(zip(['min','p1','p5','p25','p50','p75','p95','p99','max'],np.percentile(clearances,[0,1,5,25,50,75,95,99,100]).tolist())) if clearances else None)

def failure_plot(family,raw,title,output):
    maps=[s for s in read(HERE/'PILOT_GEOMETRY_FROZEN.json')[family] if s['role'] in ['train','train_new']]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1000" viewBox="0 0 1440 1000">','<rect width="1440" height="1000" fill="white"/>',f'<text x="20" y="25" font-size="18">{NAMES[family]}：{title}（红：撞墙；橙：超时；紫：未知）</text>']
    for m,s in enumerate(maps):
        ox=15+(m%8)*178;oy=55+(m//8)*115;scale=10
        parts += [f'<text x="{ox}" y="{oy-5}" font-size="11">训练几何 {m+1}</text>',f'<rect x="{ox}" y="{oy}" width="160" height="80" fill="#fafafa" stroke="#333"/>']
        for x0,y0,x1,y1 in s['rectangles']:parts.append(f'<rect x="{ox+(x0+8)*scale}" y="{oy+(4-y1)*scale}" width="{(x1-x0)*scale}" height="{(y1-y0)*scale}" fill="#999"/>')
        for r in raw:
            if r['case']['state']['geometry_seed']!=s['geometry_seed'] or r['success']:continue
            x,y=np.load(r['trace_path'])['positions'][-1,0];color='#d62728' if r['collision'] else '#e48b0f' if r['timeout'] else '#7c3db2'
            parts.append(f'<circle cx="{ox+(x+8)*scale}" cy="{oy+(4-y)*scale}" r="1.8" fill="{color}" fill-opacity=".65"/>')
    parts.append('</svg>');output.parent.mkdir(exist_ok=True);output.write_text('\n'.join(parts))

def read_stage(stage):
    assert (HERE/f'{stage}_POSTFLIGHT.json').exists()
    return [read(HERE/stage/'raw'/f"{c['rollout_uid']}.json") for c in read(HERE/f'{stage}_CASES.json')]

def analyze(family):
    result={};table=[]
    for kind,seed in POLICIES:
        rounds={}
        for r in range(4):
            stage=f'data39_{family}_round_{r+1}_{kind}_{seed}' if r<3 else f'descriptive40_{family}_training_final'
            if not (HERE/f'{stage}_POSTFLIGHT.json').exists():continue
            raw=[x for x in read_stage(stage) if x['case']['policy_kind']==kind and x['case']['policy_seed']==seed]
            assert len(raw)==64*16
            stats=summarize(raw);rounds[str(r)]=dict(stats,source_stage=stage,checkpoint=str(checkpoint(family,kind,seed,r)))
            manifest=read(HERE/'datasets39'/f'{family}_{kind}_{seed}_round_{r}.json')
            table.append(dict(family=family,kind=kind,seed=seed,round=r,**manifest['transitions'],**{k:stats[k] for k in ['episodes','success','collision','timeout','unknown','route_deviation_episode_rate','route_deviation_state_rate']},checkpoint=str(checkpoint(family,kind,seed,r))))
            if r in [0,3]:
                kind_name='确定性行为克隆策略' if kind=='deterministic' else '混合密度行为克隆策略'
                plot=HERE/'policy_diagnostics'/f'{family}_{kind}_{seed}_round_{r}_failure_positions.svg';failure_plot(family,raw,f'第{r}轮{kind_name}，种子{seed}实际失败终点',plot);rounds[str(r)]['failure_plot']=str(plot)
        result[f'{kind}_{seed}']=rounds
    old_stage=f'descriptive40_{family}_old_evaluation'
    old={f'{k}_{s}':summarize([r for r in read_stage(old_stage) if r['case']['policy_kind']==k and r['case']['policy_seed']==s]) for k,s in POLICIES} if (HERE/f'{old_stage}_POSTFLIGHT.json').exists() else None
    write(HERE/f'PILOT_POLICY_DIAGNOSTICS_{family}.json',dict(family=family,training=result,old_evaluation_descriptive=old))
    if table:
        p=HERE/f'TRAINING_ROUND_COUNTS_{family}.csv'
        with p.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    return result

if __name__=='__main__':analyze(sys.argv[1])
