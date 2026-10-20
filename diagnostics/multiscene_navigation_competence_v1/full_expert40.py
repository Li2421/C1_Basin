"""Full-family fixed expert gate; same frozen worker and criterion as both pilots."""
from common import *
from geometry import GeneratedGeometry,task
from bottleneck_family.scenario import Config
from design_generated import identities
from registry import register
from full_geometry40 import require_pilot,verify_full,read_catalog
from pilot_diagnostics40 import summarize
from deliver_failed_expert_pilot import NAMES
import math

def prepare(family):
    require_pilot();verify_full();train=[s for s in read_catalog(family) if s['role'] in ['train','train_new']];assert len(train)==64
    cases=[];rows=[];rejected=[]
    for m,s in enumerate(train):
        geo=GeneratedGeometry(Config(**s['config']),s['rectangles'])
        for j in range(32):
            base=202660000000+'ABCDEFG'.index(family)*100000000+m*100000+j*100
            for k in range(100):
                try:p,g,paths,_=task(geo,family,base+k);break
                except ValueError as error:rejected.append(dict(family=family,map=m,episode=j,seed=base+k,reason=str(error)))
            else:raise RuntimeError('Preregistered expert task block exhausted')
            i=j%4;h=int(math.ceil(2.5*np.linalg.norm(np.diff(paths[i],axis=0),axis=1).sum()/.5/.05/100)*100)
            state=dict(s,role='expert_gate',map_index=m,index=j,task_seed=base+k,alias=f'round40_expert_{family}_{m}_{j}',positions=p[i:i+1].tolist(),goals=g[i:i+1].tolist(),velocities=np.zeros((1,2)).tolist(),paths=[paths[i].tolist()],H=h,config=dict(s['config'],num_agents=1,max_steps=h))
            c,r=identities(state,'navigation_field','navigation');cases.append(c);rows+=r
    stage='full_expert_'+family;write(HERE/f'FULL_{family}_EXPERT_TASK_REJECTIONS.json',rejected);register(stage,cases,rows)
    seal(f'FULL_{family}_EXPERT_GATE_IMPLEMENTATION.receipt.json',[HERE/n for n in ['navigation_field.py','observation39.py','geometry.py','single_agent.py','full_expert40.py','expert_worker.py','NAVIGATION_UNIT_TESTS.json']],before_family_expert_labels=True,episodes=2048,success_threshold=.99,zero_collisions=True)
    return stage

def analyze(family):
    stage='full_expert_'+family;assert (HERE/f'{stage}_POSTFLIGHT.json').exists();cases=read(HERE/f'{stage}_CASES.json');raw=[read(HERE/stage/'raw'/f"{c['rollout_uid']}.json") for c in cases];assert len(raw)==2048
    stats=summarize(raw);counts={k:stats[k] for k in ['success','collision','timeout','unknown']};lower=counts['success']/2048;upper=(counts['success']+counts['unknown'])/2048
    result=dict(name=NAMES[family],episodes=2048,maps=64,episodes_per_map=32,counts=counts,success_rate=lower,success_upper=upper,collision_rate=counts['collision']/2048,timeout_rate=counts['timeout']/2048,unknown_rate=counts['unknown']/2048,passed=lower>=.99 and counts['collision']==0,uncertain=lower<.99<=upper and counts['collision']==0,minimum_wall_clearance_quantiles=stats['minimum_wall_clearance_quantiles'],route_deviation_episode_rate=stats['route_deviation_episode_rate'],route_deviation_state_rate=stats['route_deviation_state_rate'],route_deviation_definition=stats['route_deviation_definition'])
    folder=HERE/'expert_diagnostics';folder.mkdir(exist_ok=True);plot=folder/f'{family}_expert_failure_positions.svg'
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1000" viewBox="0 0 1440 1000">','<rect width="1440" height="1000" fill="white"/>',f'<text x="20" y="25" font-size="18">{NAMES[family]}：专家实际失败终点（红：撞墙；橙：超时；紫：未知）</text>']
    train=[s for s in read_catalog(family) if s['role'] in ['train','train_new']]
    for m,s in enumerate(train):
        group=[r for r in raw if r['case']['state']['map_index']==m];assert len(group)==32;ox=15+(m%8)*178;oy=55+(m//8)*115;scale=10
        parts.extend([f'<text x="{ox}" y="{oy-5}" font-size="11">训练几何 {m+1}</text>',f'<rect x="{ox}" y="{oy}" width="160" height="80" fill="#fafafa" stroke="#333"/>'])
        for x0,y0,x1,y1 in s['rectangles']:parts.append(f'<rect x="{ox+(x0+8)*scale}" y="{oy+(4-y1)*scale}" width="{(x1-x0)*scale}" height="{(y1-y0)*scale}" fill="#999"/>')
        for r in group:
            if r['success']:continue
            color='#d62728' if r['collision'] else '#e48b0f' if r['timeout'] else '#7c3db2';x,y=np.load(r['trace_path'])['positions'][-1,0]
            parts.append(f'<circle cx="{ox+(x+8)*scale}" cy="{oy+(4-y)*scale}" r="1.8" fill="{color}" fill-opacity=".65"/>')
    parts.append('</svg>');plot.write_text('\n'.join(parts));result['failure_plot']=str(plot)
    p=HERE/f'FULL_{family}_EXPERT_GATE_RESULTS.json';write(p,dict(stage=stage,families={family:result}))
    seal(f'FULL_{family}_EXPERT_GATE_RESULTS.receipt.json',[p,plot,HERE/f'{stage}_POSTFLIGHT.json'],before_family_demonstrations=True)
    return result
