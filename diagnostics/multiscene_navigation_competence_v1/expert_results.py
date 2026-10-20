"""Read-only diagnostics from saved expert traces; no controller or plant changes."""
from common import *
import csv,html
NAMES={'B':'串联瓶颈走廊','F':'房间与门'}

def analyze(stage='expert_gate_v3'):
 cases=read(HERE/f'{stage}_CASES.json');rows=[];families={};folder=HERE/'expert_diagnostics';folder.mkdir(exist_ok=True)
 for family in ['B','F']:
  selected=[c for c in cases if c['state']['family']==family];raw=[read(HERE/stage/'raw'/f"{c['rollout_uid']}.json") for c in selected];assert len(raw)==2048
  counts=dict(success=sum(r['success'] for r in raw),collision=sum(r['collision'] for r in raw),timeout=sum(r['timeout'] for r in raw),unknown=sum(r['numerical_failure'] for r in raw));assert sum(counts.values())==len(raw)
  margins=[r['minimum_wall_clearance'] for r in raw if r['minimum_wall_clearance'] is not None];q=np.percentile(margins,[0,1,5,25,50,75,95,99,100]).tolist() if margins else None
  lower=counts['success']/len(raw);upper=(counts['success']+counts['unknown'])/len(raw);route_episodes=sum(r['route_deviation_episode'] for r in raw);state_total=sum(r['episode_length']+1 for r in raw);deviating=sum(round(r['route_deviation_state_fraction']*(r['episode_length']+1)) for r in raw)
  result=dict(name=NAMES[family],episodes=len(raw),maps=64,episodes_per_map=32,counts=counts,success_rate=lower,success_upper=upper,collision_rate=counts['collision']/len(raw),timeout_rate=counts['timeout']/len(raw),unknown_rate=counts['unknown']/len(raw),passed=lower>=.99 and counts['collision']==0,uncertain=lower<.99<=upper and counts['collision']==0,minimum_wall_clearance_quantiles=dict(zip(['min','p1','p5','p25','p50','p75','p95','p99','max'],q)) if q else None,minimum_wall_clearance_mean=float(np.mean(margins)) if margins else None,route_deviation_episode_rate=route_episodes/len(raw),route_deviation_state_rate=deviating/state_total,route_deviation_definition='distance to original Geometry.path >0.5m; descriptive; clearance-weighted route may intentionally differ',failure_plot=str(folder/f'{family}_expert_failure_positions.svg'))
  families[family]=result
  parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1000" viewBox="0 0 1440 1000">','<rect width="1440" height="1000" fill="white"/>',f'<text x="20" y="25" font-size="18">{NAMES[family]}：专家实际失败终点（红：撞墙；橙：超时；紫：未知）</text>']
  for m in range(64):
   group=[r for r in raw if r['case']['state']['map_index']==m];assert len(group)==32;s=group[0]['case']['state'];ox=15+(m%8)*178;oy=55+(m//8)*115;scale=10
   parts += [f'<text x="{ox}" y="{oy-5}" font-size="11">训练几何 {m+1}</text>',f'<rect x="{ox}" y="{oy}" width="160" height="80" fill="#fafafa" stroke="#333"/>']
   for x0,y0,x1,y1 in s['rectangles']:parts.append(f'<rect x="{ox+(x0+8)*scale}" y="{oy+(4-y1)*scale}" width="{(x1-x0)*scale}" height="{(y1-y0)*scale}" fill="#999"/>')
   for r in group:
    rows.append([family,m,r['case']['state']['index'],r['outcome'],r['episode_length'],r['minimum_wall_clearance'],r['maximum_path_deviation'],r['trace_path'],r['rollout_uid']])
    if r['success']:continue
    color='#d62728' if r['collision'] else '#e48b0f' if r['timeout'] else '#7c3db2';x,y=r['final_position'][0]
    parts.append(f'<circle cx="{ox+(x+8)*scale}" cy="{oy+(4-y)*scale}" r="1.8" fill="{color}" fill-opacity=".65"/>')
  parts.append('</svg>');(folder/f'{family}_expert_failure_positions.svg').write_text('\n'.join(parts))
  # Histogram drawn directly from measured per-episode minima.
  bins=np.array([-.05,0,.005,.01,.02,.03,.05,.075,.1,.15,.2,.3,.5,1,2,4]);hist,_=np.histogram(margins,bins);write(folder/f'{family}_clearance_histogram.json',dict(edges=bins,counts=hist,below_range=int(np.sum(np.array(margins)<bins[0])),above_range=int(np.sum(np.array(margins)>bins[-1]))))
 with (HERE/'EXPERT_EPISODES.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['family','training_map_index','episode','outcome','steps','minimum_swept_wall_clearance','maximum_original_path_deviation','trace_path','rollout_uid']);w.writerows(rows)
 result=dict(stage=stage,families=families,pilot_can_enter_full=False if not all(v['passed'] for v in families.values()) else None)
 write(HERE/'EXPERT_GATE_RESULTS.json',result);seal('EXPERT_GATE_RESULTS.receipt.json',[HERE/'EXPERT_GATE_RESULTS.json',HERE/'EXPERT_EPISODES.csv',HERE/f'{stage}_POSTFLIGHT.json']+list(folder.glob('*')),frozen_before_demonstrations=True)
 print(json.dumps({k:v['counts'] for k,v in families.items()}))
if __name__=='__main__':analyze()
