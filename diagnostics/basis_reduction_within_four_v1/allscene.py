"""Conditional784-case Oracle and1176-case fixed-point comparison."""
from core import *
from analysis_results import paired,scenes

def stat(records):
 known=[r for r in records if r is not None];good=[r for r in known if r['success']]
 return dict(observed=len(known),success=len(good),mean_success_time=float(np.mean([r['completion_time']for r in good]))if good else None,mean_truncated_time=float(np.mean([r['truncated_time']for r in known]))if known else None)

def fixed_intervals(a,b):
 output=[];hard={8,10,11,15,16}
 for start,end in[(0,8),(8,12)]:
  for title,predicate in [('全部',lambda t:True),('困难五类',lambda t:t in hard),('其余类型',lambda t:t not in hard)]:
   keys=[k for k in a if start<=k[3]<end and predicate(k[0])and a[k]is not None and b.get(k)is not None]
   av={k:bool(a[k]['success'])for k in keys};bv={k:bool(b[k]['success'])for k in keys};common=[k for k in keys if av[k]and bv[k]]
   output.append(dict(scope=title,states=[start,end-1],**paired(av,bv),common_success=len(common),paired_mean_old=float(np.mean([a[k]['completion_time']for k in common]))if common else None,paired_mean_new=float(np.mean([b[k]['completion_time']for k in common]))if common else None))
 return output

def run_full(combo,make):
 from screening import search,fixed
 path=HERE/f'ALLSCENE_{combo["code"]}.json'
 if path.exists()and read(path)['complete']:return read(path)
 state=search(combo,64,make,scope='all');newfixed=fixed(combo,make,scope='all');original=next(c for c in read(HERE/'COMBINATIONS.json')['combinations']if c['code']=='four_8');oldfixed=fixed(original,make,scope='all')
 historical=read(HERE/'inputs/v3/SEARCH_STATE.json');old={(g['type_id'],g['level'],g['controller'],g['state_index']):g for g in historical['groups']if g['family']==4};safe={(g['type_id'],g['level'],g['controller'],g['state_index']):g for g in historical['baseline']}
 new={tuple(g['identity']):g for g in state['groups']};a={tuple(r['identity']):r['result']for r in oldfixed['records']};b={tuple(r['identity']):r['result']for r in newfixed['records']};rows=[];scene_map=scenes()
 def coverage(g,k,h,sc):return bool(g['success']and g['first_success']<=k and(h==6 or(g.get('solution',{}).get('result',g.get('solution',{}))['episode_length']<=sc['time_limits'][str(h)]['max_steps'])))
 for key,sc in sorted(scene_map.items()):
  keys=[identity for identity in new if identity[:2]==key];rr=[new[k]for k in keys];oo=[old[k]for k in keys]
  row=dict(type_id=key[0],level=key[1],name=f"{sc['title']}·参数{sc['knob']}，{sc['config']['num_agents']}台，宽度增量{sc['widen']}m",cases=len(keys),oracle={str(h):{method:{str(k):sum(coverage(g,k,h,sc)for g in groups)for k in(16,64)}for method,groups in [('four_column',oo),('new',rr)]}for h in(4,6)},safety_only={str(h):sum(safe[k]['success']and safe[k]['episode_length']<=sc['time_limits'][str(h)]['max_steps']for k in keys)for h in(4,6)},fixed={})
  for start,end in[(0,8),(8,12)]:
   subset=[k for k in a if k[:2]==key and start<=k[3]<end];row['fixed'][str(start)]={method:dict(stat([mapping[k]for k in subset]),success4=sum(bool(mapping[k]and mapping[k]['success']and mapping[k]['episode_length']<=sc['time_limits']['4']['max_steps'])for k in subset))for method,mapping in [('four_column',a),('new',b)]}
  row['oracle_success_mean_time']={method:float(np.mean([g.get('solution',{}).get('result',g.get('solution',{}))['completion_time']for g in groups if g['success']]))if any(g['success']for g in groups)else None for method,groups in [('four_column',oo),('new',rr)]};rows.append(row)
 types=[]
 for t in range(1,18):
  rr=[r for r in rows if r['type_id']==t];types.append(dict(type_id=t,title=next(sc['title']for k,sc in scene_map.items()if k[0]==t),cases=sum(r['cases']for r in rr),safety_only=sum(r['safety_only']['6']for r in rr),four_column={str(k):sum(r['oracle']['6']['four_column'][str(k)]for r in rr)for k in(16,64)},new={str(k):sum(r['oracle']['6']['new'][str(k)]for r in rr)for k in(16,64)}))
 regressions=[dict(identity=list(k),old_first_success=old[k]['first_success'],new_searched=new[k]['searched'],gap=new[k]['gap'])for k in new if old[k]['success']and not new[k]['success']]
 output=dict(code=combo['code'],name=combo['name'],dimension=combo['dimension'],complete=state['complete64']and newfixed['complete']and oldfixed['complete'],cases=784,scenes=rows,by_type=types,regressions=regressions,fixed_paired_intervals=fixed_intervals(a,b),old_success=sum(old[k]['success']for k in new),new_success=sum(g['success']for g in new.values()),oracle_paired=paired({k:old[k]['success']for k in new},{k:new[k]['success']for k in new}))
 lines=[f"\n## 全场景：{combo['name']}（{combo['dimension']}自由系数）\n",f"784例Oracle@64覆盖{output['new_success']}，原四列{output['old_success']}；原四列覆盖而该组合未覆盖{len(regressions)}例，逐例见本节对应JSON。",'\n| 场景 | 仅安全 | 四列16/64 | 当前16/64 | 四列固定0–7/8–11 | 当前固定0–7/8–11 | 成功均时·四列/当前 |','|---|---:|---|---|---|---|---|']
 for r in rows:
  f=lambda method:'/'.join(str(r['fixed'][str(s)][method]['success'])for s in(0,8));lines.append(f"| {r['name']} | {r['safety_only']['6']} | {r['oracle']['6']['four_column']['16']}/{r['oracle']['6']['four_column']['64']} | {r['oracle']['6']['new']['16']}/{r['oracle']['6']['new']['64']} | {f('four_column')} | {f('new')} | {r['oracle_success_mean_time']['four_column']}/{r['oracle_success_mean_time']['new']} |")
 lines+=['\n固定参数配对差按物理初态分块；4倍/6倍逐场景成败、独立状态统计与配对时间另存本节JSON。','\n| 子集 | 状态 | 固定参数成功率差 | 95%区间 |','|---|---|---:|---|']
 for r in output['fixed_paired_intervals']:lines.append(f"| {r['scope']} | {r['states']} | {r['difference']} | {r['ci95']} |")
 output['report_lines']=lines;dump(path,output)
 from analysis_results import write_report
 write_report();return output
