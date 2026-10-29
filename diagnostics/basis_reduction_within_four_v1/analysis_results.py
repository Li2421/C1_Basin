"""Evidence-only reduction decisions, paired physical-state intervals and report."""
from core import *
from collections import Counter
import datetime
_CACHE={};_CI={}

def load(name):
 path=HERE/name
 if not path.exists():return None
 stamp=path.stat().st_mtime_ns
 if name not in _CACHE or _CACHE[name][0]!=stamp:_CACHE[name]=(stamp,read(path))
 return _CACHE[name][1]

def scenes():return{(e['type_id'],e['level']):e['scene']for e in load('SCENES.json')['entries']}

def hard_keys():return{(g['type_id'],g['level'],g['controller'],g['state_index'])for g in load('inputs/previous/HARD_CASES.json')['cases']}

def metrics(code,scope='screen'):
 state=load(f'SEARCH_{code}_{scope}.json')
 if not state:return dict(coverage={'16':0,'64':0},hard64=0,hard_by_type={},complete16=False,complete64=False,search_stage_finished=False,groups=[],first_success_median=None,unknown_queries=0)
 gs=state['groups'];hard=hard_keys();good=[g for g in gs if g['success']]
 complete16=all(g['success']or g['searched']>=16 for g in gs);complete64=all(g['success']or g['searched']>=64 for g in gs)
 return dict(coverage={str(k):sum(g['success']and g['first_success']<=k for g in gs)for k in(16,64)},hard64=sum(g['success']for g in gs if tuple(g['identity'])in hard),hard_by_type={str(t):sum(g['success']for g in gs if g['identity'][0]==t and tuple(g['identity'])in hard)for t in(8,10,15,16)},complete16=complete16,complete64=complete64,search_stage_finished=complete64 or(complete16 and(HERE/f'SCREENED_{code}.json').exists()),groups=gs,first_success_median=float(np.median([g['first_success']for g in good]))if good else None,mean_first_success=float(np.mean([g['first_success']for g in good]))if good else None,mean_success_time=float(np.mean([g['solution']['result']['completion_time']for g in good]))if good else None,unknown_queries=sum(bool(q['result'].get('numerical_failure'))for g in gs for q in g['queries']),searched=sum(g['searched']for g in gs))

def qualified(code):
 m=metrics(code);r=metrics('reference')
 return bool(m['complete64']and r['complete64']and m['coverage']['64']>=r['coverage']['64']-5 and m['hard64']>=r['hard64']-1)

def vector(m,k):return{tuple(g['identity']):bool(g['success']and g['first_success']<=k)for g in m['groups']}

def paired(a,b):
 key=(tuple(sorted(a.items())),tuple(sorted(b.items())))
 if key in _CI:return _CI[key]
 blocks={}
 for identity in a.keys()&b.keys():blocks.setdefault((identity[0],identity[1],identity[3]),[]).append(int(b[identity])-int(a[identity]))
 vals=np.array([np.mean(v)for k,v in sorted(blocks.items())if len(v)==2]);rng=np.random.default_rng(2026102903)
 if len(vals):boot=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1);ci=np.quantile(boot,[.025,.975]).tolist()
 else:ci=None
 result=dict(physical_blocks=len(vals),difference=float(vals.mean())if len(vals)else None,ci95=ci,seed=2026102903,resamples=10000,unit='scene/state block, two nominal controllers kept together')
 _CI[key]=result;return result

def fixed_counts(code,scope='screen',plus=False):
 data=load(f'FIXED_{code}_{scope}'+('_plus'if plus else'')+'.json')
 if not data:return None
 result={}
 for start,end in[(0,8),(8,12)]:
  rr=[x for x in data['records']if start<=x['identity'][3]<end];known=[x['result']for x in rr if x['result']is not None];success=[r for r in known if r['success']]
  result[str(start)]=dict(cases=len(rr),observed=len(known),success=len(success),mean_success_time=float(np.mean([r['completion_time']for r in success]))if success else None,mean_truncated_time=float(np.mean([r['truncated_time']for r in known]))if known else None)
 return dict(complete=data['complete'],states=result)

def write_report():
 data=load('COMBINATIONS.json');r=metrics('reference');rows=[]
 for c in data['combinations']:
  m=metrics(c['code']);row={k:v for k,v in m.items()if k!='groups'};row.update(code=c['code'],name=c['name'],dimension=c['dimension'],parents=c['parents'],qualified=qualified(c['code']),fixed=fixed_counts(c['code']),screening=load(f'SCREENED_{c["code"]}.json'))
  if r['complete16']and m['complete16']:row['paired16']=paired(vector(r,16),vector(m,16))
  if r['complete64']and m['complete64']:row['paired64']=paired(vector(r,64),vector(m,64))
  row['coverage4x']={str(k):sum(g['success']and g['first_success']<=k and g['solution']['result']['episode_length']<=scenes()[tuple(g['identity'][:2])]['time_limits']['4']['max_steps']for g in m['groups'])for k in(16,64)}
  rows.append(row)
 q=sorted([x for x in rows if x['dimension']<=4 and x['qualified']],key=lambda x:(x['dimension'],-x['coverage']['64'],x['code']));minimum=q[0]if q else None
 out=dict(time=datetime.datetime.now().astimezone().isoformat(),reference={k:v for k,v in r.items()if k!='groups'},rows=rows,minimum=minimum,criterion=dict(total_tolerance=5,hard_tolerance=1,screen16_tolerance=10),levels=data['levels'],historical_four=next((x for x in rows if x['code']=='four_8'),None),horizon_policy='4x is truncation of executed6x first-success trajectories; no search of skipped candidates for4x.',finite_search_only=True)
 qualified_codes={x['code']for x in q};types=[]
 for t in range(1,18):
  item=dict(type_id=t,title=next(sc['title']for key,sc in scenes().items()if key[0]==t),combinations={})
  for code in ['reference','four_8']+sorted(qualified_codes):
   gs=[g for g in metrics(code)['groups']if g['identity'][0]==t];item['combinations'][code]=dict(cases=len(gs),coverage16=sum(g['success']and g['first_success']<=16 for g in gs),coverage64=sum(g['success']for g in gs),complete64=bool(gs)and all(g['success']or g['searched']==64 for g in gs))
  types.append(item)
 out['by_type']=types;dump(HERE/'REDUCTION.json',out)
 if minimum:
  opening=f"当前完成的有限候选中，最少自由系数为{minimum['dimension']}，组合是{minimum['name']}；覆盖{minimum['coverage']['64']}/272、困难案例{minimum['hard64']}/40。参照为{r['coverage']['64']}/272、{r['hard64']}/40；原四列为{metrics('four_8')['coverage']['64']}/272、{metrics('four_8')['hard64']}/40。尚未完成的下一级不能据此排除。"
 else:opening='当前尚未验证出满足判据的四系数以内组合；只有已完成规定搜索的组合才能作删减结论。'
 binding_results=[];replacement=[]
 combo_map={c['code']:c for c in data['combinations']}
 for x in q:
  c=combo_map[x['code']]
  if c['bindings']:binding_results.append(x['name'])
  for parent in x['parents']:replacement.append(combo_map[parent]['name']+' → '+x['name'])
 lines=['# 四系数以内的删减与合并\n',opening,'\n可合并的已验证组合：'+('；'.join(binding_results)if binding_results else'尚无完成证据')+'。可替代关系：'+('；'.join(replacement)if replacement else'尚无完成证据')+'。',
        '\n名义控制器为集中式随机采样模型预测控制、梯度下降模型预测控制。原两次安全投影、动力学、场景和时限保持冻结。所有Oracle是直接按顺序搜索系数；首次成功即停，不使用生成器或评判器。',
        '\n| 组合（自由系数） | 父组合 | 覆盖16/64·272例 | 困难64：窄门/长走廊/四房间/仓库 | 与参照64配对差及95%区间 | 不比参照差 | 首次成功序号中位数 | 固定参数0–7/8–11 |',
        '|---|---|---|---|---|---|---|---|']
 for x in rows:
  parent='；'.join(combo_map[n]['name']for n in x['parents'])or'第一级预定组合';cov64=str(x['coverage']['64'])if x['complete64']else('未补搜'if x['screening']else'未完成')
  ci=x.get('paired64');cistr=f"{ci['difference']:.4f} {ci['ci95']}"if ci else'待完整64次证据';hard='/'.join(str(x['hard_by_type'].get(str(t),0))for t in(8,10,15,16))if x['complete64']else'未完成'
  fp=x['fixed'];fix=f"{fp['states']['0']['success']}/{fp['states']['0']['cases']}；{fp['states']['8']['success']}/{fp['states']['8']['cases']}"if fp else'未运行'
  lines.append(f"| {x['name']}（{x['dimension']}） | {parent} | {x['coverage']['16']}/{cov64} | {hard} | {cistr} | {x['qualified']} | {x['first_success_median']} | {fix} |")
 lines+=['\n配对区间按物理场景/初态分块，两个控制器保留在同一块，10000次自助抽样。未补到64次的组合不被当作64次失败；其与参照的16次配对差保存在REDUCTION.json。目标列删除后为零，保留时系数下界0.5，两个参数域严格分开。',
         '\n## 逐类替代效果\n','| 场景类型 | '+ ' | '.join(combo_map[c]['name']for c in ['reference','four_8']+sorted(qualified_codes))+' |','|---|'+'---:|'*(2+len(qualified_codes))]
 for x in types:lines.append('| '+x['title']+' | '+' | '.join(str(x['combinations'][code]['coverage64'])if x['combinations'][code]['complete64']else'未完成'for code in ['reference','four_8']+sorted(qualified_codes))+' |')
 lines+=['\n## 固定点与时限\n','主固定点为朝目标0.875、互斥分离0.375、目标旋转−0.75、净空0、独立放行1，再按组合规则投影。状态8–11未参加搜索。上一轮+0.75点仅作衔接，另列，不参与判据。4倍成败仅截取同次6倍搜索中的首次成功轨迹。','\n| 衔接对象 | +0.75固定点状态0–7/8–11 |','|---|---|']
 for code in ['reference']+([minimum['code']]if minimum else[]):
  f=fixed_counts(code,plus=True)
  if f:lines.append('| '+combo_map[code]['name']+' | '+ '/'.join(str(f['states'][str(k)]['success'])for k in(0,8))+' |')
 for name,title in [('CORRECTIONS_ANALYSIS.json','两处定义修正与旧伤害案例'),('FAILURE_ANALYSIS.json','失败分型')]:
  if(HERE/name).exists():lines.append(f'\n[{title}]({name})\n')
 for path in sorted(HERE.glob('ALLSCENE_*.json')):
  full=read(path);lines+=full.get('report_lines',[])
 lines+=['\n每个组合的成功任务条件均时、首次成功组数均值、原始候选计算开销另见REDUCTION.json和candidates.jsonl。超时只标为64次内未找到，失败分型是已执行代表轨迹的诊断，不是整个连续修正族的不可能性证书。']
 (HERE/'REPORT.md').write_text('\n'.join(lines).replace('None','—')+'\n');return out

def finalize(budget_exhausted=False):
 out=write_report();rows=out['rows'];gaps=[]
 for row in rows:
  if not row['search_stage_finished']:gaps.append(row['name']+'搜索未完成。')
  if row['fixed']is None or not row['fixed']['complete']:gaps.append(row['name']+'固定参数未完成。')
 if budget_exhausted:gaps.append('80000局新运行硬上限耗尽；所有未执行候选保留为缺口。')
 from final_checks import finish
 finish(out,gaps)
