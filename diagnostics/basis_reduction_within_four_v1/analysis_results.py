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

def scene_title(sc):
 t=sc['type_id'];n=sc['config']['num_agents'];w=sc['widen'];r=sc['config']['agent_radius']
 detail=f'{n}台，半径{r:g}m'
 if t in(8,10,11):detail=f'每侧{n//2}台，主道宽{.3+w:.2f}m'
 elif t==9:detail=f'{n}台，门宽{.3+w:.2f}m'
 elif t==7:detail=f'{n}台，{sc["knob"]}个0.3m障碍'
 elif t==14:detail=f'{n}台，通道宽{float(sc["knob"])+w:.2f}m'
 elif t==2:detail=f'每股{n//2}台，半径{r:g}m'
 elif t==5:detail=f'每向{n//4}台'
 elif t==12:detail=f'每侧{n//2}台，通道宽{.5+w:.2f}m'
 elif t==13:detail=f'每向{n//3}台，通道宽{.6+w:.2f}m'
 return sc['title']+'，'+detail

def hard_keys():return{(g['type_id'],g['level'],g['controller'],g['state_index'])for g in load('inputs/previous/HARD_CASES.json')['cases']}

def metrics(code,scope='screen'):
 state=load(f'SEARCH_{code}_{scope}.json')
 if not state:return dict(coverage={'16':0,'64':0},hard64=0,hard_by_type={},complete16=False,complete64=False,search_stage_finished=False,groups=[],first_success_median=None,unknown_queries=0)
 gs=state['groups'];hard=hard_keys();good=[g for g in gs if g['success']]
 complete16=all(g['success']or g['searched']>=16 for g in gs);complete64=all(g['success']or g['searched']>=64 for g in gs)
 return dict(coverage={str(k):sum(g['success']and g['first_success']<=k for g in gs)for k in(16,64)},hard64=sum(g['success']for g in gs if tuple(g['identity'])in hard),hard_by_type={str(t):sum(g['success']for g in gs if g['identity'][0]==t and tuple(g['identity'])in hard)for t in(8,10,15,16)},complete16=complete16,complete64=complete64,search_stage_finished=complete64 or bool(state.get('user_early_stop'))or(complete16 and(HERE/f'SCREENED_{code}.json').exists()),user_early_stop=state.get('user_early_stop'),groups=gs,first_success_median=float(np.median([g['first_success']for g in good]))if good else None,mean_first_success=float(np.mean([g['first_success']for g in good]))if good else None,mean_success_time=float(np.mean([g['solution']['result']['completion_time']for g in good]))if good else None,unknown_queries=sum(bool(q['result'].get('numerical_failure'))for g in gs for q in g['queries']),searched=sum(g['searched']for g in gs))

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

def write_report(final=False):
 data=load('COMBINATIONS.json');r=metrics('reference');rows=[]
 for c in data['combinations']:
  m=metrics(c['code']);row={k:v for k,v in m.items()if k!='groups'};row.update(code=c['code'],name=c['name'],dimension=c['dimension'],parents=c['parents'],qualified=qualified(c['code']),fixed=fixed_counts(c['code']),screening=load(f'SCREENED_{c["code"]}.json'))
  if r['complete16']and m['complete16']:row['paired16']=paired(vector(r,16),vector(m,16))
  if r['complete64']and m['complete64']:
   av,bv=vector(r,64),vector(m,64);row['paired64']=paired(av,bv)
   row['lost_vs_reference']=[list(k)for k in sorted(av)if av[k]and not bv[k]]
   row['gained_vs_reference']=[list(k)for k in sorted(av)if not av[k]and bv[k]]
  row['coverage4x']={str(k):sum(g['success']and g['first_success']<=k and g['solution']['result']['episode_length']<=scenes()[tuple(g['identity'][:2])]['time_limits']['4']['max_steps']for g in m['groups'])for k in(16,64)}
  row['budget_curve']=[dict(K=k,verified_coverage=sum(g['success']and g['first_success']<=k for g in m['groups']),complete=bool(m['groups'])and all(g['success']or g['searched']>=k for g in m['groups']))for k in range(1,65)]
  rows.append(row)
 q=sorted([x for x in rows if x['dimension']<=4 and x['qualified']],key=lambda x:(x['dimension'],-x['coverage']['64'],x['code']));minimum=q[0]if q else None
 out=dict(time=datetime.datetime.now().astimezone().isoformat(),reference={k:v for k,v in r.items()if k!='groups'},rows=rows,minimum=minimum,criterion=dict(total_tolerance=5,hard_tolerance=1,screen16_tolerance=10),levels=data['levels'],historical_four=next((x for x in rows if x['code']=='four_8'),None),horizon_policy='4x is truncation of executed6x first-success trajectories; no search of skipped candidates for4x.',finite_search_only=True)
 old_four=metrics('four_8');old_by_id={tuple(g['identity']):g for g in old_four['groups']}
 late_old=[g for g in r['groups']if g['success']and g['first_success']>16 and old_by_id.get(tuple(g['identity']),{}).get('success')]
 same_index=[g for g in late_old if g['first_success']==old_by_id[tuple(g['identity'])]['first_success']]
 rescued=[g for g in r['groups']if g['success']and tuple(g['identity'])in old_by_id and not old_by_id[tuple(g['identity'])]['success']]
 lost_old=[g for g in r['groups']if not g['success']and old_by_id.get(tuple(g['identity']),{}).get('success')]
 out['search_budget_diagnosis']=dict(reference_complete64=r['complete64'],historical_complete64=old_four['complete64'],reference_late_success_already_covered_by_historical64=[g['identity']for g in late_old],same_first_success_index=[g['identity']for g in same_index],rescued_vs_historical64=[g['identity']for g in rescued],interpretation='Late successes already covered by historical64 are not independent coverage gains of admission; matching first indices does not by itself establish identical trajectories.')
 out['search_budget_diagnosis']['lost_vs_historical64']=[g['identity']for g in lost_old]
 qualified_codes={x['code']for x in q};types=[]
 for t in range(1,18):
  selected=next(x for x in load('REPRESENTATIVE17_FROZEN.json')['selected']if x['type_id']==t)
  item=dict(type_id=t,title=scene_title(scenes()[(t,selected['level'])]),combinations={})
  for code in ['reference','four_8']+sorted(qualified_codes):
   gs=[g for g in metrics(code)['groups']if g['identity'][0]==t];item['combinations'][code]=dict(cases=len(gs),coverage16=sum(g['success']and g['first_success']<=16 for g in gs),coverage64=sum(g['success']for g in gs),complete64=bool(gs)and all(g['success']or g['searched']==64 for g in gs))
  types.append(item)
 out['by_type']=types;dump(HERE/'REDUCTION.json',out)
 if minimum:
  opening=f"当前完成的有限候选中，最少自由系数为{minimum['dimension']}，组合是{minimum['name']}；覆盖{minimum['coverage']['64']}/272、困难案例{minimum['hard64']}/40。参照为{r['coverage']['64']}/272、{r['hard64']}/40；原四列为{metrics('four_8')['coverage']['64']}/272、{metrics('four_8')['hard64']}/40。尚未完成的下一级不能据此排除。"
 else:opening=('本轮冻结规则筛选的候选中未找到满足判据的四系数以内组合。'if final else'当前尚未验证出满足判据的四系数以内组合；只有已完成规定搜索的组合才能作删减结论。')
 if final and minimum:opening=opening.replace('尚未完成的下一级不能据此排除。','结论限于按预定规则实际完成的组合与有限候选；未完成部分见证据缺口。')
 binding_results=[];replacement=[]
 combo_map={c['code']:c for c in data['combinations']}
 for x in q:
  c=combo_map[x['code']]
  if c['bindings']:binding_results.append(x['name'])
  if not x['parents']and x['dimension']==4:replacement.append(combo_map['reference']['name']+' → '+x['name'])
  for parent in x['parents']:replacement.append(combo_map[parent]['name']+' → '+x['name'])
 lines=['# 四系数以内的删减与合并\n',opening,'\n可合并的已验证组合：'+('；'.join(binding_results)if binding_results else'尚无完成证据')+'。可替代关系：'+('；'.join(replacement)if replacement else'尚无完成证据')+'。',
        '\n名义控制器为集中式随机采样模型预测控制、梯度下降模型预测控制。原两次安全投影、动力学、场景和时限保持冻结。所有Oracle是直接按顺序搜索系数；首次成功即停，不使用生成器或评判器。本轮维度指自由系数数目；通道项包含状态与输入缩放。因此，相对原四列的比较同时引入通行权记忆，覆盖提升不能单独解释为静态修正矩阵增加一列的效果。',
        '\n| 组合（自由系数） | 父组合 | 覆盖16/64·272例 | 困难64：窄门/长走廊/四房间/仓库 | 与参照64配对差及95%区间 | 不比参照差 | 首次成功序号中位数 | 固定参数0–7/8–11 |',
        '|---|---|---|---|---|---|---|---|']
 for x in rows:
  parent='；'.join(combo_map[n]['name']for n in x['parents'])or(combo_map['reference']['name']if x['dimension']==4 else'五项参照');cov64=str(x['coverage']['64'])if x['complete64']else('未补搜'if x['screening']else'未完成')
  ci=x.get('paired64');cistr=f"{ci['difference']:.4f} {ci['ci95']}"if ci else'待完整64次证据';hard='/'.join(str(x['hard_by_type'].get(str(t),0))for t in(8,10,15,16))if x['complete64']else'未完成'
  fp=x['fixed'];fix=f"{fp['states']['0']['success']}/{fp['states']['0']['cases']}；{fp['states']['8']['success']}/{fp['states']['8']['cases']}"if fp else'未运行'
  lines.append(f"| {x['name']}（{x['dimension']}） | {parent} | {x['coverage']['16']}/{cov64} | {hard} | {cistr} | {x['qualified']} | {x['first_success_median']} | {fix} |")
 lines+=['\n配对区间按物理场景/初态分块，两个控制器保留在同一块，10000次自助抽样。未补到64次的组合不被当作64次失败；其与参照的16次配对差保存在REDUCTION.json。目标列删除后修正系数为零，原名义安全动作仍存在；保留时系数下界0.5，两个参数域严格分开。达标是规定的总覆盖判据，逐案例新增和损失另存JSON，不能解释成成功集合的包含关系。',
         '\n独立放行强度的参数盒为[0.5,1]；绑定到目标旋转绝对值后可低于0.5，删除放行项时则为0。这些是说明规定的不同参数域，较少系数的结果不能单独证明一个连续线性基的秩结论。各组合的64点按规定投影、去重和补齐，并非彼此嵌套的完整连续参数域；较小组合的有限搜索覆盖高于五项参照，不表示其连续表达能力更强。',
         '\n## 逐类替代效果\n','| 场景类型 | '+ ' | '.join(combo_map[c]['name']for c in ['reference','four_8']+sorted(qualified_codes))+' |','|---|'+'---:|'*(2+len(qualified_codes))]
 for x in types:lines.append('| '+x['title']+' | '+' | '.join(str(x['combinations'][code]['coverage64'])if x['combinations'][code]['complete64']else'未完成'for code in ['reference','four_8']+sorted(qualified_codes))+' |')
 lines+=['\n## 成功时间与搜索成本\n','下表均值对应各组合实际搜索预算与各自成功子集；不同子集的条件时间不能直接作配对速度排名。','\n| 组合 | 首次成功组数均值 | 成功任务完成均时·秒 | 已完成逻辑查询数 |','|---|---:|---:|---:|']
 for x in rows:lines.append(f"| {x['name']} | {x.get('mean_first_success')} | {x.get('mean_success_time')} | {x.get('searched',0)} |")
 lines+=[f'\n参照在第17项之后才成功、但原四列64项内也已成功的案例目前有{len(late_old)}例，其中{len(same_index)}例首次成功序号也相同；这些不计为通道放行对原四列的独立覆盖补救。目前参照覆盖而原四列64项未覆盖的有{len(rescued)}例；原四列覆盖而参照未覆盖的有{len(lost_old)}例，损失逐例也保存在REDUCTION.json。参照完整64项状态为{r["complete64"]}，未完成时上述数量只是当前观察。序号相同本身不证明轨迹逐位相同。']
 lines+=['\n## 固定点与时限\n','主固定点为朝目标0.875、互斥分离0.375、目标旋转−0.75、净空0、独立放行1，再按组合规则投影。状态8–11未参加搜索。上一轮+0.75点仅作衔接，另列，不参与判据。4倍成败仅截取同次6倍搜索中的首次成功轨迹。','\n| 衔接对象 | +0.75固定点状态0–7/8–11 |','|---|---|']
 for code in ['reference']+([minimum['code']]if minimum else[]):
  f=fixed_counts(code,plus=True)
  if f:lines.append('| '+combo_map[code]['name']+' | '+ '/'.join(str(f['states'][str(k)]['success'])for k in(0,8))+' |')
 for name,title in [('CORRECTIONS_ANALYSIS.json','两处定义修正与旧伤害案例'),('FAILURE_ANALYSIS.json','失败分型')]:
  if(HERE/name).exists():lines.append(f'\n[{title}]({name})\n')
 correction=load('CORRECTIONS_ANALYSIS.json')
 if correction:
  lines+=['\n## 两处定义修正的实测效果\n','旧合成一列与修正后五项参照使用相同的四项系数和放行强度，比较限于该冻结有限池。历史8个让行湾伤害来自旧三项菜单，单列保留，不能与旧合成一列混称。','\n| 场景 | 旧合成一列16/64 | 新参照16/64 |','|---|---|---|']
  for x in correction['by_type']:lines.append(f"| {x['title']} | {x['old']['16']}/{x['old']['64']} | {x['new']['16']}/{x['new']['64']} |")
  lines+=['\n| 让行湾原伤害案例 | 新参照16次 | 新参照64次 | 首次成功序号 |','|---|---|---|---|']
  controller_names={'centralized_sampling':'集中式随机采样模型预测控制','gradient_mpc':'梯度下降模型预测控制'}
  for x in correction['eight_bay_harms']:lines.append(f"| {controller_names[x['identity'][2]]}·初态{x['identity'][3]} | {x['new_success16']} | {x['new_success64']} | {x['new_first_success']} |")
 witness=load('DOMAIN_WITNESSES.json')
 if witness:
  wa,wb=witness['reference'],witness['bound']
  lines+=['\n## 参数范围与方向贡献的区分\n',f"货架仓库10台、初态3、集中式随机采样模型预测控制的已执行配对：其余四项系数、初态和种子完全相同，放行强度{wa['q']}时超时，强度{wb['q']:.9f}时在{wb['completion_time']}秒完成。初始位置、速度、目标及首次动作分歧前的名义输入、安全参考动作均逐位相同。该低强度由目标侧转绝对值绑定产生，落在五项参照[0.5,1]盒之外；此见证不能单独计为较少维数增加表达能力，也不能推出整个原参数盒数学上无解。原始UID与轨迹见[范围敏感性见证](DOMAIN_WITNESSES.json)。"]
 if final:
  from coverage_complementarity import audit
  complement=audit();leader=complement['leader']
  lines+=['\n## 已完成组合的互补性\n',f"按总覆盖展示的单族最高为{leader['name']}，{leader['coverage']}/272；本节不替代开头的最少自由系数判据。全部完整64次搜索族的成功并集为{complement['diagnostics_only_union']}/272，仅用于定位互补案例，不计为一个固定族的成绩。逐例补救组合、首次成功序号及参数保存在[COVERAGE_COMPLEMENTARITY.json](COVERAGE_COMPLEMENTARITY.json)。",'\n| 最高覆盖族的未覆盖场景 | 未覆盖案例 | 其他完整搜索族补回 | 仍未覆盖 |','|---|---|---|---|']
  for row in complement['by_type']:lines.append(f"| {row['title']} | {row['uncovered']} | {row['rescued']} | {row['still_uncovered']} |")
  if(HERE/'failure_examples.png').exists():lines.append('\n[阶段性四自由系数组合失败示例](failure_examples.png)：读取已保存的第64候选轨迹绘图，不改变控制器、物理或判据。示例不推广为全部失败候选的机制证书。')
 failure=load('FAILURE_ANALYSIS.json')
 if failure:
  lines+=['\n## 困难案例的失败轨迹分型\n','以下只分型64次内未找到成功的案例所执行的末次候选，不把其诊断推广成整个修正族的数学缺陷。首次候选和独立标志同时保存在JSON。','\n| 组合 | 末次候选分类计数 |','|---|---|']
  for code,counts in failure['counts'].items():lines.append('| '+combo_map[code]['name']+' | '+json.dumps(counts['final_candidate'],ensure_ascii=False)+' |')
 for path in sorted(HERE.glob('ALLSCENE_*.json')):
  full=read(path);lines+=full.get('report_lines',[])
 context=load('PAPER_CONTEXT.json')
 if context:
  combo_names={c['code']:c['name']for c in data['combinations']}
  paper_lines=context.get('report_lines',[])
  for line in paper_lines:
   for code,name in sorted(combo_names.items(),key=lambda kv:-len(kv[0])):line=line.replace('| '+code+' |','| '+name+' |')
   lines.append(line.replace('# 论文端口历史对照','## 论文端口历史对照'))
 certificate=load('ACCELERATION_CERTIFICATE.json')
 if certificate:
  original=certificate.get('original_seconds');fast=certificate.get('fast_seconds')
  saving=f'原路径{original:.2f}秒，加速路径{fast:.2f}秒，节省{original-fast:.2f}秒（{100*(original-fast)/original:.2f}%）'if original and fast is not None else'计时见证书'
  lines+=['\n## 保持结果不变的工程加速\n',
          '启用四项：复用相同稀疏约束矩阵数组、以NumPy计算相同Threefry随机键折叠、剔除不可能贡献的远墙代价、预筛不可能活跃的远墙约束。加速模块哈希与上一轮一致。',
          f"旧路径{certificate.get('old_replays',0)}局完整轨迹重放、新路径{certificate.get('new_pairs',0)}对原路径与加速路径对跑，逐步数组与结局核对{'通过'if certificate.get('passed')else'未通过'}。认证对跑样本的实测墙钟合计：{saving}。该计时不包含导入、排队与协调开销，也不外推为整轮相同加速比例。详见[加速证书](ACCELERATION_CERTIFICATE.json)。"]
 if(HERE/'USER_EARLY_STOP_POLICY.json').exists():
  lines+=['\n## 用户授权的计算价值早停\n','用户在运行中授权对没有继续价值的删减组合提前停止。8组预算明显落后50例，或8组和12组预算均落后30例，且未补救当前最好四参数族的任何失败案例时，可提前结束该组合，并省略其后续固定点对局。策略在已观察的前缀之后制定，详细决策见USER_EARLY_STOP_POLICY.json与value_reviews/。已执行结果保持，未执行候选为未知，不计为16/64次失败。最终最少自由系数结论只覆盖实际验证的族，不能排除提前停止族的连续参数能力。']
 lines+=['\n每个组合的成功任务条件均时、首次成功组数均值、原始候选计算开销另见REDUCTION.json和candidates.jsonl。超时只标为64次内未找到，失败分型是已执行代表轨迹的诊断，不是整个连续修正族的不可能性证书。']
 (HERE/'REPORT.md').write_text('\n'.join(lines).replace('None','—')+'\n');return out

def finalize(budget_exhausted=False):
 out=write_report();rows=out['rows'];gaps=[]
 for row in rows:
  if not row['search_stage_finished']:gaps.append(row['name']+'搜索未完成。')
  if not row.get('user_early_stop')and not(HERE/f'FIXED_SKIPPED_USER_{row["code"]}.json').exists()and(row['fixed']is None or not row['fixed']['complete']):gaps.append(row['name']+'固定参数未完成。')
 if budget_exhausted:gaps.append('80000局新运行硬上限耗尽；所有未执行候选保留为缺口。')
 for code in ['reference']+([out['minimum']['code']]if out['minimum']else[]):
  fp=fixed_counts(code,plus=True)
  if fp is None or not fp['complete']:gaps.append(code+'的+0.75衔接固定参数结果未完成。')
 from final_checks import finish
 finish(out,gaps)
