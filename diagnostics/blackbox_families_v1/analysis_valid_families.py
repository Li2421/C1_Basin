"""Read evaluation only after verifying every pre-label choice receipt."""
from common import *
from design import FAMILIES
from analysis_statistics import summarize,ARMS
import csv
FULL_NAMES={'centralized_sampling':'集中式随机采样模型预测控制','cross_entropy':'交叉熵方法模型预测控制','prioritized_astar':'优先级时空 A* 规划','deterministic':'确定性行为克隆策略','mixture':'混合密度行为克隆策略','diffusion':'扩散行为克隆策略'}
METHOD_NAMES={'nominal_only':'只执行控制器动作','safety_baseline':'只加安全投影基线','F10':'全程修正参数十','F13':'全程修正参数十三','S500':'按比例切换时间表','Sprog':'按进度切换时间表','RHR':'固定参数的持续右手规则','RHR_fair':'公平调参的持续右手规则','original':'原冻结控制器特征选择模型','controller_blind':'原冻结不看控制器的选择模型','probe_rule':'探测效果规则','expanded':'扩展选择模型','always_schedule':'一律按比例切换时间表','always_right_hand':'一律固定参数的持续右手规则','pilot_2':'两个标签状态的选择','pilot_4':'四个标签状态的选择','pilot_8':'八个标签状态的选择','pilot_16':'十六个标签状态的选择','fixed_oracle':'固定修正参数交叉拟合最佳臂','crossfit_oracle':'完整菜单交叉拟合最佳臂','in_sample_oracle':'完整菜单样本内最佳臂'}

def main():
 receipt=read(HERE/'ALL_CHOICES_FROZEN.receipt.json')
 for path,digest in receipt['files'].items():assert sha(path)==digest,path
 f=read(HERE/'FROZEN_CHOICES.json');assert not f['evaluation_labels_read'];names=[n for n in f['controller_order'] if not n.startswith('prioritized_astar')];states=read(HERE/'EVAL_STATES.json');lo=np.zeros((len(names),32,8,2));hi=lo.copy();events={k:lo.copy() for k in ['collision','timeout','unknown']};seen=set();raw_index=[];individual_transitions=[]
 # This is the first access to main evaluation outcomes by this analysis.
 for c in read(HERE/'main_CASES.json'):
  if c['family']=='prioritized_astar':continue
  path=HERE/'main/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];ci=names.index(c['base_controller_name']);si=c['state_index'];ai=ARMS.index(c['arm']);seed=c['seed'];ix=(ci,si,ai,seed);assert ix not in seen;seen.add(ix);unknown=r['numerical_failure'];lo[ix]=r['success'];hi[ix]=r['success']+unknown;events['collision'][ix]=r['collision'];events['timeout'][ix]=r['timeout'];events['unknown'][ix]=unknown;raw_index.append(dict(rollout_uid=c['rollout_uid'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256'],unknown=unknown,physical_reuse=bool(r.get('exact_inactive_reuse'))))
 assert len(seen)==len(names)*32*8*2
 choices={a:ARMS.index(a) for a in ARMS}
 for method in ['original','controller_blind','probe_rule','expanded','always_schedule','always_right_hand','pilot_2','pilot_4','pilot_8','pilot_16']:choices[method]=[ARMS.index(f['controllers'][n][method]) for n in names]
 np.savez_compressed(HERE/'EVALUATION_LABELS.npz',lower=lo,upper=hi,controller_names=names,arms=ARMS,**events);write(HERE/'EVALUATION_RAW_INDEX.json',raw_index);results={};passed=[];family_status={};preferences={}
 for stage,cap in f['capabilities'].items():family_status.update(cap['family_status'])
 observed_capability=dict(family_status);family_status['prioritized_astar']='GAP_IMPLEMENTATION'
 for j,family in enumerate(FAMILIES):
  indices=np.asarray([i for i,name in enumerate(names) if name.startswith(family)],int);status=family_status.get(family,'GAP')
  if len(indices):
   ch={k:(v if np.asarray(v).ndim==0 else np.asarray(v)[indices]) for k,v in choices.items()};out=summarize(FULL_NAMES[family],lo[indices],hi[indices],{k:v[indices] for k,v in events.items()},ch,202610173800+j);out['family_capability']=status;out['controller_names']=[names[i] for i in indices]
   if status!='PASS':out['endpoint1']='INCONCLUSIVE';out['evidence_level']='DEVELOPMENT capability-inconclusive descriptive results'
   else:passed.extend(indices.tolist())
   comparison=out['paired']['right_hand_minus_schedule'];bestarm=max(ARMS[2:7],key=lambda a:out['methods'][a]['success_pct'][0]);preferences[family]=dict(fixed_rule_minus_schedule_pp=comparison['estimate_pp'],CI95_pp=comparison['CI95_pp'],preference='持续右手规则' if comparison['CI95_pp'][0]>0 else '时间表' if comparison['CI95_pp'][1]<0 else '未分出可靠偏好',descriptive_best_menu_arm=METHOD_NAMES[bestarm],fair_rule_minus_schedule_pp=out['paired']['fair_right_hand_minus_schedule']['estimate_pp'])
  else:out=dict(unit=FULL_NAMES[family],family_capability=status,status='INCONCLUSIVE',methods={},reason='Required oblique empty-map straightness unit test failed; initial axis-aligned test was insufficient. Observed pilot capability PASS 6/6, but family excluded from confirmatory evaluation.',observed_family_capability=observed_capability.get(family));preferences[family]=dict(preference='缺口，无法判定')
  results[family]=out
 if passed:
  indices=np.asarray(passed);ch={k:(v if np.asarray(v).ndim==0 else np.asarray(v)[indices]) for k,v in choices.items()};combined=summarize('通过能力门的族合并',lo[indices],hi[indices],{k:v[indices] for k,v in events.items()},ch,202610173899)
  rg=combined['methods']['expanded']['mean_regret_pp'];harm=combined['methods']['expanded']['harm_possible'];endpoint2a='SUPPORTED' if rg[1]<=3 and harm==0 else 'REJECTED' if rg[0]>3 or combined['methods']['expanded']['harm_certain']>0 else 'INCONCLUSIVE';headroom=[family for family in FAMILIES if results[family].get('family_capability')=='PASS' and results[family]['paired']['right_hand_minus_schedule']['estimate_pp'][0]>=10-1e-12];ci=combined['paired']['expanded_regret_minus_always_schedule']['CI95_pp'];endpoint2b='INCONCLUSIVE' if not headroom else 'SUPPORTED' if ci[1]<0 else 'REJECTED';p4=combined['methods']['pilot_4']['mean_regret_pp'];endpoint3='SUPPORTED' if p4[1]<=3 else 'REJECTED' if p4[0]>3 else 'INCONCLUSIVE'
 else:combined=dict(methods={});rg=None;harm=None;headroom=[];endpoint2a=endpoint2b=endpoint3='INCONCLUSIVE';p4=None
 result=dict(utc=utc(),integrity='See INTEGRITY_COMPARISON.json',families=results,combined=combined,preferences=preferences,endpoint1=dict(supported_families=sum(results[k].get('endpoint1')=='SUPPORTED' for k in FAMILIES),capability_pass_families=sum(observed_capability.get(k)=='PASS' for k in FAMILIES),implementation_valid_families=sum(family_status.get(k)=='PASS' for k in FAMILIES)),endpoint2=dict(a=endpoint2a,expanded_regret_pp=rg,harm_possible=harm,b=endpoint2b,headroom_families=[FULL_NAMES[k] for k in headroom]),endpoint3=dict(status=endpoint3,pilot4_regret_pp=p4),unknown_count=int(events['unknown'].sum()),unknown_is_not_known_failure=True,timeout_is_not_deadlock=True,all_choices_receipt_sha256=sha(HERE/'ALL_CHOICES_FROZEN.receipt.json'))
 write(HERE/'results.json',result)
 for ci,name in enumerate(names):
  for si in range(32):
   for seed in [0,1]:
    def event(a):return 'unknown' if events['unknown'][ci,si,a,seed] else 'collision' if events['collision'][ci,si,a,seed] else 'success' if lo[ci,si,a,seed] else 'timeout'
    individual_transitions.append(dict(controller=name,state_uid=states[si]['state_uid'],seed=seed,nominal=event(0),safety=event(1)))
 write(HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.json',individual_transitions);table(result)
 print(json.dumps(dict(endpoint1=result['endpoint1'],endpoint2=result['endpoint2'],endpoint3=result['endpoint3']),ensure_ascii=False))

def table(result):
 methods=list(METHOD_NAMES);columns=['族','能力门']+[METHOD_NAMES[k] for k in methods]+['证据等级'];rows=[];detail=[]
 for family,out in list(result['families'].items())+[('combined',result['combined'])]:
  row={'族':FULL_NAMES.get(family,'能力门通过且实现合格的族合并'),'能力门':'六个实例通过；实现单元测试失败，确认性评估缺口' if out.get('family_capability')=='GAP_IMPLEMENTATION' else out.get('family_capability','合并')}
  for k in methods:
   m=out['methods'].get(k)
   if m:
    low,high=m['success_pct'];a,b=m['success_CI95_pct'];row[METHOD_NAMES[k]]=f'{low:.2f}'+(f'–{high:.2f}' if low!=high else '')+f' [{a:.2f}, {b:.2f}]';detail.append(dict(family=row['族'],method=METHOD_NAMES[k],**m))
   else:row[METHOD_NAMES[k]]='缺口'
  row['证据等级']=out.get('evidence_level','INCONCLUSIVE');rows.append(row)
 with (HERE/'AXIS_TABLE.csv').open('w') as file:
  writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows)
 lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(str(row[k]) for k in columns)+' |' for row in rows]
 lines+=['','单元格为成功率百分比 [95% 配对分层置信区间]；有未知时保留下界与上界。碰撞率、超时率、未知率、平均及最大遗憾、灾难数、伤害数见 METHOD_METRICS.json。样本内最佳臂与按状态奇偶交叉拟合最佳臂分列。专家没有本轮兼容评估标签，标为不适用。公平调参的持续右手规则为特权对照，不进入选择菜单。'];(HERE/'AXIS_TABLE.md').write_text('\n'.join(lines)+'\n');write(HERE/'METHOD_METRICS.json',detail)
if __name__=='__main__':main()
