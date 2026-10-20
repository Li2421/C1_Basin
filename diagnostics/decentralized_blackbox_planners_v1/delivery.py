from common import *
from audit_utils import source_inventory
import subprocess,csv
G=guard()
FULL={'DM':'分布式模型预测控制（DM）','AS':'A*规划控制器（AS）','G':'高斯行为克隆策略（G）'}
def pct(x):return f'{x*100:.2f}%'
def interval(x):return '['+', '.join(f'{100*v:.2f}' for v in x)+'] 个百分点'
def main():
 r=read(HERE/'results.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');assert audit['passed'];cap=read(HERE/'CAPABILITY_GATE.json');groups=r['groups'];checks={};gaps=[]
 subprocess.run([sys.executable,'-B',str(HERE/'tables_extra.py')],check=True,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 plot_python='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
 subprocess.run([plot_python,'-B',str(HERE/'plots.py')],check=True,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 subprocess.run([sys.executable,'-B',str(HERE/'mechanism_supplement.py')],check=True,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 strict_four=read(HERE/'STRICT_FOUR_DIRECTION.json')['groups']
 # Receipts remain immutable and all frozen files must still match.
 receipt_names=['PREREGISTRATION.receipt.json','IMPLEMENTATION_FROZEN.receipt.json','ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','R5_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','CONSTANT_CHOICES_FROZEN.receipt.json','CAPABILITY_FROZEN.receipt.json','RHR_FROZEN.receipt.json','ANALYSIS_COMPLETE.receipt.json']
 for name in receipt_names:
  rec=read(HERE/name);checks[name]=all(Path(p).exists() and sha(p)==h for p,h in rec['files'].items())
 for phase in ['probe','pilot','tune','main']:
  checks[phase+'_preflight_frozen']=all(sha(p)==h for p,h in read(HERE/f'{phase}_PREFLIGHT.receipt.json')['files'].items());checks[phase+'_official_merge']=(HERE/f'{phase}_MERGE_RECEIPT.json').exists()
  checks[phase+'_guarded']=read(HERE/f'{phase}_SUBMISSION_PREFLIGHT.receipt.json')['guard']==guard()
  recs=[read(f) for f in (HERE/phase/'raw').glob('*.json')];checks[phase+'_no_duplicate_journal']=True
  counts={}
  exp=read(HERE/f'{phase}_protocol.json')['experiment_uid'];journals=ROOT/'shared_rollout_db/journals'/exp
  for f in list(journals.glob('*.merged'))+list(journals.glob('*.jsonl')):
   for line in f.open():
    row=json.loads(line);uid=row.get('rollout_uid');counts[uid]=counts.get(uid,0)+1
  checks[phase+'_no_duplicate_journal']=len(counts)==len(recs) and all(n==1 for n in counts.values())
  if phase!='probe':
   dbrows=read(HERE/f'{phase}_DB_ROWS.json');assert len(dbrows)==len(read(HERE/f'{phase}_CASES.json'))
   u=sum(row['numerical_failure'] for row in dbrows)
   if u:gaps.append(dict(phase=phase,kind='PRESERVED_UNKNOWN',count=u,rerun=False))
 checks['round28_bitwise_16']=read(HERE/'round28_REPLAY_GATE.json')['completed']==16 and read(HERE/'round28_REPLAY_GATE.json')['passed'];checks['round33_bitwise_16']=read(HERE/'round33_REPLAY_GATE.json')['completed']==16 and read(HERE/'round33_REPLAY_GATE.json')['passed'];checks['unit_tests']=read(HERE/'UNIT_TESTS.json')['passed'];checks['statistics_tests']=read(HERE/'STATISTICS_TESTS.json')['passed'];checks['astar_seed_identity']=read(HERE/'ASTAR_SEED_IDENTITY.json')['passed']
 subprocess.run([sys.executable,'-B',str(HERE/'source_start_coverage.py')],check=True,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 checks['all_research_imports_have_start_hash']=read(HERE/'SOURCE_START_COVERAGE.json')['passed']
 # All imported sources are inventoried and unchanged relative to each recorded import.
 inv={};conflicts=[]
 for f in [HERE/'IMPORTED_SOURCE_BEGIN.json']+sorted(HERE.glob('imports_*.json')):
  for path,h in read(f).get('imported_sources',{}).items():
   if Path(path).is_relative_to(HERE):continue
   if path in inv and inv[path]!=h:conflicts.append(path)
   inv[path]=h
 changes=[p for p,h in inv.items() if not Path(p).exists() or sha(p)!=h];write(HERE/'IMPORTED_SOURCE_END.json',dict(utc=utc(),files=inv,changed=changes,conflicting_import_hashes=conflicts));checks['imported_sources_unchanged']=not changes and not conflicts
 # Frozen choice and prereg timestamps must precede every pilot/tuning/main trace.
 freeze=read(HERE/'CONSTANT_CHOICES_FROZEN.receipt.json')['utc'];starttimes={phase:read(HERE/f'{phase}_SUBMISSION.json')['utc'] for phase in ['pilot','tune','main']};checks['choice_freeze_before_all_evaluations']=all(freeze<t for t in starttimes.values());checks['rhr_freeze_before_main']=read(HERE/'RHR_FROZEN.receipt.json')['utc']<starttimes['main']
 checks['probe_db_postflight']=(HERE/'PROBE_DB_POSTFLIGHT.receipt.json').exists()
 checks['no_resource_exception']=not (HERE/'RESOURCE_EXCEPTION.json').exists()
 before_status=set(audit['git_status_before'].splitlines());after_status=set(audit['git_status_after'].splitlines());external_status=sorted(x for x in after_status-before_status if 'decentralized_blackbox_planners_v1' not in x);removed_status=sorted(x for x in before_status-after_status if 'decentralized_blackbox_planners_v1' not in x);checks['no_external_git_status_removals']=not removed_status;write(HERE/'EXTERNAL_GIT_STATUS.json',dict(new_entries_not_written_by_this_round=external_status,removed_entries=removed_status))
 from integrity import HIST
 protected_roots=[FIELD]+[D/x for x in HIST];new_in_protected=[p for p in audit['new_external_files'] if any(Path(p).is_relative_to(root) for root in protected_roots)];checks['protected_tree_file_sets_unchanged']=not new_in_protected
 write(HERE/'FINAL_VERIFICATION.json',dict(utc=utc(),passed=all(checks.values()),checks=checks,freeze=freeze,submission_times=starttimes,guard=guard()));assert all(checks.values()),checks
 for family in ['DM','AS','G']:
  if groups[family]['status']=='INCONCLUSIVE':gaps.append(dict(family=family,kind='FEWER_THAN_FOUR_ELIGIBLE',count=len(groups[family]['controllers'])))
 write(HERE/'GAPS.json',gaps)
 mechs=read(HERE/'MECHANISMS.json');opening=['完整性审计通过，Basin_C1 共享数据库与默认 journal 路径断言通过。']
 for f,full in FULL.items():
  g=groups[f]
  if not g.get('controllers'):opening.append(full+'无可评估实例，终点不确定。');continue
  e1=g['endpoints']['one'];e2=g['endpoints']['two'];to=g['arms']['safety baseline']['timeout']['lower']['estimate'];a3=g['choices']['A3_full'];reg=a3['oracle_regret']['lower']['estimate'];s=g['arms']['S500']['success']['lower']['estimate'];b=g['arms']['safety baseline']['success']['lower']['estimate'];hr=g['arms']['RHR-fair']['success']['lower']['estimate']
  verdict='通过' if e1['decision']=='PASS' else '不确定（不足4实例）' if e1['decision']=='INCONCLUSIVE' else '未通过'
  fair='通过' if e2['decision']=='PASS' else '不确定（不足4实例）' if e2['decision']=='INCONCLUSIVE' else '未通过'
  if e2['decision']=='NOT_PASS' and e2['CI_unknown_envelope'][1]<0:fair+='（配对区间整体为负，所测右手规则更好）'
  critic_verdict='达到所测四臂样本内最优' if abs(reg)<1e-12 else '未达到所测四臂样本内最优'
  opening.append(f'{full}在分布式安全层下超时率{pct(to)}；集中式确定的前500步第13组、之后第10组修正时间表，成功率从{pct(b)}变为{pct(s)}，终点一{verdict}；公平调参右手规则成功率{pct(hr)}，时间表优于它的终点二{fair}；冻结的通用critic相对控制器级四臂事后最优的平均遗憾为{100*reg:.2f}个百分点，{critic_verdict}。')
 new_choices=read(HERE/'FROZEN_CHOICES.json')['controllers']
 if all(all(c[k]=='S500' for k in ['A3_full','A0_full','G_E']) for c in new_choices.values()):opening.append('三种冻结选择器对18个新实例均选择同一个前500步时间表，因此没有相对该固定默认方案的实际选择收益；其是否达到四修正臂的样本内最优另按上述遗憾评价，选择菜单不含右手规则。')
 opening.append('超时和停滞诊断不等于已证明死锁。')
 lines=[''.join(opening),'','本轮按原任务执行，三新族的主终点为CONFIRMATORY；历史学生与合并总表为DEVELOPMENT。能力门选择会改变适用实例集合；每族少于4个实例时，不能据其成功率下结论。','', '## 冻结、复用与信息条件','', '原任务全文、固定网格、阈值、平局规则和统计方案见PREREGISTRATION.md。18个实例的576条无终止R5探测完成后才冻结选择，随后才提交能力门、调参和主评估。分布式安全层和模型预测控制各16条历史记录逐位复现；高斯策略checkpoint及归一化hash逐项核验，无训练。机制代码逐字复制round30g，只更改支持模块的导入路径，未沿用其事后敏感性指标。','', '分布式模型预测控制各机器人自行随机采样优化，但冻结实现使用其余三机器人的位置和速度预测。高斯行为克隆策略输入四机器人的联合观测，需要共享观测。A*只使用自身位置、自身目标与1.5米内邻居位置。三者都使用相同的分布式安全层与固定修正；不能将“自行计算动作”误写成三者都不通信或仅局部感知。','', 'Round33主评估使用另一组物理状态和随机别名，不能复用为本轮Round28评估标签。14个原学生的6272条角色记录与原冻结选择全部复用，没有为它们重新训练或生成轨迹。新三族使用24候选调参；学生按任务要求复用历史20候选调参结果，没有补跑新增4候选。','', '## 能力门与右手规则调参','']
 lines+=['| 实例 | 只加安全层成功率未知界 | 前500步修正时间表成功率未知界 | 能力门 |','|---|---:|---:|---|']
 for n,rr in cap['rates'].items():
  outcome='能力不足' if n in cap['excluded_low'] else '自身已处理交互、无需修正' if n in cap['excluded_high'] else '进入主评估';a=rr['safety baseline'];b=rr['S500'];lines.append(f"| {n} | {pct(a['lower'])}–{pct(a['upper'])} | {pct(b['lower'])}–{pct(b['upper'])} | {outcome} |")
 lines+=['', '能力不足排除名单：'+(', '.join(cap['excluded_low']) or '无')+'。自身已处理交互排除名单：'+(', '.join(cap['excluded_high']) or '无')+'。','']
 for f,choice in r['rhr_selected'].items():lines.append(f"{FULL[f]}冻结右手规则：{json.dumps(choice['rule'],ensure_ascii=False)}；调参成功率{pct(choice['lower'])}，未知界上限{pct(choice['upper'])}，{choice['n']}条。完整24候选见RHR_GRID_RESULTS.json。")
 lines+=['', '## 逐族终点与失败机制','']
 for f,full in FULL.items():
  g=groups[f];lines += ['### '+full,'']
  if not g.get('controllers'):lines+=['无主评估实例；INCONCLUSIVE。',''];continue
  for lab in ['one','two']:
   e=g['endpoints'][lab];lines.append(f"终点{'一（时间表减只加安全层）' if lab=='one' else '二（时间表减公平右手规则）'}：{e['decision']}，保守95%CI {interval(e['CI_unknown_envelope'])}。")
  a=g['choices']['A3_full'];lines.append(f"冻结通用critic的选择：{a['selected']}。遗憾{100*a['oracle_regret']['lower']['estimate']:.2f}个百分点，未知包络95%CI {interval(a['oracle_regret']['CI_unknown_envelope'])}；相对一律前500步时间表的差CI {interval(a['minus_S500']['CI_unknown_envelope'])}；相对一律全程第10组的差CI {interval(a['minus_F10']['CI_unknown_envelope'])}。这些比较为描述性。")
  critic_ci=a['minus_S500']['CI_unknown_envelope']
  lines.append('描述性配对区间显示critic逊于固定时间表；在这两种零样本方案之间优先固定时间表，不据此声称其优于该族所有方法。' if critic_ci[1]<0 else '描述性配对区间支持critic优于固定时间表。' if critic_ci[0]>0 else '尚未证实critic相对固定时间表有选择收益，也不把区间跨零解读为两者等效。')
  m=mechs[f];base=m['safety baseline'];nom=m['nominal only'];lines.append(f"只跑控制器的碰撞对象（可重叠）：中心障碍{nom['collision_objects_overlapping']['center']}，外环{nom['collision_objects_overlapping']['outer']}，机器人{nom['collision_objects_overlapping']['agent']}；几何核验{nom['collision_geometry_verified']}条。只跑控制器绕行标签计数：{nom['circulation_counts']}（0静止，1顺时针一致，2逆时针一致，3混合）。")
  lines.append('绕行模式1/2表示出现非零绕行且没有反向机器人，可包含零标签机器人；严格四机器人同向非零另由既有four_same标记统计：'+json.dumps({a:strict_four[f][a]['all_four_same_nonzero']['estimate'] for a in ['nominal only','safety baseline','S500']},ensure_ascii=False)+'。完整置信区间见STRICT_FOUR_DIRECTION.json。')
  if base['timeout_count']:
   mm=base['timeout_metrics'];val=lambda k:'未定义' if mm[k]['estimate'] is None else f"{mm[k]['estimate']:.3f}";lines.append(f"只加安全层的{base['timeout_count']}条超时局：在动但无进展平均{val('moving_no_progress_robot_seconds')}机器人秒，严格低速停滞平均{val('stall_robot_seconds')}机器人秒、{val('stall_runs')}段，最后显著进展平均发生在{val('last_progress_seconds')}秒，最终未到达{val('final_unfinished')}个机器人。任何严格低速停滞的比例为{base['timeout_any_strict_stall']['estimate']}；混合方向比例为{base['timeout_mixed']['estimate']}。")
  failure_time=m['S500']['nominal_failure_time_seconds'];lines.append('时间表残留失败的平均终止时间（秒）：'+str(failure_time['estimate'])+'，95%CI '+str(failure_time['CI95'])+'；应结合碰撞对象与超时类型阅读，不能把35秒预算耗尽称作死锁事件。')
  gain=m['S500']['early_100_progress_minus_safety_m'];lines.append(f"时间表在前100步相对只加安全层多取得总目标距离进展{gain['estimate']:.3f}米，95%CI {gain['CI95']}。已知提前终止按原定义吸收。")
  fail=read(HERE/f'{f}_FAILURE_COMPOSITION.json');lines.append(f"自身名义超时且无碰撞{fail['own_nominal_stall_total']}条，其中加安全层后仍超时{fail['own_stall_persisting_under_safety']}条；名义碰撞转为安全层超时{fail['safety_induced_stall_total']}条。时间表救回计数：{fail['rescues']['S500']}。完整逐条转移及各臂救回计数见{f}_FAILURE_COMPOSITION.json及FAILURE_RESCUES.csv。")
  lines+=['',f"论文表：[PAPER_TABLE_{f}.md](PAPER_TABLE_{f}.md)。",'']
  e1=g['endpoints']['one']['decision'];e2=g['endpoints']['two']['decision']
  conclusion='实例不足，不能确认该族的修正效果。' if g['status']=='INCONCLUSIVE' else '修正在该族通过两个确认性终点。' if e1==e2=='PASS' else '修正改善成功率，但未确认优于逐族调参右手规则；调参成本差异仍存在，不能宣称性能优势。' if e1=='PASS' else '修正未通过该族有效性终点；这是当前固定时间表的适用边界，不能解读为效应严格为零。'
  if g['status']!='INCONCLUSIVE' and g['endpoints']['two']['CI_unknown_envelope'][1]<0:conclusion+=' 时间表减右手规则的配对区间整体为负，不能仅写成未证实优势；本轮所测公平右手规则的成功率更高。'
  if g['status']!='INCONCLUSIVE' and g['endpoints']['one']['CI_unknown_envelope'][1]<0:conclusion+=' 时间表减只加安全层的配对区间整体为负，所测时间表存在成功率退化。'
  lines += [conclusion,'']
 lines+=['## 合并总表与限制','', 'PAPER_TABLE_ALL.md/csv合并三新族符合能力门的实例与14个原学生，仅DEVELOPMENT；PAPER_TABLE_V0.md保留学生参照。总体均值按控制器等权，不能掩盖逐族结果或将共享观测高斯策略宣称为不通信。unknown按成功下界0与上界1报告，unknown不重跑且不记作已知失败。事后最优是同一评估集上的控制器级四臂选择，属于描述性上界，未做交叉验证。','', 'A*两个seed的动作/位置/速度逐位一致核验见ASTAR_SEED_IDENTITY.json。没有改变物理、事件、成功标准、安全层、修正、冻结权重，没有安装包，没有git写操作，不交付视频。', '', '## 审计、数据库与执行记录','', f"保护审计核查{len(read(HERE/'INTEGRITY_BEGIN.json')['files'])}个已有文件。结束审计通过，新增外部Python文件{audit['new_external_files']}；Git状态前后详见INTEGRITY_COMPARISON.json，轮外新增不归因于本轮。导入源文件逐项hash一致，见IMPORTED_SOURCE_END.json。Git轮外新增条目另列EXTERNAL_GIT_STATUS.json，并明确不由本轮写入。",'', 'DB采用只追加新身份、封存preflight、缺失任务、默认journal、单进程官方merger和postflight；registration/每阶段保存既有登记行前后hash及插入行hash。各阶段的journal唯一性已核验；行为探测仅作为SUMMARY_ONLY来源，没有任务rollout标签。', '', '最初复现作业22825因debug分区时限配置而一直PENDING，取消后改用cpu分区22826，未产生轨迹且没有科学重跑；预注册的路径适配补充发生在新轨迹之前并保留初始收据。资源以全部用户RUNNING/PENDING数组元素预留，提交最多18个CPU shard，没有GPU shard。详细提交和队列记录见各阶段SUBMISSION与SUBMISSION_PREFLIGHT收据。', '', '科研图件见figures/family_success.png、figures/family_endpoints.png（均附PDF），仅根据封存仿真证据绘制。','当前缺口：'+json.dumps(gaps,ensure_ascii=False)+'。所有原始证据保留。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'PAPER_TABLE.md').write_text('\n\n'.join((HERE/f'PAPER_TABLE_{f}.md').read_text() for f in ['DM','AS','G','V0','ALL']))
 handoff='''本轮执行 docs/prompt_round30e_decentralized_blackbox_planners_20261016.md。全部新代码与产物在 diagnostics/decentralized_blackbox_planners_v1，唯一目录外写入为官方共享数据库登记、journal和官方merger。不要重启coordinator.py；所有unknown与轨迹保留，禁止重新运行科学任务或覆盖证据。

阅读 REPORT.md 首段与逐族终点，然后阅读 results.json。主论文表 PAPER_TABLE.md、PAPER_TABLE.csv；三个族独立表与合并DEVELOPMENT表另存。CAPABILITY_GATE.json 有能力不足和无需修正两类排除，RHR_GRID_RESULTS.json 有完整24候选、RHR_SELECTED.json 为每族预先冻结选择。FROZEN_CHOICES.json、CONSTANT_CHOICES.json 及收据证明先于pilot/tune/main；无训练。

pipeline.py、episode_frozen.py、fair_rhr.py 从round30d逐字复制；controllers.py、legacy_controllers.py、gaussian.py 从round33逐字复制；metrics.py、stats.py 从round30g复制，仅支持模块导入路径改变。来源与diff见 COPY_PROVENANCE.json、VENDOR_PROVENANCE.json。A*实现 astar.py 是本轮新写。分布式模型预测控制需要其余机器人的位置和速度；高斯行为克隆需要四机器人联合观测；A*是自身/局部邻居接口。不得统称三族都不通信。

数据库登记行在registration/，每个phase都有protocol、CASES、MISSING_CASES、PLANNED_ROLLOUTS、官方CACHE_PREFLIGHT/POSTFLIGHT（探测的POSTFLIGHT报告无任务标签）、MERGE_RECEIPT、SUBMISSION。原始轨迹在probe/pilot/tune/main/raw；32条校验重放另存，均逐位一致。不要把timeout称deadlock。EPISODE_METRICS.jsonl、MECHANISMS.json复用round30g机制定义；各族_FAILURE_COMPOSITION.json与FAILURE_RESCUES.csv含来源和救回计数。

完整性审计、导入来源及冻结时序最终核验见 INTEGRITY_COMPARISON.json、IMPORTED_SOURCE_END.json、FINAL_VERIFICATION.json。DELIVERY.receipt.json 是最终交付有效性依据；PREREGISTRATION_initial.receipt.json 是执行前路径适配的历史收据，当前收据为PREREGISTRATION.receipt.json。所有新族终点CONFIRMATORY，历史学生和合并分析DEVELOPMENT；GAPS.json逐项列出未知或能力缺口。没有修改旧源码、重训、包安装、git写操作或视频任务。
'''
 (HERE/'HANDOFF.md').write_text(handoff);source_inventory('imports_delivery.json');seal('DELIVERY.receipt.json',[HERE/n for n in ['REPORT.md','HANDOFF.md','PAPER_TABLE.md','PAPER_TABLE.csv','results.json','MECHANISMS.json','GAPS.json','FINAL_VERIFICATION.json','INTEGRITY_COMPARISON.json','IMPORTED_SOURCE_END.json']]+sorted((HERE/'figures').glob('*')),passed=True,guard=guard());write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='COMPLETE_WITH_GAPS' if gaps else 'COMPLETE',gaps=gaps,guard=guard()));print('DELIVERY PASS',flush=True)
if __name__=='__main__':main()
