"""Complete Chinese report and audited research handoff; no optional plotting dependency."""
from common import *
from design import *
from statistics_pool import METHOD_NAMES
import numpy as np,difflib,subprocess

def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}个百分点'
def ci(v):return '['+', '.join(f'{100*x:+.2f}'for x in v)+']个百分点'
def eta(c):return describe_eta(c['key'],c.get('eta_a',c.get('eta')))
def yes(v):return '是'if v else'否'
def update_provenance():
 r=read(HERE/'TOOLING_COPY.receipt.json')
 for row in r['files']:
  assert sha(row['source'])==row['source_sha256'];row['target_sha256']=sha(row['target']);row['changed']=row['source_sha256']!=row['target_sha256']
  if row['changed']:
   p=Path(row['target']);p.with_suffix(p.suffix+'.diff').write_text(''.join(difflib.unified_diff(Path(row['source']).read_text().splitlines(True),p.read_text().splitlines(True),fromfile=row['source'],tofile=row['target'])))
 write(HERE/'TOOLING_COPY.receipt.json',r)

def geometry_final():
 reports=read(HERE/'GEOMETRY_BASE_REPORTS.json');tables=read(HERE/'LABEL_TABLES.json');cap=read(HERE/'CAPABILITY_TABLE.json')
 for n,r in reports.items():
  r['joint_solvable']='未认证';r['deadlock_resolution']='未认证'
  for state in r['states']:
   state['joint_solvable']='未认证';state['deadlock_resolution']='未认证'
   if n in tables:
    i=state['index'];y=np.asarray(tables[n]['y'])[:,i];u=np.asarray(tables[n]['unknown'])[:,i];state['controller_success']=dict(successful_cells=int(y.sum()),total_cells=int(y.size),unknown_cells=int(u.sum()),per_controller={k:int(y[j].sum())for j,k in enumerate(CONTROL_IDS)})
   else:state['controller_success']='未运行联合主标签；预定mild替代或能力门不适用'
 write(HERE/'GEOMETRY_REPORTS.json',reports)

def report():
 s=read(HERE/'STATISTICS.json');d=s['decisions'];r=s['comparisons']['main'];a=r['comparisons']['global_fixed'];b=r['comparisons']['knn2'];c=r['comparisons']['old_pool'];names=s['scenes'];methods=s['method_order'];pooldata=read(HERE/'TRAINING_POOL.json');cap=read(HERE/'CAPABILITY_TABLE.json')
 first=(f"critic是否修好（本验证集上）：{yes(d['critic_fixed_on_this_validation_set'])}。有用：{yes(d['useful'])}，主方法减一个全局η{pp(a['difference'])}，只重抽状态95%区间{ci(a['ci_states'])}，场景→状态95%区间{ci(a['ci_scene_states'])}。超过抄近邻：{d['beats_neighbour']}，减抄最近2个训练场景{pp(b['difference'])}，只重抽状态95%区间{ci(b['ci_states'])}，场景→状态95%区间{ci(b['ci_scene_states'])}。补场景有用：{yes(d['supplement_useful'])}，比补充前训练池{pp(c['difference'])}，只重抽状态95%区间{ci(c['ci_states'])}。不伤：{yes(d['not_harmful'])}，最差验证场景相对全局η为{pp(d['worst_scene_difference'])}。四条分开判定，均使用未四舍五入JSON。")
 lines=['# critic训练池补充 v1\n',first+'\n','这8个验证场景在先前开发中已用于比较配方，本轮不把它们用于训练、标准化、近邻表或超参数选择；最终宣称需要另一批未碰过的场景做测试。本轮只训练与评估critic，不训练生成器，不运行生成器提案。\n',
 '新critic·近邻输入（本轮主方法，main）预测每个状态、控制器、η的成功logit；10种子平均logit后转概率，对整个场景全部状态和3控制器平均预测概率选一组η。输入为尺度无关场景描述c′12维、状态特征h6维、控制器3维独热、归一化η3维，以及排除自己后最近3个源场景的响应输入3维；输出无先验锚定、压缩或超范围退回。\n',
 '集中式随机采样模型预测控制（centralized_sampling0）、梯度下降模型预测控制（gradient_mpc0）、分布式模型预测控制（DM0）等权；计分状态等权，8场景等权。η三个数依次为朝目标、侧移、互斥。新critic·无近邻输入、原网络结构、补充前训练池与逐类加入均严格按固定配方。\n',
 f"训练池含{sum(not p['new']for p in pooldata)}个已有版本与{sum(p['new']for p in pooldata)}个通过能力门的新场景，共{sum(p['cells']for p in pooldata)}格；包含菜单外精确η，逐格软标签是全部兼容rollout的平均成功。训练按场景等权、场景内格等权抽样，unknown计失败。已有数据与本轮新标签来源、不同η数和c′见TRAINING_POOL.json。\n",
 '主口径仅计核对状态：十字路口与方形场地8–23、对穿12–27、文献已有16状态（圆形五机原编号16–31）。另报全部已标注状态。模型选η使用无标签输入中的全部状态；没有以验证标签重选任何学习方法。\n',
 '## 主比较\n','| 方法 | 核对状态平均成功率 | unknown计成功上界 | 全部标注状态平均率 | 减全局η（状态区间；场景→状态区间） | 减最近2个（状态区间；场景→状态区间） |\n|---|---:|---:|---:|---|---|']
 for m in methods:
  v=s['comparisons'][m];g=v['comparisons']['global_fixed'];k=v['comparisons']['knn2'];whole='旧报告无此口径'if v['full_state_mean_rate']is None else pct(v['full_state_mean_rate'])
  lines.append(f"| {METHOD_NAMES[m]}（{m}） | {pct(v['rate'])} | {pct(v['unknown_success_upper'])} | {whole} | {pp(g['difference'])}；{ci(g['ci_states'])}；{ci(g['ci_scene_states'])} | {pp(k['difference'])}；{ci(k['ci_states'])}；{ci(k['ci_scene_states'])} |")
 lines.append('\n上一轮学习选η v2的点估计逐字沿用其已有STATISTICS数值；本轮新增对比的配对区间只使用旧冻结选择对应的既有逐状态结果，不重训、不重新推断、不重选、不重跑。参照方法使用验证标签，不是零样本方法。覆盖上界限定于训练场景在49菜单中有≥50%格覆盖的最好候选；选择能在验证场景计分的49菜单，不添加rollout。\n')
 scene_table=['# 逐验证场景结果\n']
 for n in names:
  scene_table.extend([f'## {n}\n','| 方法 | 选出的η（三个数） | 核对成功率 | unknown上界 |\n|---|---|---:|---:|'])
  for m in methods:
   v=s['per_scene'][n]['methods'][m];chosen=eta(v['candidate'])if v['candidate']is not None else'逐核对状态留一选择，完整索引见STATISTICS.json'
   scene_table.append(f"| {METHOD_NAMES[m]} | {chosen} | {pct(v['rate'])} | {pct(v['unknown_success_upper'])} |")
  scene_table.append('')
 (HERE/'PER_SCENE_RESULTS.md').write_text('\n'.join(scene_table)+'\n');lines.extend(['\n'+v for v in scene_table])
 lines.extend(['\n## 上一轮仍错的四个场景\n','| 场景 | 主方法η | 主方法成功率 | 全局η率 | 最近2个率 | 补充前池率 |\n|---|---|---:|---:|---:|---:|'])
 for n in ['lit_swap8_obstacles','xsc_swap_n10','lit_swap8_free','lit_circle_five']:
  v=s['per_scene'][n]['methods'];lines.append(f"| {n} | {eta(v['main']['candidate'])} | {pct(v['main']['rate'])} | {pct(v['global_fixed']['rate'])} | {pct(v['knn2']['rate'])} | {pct(v['old_pool']['rate'])} |")
 additions=['# 补场景作用\n','A：无障碍对穿；B：静态障碍对穿；C：方形场地；D：大机器人方形场地；E：十字路口。逐类加入各5种子，补充前池与完整主方法各10种子；没有据验证表现挑选类别替代主方法。\n','| 验证场景 | 补充前池率 | 完整池主方法率 | 完整池减补充前 | 加A变化 | 加B变化 | 加C变化 | 加D变化 | 加E变化 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in names:
  v=s['per_scene'][n]['methods'];changes=' | '.join(pp(s['class_additions'][g]['per_scene'][n])for g in 'ABCDE');additions.append(f"| {n} | {pct(v['old_pool']['rate'])} | {pct(v['main']['rate'])} | {pp(v['main']['rate']-v['old_pool']['rate'])} | {changes} |")
 additions.append(f"\n照抄对的训练场景覆盖上界：补充前{pct(s['comparisons']['copy_old_oracle']['rate'])}，完整池{pct(s['comparisons']['copy_oracle']['rate'])}。具体可照抄的源场景与η见STATISTICS.coverage_upper_details。\n")
 (HERE/'SUPPLEMENT_EFFECTS.md').write_text('\n'.join(additions)+'\n');lines.extend(['\n'+v for v in additions])
 # New scenes: all capability results and complete49-candidate curves.
 tables=read(HERE/'LABEL_TABLES.json');library={r['name']:r for r in read(HERE/'SCENE_LIBRARY.json')};newlines=['# 新场景与能力门\n','| 场景 | 类别 | 3控制器能力门成功/总数 | 主标签样本内前3候选η与成功率 |\n|---|---|---|---|'];rates=['# 新场景49候选成功率\n'];best_by_group={g:[]for g in 'ABCDE'}
 for base,status in cap['scene_status'].items():
  n=status['active_name'];gate=cap['per_scene'][n];gatetext='；'.join(CONTROL_NAMES[k]+f" {gate[k]['success']}/{gate[k]['total']}"for k in CONTROL_IDS)
  if n not in tables:newlines.append(f'| {n} | {status["group"]} | {gatetext} | 不适用；未运行联合标签 |');continue
  y=np.asarray(tables[n]['y']);u=np.asarray(tables[n]['unknown']);q=y.mean((0,1));qu=(y+u).mean((0,1));order=np.lexsort((np.arange(49),-q));top='；'.join(eta(menu()[int(j)])+' '+pct(q[j])for j in order[:3]);newlines.append(f'| {n} | {status["group"]} | {gatetext} | {top} |');best_by_group[status['group']].append(dict(scene=n,candidate=menu()[int(order[0])],rate=float(q[order[0]])))
  rates.extend([f'## {n}\n','| 候选η（三个数） | 成功率 | unknown计成功上界 |\n|---|---:|---:|'])
  for j,a in enumerate(menu()):rates.append(f'| {eta(a)} | {pct(q[j])} | {pct(qu[j])} |')
 for group,title in [('B','静态障碍对穿是否偏好负侧移、大互斥'),('D','大机器人方形场地是否偏好小互斥、大朝目标')]:
  newlines.append('\n'+title+'：按各场景实测最好候选的三个分量直接核对，不另设事后阈值。')
  for v in best_by_group[group]:newlines.append(v['scene']+'：'+eta(v['candidate'])+'，'+pct(v['rate'])+'。')
 (HERE/'NEW_SCENES.md').write_text('\n'.join(newlines)+'\n');(HERE/'CANDIDATE_RATES.md').write_text('\n'.join(rates)+'\n');write(HERE/'NEW_SCENE_BEST.json',best_by_group);lines.extend(['\n'+v for v in newlines]);lines.append('\n完整49候选表见CANDIDATE_RATES.md。几何有效、单机器人路径、联合可解、控制器成功、死锁解决分别见GEOMETRY_REPORTS；后两类认证不由timeout推出，联合可解与死锁解决保持未认证。\n')
 # Per-cell policy, kept distinct from the scene-level main method.
 cell=['# 每个状态和控制器各自选η（附报）\n','| critic变体 | 核对状态平均成功率 | 全部标注状态平均成功率 |\n|---|---:|---:|']
 for m in RUN_METHODS_LOCAL:
  cell.append(f"| {METHOD_NAMES[m]}·逐格选择 | {pct(np.mean([s['per_cell'][n][m]['rate']for n in names]))} | {pct(np.mean([s['per_cell'][n][m]['full_state_rate']for n in names]))} |")
 cell.append('\n逐格η索引与完整菜单三分量见ALL_CHOICES.json和ETA_MENU.json；主判定使用整个场景一组η。')
 (HERE/'PER_CELL_RESULTS.md').write_text('\n'.join(cell)+'\n');lines.extend(['\n'+v for v in cell])
 # Grouped CV descriptions.
 cv=['# 训练池内分组交叉验证\n','拥挤随机7版本始终同组；组名排序后按固定种子打乱分6折。每个训练变体10种子平均logit。只在≥40个候选有≥50%状态/控制器格覆盖的场景上计遗憾；被选候选自身覆盖不足时单列缺口，不补rollout、不填零。\n','| 折 | 方法 | 可计分场景数 | 平均遗憾（百分点） |\n|---|---|---:|---:|']
 for fold in list(range(6))+['全部']:
  for m in ['main','no_neighbour','global_fixed','knn1','knn2','knn3']:
   vals=[r['methods'][m]['regret']for r in s['training_pool_cv']if r['eligible']and(fold=='全部'or r['fold']==fold)and r['methods'][m]['regret']is not None];value='无法计算'if not vals else f'{100*np.mean(vals):.3f}';cv.append(f'| {fold} | {METHOD_NAMES[m]} | {len(vals)} | {value} |')
 (HERE/'TRAINING_POOL_CV.md').write_text('\n'.join(cv)+'\n');lines.extend(['\n'+v for v in cv])
 curve=['# 学习曲线\n','每个比例独立抽5次源场景，每次主方法3种子平均logit；抄近邻与全局η都只用该次子集。\n','| 训练场景比例 | 子集次数 | 每次场景数 | critic成功率均值±子集标准差 | 最近2个均值 | 全局η均值 |\n|---|---:|---|---|---:|---:|']
 for fraction in [.25,.5,.75]:
  rows=[v for v in s['learning_curves']if v['fraction']==fraction];values=[v['rates']['critic']for v in rows];curve.append(f"| {fraction:.0%} | {len(rows)} | {sorted(set(v['training_scenes']for v in rows))} | {pct(np.mean(values))} ± {100*np.std(values):.2f}个百分点 | {pct(np.mean([v['rates']['knn2']for v in rows]))} | {pct(np.mean([v['rates']['global_fixed']for v in rows]))} |")
 curve.append(f"| 100%完整池 | 1 | {len(pooldata)} | {pct(s['comparisons']['main']['rate'])}（10种子） | {pct(s['comparisons']['knn2']['rate'])} | {pct(s['comparisons']['global_fixed']['rate'])} |")
 (HERE/'LEARNING_CURVES.md').write_text('\n'.join(curve)+'\n');lines.extend(['\n'+v for v in curve]);lines.append('\n曲线逐次抽样与逐场景数值见STATISTICS.learning_curves，来源清单见LEARNING_CURVE_SUBSETS。曲线不用于改变主方法配方。\n')
 plan=read(HERE/'TRAINING_PLAN.json');lines.append(f"\n实际计划{plan['fits']}次拟合：完整4变体40、逐类加入25、曲线45、分组CV120。CV按一般10种子规则执行，超出prompt中约120的粗略规模估计；没有额外配方搜索或生成器。原网络4,000步，其余3,000步；余弦学习率在各自规定步数结束时衰减至0，无早停或挑种子。训练最大14并发，总18CPU。\n")
 lines.append('5000次配对自助法：固定8场景只重抽核对状态种子2026102671，场景→状态种子2026102672；全部方法共享抽样索引。区间条件于已训练模型、已冻结选择与这8个开发中触及的场景，不包括再选训练池、配方、模型种子或新场景总体的不确定性。\n')
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');return first
RUN_METHODS_LOCAL=['main','no_neighbour','original','old_pool']+['add_'+g for g in 'ABCDE']

def main():
 from audit_pool import database,historical_database,integrity
 geometry_final();first=report();update_provenance();database();historical_database()
 cv=read(HERE/'STATISTICS.json')['training_pool_cv'];gaps=[r for r in cv if not r['eligible']or any(v['regret']is None for v in r['methods'].values())];cap=read(HERE/'CAPABILITY_TABLE.json');excluded=[v for v in cap['scene_status'].values()if not v['test_applicable']]
 (HERE/'GAPS.md').write_text('# 证据缺口与范围\n\n'+f'能力门不适用场景{len(excluded)}个，记录见CAPABILITY_TABLE。CV中{len(gaps)}个场景因候选覆盖不足或所选候选不可计分而有描述性缺口，记录见STATISTICS.training_pool_cv；未填零、未为其追加rollout。\n\n验证8场景参与过配方开发；最终宣称需要新的未触及场景。当前只有3控制器、种子0的新训练场景标签，未证明任意控制器/新η/新场景总体上的表现。联合可解与死锁解决未认证。\n')
 integrity();post=read(HERE/'DB_POSTFLIGHT.json');assert post['passed']and post['all_requests_present']
 with(HERE/'REPORT.md').open('a')as f:f.write(f"\n新正式主标签{post['new_main_requests']}条，含能力门共{post['new_requests']}条新请求；默认共享数据库身份、结果与轨迹逐项核对通过，无未合并journal。已有训练行、旧源码和全部冻结文件审计通过。\n")
 (HERE/'STATUS.md').write_text('# 当前状态\n\n'+utc()+'：COMPLETE。标注、14版本抽取、critic训练、冻结、验证、统计及最终审计全部完成。\n\n'+first+'\n')
 files=['REPORT.md','HISTORY_CHECK.md','SCENES.json','GEOMETRY_REPORTS.json','CAPABILITY_TABLE.json','LABEL_TABLES.json','TRAINING_POOL.json','FROZEN.receipt.json','TRAINING_FROZEN.receipt.json','STATISTICS.json','TEST_DECISION.json','STATUS.md','GAPS.md','ITERATION_LOG.md','DB_POSTFLIGHT.json','HISTORICAL_ROWS_END.receipt.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','PARALLEL_METADATA_END.json','PER_SCENE_RESULTS.md','SUPPLEMENT_EFFECTS.md','NEW_SCENES.md','CANDIDATE_RATES.md','PER_CELL_RESULTS.md','TRAINING_POOL_CV.md','LEARNING_CURVES.md','TOOLING_COPY.receipt.json','VALIDATION_LABEL_ACCESS.receipt.json','TRAINING_PLAN.json']
 write(HERE/'DELIVERY_CODE_SHA256.json',{str(p):sha(p)for p in HERE.glob('*.py')});files.append('DELIVERY_CODE_SHA256.json')
 write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',decisions=read(HERE/'STATISTICS.json')['decisions'],new_main_requests=post['new_main_requests'],new_requests=post['new_requests'],source_and_database_audits_passed=True,reports={n:sha(HERE/n)for n in files}));write(HERE/'RUN_STATE.json',dict(status='COMPLETE',active_jobs=[],report=str(HERE/'REPORT.md')));print(first,flush=True)
if __name__=='__main__':main()
