from common import *
from analyze import METHODS
import csv

def history():
 z=np.load(D/'method_benchmark_v1/T2_ANALYSIS.npz');nn=z['methods'].tolist();ar=z['arms'].tolist();rng=np.random.default_rng(202610153203);r={}
 for label,method in [('F10','M0'),('S500','M2'),('A3 zero-shot','M4_full'),('G_E','M5')]:
  w=z['weights'][:,nn.index(method)];lo=(z['lo']*w[:,None]).sum(-1);hi=(z['hi']*w[:,None]).sum(-1);boot=[]
  for _ in range(5000):
   cc=rng.integers(14,size=14);ss=rng.integers(32,size=(14,32));boot.append([lo[cc[:,None],ss].mean()*100,hi[cc[:,None],ss].mean()*100])
  r[label]=dict(success_pct=[float(lo.mean()*100),float(hi.mean()*100)],success_95CI_pct=[float(np.quantile(np.asarray(boot)[:,0],.025)),float(np.quantile(np.asarray(boot)[:,1],.975))])
 write(HERE/'HISTORICAL_DEVELOPMENT.json',dict(R26_T2=r,R16_T2_legacy_mean_regret_pp=.75,R26_aligned_T2_mean_regret_pp=next(x['mean_regret'] for x in read(D/'method_benchmark_v1/T2_POINT_RESULTS.json')['rows'] if x['method']=='M4_full')*100,R2=dict(gain_pp=5.21,CI_pp=[2.65,7.88]),note='R16 legacy T2 used obsolete state file; R26 aligned predictions preferred. R26 uses 32 evaluation states and split/calibration/menu unlike current96-state four-arm cohort. No new rollout/training/probes. Historical rate CI recomputed from frozen analysis arrays, paired nested controller/state bootstrap5000, seed202610153203.'))
 return r

def main():
 result=read(HERE/'results.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');hist=history();gaps=[dict(item='legacy state coordinates',reason='DB state rows without recoverable physical triples cannot be proven coordinate-disjoint',condition='recover original immutable physical coordinates'),dict(item='initial import-source audit timing',reason='Additional PoC source/ and validation_stage2/ hashes were supplemented after gate and main launch; initial poc/*.py and all Basin Python snapshots exist',condition='cannot retrospectively create a pre-launch receipt; disclose limitation')]
 if not audit['passed']:gaps.append(dict(item='source integrity comparison',reason=str(audit['changed_sections']),condition='inspect recorded diffs/identities; no automatic restoration'))
 gaps.append(dict(item='strict State-axis seen-controller premise',reason='Exact checkpoint/parameter overlap0/14 with frozen A3 training56; only14 recipe families overlap',condition='To isolate State-only axis, evaluate exact critic-training checkpoints on held-out states in a separately authorized/preregistered study; no extra rollout or model change in this fixed14 prompt'))
 result['scope']={'strict_state_axis':'GAP: exact controllers unseen','seen_recipe_count':14,'seen_exact_checkpoint_count':0,'confirmatory_numerical_scope':'Original14 checkpoints and prespecified new96 states; combined controller/state transfer'}
 write(HERE/'GAPS.json',gaps);result['integrity']=audit;result['integrity_timing_qualification']=True;result['gaps']=gaps;result['status']='NUMERICAL_COMPLETE_WITH_AUDIT_QUALIFICATION';write(HERE/'results.json',result)
 columns=['subitem','evidence']+METHODS+['note'];rows=[]
 def cell(x):
  a,b=x['success_pct'];ci=x['success_95CI_pct'];return f'{a:.2f}–{b:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]'
 for cohort,label in [('1a','1a same-distribution new states; seen recipes/unseen checkpoints'),('1b','1b shifted states; seen recipes/unseen checkpoints')]:
  rows.append(dict(subitem=label,evidence='CONFIRMATORY specified cohort; State-only isolation GAP',**{m:cell(result[cohort]['methods'][m]) for m in METHODS},note='cross-fitted oracle; unknown intervals retained'))
 rows.append(dict(subitem='1c state selection',evidence='CONFIRMATORY specified cohort; State-only isolation GAP',**{m:'see 1a/1b rows' if m in ['A3 zero-shot','A3 per state','A0','A0 per state','state oracle'] else 'N/A: comparison endpoint' for m in METHODS},note=json.dumps(result['ST3_pooled'],ensure_ascii=False)))
 rows.append(dict(subitem='historical T2 R26 aligned',evidence='DEVELOPMENT',**{m:cell(hist[m]) if m in hist else 'N/A: not in extracted historical panel' for m in METHODS},note='Existing wider R10 menu and32-state evaluation split; differs from current four-arm protocol'))
 rows.append(dict(subitem='historical T2 R16',evidence='DEVELOPMENT',**{m:'N/A: prefer aligned R26 panel' for m in METHODS},note='Legacy reported A3 regret0.75pp; obsolete state-input qualification'))
 rows.append(dict(subitem='historical R2 per-state selection',evidence='DEVELOPMENT',**{m:'N/A: different single-controller/menu/Q512 design' for m in METHODS},note='Per-state gain+5.21pp [2.65,7.88] relative best fixed; see source REPORT'))
 with (HERE/'AXIS_TABLE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
 md='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
 for row in rows:md+='| '+' | '.join(str(row.get(k,'')).replace('|','/') for k in columns)+' |\n'
 md+='\nSuccess % is lower–upper unknown envelope [95% paired nested bootstrap CI]. All raw unknowns are retained.\n\n'
 md+='| cohort | method | collision% | timeout% | unknown% | mean regret pp | max regret pp | catastrophes certain–possible | harm certain–possible |\n|---|---|---|---|---|---|---|---|---|\n'
 for cohort in ['1a','1b']:
  for method,x in result[cohort]['methods'].items():
   e=x['events_pct'];md+=f"|{cohort}|{method}|{e.get('collision')}|{e.get('timeout')}|{e.get('unknown')}|{x['mean_regret_pp']}|{x['max_regret_pp']}|{x['catastrophes_certain']}–{x['catastrophes_possible']}|{x['harm_certain']}–{x['harm_possible']}|\n"
 (HERE/'AXIS_TABLE.md').write_text(md)
 first=f"完整性比对{'通过' if audit['passed'] else '未通过'}；额外导入源码的首轮哈希补录晚于启动，完整性审计存在时间收据局限。ST1：{result['1a']['endpoint']['status']}，A3交叉拟合遗憾{result['1a']['endpoint']['A3_vs_crossfit_oracle_regret_pp']}pp，A3−安全基线95%CI {result['1a']['endpoint']['A3_minus_safety_CI_pp']}pp。ST2：{result['1b']['endpoint']['status']}，对应数字{result['1b']['endpoint']['A3_vs_crossfit_oracle_regret_pp']}pp和{result['1b']['endpoint']['A3_minus_safety_CI_pp']}pp。ST3（合并）：{result['ST3_pooled']['status']}，95%CI {result['ST3_pooled']['CI_pp']}pp。\n"
 first=first.rstrip()+' 设计前提审计：14个测试checkpoint与A3训练56个checkpoint/参数重合均为0，仅14个配方已见；因此下列ST数值结论针对指定cohort，纯State轴尚不能判定。\n'
 text=first+'\n本轮评估指定14个原Ring Flow在新state上的表现：FF、H700、seed0–1、四臂修正菜单，以及nominal only和安全基线；无训练、无新探测。16/16历史复现通过，首动作、结局、步数及完整位置轨迹逐字节一致。\n\n'
 text+='新state按预注册生成并扫描全部DB的UID/content/可恢复物理坐标；无法恢复的旧坐标仍是缺口。所有A3/A0/G_E选择先于新评估标签冻结。共16128条记录经官方journal与merger流程；unknown保留，postflight将unknown标成missing时不重跑。见main_POSTFLIGHT.json、DB_COMPLETENESS.json。\n\n'
 text+='按controller选取的oracle以state半划分交叉拟合；逐state oracle以seed半划分。同时报告样本内上界。置信区间按controller再按state重抽5000次，种子保留在簇内。unknown的不确定性使用保守上下界，不称作失败。\n\n'
 text+='逐state选择的分组结果、nominal→安全基线逐条转移、偏移分层、成功率变化，以及所有方法的碰撞/超时/unknown/遗憾/灾难/伤害见results.json和AXIS_TABLE.md。timeout仅为timeout；无deadlock结论。\n\n'
 text+='历史证据仅标DEVELOPMENT：R2收益5.21pp [2.65,7.88]；R16旧T2报告遗憾0.75pp但有旧state输入局限；R26对齐T2的实际均值见HISTORICAL_DEVELOPMENT.json。历史与本轮菜单、评价state划分不相同。专家在本轮无逐state评估标签，N/A；不为参照额外生成rollout。\n\n'
 text+='源码保护与收据：INTEGRITY_BEGIN/END/COMPARISON、COPIED_SOURCES和各.diff保留。PoC source补录时点明确披露于POC_SOURCE_SUPPLEMENT.json；不能回补为运行前凭证。没有修改已有源码、环境、物理、安全、事件或成功标准，没有git写操作。外部并发修改若存在，以审计具体差异为准。\n\n'
 text+='round33–36尚未开始；按用户要求一次仅执行一个md。\n'

 initial=read(HERE/'INTEGRITY_BEGIN.json');ending=read(HERE/'INTEGRITY_END.json');changed_paths={}
 for section in ['existing_source_sha256','poc_py']:
  a=initial[section];b=ending[section];changed_paths[section]={'added':[p for p in b if p not in a],'removed':[p for p in a if p not in b],'modified':[p for p in a if p in b and a[p]!=b[p]]}
 write(HERE/'EXACT_CHANGED_PATHS.json',changed_paths)
 text+='\n完整性比对发现的目录外变化（不回滚，不修改这些文件）：\n'
 for section,changes in changed_paths.items():
  for kind,paths in changes.items():
   for path in paths:text+='- '+kind+': `'+path+'`\n'
 text+='\n以上为审计检测到的目录外变化；本轮没有执行修改它们的操作。完整哈希与git状态依据保留于INTEGRITY_BEGIN/END及EXACT_CHANGED_PATHS.json。\n'
 text+='\n资源调度补充：用户要求总计使用18个CPU；原6个worker仅短暂暂停以移交未开始的case，保留当前episode和全部已完成证据。15416个未执行case经追加preflight转入互斥共享队列，数值episode函数逐字不变。raw目录内预留标记不是证据，最终已逐条校验全部16128条真实记录且预留标记清零。动态并发维持其他用户作业计入后的总18CPU目标；收据见SPEEDUP_OWNERSHIP、SPEEDUP_PREFLIGHT及SPEEDUP_IMPLEMENTATION_AUDIT。\n'

 contrast=result['state_shift_contrasts']['A3 zero-shot']
 text+='\n从1a到1b，A3成功率变化区间为'+str(contrast['success_change_pp'])+'pp，平均遗憾变化区间为'+str(contrast['regret_change_pp'])+'pp（描述性unknown包络）。\n'
 text+='\n按预注册偏移参数分层的结果：\n'
 for key,value in result['shift_strata'].items():text+='- '+key+'：n='+str(value['states'])+'，A3成功率'+str(value['A3_success_pct'])+'%，交叉拟合遗憾'+str(value['A3_crossfit_regret_pp'])+'pp。\n'
 text+='\n逐controller/方法的成功率与遗憾见CONTROLLER_DETAILS.json；nominal碰撞后的转移比例已列于results.json的fractions字段。\n'
 if result['1a']['endpoint']['status']=='SUPPORTED':text+='\nST1数值标准通过：指定原14 checkpoint在同分布新state上的标准成立；这些精确checkpoint未在critic训练集中，不能映射为已见controller上的纯State泛化成立。\n'
 if result['1b']['endpoint']['status']=='SUPPORTED':text+='\nST2数值标准通过：本轮指定state偏移下的标准成立；含精确checkpoint未见这一共同条件，不能单独归因为state偏移稳健性。\n'
 else:text+='\nST2未获支持：仅按以上已预注册分层描述失效或不确定性，不能据此调整生成器或阈值。\n'
 label={'gain':'有增益','harm':'有损','undetected':'未检出差别'}
 text+='\nST3合并结论：'+label[result['ST3_pooled']['status']]+'；逐组oracle潜在增益与置信区间见results.json。该潜在增益来自交叉拟合，不能把样本内oracle当作可实现收益。\n'
 text+='\nController身份前提核查（在主结果打开前完成）：R16 source_data读取R15的56个新seed controller，R15 CHECKPOINT_NOVELTY_AUDIT及protocol表明测试原14与训练56的checkpoint/参数哈希均无重合，只有配方相同。证据与收据见CONTROLLER_MEMBERSHIP_AUDIT.json及.receipt.json。此前prompt和本轮初始预注册中的已见controller表述错误，保留原文而不追改；ST1/ST2/ST3按原规则计算，不替换模型或controller、不追加rollout。历史T2亦适用该范围局限。纯State轴属于设计缺口。\n'
 (HERE/'REPORT.md').write_text(text)
 (HERE/'HANDOFF.md').write_text('Round32 numerical pipeline complete. Read REPORT.md first.\n\nDirectory: '+str(HERE)+'\nPython: /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python\nAll16128 raw records and traces preserved. LABELS.npz/results.json/AXIS_TABLE.{md,csv} contain finalized evidence. Official DB registration and merged journal receipts are retained. Never rerun unknown or cached identities. Source integrity timing qualification is documented; do not erase it.\n\nThe next requested task, only after reviewing round32 completion and audit, is to read docs/prompt_round33_axis2_controller_20261015.md in full and execute it in its own new directory. Rounds34–36 follow sequentially; none has been started by this pipeline.\n')
 seal('NUMERICAL_COMPLETE.receipt.json',[HERE/'results.json',HERE/'AXIS_TABLE.md',HERE/'AXIS_TABLE.csv',HERE/'REPORT.md',HERE/'HANDOFF.md'],status=result['status'])
 print(result['status'])
if __name__=='__main__':main()
