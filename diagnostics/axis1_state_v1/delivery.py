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
 write(HERE/'GAPS.json',gaps);result['integrity']=audit;result['integrity_timing_qualification']=True;result['gaps']=gaps;result['status']='NUMERICAL_COMPLETE_WITH_AUDIT_QUALIFICATION';write(HERE/'results.json',result)
 columns=['subitem','evidence']+METHODS+['note'];rows=[]
 def cell(x):
  a,b=x['success_pct'];ci=x['success_95CI_pct'];return f'{a:.2f}–{b:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]'
 for cohort,label in [('1a','1a same-distribution new states'),('1b','1b shifted states')]:
  rows.append(dict(subitem=label,evidence='CONFIRMATORY (audit timing qualified)',**{m:cell(result[cohort]['methods'][m]) for m in METHODS},note='cross-fitted oracle; unknown intervals retained'))
 rows.append(dict(subitem='1c state selection',evidence='CONFIRMATORY (audit timing qualified)',**{m:'see 1a/1b rows' if m in ['A3 zero-shot','A3 per state','A0','A0 per state','state oracle'] else 'N/A: comparison endpoint' for m in METHODS},note=json.dumps(result['ST3_pooled'],ensure_ascii=False)))
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
 text=first+'\n本轮仅评估见过的14个Ring Flow在新state上的泛化：FF、H700、seed0–1、四臂修正菜单，以及nominal only和安全基线；无训练、无新探测。16/16历史复现通过，首动作、结局、步数及完整位置轨迹逐字节一致。\n\n'
 text+='新state按预注册生成并扫描全部DB的UID/content/可恢复物理坐标；无法恢复的旧坐标仍是缺口。所有A3/A0/G_E选择先于新评估标签冻结。共16128条记录经官方journal与merger流程；unknown保留，postflight将unknown标成missing时不重跑。见main_POSTFLIGHT.json、DB_COMPLETENESS.json。\n\n'
 text+='按controller选取的oracle以state半划分交叉拟合；逐state oracle以seed半划分。同时报告样本内上界。置信区间按controller再按state重抽5000次，种子保留在簇内。unknown的不确定性使用保守上下界，不称作失败。\n\n'
 text+='逐state选择的分组结果、nominal→安全基线逐条转移、偏移分层、成功率变化，以及所有方法的碰撞/超时/unknown/遗憾/灾难/伤害见results.json和AXIS_TABLE.md。timeout仅为timeout；无deadlock结论。\n\n'
 text+='历史证据仅标DEVELOPMENT：R2收益5.21pp [2.65,7.88]；R16旧T2报告遗憾0.75pp但有旧state输入局限；R26对齐T2的实际均值见HISTORICAL_DEVELOPMENT.json。历史与本轮菜单、评价state划分不相同。专家在本轮无逐state评估标签，N/A；不为参照额外生成rollout。\n\n'
 text+='源码保护与收据：INTEGRITY_BEGIN/END/COMPARISON、COPIED_SOURCES和各.diff保留。PoC source补录时点明确披露于POC_SOURCE_SUPPLEMENT.json；不能回补为运行前凭证。没有修改已有源码、环境、物理、安全、事件或成功标准，没有git写操作。外部并发修改若存在，以审计具体差异为准。\n\n'
 text+='round33–36尚未开始；按用户要求一次仅执行一个md。\n'
 (HERE/'REPORT.md').write_text(text)
 (HERE/'HANDOFF.md').write_text('Round32 numerical pipeline complete. Read REPORT.md first.\n\nDirectory: '+str(HERE)+'\nPython: /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python\nAll16128 raw records and traces preserved. LABELS.npz/results.json/AXIS_TABLE.{md,csv} contain finalized evidence. Official DB registration and merged journal receipts are retained. Never rerun unknown or cached identities. Source integrity timing qualification is documented; do not erase it.\n\nThe next requested task, only after reviewing round32 completion and audit, is to read docs/prompt_round33_axis2_controller_20261015.md in full and execute it in its own new directory. Rounds34–36 follow sequentially; none has been started by this pipeline.\n')
 seal('NUMERICAL_COMPLETE.receipt.json',[HERE/'results.json',HERE/'AXIS_TABLE.md',HERE/'AXIS_TABLE.csv',HERE/'REPORT.md',HERE/'HANDOFF.md'],status=result['status'])
 print(result['status'])
if __name__=='__main__':main()
