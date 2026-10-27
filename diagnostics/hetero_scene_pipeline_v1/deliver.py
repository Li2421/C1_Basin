"""Chinese scientific report with exact frozen selections, uncertainty and audit handoff."""
from common import *
from design import *
from statistics_hetero import METHODS,NAMES
import numpy as np,difflib,subprocess

def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}个百分点'
def interval(v):return '['+', '.join(f'{100*x:+.2f}'for x in v)+']个百分点'
def candidate(r):return describe_eta(r.get('key',''),r.get('eta',r.get('eta_a')))
def chosen_eta(row,m):
 if m=='scene_xstate':return '随200次对半划分选择；最常选'+row['reference_most_frequent_eta']
 a=row['methods'][m]['selected_index'];return describe_eta(row['candidate_keys'][a],row['candidate_etas'][a])

def report():
 s=read(HERE/'STATISTICS.json');hc=read(HERE/'HETEROGENEITY_CHECK.json');t=read(HERE/'LABEL_TABLES.json');ev=read(HERE/'EVALUATION_STATES.json')['scenes'];d=s['decisions'];p=s['comparisons']['v2'];h=hc['groups']['all8'];strong=hc['groups']['strongest3'];names=s['scenes']
 references={};rng=np.random.default_rng(2026102652)
 counts=np.stack([np.asarray(t[n]['y'],dtype=np.int16)[:,[t[n]['state_indices'].index(i)for i in ev[n]]].sum(0,dtype=np.int16)for n in names])
 order=np.argsort(rng.random((len(names),200,16)),axis=-1,kind='stable')[:,:,:8];train=counts[np.arange(len(names))[:,None,None],order].sum(2,dtype=np.int16);selected=train.argmax(-1)
 for i,n in enumerate(names):
  frequency=np.bincount(selected[i],minlength=49);a=int(frequency.argmax());s['per_scene'][n]['reference_most_frequent_eta']=candidate(menu()[a]);references[n]=dict(candidate_indices_by_split=selected[i].tolist(),selected_candidates=[dict(candidate=menu()[j],count=int(frequency[j]))for j in range(49)if frequency[j]])
 write(HERE/'REFERENCE_SELECTIONS.json',references)
 yn=lambda x:'是'if x else'否'
 title='场景集在规定核对状态上成立'if d['scene_set_established']else'场景集未在新状态上成立，第二阶段结果解释受限'
 winning=d['learned_methods'];learn_details='；'.join(NAMES[m]+f"的D={s['scene_matching'][m]['D']:.6f}、p={s['scene_matching'][m]['p_one_sided']:.6g}"for m in ['v2_49','v2_49_no_fallback'])
 first=(f"{title}：全局缺口{pp(h['global_gap'])}，只重抽状态95%区间{interval(h['ci_states'])}，场景→状态区间{interval(h['ci_scene_states'])}。学到了：{yn(d['learned'])}（{learn_details}；通用输入版本D={s['scene_matching']['generic_input']['D']:.6f}）。有用：{yn(d['useful'])}；学习选η v2减一个全局η为{pp(p['difference_from_global'])}，只重抽状态95%区间{interval(p['ci_states'])}，场景→状态95%区间{interval(p['ci_scene_states'])}。不伤：{yn(d['not_harmful'])}，最差场景差值{pp(d['worst_scene_difference'])}。三条判定分开，均读未四舍五入数值。")
 lines=['# 异质场景 pipeline v1\n',first+'\n',
 '学习选η v2（本轮主方法，v2）为整个场景生成5个提案，与49个旧候选合并；critic以训练场景候选成功率为先验，加λ×3×tanh(raw/3)修正。λ在训练场景内部留一、仅种子17选择；外层critic和生成器分别从17、23、41中按源场景内部验证损失选种子。使用尺度无关场景描述，超范围按原v2规则退回训练场景的一个全局η。未训练原始描述或无先验消融。\n',
 '集中式随机采样模型预测控制（centralized_sampling0）、梯度下降模型预测控制（gradient_mpc0）、分布式模型预测控制（DM0）等权，状态等权，场景等权。候选η的三个数依次为朝目标、侧移、互斥，完整精确值见ETA_MENU.json。\n',
 '十字路口与方形场地核对及留出计分仅用状态8–23，对穿6/10机仅用状态12–27；三个文献场景仍用参与过场景挑选的已有16状态，不能称为全新的独立场景验证。训练源场景使用全部标注状态；没有留出场景标签进入对应折的拟合、标准化、先验或λ选择。\n',
 '## 第一阶段全局缺口\n',
 '| 组 | 场景数 | 每个场景自己的最好η | 一个全局η | 全局缺口 | 只重抽状态95%区间 | 场景→状态95%区间 |\n|---|---:|---:|---:|---:|---|---|']
 for g in [h,strong]:lines.append(f"| {'全部保留场景'if g is h else'预先指定最强3场景'} | {len(g['scenes'])} | {pct(g['own_best_crossvalidated_rate'])} | {pct(g['global_crossvalidated_rate'])} | {pp(g['global_gap'])} | {interval(g['ci_states'])} | {interval(g['ci_scene_states'])} |")
 lines.extend(['\n全局候选在每次划分的所有场景训练半上等权选择，在相同计分半上计分；每个场景自己的最好η也只用自己的训练半选择。200次随机对半划分，bootstrap每次重抽内部重新划分。\n','样本内全部核对状态上最好的全局候选为'+candidate(h['global_full_sample_candidate'])+'；其身份用于描述，交叉验证全局选择可能随划分变化，全部200次选择及每个场景损失见HETEROGENEITY_CHECK.json。\n','| 场景 | 自己的最好η计分率 | 全局η计分率 | 损失 |\n|---|---:|---:|---:|'])
 for n,r in h['per_scene'].items():lines.append(f"| {SCENE_NAMES[n]}（{n}） | {pct(r['own_best_rate'])} | {pct(r['global_rate'])} | {pp(r['loss'])} |")
 lines.extend(['\n## 留一场景的方法比较\n','| 方法 | 场景平均成功率 | unknown计成功上界 | 减一个全局η | 只重抽状态95%区间 | 场景→状态95%区间 |\n|---|---:|---:|---:|---|---|'])
 for m in METHODS:
  r=s['comparisons'][m];lines.append(f"| {NAMES[m]}（{m}） | {pct(r['rate'])} | {pct(r['unknown_success_upper'])} | {pp(r['difference_from_global'])} | {interval(r['ci_states'])} | {interval(r['ci_scene_states'])} |")
 methods=[]
 for n,r in s['per_scene'].items():
  methods.extend([f'## {SCENE_NAMES[n]}（{n}）\n','| 方法 | 选出的η（三个数） | 成功率 | 减一个全局η |\n|---|---|---:|---:|'])
  for m in METHODS:methods.append(f"| {NAMES[m]} | {chosen_eta(r,m)} | {pct(r['rates'][m])} | {pp(r['differences_from_global'][m])} |")
  methods.append('')
 (HERE/'PER_SCENE_METHODS.md').write_text('\n'.join(methods)+'\n');lines.extend(['\n'+x for x in methods])
 match=['# 场景对号检验\n','行是被计分的目标场景，列是为某个场景所选择的η；D为对角平均减去其余选择平均，全部场景重新分配精确单侧检验。三控制器×16状态的整数成功数用于精确处理排列同分。\n','| 方法 | D | 单侧p | 排列数 | 自己选择率 | 其他选择率 |\n|---|---:|---:|---:|---:|---:|']
 for m,r in s['scene_matching'].items():match.append(f"| {NAMES[m]} | {r['D']:.8f} | {r['p_one_sided']:.8g} | {r['permutations']} | {pct(r['diagonal_rate'])} | {pct(r['other_choice_rate'])} |")
 match.append('\n通用输入版本复用同一折冻结的critic、先验、λ和控制器编码，仅把c′换成源场景等权平均、h换成实际源训练状态均值；按替换后输入重新计算相同超范围规则，没有重新训练。\n')
 (HERE/'SCENE_MATCHING.md').write_text('\n'.join(match)+'\n');lines.extend(['\n'+x for x in match])
 transfer=['# 同类迁移\n','| 留出场景 | 同类源场景 | 主方法选出的η | 同类源场景前3名η | 是否前3名 | 含第3名并列 | 主方法成功率 | 减一个全局η |\n|---|---|---|---|---|---|---:|---:|']
 for n,r in s['same_type_transfer'].items():
  if not r['available']:transfer.append(f"| {SCENE_NAMES[n]} | {SCENE_NAMES[r['peer']]} | 同类不适用 | — | — | — | — | — |");continue
  transfer.append(f"| {SCENE_NAMES[n]} | {SCENE_NAMES[r['peer']]} | {candidate(r['selected_candidate'])} | {'；'.join(candidate(c)for c in r['peer_top3'])} | {yn(r['in_peer_canonical_top3'])} | {yn(r['in_peer_top3_including_boundary_ties'])} | {pct(r['v2_rate'])} | {pp(r['difference_from_global'])} |")
 (HERE/'SAME_TYPE_TRANSFER.md').write_text('\n'.join(transfer)+'\n');lines.extend(['\n'+x for x in transfer])
 cross=['# 交叉使用损失\n','为行场景的训练半所选η用到列场景计分半；数字为相对列场景自己的选择少的百分点。\n','| 来源→目标 | '+' | '.join(SCENE_NAMES[n]for n in names)+' |','|---|'+'---:|'*len(names)]
 for a in names:cross.append('| '+SCENE_NAMES[a]+' | '+' | '.join('—'if a==b else f"{100*h['cross_use_losses'][a][b]:+.2f}"for b in names)+' |')
 (HERE/'CROSS_USE_LOSSES.md').write_text('\n'.join(cross)+'\n');lines.extend(['\n'+x for x in cross])
 lines.extend(['\n## 每折训练选择与超范围退回\n','| 留出场景 | critic种子 | 生成器种子 | λ | 带宽 | 场景超范围维度（1起） | 超范围状态数 | 主方法选中生成器提案 |\n|---|---:|---:|---:|---:|---|---:|---|'])
 for n in names:
  folder=HERE/'folds'/('loso__'+n);pr=read(folder/'PRIORS.json');cr=read(folder/'full'/'SELECTED.json')['selected'];ge=read(folder/'generator'/'SELECTED.json')['selected'];r=s['per_scene'][n]
  lines.append(f"| {SCENE_NAMES[n]} | {cr['seed']} | {ge['seed']} | {r['shrinkage']} | {pr['bandwidth']} | {r['guard']['scene_trigger_dimensions']} | {sum(r['guard']['states_out_of_range'])} | {yn(n in s['generator_selected_scenes'])} |")
 rates=['# 49个固定候选的成功率\n'];bestrows=['# 每个场景自己的最好固定候选\n','| 场景 | 全部标注状态最好η | 全部状态成功率 | 核对状态最好η | 核对成功率 | 核对前3名 |\n|---|---|---:|---|---:|---|'];ctlrows=['# 按控制器的最好固定候选\n','| 场景 | 控制器 | 全部标注状态最好η | 全部成功率 | 核对状态最好η | 核对成功率 |\n|---|---|---|---:|---|---:|']
 for n in names:
  y=np.asarray(t[n]['y']);u=np.asarray(t[n]['unknown']);pos=[t[n]['state_indices'].index(i)for i in ev[n]];q=y.mean((0,1));qe=y[:,pos].mean((0,1));qu=(y+u)[:,pos].mean((0,1));a=int(q.argmax());b=int(qe.argmax());order=np.lexsort((np.arange(49),-qe))
  bestrows.append(f"| {SCENE_NAMES[n]} | {candidate(menu()[a])} | {pct(q[a])} | {candidate(menu()[b])} | {pct(qe[b])} | {'；'.join(candidate(menu()[int(j)])+' '+pct(qe[j])for j in order[:3])} |")
  rates.extend([f'## {SCENE_NAMES[n]}（{n}）\n','| 候选η（三个数） | 全部标注状态 | 核对状态 | 核对unknown计成功上界 |\n|---|---:|---:|---:|'])
  for j,c in enumerate(menu()):rates.append(f'| {candidate(c)} | {pct(q[j])} | {pct(qe[j])} | {pct(qu[j])} |')
  for ci,ctl in enumerate(CONTROL_IDS):
   cq=y[ci].mean(0);ce=y[ci,pos].mean(0);a=int(cq.argmax());b=int(ce.argmax());ctlrows.append(f'| {SCENE_NAMES[n]} | {CONTROL_NAMES[ctl]}（{ctl}） | {candidate(menu()[a])} | {pct(cq[a])} | {candidate(menu()[b])} | {pct(ce[b])} |')
 (HERE/'CANDIDATE_RATES.md').write_text('\n'.join(rates)+'\n');(HERE/'BEST_CANDIDATES.md').write_text('\n'.join(bestrows)+'\n');(HERE/'PER_CONTROLLER_BEST.md').write_text('\n'.join(ctlrows)+'\n')
 lines.extend(['\n'+x for x in bestrows]);lines.extend(['\n'+x for x in ctlrows]);lines.append('\n各场景49候选成功率见CANDIDATE_RATES.md。能力门与五层证据见CAPABILITY_TABLE.json和GEOMETRY_REPORTS.json；联合可解与死锁解决保持未认证，timeout不当deadlock。探索0–7取值重建差异仅报告、不停止，见EXPLORATION_RECONSTRUCTION_CHECK.json。\n')
 lines.append('区间为5000次配对自助法，固定场景时仅重抽状态，另一口径按场景→状态分层重抽；方法共用抽样索引。只支持这8个预选场景、三个控制器与种子0，不把场景挑选后的结果当作未选择场景总体上的无偏验证。\n')
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');return first

def update_copy_provenance():
 receipt=read(HERE/'TOOLING_COPY.receipt.json')
 for r in receipt['files']:
  assert sha(r['source'])==r['source_sha256'];r['target_sha256']=sha(r['target']);r['changed']=r['source_sha256']!=r['target_sha256']
  if r['changed']:
   before=Path(r['source']).read_text();after=Path(r['target']).read_text();p=Path(r['target']);p.with_suffix(p.suffix+'.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=r['source'],tofile=r['target'])))
 write(HERE/'TOOLING_COPY.receipt.json',receipt)

def main():
 from audit import database,integrity
 from geometry_final import main as geometry
 geometry();first=report();update_copy_provenance();database()
 cap=read(HERE/'CAPABILITY_TABLE.json');excluded={n:v for n,v in cap['scene_status'].items()if not v['test_applicable']}
 (HERE/'GAPS.md').write_text('# 证据缺口与范围\n\n'+('没有能力门剔除场景、未完成请求或基础设施失败。\n'if not excluded else'能力门不适用场景：'+canonical(excluded)+'\n')+'\n文献3场景已有16状态参与过场景选择；三个新场景0–7属于探索。独立新状态只包括三个新场景8–23、两个对穿12–27；报告明确分开。场景集经过探索选择，不能外推至未经选择的场景总体。联合可解与死锁解决未认证。\n')
 integrity();db=read(HERE/'DB_POSTFLIGHT.json');assert db['passed']and db['all_requests_present'];assert not read(HERE/'INTEGRITY_COMPARE.json')['changed_inputs']
 cost=read(HERE/'ACTUAL_COST.json');new=cost['new_execution_count'];assert new<=20000
 with(HERE/'REPORT.md').open('a')as f:f.write(f"\n新仿真共{new}条；单CPU局内墙钟合计{cost['single_cpu_episode_wall_hours_sum']:.4f}小时（不含导入、压缩、调度与训练）；历史能力门与标签未重跑。所有身份、原始记录哈希、轨迹哈希、冻结文件和旧源码审计通过。\n")
 (HERE/'STATUS.md').write_text('# 当前状态\n\n'+utc()+'：COMPLETE。两阶段、全部折冻结、提案、统计、数据库与源保护审计完成。\n\n'+first+'\n')
 reports=['REPORT.md','HISTORY_CHECK.md','SCENES.json','GEOMETRY_REPORTS.json','CAPABILITY_TABLE.json','HETEROGENEITY_CHECK.json','STATISTICS.json','TEST_DECISION.json','STATUS.md','GAPS.md','ITERATION_LOG.md','DB_POSTFLIGHT.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','ALL_FOLDS_FROZEN.receipt.json','TRAINING_PLAN.json','METHOD_PARITY_CHECK.json','CANDIDATE_RATES.md','REFERENCE_SELECTIONS.json','PER_SCENE_METHODS.md','SCENE_MATCHING.md','SAME_TYPE_TRANSFER.md','CROSS_USE_LOSSES.md','PER_CONTROLLER_BEST.md','TOOLING_COPY.receipt.json']
 write(HERE/'DELIVERY_CODE_SHA256.json',{str(p):sha(p)for p in HERE.glob('*.py')});reports.append('DELIVERY_CODE_SHA256.json')
 complete=dict(utc=utc(),status='COMPLETE',decisions=read(HERE/'STATISTICS.json')['decisions'],new_execution_count=new,all_audits_passed=True,reports={n:sha(HERE/n)for n in reports})
 write(HERE/'COMPLETE.json',complete);write(HERE/'RUN_STATE.json',dict(status='COMPLETE',active_jobs=[],report=str(HERE/'REPORT.md')));print(first,flush=True)
if __name__=='__main__':main()
