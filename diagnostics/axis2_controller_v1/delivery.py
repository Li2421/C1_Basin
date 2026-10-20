from common import *
import csv,subprocess

def textfile(name,text):
 p=HERE/name;assert p.resolve().is_relative_to(HERE);p.write_text(text)
def cell(v):
 if not v:return 'N/A（见缺口）'
 a,b=v['success_pct'];ci=v['success_95CI_pct'];point=f'{a:.2f}' if abs(a-b)<1e-10 else f'{a:.2f}–{b:.2f}';return f'{point} [{ci[0]:.2f}, {ci[1]:.2f}]'
def main():
 # Historical-only descriptive tables do not affect frozen primary endpoints.
 with (HERE/'HISTORICAL_SUPPLEMENT.log').open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/'historical_supplement.py'),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (HERE/'DEPLOYMENT_METRICS.log').open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/'deployment_metrics.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 results=read(HERE/'results.json');hist=read(HERE/'HISTORICAL_RESULTS.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');start=read(HERE/'INTEGRITY_BEGIN.json');end=read(HERE/'INTEGRITY_END.json');details={}
 for section in audit['changed_sections']:
  a=start[section];b=end[section]
  if isinstance(a,dict) and isinstance(b,dict):details[section]={k:dict(before=a.get(k),after=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
  else:details[section]=dict(before=a,after=b)
 write(HERE/'INTEGRITY_DETAILS.json',details)
 names=[('2a','2a 新 seed'),('2b','2b 新 checkpoint'),('2cN2','2c 新架构 R17 N2'),('2cTEST','2c 新架构 R18 TEST'),('M','2d MPC 历史'),('P','2d 势场 历史'),('G','2d 高斯BC'),('DM','2d 分散MPC'),('VO','2d VO'),('PT','2d 切向势场'),('2e_central','2e 集中TT'),('2e_shared','2e D-shared'),('2e_local','2e D-local'),('2e_student','2e R30学生')]
 standard=['nominal only','safety baseline','F10','S500','A3 zero-shot','A0','G_E','best fixed oracle','full menu oracle','expert','original frozen method'];table=[];metrics=[];gaps=[]
 for key,label in names:
  if key in hist['deployment']:r=hist['deployment'][key]
  elif key in hist['cohorts']:r=hist['cohorts'][key]
  else:r=results[key]
  methods=dict(r['methods'])
  if key in ['2a','2b','2cN2','2cTEST']:
   # Keep original method explicitly as well as added R26 frozen-model summaries.
   methods.update(hist['cohorts'][key]['methods']);methods['original frozen method']=results[key]['methods']['original frozen method']
  if 'A3 full' in methods:methods['A3 zero-shot']=methods['A3 full']
  row={'subaxis':label,'evidence':r['evidence']}
  for method in standard:
   if method=='expert':row[method]='100.00 [100.00, 100.00]（特权）' if key.startswith('2e_') and key!='2e_student' else 'N/A（无匹配专家标签）'
   else:row[method]=cell(methods.get(method))
   if method not in methods and method not in ['expert','original frozen method']:gaps.append(dict(subaxis=label,method=method,reason=r.get('gaps','No frozen compatible method/labels in reused source menu'),required='A separately preregistered compatible model/probe or missing baseline; no unplanned rollout in this round'))
  table.append(row)
  for method,v in methods.items():metrics.append(dict(subaxis=label,method=method,evidence=r['evidence'],**v))
 write(HERE/'METHOD_METRICS.json',metrics)
 with (HERE/'METHOD_METRICS.csv').open('w',newline='') as f:
  columns=['subaxis','method','evidence','success_pct','success_95CI_pct','mean_regret_pp','max_regret_pp','catastrophes_certain','catastrophes_possible','harm_certain','harm_possible','events_pct'];w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items() if k in columns} for r in metrics)
 columns=['subaxis']+standard+['evidence']
 with (HERE/'AXIS_TABLE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(table)
 md='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in columns)+' |' for r in table)
 textfile('AXIS_TABLE.md',md+'\n\n单位为 % [95% CI]；范围为 unknown 识别界。旧部署标签仅作 DEVELOPMENT；本轮补充重算交叉拟合与原样本内参照，不改原报告。原冻结方法的配对 CT4 见 results.json；R26 补充 A3/A0 为 DEVELOPMENT。M/P 行只用旧标签重新计算交叉拟合。细项见 METHOD_METRICS.json/csv；缺项见 GAPS.json。\n')
 gaps.extend([dict(subaxis='all',method='expert',reason='No privileged expert labels on the new32 or historical subset states; R28 32/32 applies only to its own states',required='Separately preregistered privileged planning if wanted'),dict(subaxis='2e central/shared',method='harm/transitions',reason='No matched safety/nominal baseline labels for these two projection pipelines; old four-arm labels permit success, regret and catastrophe recomputation only',required='Separately preregister baseline labels if needed'),dict(subaxis='source scope',reason='R27 seed-overlap prereg amendment, R29 private DB, R28 timing deviation, R30 out-of-scope journals/chart mismatch/6step0 students retained',required='Interpret existing results with original qualifications')])
 write(HERE/'GAPS.json',gaps)
 primary=results['2d_all'];ct1=primary['CT1'];ct2=primary['CT2'];ct3=primary['CT3'];ct4='；'.join(f"{k} 原冻结方法−safety CI={v['paired']['original_minus_safety']['CI_pp']}pp" for k,v in results.items() if k in ['2a','2b','2cN2','2cTEST'])
 first=f"完整性审计{'通过' if audit['passed'] else '未完全通过（详见 INTEGRITY_DETAILS.json）'}。CT1 {ct1['status']}：A3 平均交叉拟合遗憾 {ct1['regret_pp']}pp，A3−safety 95%CI {ct1['A3_minus_safety_CI_pp']}pp，伤害确定/可能 {ct1['harm_certain']}/{ct1['harm_possible']}。CT2 {ct2['status']}：固定偏好双向各 {ct2['prefers_F10_by_10pp']}/{ct2['prefers_F13_by_10pp']} 个，遗憾差 CI {ct2['regret_difference_CI_pp']}pp。CT3 {ct3['status']}：混合−A3 遗憾差 CI {ct3['regret_difference_CI_pp']}pp。2a 新seed、2b兄弟checkpoint、2c新架构均补齐配对基线，{ct4}；2d 四类各6实例全部通过能力门，是否支持迁移以各CT门槛为准；2e仅整理历史三管线和R30学生，其流程与蒸馏限制保留。"
 body='\n\n本轮新增 baseline6784、pilot768、main12288（全部24实例保留）；历史复现80，R5无标签probe768。训练6个Gaussian BC与混合类型critic，后者选step500；全部在新类型pilot之前冻结。probe后阶段曾因分析实现尚未封存而按保护检查退出；封存完成后只提交main一次，没有重跑探测或冻结选择，原错误记录和PIPELINE_RESUME.json保留。方法选择与分析实现冻结均早于main提交。本轮提交18 CPU分片，最多2 GPU训练分片，无本任务跨轮并行。其他r30b任务中途增加占用，监控响应前曾短时合计19；初版只暂停1分片的局限随后修正为按总占用动态暂停相应数量的本轮分片。后处理也按空位暂停/恢复整个进程组；所有在途轨迹和统计状态保留，见RESOURCE_REBALANCE.json与STAGE_RESOURCE_REBALANCE.json。不得把这些记录说成全程无资源偏差。\n\n成功、碰撞、timeout、unknown按原仿真语义；unknown上下界保留且不重跑，timeout不称deadlock。所有CI5000次controller→state簇配对bootstrap，双seed保持配对；oracle每次重抽按原state索引奇偶交叉拟合，同时给出样本内上界。CT2/3未通过显著性门槛并不证明效应为零。oracle事件率按已确认成功下界训练后的确定性首选臂计算；unknown引起的可选臂不确定性另体现在成功率和遗憾包络。\n\nCT4使用历史原冻结选择，按原state_uid排序前32、seed0/1与新基线配对。补充A3/A0/GE只读R26冻结预测，沿用原预测state全集聚合并限制到有标签菜单，标DEVELOPMENT；不得把额外事后整理升级为预注册终点。旧六臂的F4等无critic head时不做映射臆造。AXIS_TABLE增加original frozen method列展示CT4的原规则；额外A3/A0/GE列不代替CT4原规则。\n\nDB使用官方登记→提交前preflight→journal→单进程merger→postflight；原含timestep的state行保留，规范三元组使用新的transport身份。原uid、物理、key_for随机流保持不变，80条全轨迹复现证明兼容；不能把这种表示规范化当作旧标签无效的理由。源DB无法恢复的历史物理状态去重范围见 NEW_STATES_DEDUP.json。R5只进入SUMMARY_ONLY来源，不生成success/terminal标签。\n\n混合训练数据只有旧R15 Flow56与R27 M/P12，不含任何新类型pilot或eval标签。MIXED_LOTO_RESULTS.json为描述性留一类型遗憾，预处理/PCA和step1000有旧Flow先验，不是独立前瞻确认。混合模型相对A3同时改变训练类型覆盖与FF/TT输入标志，CT3只支持这一整套预注册配置的效果，不单独识别标志或数据多样性的因果贡献。\n\n历史M/P继续保留R27状态区间修订、R29私有DB及single-projection baseline差异。部署行保留R28 preflight提交时序偏差，R30的448PoC journal越界、chart信息差与6个step0学生。R28特权专家32/32只对应其原32states，不能转作本轮新states的专家成功率。\n\n主交付：AXIS_TABLE.md/csv、METHOD_METRICS.json/csv、results.json、HISTORICAL_RESULTS.json、MIXED_LOTO_RESULTS.json、GAPS.json。完整性差异逐项见 INTEGRITY_DETAILS.json。历史库存、复现、登记/预检/后检、原始轨迹、训练、能力门和冻结收据均在本目录；独立机械核验见 INDEPENDENT_DELIVERY_REVIEW.json。未更改控制器参数、物理、事件和成功标准；无视频要求、无git写操作。\n'
 source_changes=[p for p in set(start['existing_source_sha256'])|set(end['existing_source_sha256']) if start['existing_source_sha256'].get(p)!=end['existing_source_sha256'].get(p)]
 if source_changes and all(p not in start['existing_source_sha256'] for p in source_changes):body+='\n源码清单差异全部是轮外新增Python文件；本轮开始时已有的Python内容hash无变化。严格的起止清单一致条件仍未通过，不把新增文件忽略后改判通过。\n'
 change_lines=[]
 for section in audit['changed_sections']:
  a=start[section];b=end[section]
  if section in ['Basin','MACFlow']:
   for field in ['HEAD','diff_sha256']:
    if a.get(field)!=b.get(field):change_lines.append(f'- {section}/{field}: {a.get(field)} → {b.get(field)}')
   for line in sorted(set(b['status'].splitlines())-set(a['status'].splitlines())):change_lines.append(f'- {section} status added: {line}')
   for line in sorted(set(a['status'].splitlines())-set(b['status'].splitlines())):change_lines.append(f'- {section} status removed: {line}')
  elif isinstance(a,dict) and isinstance(b,dict):
   for path in sorted(set(a)|set(b)):
    if a.get(path)!=b.get(path):change_lines.append(f'- {section}: {path}: {a.get(path)} → {b.get(path)}')
  else:change_lines.append(f'- {section}: {a} → {b}')
 textfile('REPORT.md',first+body+('\n完整性差异清单：\n\n'+'\n'.join(change_lines)+'\n' if change_lines else '\n起止完整性记录一致。\n'))
 textfile('HANDOFF.md',first+'\n\n工作目录 diagnostics/axis2_controller_v1。指定Python环境不变。所有raw及trace保留，不重跑unknown。SOURCE_INVENTORY.json映射旧标签与冻结选择，main_CASES.json映射新标签。results.json为预注册CT1–4；历史追加表为DEVELOPMENT。INDEPENDENT_DELIVERY_REVIEW.json通过且NUMERICAL_COMPLETE.receipt.json存在才算本轮数值完成。下一步完整阅读round34 prompt，顺序执行；不得提前并行下一轮。\n')
 write(HERE/'DELIVERY_STATUS.json',dict(utc=utc(),numerical_analysis_complete=True,independent_review_pending=True,audit_passed=audit['passed']))
if __name__=='__main__':main()
