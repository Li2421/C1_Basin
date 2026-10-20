from common import *
import csv
METHODS=[('nominal only','只执行控制器动作'),('safety baseline','只加安全投影、不加修正'),('F10','全程固定用η10的修正'),('S500 class','按时间切换的时间表'),('A3 zero-shot','使用控制器特征的冻结选择模型'),('A0','不使用控制器特征的冻结选择模型'),('G_E','探测、估计有效响应并选择的规则'),('best fixed oracle','交叉拟合最佳固定修正'),('full menu oracle','交叉拟合完整菜单最佳动作'),('expert','特权专家参照'),('continuing right-hand','持续右手规则'),('source per-controller transfer','源机器人数量冻结的同控制器选择'),('Sprog','按进度切换时间表')]
ALIASES={'S500 class':['S500 class','S500','Srel'],'A3 zero-shot':['A3 zero-shot','A3','A3 full'],'A0':['A0','A0_full'],'G_E':['G_E']}
DNAMES={'D1':'持续右手规则：同分布新状态','D2':'持续右手规则：分布偏移状态','D3_H500':'持续右手规则：500步时间预算','D3_H1000':'持续右手规则：1000步时间预算','D5':'持续右手规则：新类型黑盒合并','D5_Gaussian':'持续右手规则：高斯行为克隆策略','D5_distributed':'持续右手规则：分散式模型预测控制'}
def dname(k):return DNAMES.get(k,'持续右手规则：障碍半径'+k.split('r')[-1])
def fmt(v):return '–'.join(f'{x:.2f}'for x in v)
def rate(q):return f"{fmt(q['success_pct'])}% [{fmt(q['success_95CI_pct'])}]"
def verdict(q,eligible=True):
 if not eligible:return '未通过该半径的能力标准，差异仅作描述'
 return {'SCHEDULE_BETTER':'时间表更好','RIGHT_HAND_BETTER':'持续右手规则更好','UNDETECTED':'未检出差别'}[q['status']]+f"（配对差95%置信区间 {fmt(q['CI_pp'])} 个百分点）"
def main():
 verification=read(HERE/'FINAL_VERIFICATION.json');A=read(HERE/'A_RESULTS.json');known=read(HERE/'A_FAILURE_DENOMINATOR_CLARIFICATION.json');A['unknown_denominator_clarification']=known;B=read(HERE/'B_RESULTS.json');C=read(HERE/'N2_GATE_AUDIT.json');DD=read(HERE/'D_RESULTS.json');E=read(HERE/'E_RESULTS.json');E['per_instance']=read(HERE/'E_PER_INSTANCE_RESULTS.json');F=read(HERE/'BLACKBOX_SUBSET_RESULTS.json');AS=read(HERE/'A_STANDARD_RESULTS.json');gaps=[]
 def gap(part,reason,condition):gaps.append(dict(part=part,reason=reason,condition_to_complete=condition))
 gap('历史数据库去重可见性','已对全部具有位置记录的旧状态排除调参状态的位置重复；3200条仅别名的旧记录没有位置，不能给出全库物理唯一性的绝对保证，按原场景逐项列出。','补全这些历史身份的物理状态后可增强去重证明；本轮不补造缺失物理字段、不按结果替换已冻结调参状态。')
 gap('双机器人流匹配策略只读审计','既有能力门没有只执行控制器动作的逐条终止标签；保持只读，没有补跑。','另行预注册补充该臂；现有缺失不阻止执行正确与原能力门结论。')
 gap('几何标准方法','阳性对照和旧几何门没有只加安全投影的标签；同状态换半径没有全程固定η13的标签，完整菜单最佳值只覆盖实际可用菜单。','如需要这些额外标准方法，另行设计补充对应标签；本轮没有把受限菜单写成完整原菜单。')
 for r,v in A['same_state_radii'].items():
  if v['median_max_success_pct'][0]<20:gap('障碍半径'+r,'同状态换半径的能力标准未通过；持续右手规则比较只作描述，条件性几何主评估未触发。','新的、提前冻结的几何泛化研究；不得用本轮评估标签改变门或状态。')
 for n,unit in E['units'].items():
  if unit.get('status')=='GAP':
   details=[]
   for instance,c in read(HERE/'MPC_CAPABILITY_GATE.json')[n].items():details.append(PARAM_DESCRIPTION(instance)+'：'+('三臂最大成功率界'+fmt([100*x for x in c['max_success_bounds']])+'%，未达到已知下界20%的资格标准' if 'max_success_bounds' in c else c.get('reason','缺口')))
   gap('机器人数量'+n,'；'.join(details),'需在新的预注册研究中取得同参数控制器在源数量和目标数量上的合格能力证据；本轮参数、时域和样本数保持冻结。')
  elif unit.get('clean_same_controller_transfer',{}).get('status')=='GAP':gap('机器人数量'+n+'迁移选择',unit['clean_same_controller_transfer']['reason'],'同一个固定参数实例须在源数量和目标数量均通过能力门。')
 if not any(int(n)!=(E['source_N'] or -1) and 'correction_status' in v.get('clean_same_controller_transfer',{}) for n,v in E['units'].items()):gap('跨机器人数量迁移','没有同时通过源数量与更大目标数量能力门的同一固定参数实例；源数量内交叉拟合不算跨数量证据。','须在新的预注册实验中先获得同实例源与目标能力，再检验冻结选择的迁移。')
 gap('机器人数量轴冻结选择模型','原环形交换场景冻结模型缺少经认证的可变机器人数量Gap输入接口；本轮未填充、未重新拟合。','另行预注册数量无关表示、探测及冻结模型验证。')
 gap('五层证据','无论导航成功与否，均没有建立正式死锁解除的第五层证据。','独立的死锁定义、识别与解除证据；不能把超时或停滞候选改称死锁。')
 gap('黑盒子集跨类型关系','固定η10与固定η13的双向偏好分歧前提不满足，跨类型关系终点不确定。','新的预注册控制器覆盖应具备足够双向偏好分歧；本轮保留不确定。')
 comp=verification['integrity'];external=comp['external_added_status'];added=comp['external_added_sources'];audit_text='已有源码、模型、导入来源和冻结收据的完整性审计通过' if verification['source_protection_passed'] and verification['receipt_hashes_passed'] else '完整性审计存在差异，详见逐项审计'
 if external or added:
  origins=sorted({str(Path(path).relative_to(ROOT).parts[1]) for path in added if Path(path).is_relative_to(D)});status_paths=[x[3:] for x in external];audit_text+='；外部差异来自并行学生实验目录'+('、'.join(origins) if origins else '（见状态清单）')+'，以及新增状态路径'+('、'.join(status_paths) if status_paths else '（无额外状态）')+f'，共新增{len(added)}个源码文件，逐路径哈希已保存'
 elif verification['source_protection_passed']:audit_text+='；工作树差异仅来自本轮目录，未改动已有文件'
 else:audit_text+='；差异来源为'+str(list(comp['other_changes']))+'及已有源码逐路径清单'+str(list(comp['changed_existing_sources']))
 delta=B['progress_minus_relative'];Dtext='；'.join(dname(k)+'：'+verdict(v['paired']['schedule_minus_right_hand'],v.get('inferential_endpoint_eligible',True))for k,v in DD.items()if k in ['D1','D2','D3_H500','D3_H1000','D5']);missing_or_failed=read(HERE/'MPC_FIVE_LAYER_EVIDENCE.json')['first_observed_failures'];first=next((x for x in missing_or_failed if x.get('layer')),None) or E['first_observed_failure']
 if first and first['layer']:etext=f"机器人数量轴首次观测到的失败为{first['N']}个机器人、控制器能力第四层，参数实例为"+str(PARAM_DESCRIPTION(first['instance']))
 elif first:etext=f"机器人数量轴首先在{first['N']}个机器人处出现证据缺口，不能把缺测称为能力失败"
 else:etext='机器人数量轴前三层与能力门未出现已观测失败；正式死锁解除第五层仍未建立'
 transfers=[]
 for n,v in E['units'].items():
  t=v.get('clean_same_controller_transfer',{})
  if 'correction_status'in t:
   transfers.append(n+'个机器人：修正'+t.get('correction_status','缺口')+('，这里只是源数量内交叉拟合参照，不构成跨数量迁移' if int(n)==E['source_N'] else '、迁移选择'+t.get('transfer_status','缺口')))
  else:transfers.append(n+'个机器人：修正和迁移选择缺口')
 if all(DD[k]['paired']['schedule_minus_right_hand']['status']=='SCHEDULE_BETTER'for k in ['D1','D2','D3_H500','D3_H1000','D5']):Dtext='时间表在同分布新状态、分布偏移状态、500步和1000步预算、新类型黑盒合并结果中均优于持续右手规则；同分布主比较的配对差95%置信区间为'+fmt(DD['D1']['paired']['schedule_minus_right_hand']['CI_pp'])+'个百分点'
 caps=read(HERE/'MPC_CAPABILITY_GATE.json')
 if all(q.get('status')=='GAP'for q in E['units'].values()):
  maxima=[max(v['max_success_bounds'][1]for v in caps[str(n)].values())*100 for n in [2,10,20]];etext+='；两实例在2、10、20个机器人上的最高三臂成功率分别为'+ '、'.join(f'{x:.2f}%'for x in maxima)+'，均未达到20%的能力门';transfers=['主评估中的修正效应与跨数量选择迁移均为缺口，正式死锁解除第五层未建立']
 # Convert machine verdicts in all report prose to complete wording.
 translate=lambda s:s.replace('SUPPORTED','满足预注册标准').replace('REJECTED','未满足预注册标准').replace('INCONCLUSIVE','不确定')
 p=(audit_text+'。几何阳性对照中位数为93.75%，原几何能力门成立；修正失败中的停滞候选占'+f"{known['stagnation_fraction_among_known_failures']:.4%}"+'。按进度切换时间表已由兼容历史实验执行，本轮独立核算后复用全部2688条，相对按比例切换的合并增益置信区间为'+fmt(delta['CI_pp'])+'个百分点。双机器人流匹配策略（Flow）审计结论是“门的执行正确，N=2 Flow 在相向任务上能力不足”。'+Dtext+'；但分散式模型预测控制子组中持续右手规则达到100%成功率，显著优于时间表，时间表减持续右手规则的95%置信区间为−29.69至−11.20个百分点。四个扰动半径的比较均只作描述。'+etext+'；'+translate('；'.join(transfers))+'。黑盒事后子集属于DEVELOPMENT（事后子集）证据：方法可用性满足标准，跨类型关系不确定，加入更多类型的训练优势未满足标准。')
 units=[]
 def add(label,q,grade=None,reason=''):
  units.append((label,q,'缺口（无主评估标签）' if q.get('status')=='GAP' else grade or q.get('evidence','描述性'),reason))
 for k,q in AS.items():add('几何：'+('生成器阳性对照'if k=='A1_positive'else '同状态半径'+k.split('r')[-1] if k.startswith('A2')else '原能力门半径'+k.split('r')[-1]),q,reason='仅实际可用菜单；缺失方法不推断。')
 cm={};rng=np.random.default_rng(202610163705);ss=rng.integers(16,size=(5000,16))
 for arm,key in [('safety_baseline','safety baseline'),('F10','F10'),('S13t2','S500 class')]:
  v=C['arms'][arm];success=np.isin(np.arange(16),v['success_state_indices']);boot=success[ss].mean(1)*100;point=v['success_rate']*100;cm[key]=dict(success_pct=[point,point],success_95CI_pct=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],events_pct=dict(collision=0,timeout=100-point,unknown=0))
 add('双机器人流匹配策略只读审计',dict(methods=cm),'历史门的描述性审计','只读，不重判门；缺失原始动作基线和其他菜单标签。')
 for h,q in B['horizons'].items():add('按进度切换：'+h+'步',q,'CONFIRMATORY（定义修订，继承历史证据）')
 for k,q in DD.items():add(dname(k),q,reason=q.get('unavailable',''))
 for n,q in E['units'].items():add('机器人数量：'+n,q,'CONFIRMATORY',q.get('reason','只覆盖通过能力门的实例：'+'、'.join(PARAM_DESCRIPTION(k)for k in q.get('classes',[]))+'；原环形场景冻结模型无可变数量瓶颈场景接口。'))
 for n,instances in read(HERE/'E_PER_INSTANCE_RESULTS.json').items():
  for instance,q in instances.items():add('机器人数量：'+n+'，'+PARAM_DESCRIPTION(instance),q,'CONFIRMATORY',q.get('reason','单个固定参数实例，仅有状态层面的证据。'))
 for key in ['黑盒子集','高斯行为克隆策略','分散式模型预测控制']:add('事后重算：'+key,F[key],'DEVELOPMENT（事后子集）')
 rows=[];metrics=[]
 for label,q,grade,reason in units:
  methods=q.get('methods',{});row={'子项':label}
  for key,title in METHODS:
   candidate=next((methods[k]for k in ALIASES.get(key,[key])if k in methods),None)
   if key=='expert' and label.startswith('机器人数量：'):
    n=int(label.split('：')[-1].split('，')[0]);layer=next(x for x in read(HERE/'MPC_FIVE_LAYER_EVIDENCE.json')['layers']if x['N']==n);candidate=layer['joint_reference']
   row[title]=rate(candidate)if candidate else '缺口'if q.get('status')=='GAP'else '不适用／无标签'
  primary=next((methods[k]for k in ['S500 class','S500','Srel','F10']if k in methods),{});events=primary.get('events_pct',{});row.update({'碰撞率（时间表）':events.get('collision','不适用'),'超时率（时间表）':events.get('timeout','不适用'),'未知率（时间表）':events.get('unknown','不适用'),'平均遗憾（时间表）':str(primary.get('mean_regret_pp','不适用')),'最大遗憾（时间表）':str(primary.get('max_regret_pp','不适用')),'灾难数（时间表）':str([primary.get('catastrophes_certain'),primary.get('catastrophes_possible')]),'伤害数（时间表）':str([primary.get('harm_certain'),primary.get('harm_possible')]),'证据等级':grade,'缺口说明':reason});rows.append(row)
  for key,v in methods.items():metrics.append(dict(subunit=label,method=next((title for kk,title in METHODS if key in ALIASES.get(kk,[kk])),key),**v))
 with (HERE/'AXIS_TABLE.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 table='本轮新计算只含明确黑盒控制器。成功率为已知下界–上界%，方括号为配对分层重抽样5000次的95%置信区间。附加指标列对应该行时间表；全部方法的完整指标另见METHOD_METRICS.json。几何仅有实际可用菜单的最佳值，未假定缺失标签。持续右手规则只在本轮规定的五组环形任务中评估；专家列只复用同状态、同物理的既有特权参照，其他行保留不适用。\n\n'+'| '+' | '.join(rows[0])+' |\n|'+'|'.join(['---']*len(rows[0]))+'|\n'+'\n'.join('| '+' | '.join(str(v).replace('|','／')for v in row.values())+' |'for row in rows)+'\n';(HERE/'AXIS_TABLE.md').write_text(table);write(HERE/'METHOD_METRICS.json',metrics);write(HERE/'GAPS.json',gaps);(HERE/'GAPS.md').write_text('\n\n'.join(f"{v['part']}：{v['reason']} 补齐条件：{v['condition_to_complete']}"for v in gaps)+'\n')
 body=[p,'','本轮范围为十四个环形交换场景流匹配策略、六个高斯行为克隆策略、六个分散式模型预测控制控制器，以及两个固定参数的瓶颈场景移植实例。原四类合并结果只在事后子集对照表中引用，未重算未纳入的类型。','', '除官方共享数据库流程外，全部新代码和输出位于本轮目录；未进行任何Git写操作或包安装。共享数据库使用官方追加登记、提交前封存检查、缺失项运行、默认日志、单进程合并和完成后检查。所有数值失败保留为未知，未重跑；超时和停滞候选均不等同死锁。','', '几何传参逐字段测试通过，三十二条历史复现逐位一致；半径0.90没有行为漂移。逐控制器曲线同时显示异质性：编号十三的环形交换场景流匹配策略在半径0.85和0.95下，两臂较好成功率分别为50%和87.5%；中位数能力门仍按十四个控制器的冻结标准判定。复用了448条阳性对照、1792条同状态修正标签和2688条按进度切换标签。另补2912条几何下缺失的动作或安全投影基线。几何修正执行中有3531条已知超时，另有65条未知；已知失败中停滞候选为1条。原三类分类表的3596条分母表示“未成功或未知”，不能把其中未知当作已知失败；分母澄清收据单列，分类阈值和逐条诊断均未改变。不能据此声称修正解除正式死锁，也没有把这1条归为控制器能力局限。','', '进度门槛由确实到达第500步的699条独立重算为0.9681620884830543，浮点位模式相同；排除195条提前成功和2条提前数值失败。它低于把提前成功保持至第500步的约0.97281定义，切换会略早。在500、700与1000步预算下，冻结的控制器特征选择模型均满足预注册的方法适用性标准。本轮收据先于任何本轮新进度切换执行；由于原身份已完整运行，本轮没有新增该类执行。原历史实验自己的决定收据早于其提交，予以保留。','', '双机器人流匹配策略的已有三臂成功率分别为6.25%、12.50%、18.75%。历史能力证据来自非相向导航与有序释放，不能替代同时相向任务。检查器修正只改变存储精度的比较方式，执行代码与实际轨迹未改变；本轮没有重新判门或补跑。','', '环形交换场景流匹配策略的修正保持在流积分内部；高斯策略与分散式模型预测控制保持“动作投影、加修正、再投影”的原位置。持续右手规则先取得原执行位置的零修正安全动作，触发时加入世界坐标顺时针切向速度，再使用同一个原安全投影。触发器逐字复用，未激活的三十二条轨迹逐位等于原安全基线。二十组参数在十六个新调参状态上分别对三组黑盒控制器调参，选择收据早于6144条新方法评估身份；其中3908条通过严格未激活证明复用原轨迹，其余2236条按新方法运行。既有安全轨迹若经逐步检查证明触发器从未激活，就严格复用轨迹和终止标签，不重复生成物理过程。种子区间内前十六个候选因与历史跨场景研究的环形交换场景能力门状态重复而剔除，实际使用202610370016至202610370031；全部可见旧位置均已排除重复，3200条仅别名历史身份的物理去重可见性限制保留为缺口。','']
 for k,v in DD.items():body +=[dname(k)+'：'+verdict(v['paired']['schedule_minus_right_hand'],v.get('inferential_endpoint_eligible',True))+'。持续右手规则相对只加安全投影的配对差置信区间为'+fmt(v['paired']['right_hand_minus_safety']['CI_pp'])+'个百分点；逐条救回交叉表见交付文件。','']
 resource=read(HERE/'RESOURCE_MONITOR.json');body+=['资源监控记录的CPU分片分配峰值为'+str(max(x['allocated_total']for x in resource['samples']))+'，已知活跃CPU分片峰值为'+str(max(x['known_actual_total']for x in resource['samples']))+'。本轮GPU分片为零；其他任务退出时仅调整本轮数组并发，必要时暂停本轮进程并保留其状态。监控记录包含全部响应事件。','']
 body+=['瓶颈场景的执行顺序为规划动作、原速度界与到达目标后的停止、原经认证的硬安全投影、原OrthoFlow3修正、同一安全投影、原生动力学与事件。第四层是冻结三臂组合的操作性能力门，与几何有效、联合可解及正式死锁解除分层解释。瓶颈场景移植保留原目标、机器人间避碰、平滑代价、采样和随机流，仅推广机器人数量与真实墙段代价。四机器人环形交换场景的48次动作对照逐位一致；复现门、测时、每种机器人数量合计432个CPU小时的计算预算、能力门与五层证据见对应交付；本轮没有合格源数量，源选择冻结不适用。最省计算的参数为规划步长10、样本数64、避碰权重100；第二组为规划步长10、样本数256、避碰权重100。两者在原环形交换场景最佳修正臂成功率上并列最高，第二组避免重复同一个实例；原聚合报告与实例标签的适用差别已在预注册前写入来源选择文件。','', translate('；'.join(transfers))+'。','', '事后黑盒子集的划分来自用户2026年10月16日给出的黑盒定义。全部子集重算为DEVELOPMENT（事后子集）证据；关系前提不满足和训练优势未达显著性，均不解释为已证明零效果。','', '数量轴首批复现曾因通用入口预加载了实验副本中的同名模块而在执行前退出，零轨迹、零物理步、零未知执行；改用独立瓶颈场景入口并锁定原仓库模块后重新通过提交前检查，只补跑仍缺失的身份。纯速度限幅函数逐字复制，数值核未改，四机器人逐位对照重新通过。该失败尝试、修复收据和官方完成后检查均保留。完整性差异来源及逐路径清单见INTEGRITY_COMPARISON.json；导入源哈希、复制差异、冻结收据和共享数据库核查见FINAL_VERIFICATION.json。其他并行任务的新增文件没有被本轮修改或删除。','', '缺口清单见GAPS.md。持续右手规则未纳入原冻结选择模型的动作菜单，其相对该菜单的遗憾可为负。交叉拟合最佳动作按训练半组选臂，其评估半组值也可能低于另一个预先固定的方法；样本内最佳值另报。标准方法、可用菜单限制、未知上下界、交叉拟合与样本内最佳值，以及各方法事件、遗憾、灾难与伤害数见AXIS_TABLE和METHOD_METRICS.json。']
 (HERE/'REPORT.md').write_text('\n'.join(body)+'\n');handoff=['本目录是Basin_C1 / OrthoFlow3的黑盒控制器补充实验，可独立于对话复核。研究问题、顺序、阈值、种子、菜单和资源上限已在统一预注册及其收据中冻结。','',p,'','先读REPORT.md，再查results.json、AXIS_TABLE.md、GAPS.md和FINAL_VERIFICATION.json。原始轨迹分阶段保存在本目录，历史复用清单记录原始文件、轨迹和数据库身份的哈希；不要根据新命名空间重复训练或运行兼容执行。','', '持续右手规则使用run_D_reuse_v3.py；数量轴复现与测时使用run_E_repaired.py，后续能力门与主评估使用run_E_after_timing.py。早期版本和修复收据保留用于审计；参数、阈值、种子和终点没有因修复而改变。封存完成后以COMPLETION.receipt.json核对交付文件，不重新提交已完成身份。源文件保护适用于本目录之外的全部已有文件。共享数据库的唯一写入口是官方登记、默认日志与官方单进程合并。运行入口先锁定官方数据库模块，全部导入后显式关闭JAX的64位模式，所有检查与输出均留在本目录。','', '持续右手规则的选择冻结后不得据评估结果调参。瓶颈场景移植实例参数固定，若预算超限或能力门失败，对应主评估缺失是设计要求；不得减少样本或缩短时域补出通过。旧环形交换场景模型缺少可变数量瓶颈场景输入接口，因此保留不适用。','', '数量轴的16条历史复现、12条测时和288条能力门记录均已完成；三个数量的两个参数实例全部未过门，没有运行主评估或生成源选择冻结收据。缺失的迁移与动作族检验保留为缺口，前三层32个评估状态的见证来自原参考。','', '本轮所有六部分的执行与缺口已封存。后续工作需保留未知、区分几何有效、单机器人有路、联合可解、控制器成功与死锁解除五层，按缺口条件另立预注册。不要重跑已存在的未知执行，也不要把超时改称死锁。']
 (HERE/'HANDOFF.md').write_text('\n'.join(handoff)+'\n');write(HERE/'results.json',dict(utc=utc(),parts=dict(A=A,B=B,C=C,D=DD,E=E,F=F),gaps=gaps,integrity=verification['integrity'],completion={k:'COMPLETE_WITH_GAPS' if k in ['C','D','E','F'] else 'COMPLETE'for k in ['A','B','C','D','E','F']}));print(p)
def PARAM_DESCRIPTION(name):return {'DM1':'规划步长10、样本数64、避碰权重100','DM3':'规划步长10、样本数256、避碰权重100'}[name]
if __name__=='__main__':main()
