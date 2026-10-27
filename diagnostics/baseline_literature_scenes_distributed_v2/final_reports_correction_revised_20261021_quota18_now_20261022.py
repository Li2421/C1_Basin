"""Final user-requested original/revised report from sealed evidence only."""
from experiment import *
from registry import verify_receipt
from final_reports_flat_static_allcpu_delivery import pct,cell,delta,table,METHODS,LAYERS,pick,geometry,dynamics
import csv,collections
SEM={'original':'原修正语义（预注册）','revised':'修订修正语义（看过预注册结果后提出的事后修订，开发证据）'}

def save(name,text):
 path=HERE/name;assert not path.exists() or name in ('HANDOFF.md',),name;path.write_text(text.rstrip()+'\n')

def axis(results,additional):
 chunks=[];flat=[];types={r['type_code']:r['type'] for r in read(HERE/'CONTROLLER_MANIFEST.json')}
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised']
   for scope in ('fixed_full','learned_evaluation'):
    summary=entry[scope]
    if summary.get('status')!='complete':continue
    stats=list(summary['statistics'])
    if scope=='learned_evaluation':
     refs=additional[sem]['evaluation_fixed_reference_tables'].get(scene,{})
     keys={(r['type'],r['layer'],r['method']) for r in stats};stats += [r for r in refs.get('statistics',[]) if (r['type'],r['layer'],r['method']) not in keys]
    typenames=sorted({r['type'] for r in stats}-{'type_equal'})+['type_equal'];rows=[]
    for typ in typenames:
     values=[]
     for layer in ('central','distributed'):
      for method in METHODS:
       row=next((r for r in stats if (r['type'],r['layer'],r['method'])==(typ,layer,method)),None)
       values.append(cell(row));flat.append(dict(scene=scene,state_scope=scope,semantics=sem,type=typ,layer=layer,method=method,success_rate=None if row is None else row['success_rate'],ci95=None if row is None else row['ci95'],success_upper=None if row is None else row['success_upper']))
     rows.append([types.get(typ,'类型等权合并'),*values])
    chunks.append(NAMES[scene]+'；'+SEM[sem]+'；'+('全部主state' if scope=='fixed_full' else '独立学习评估state')+'\n\n'+table(['controller类型',*[LAYERS[l]+'：'+METHODS[m] for l in ('central','distributed') for m in METHODS]],rows))
 for label,key in [('原语义四场景类型等权合并','four_scene_type_equal'),('修订语义四场景类型等权合并','four_scene_revised_type_equal')]:
  summary=results[key];rows=[[LAYERS[r['layer']],METHODS[r['method']],cell(r)] for r in summary.get('statistics',[]) if r['type']=='type_equal'];chunks.append(label+'\n\n'+table(['安全层','做法','成功率% [95% CI]'],rows))
 save('AXIS_TABLE.md','全部为开发证据。固定主state与学习评估子集分别比较；类型→实例→可用state等权，seed成对，5000次bootstrap；成功下界将unknown记失败，上界另见CSV/JSON。N/A表示未要求或已明确停止。旧12动作与新15动作结果、原语义与事后修订分别保留。\n\n'+'\n\n'.join(chunks))
 with (HERE/'AXIS_TABLE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in flat)

def decomposition(data,additional):
 rows=[]
 for sem in SEM:
  for scope,arms in additional[sem]['weighted_outcome_and_diagnosis_rates'].items():
   for key,value in arms.items():
    layer,method=key.split('|');r=value['rates'];hover=data['near_goal_wandering'][sem].get(scope,{}).get(key,{}).get('rates',{}).get('near_goal_wandering')
    rows.append([NAMES.get(scope,'合格场景合并'),SEM[sem],LAYERS[layer],METHODS[method],*[pct(r[k]) for k in ('success','upper','collision','timeout','unknown','deadlock_episode','wall_episode')],pct(hover)])
 save('FAILURE_DECOMPOSITION.md','类型等权率，保留全部合格实例。4a′/4a″是保存轨迹上的有限窗口判据；timeout本身不是deadlock。目标附近徘徊为超时、至少一个未到达机器人、且所有未到达机器人到目标距离≤3倍原容差。原始计数与CI见MIGRATION_RESULTS.json。\n\n'+table(['范围','修正语义','安全层','做法','成功下界','成功上界','碰撞','超时','unknown','逐机器人4a′','墙前4a″','目标附近徘徊'],rows)+'\n原集中式联合锥参照、实际乘积联合锥与九格敏感性见原/修订MAIN_ADDITIONAL_METRICS与DIAGNOSTICS。圆形低速度下有限窗口静止条件较弱，不单独证明无限时域死锁。')

def class_gain_tables(results):
 chunks=[]
 for scene in NAMES:
  rows=[]
  for sem in SEM:
   for layer,scopes in results[scene]['instance_class_gain_splits'][sem].items():
    for scope,entry in scopes.items():
     for kind in ('fixed_full','learned_evaluation'):
      for r in entry[kind].get('statistics',[]):
       if r['type']!='type_equal' or r['method']=='safety':continue
       if kind=='learned_evaluation' and r['method']!='learned_scene':continue
       rows.append([SEM[sem],LAYERS[layer],scope,'全部主state' if kind=='fixed_full' else '独立评估state',METHODS[r['method']],cell(r),delta(r.get('delta_vs_safety')),entry['gain_interpretation']])
  chunks.append(NAMES[scene]+'\n\n'+table(['修正语义','安全层','实例范围','state范围','做法','成功率','相对只加安全投影增益','解释'],rows))
 save('CORRECTION_GAIN_SPLITS_20261021.md','只去除新归类中的controller级失败；安全层导致的全局失败保留并单列。每层独立归类与取样，不能把两层失败集合混为一谈。controller级失败增益标为“修正推动controller自身停住的实例”，不计为解开死锁。\n\n'+'\n\n'.join(chunks))

def main():
 for proof in ('MIGRATION_NUMBERS.receipt.json','MAIN_ADDITIONAL_METRICS.receipt.json','MAIN_ADDITIONAL_METRICS_REVISED_20261021.receipt.json','OPTIONAL_NUMBERS.receipt.json','DB_POSTFLIGHT.receipt.json','INTEGRITY_END.receipt.json','CORRECTION_OPTIONAL_READY_20261021.receipt.json'):verify_receipt(proof)
 audit=read(HERE/'INTEGRITY_END.json');db=read(HERE/'DB_POSTFLIGHT.json');assert audit['passed'] and db['passed']
 data=read(HERE/'MIGRATION_RESULTS.json');results=data['results'];additional={'original':read(HERE/'MAIN_ADDITIONAL_METRICS.json'),'revised':read(HERE/'MAIN_ADDITIONAL_METRICS_REVISED_20261021.json')};classes=read(HERE/'INSTANCE_CLASSES_REVISED.json');optional=read(HERE/'OPTIONAL_RESULTS.json')
 axis(results,additional);decomposition(data,additional);class_gain_tables(results)
 geometry();dynamics(optional)
 revised_geometry=read(HERE/'GEOMETRY_CENSUS_REVISED_20261021.json');geometry_groups=collections.defaultdict(list)
 for item in revised_geometry:geometry_groups[(item['scene'],item['layer'])].append(item)
 save('GEOMETRY_CENSUS_REVISED_20261021.md','修订语义，仅读保存仿真证据；候选几何不证明物理可解或无限时域死锁。完整约束法向、条件下界和逃逸量见JSON。\n\n'+table(['场景','层','候选轨迹','候选agent-step','最大安全速度'],[[NAMES[scene],LAYERS[layer],len(rr),sum(r['candidate_agent_steps'] for r in rr),max(r['maximum_safe_speed'] for r in rr)] for (scene,layer),rr in sorted(geometry_groups.items())]))
 # The legacy JSON is kept for historical equality checks; revised taxonomy is primary.
 if not (HERE/'CONTROLLER_LEVEL_FAILURES.md').exists():save('CONTROLLER_LEVEL_FAILURES.md','本轮最终口径以INSTANCE_CLASSES_REVISED.md/.json为准。只加安全投影成功上界≤5%是候选条件，随后以失败局未到达机器人·局的4a′/4a″多数分类。只去掉controller级失败，安全层全局失败不排除；墙前陷阱单列。CONTROLLER_LEVEL_FAILURES.json保留原阈值口径用于核对已封存阶段报告，不是最终排除定义。\n\n'+table(['场景','层','合格数','controller级失败','state相关失败','安全层全局失败','墙前陷阱'],[[NAMES[r['scene']],LAYERS[r['layer']],r['qualified_instances'],r['controller_level_failure_instances'],r['state_related_failure_instances'],r['safety_layer_global_failure_instances'],r['wall_trap_instances']] for r in classes['discrimination']]))
 with (HERE/'DYNAMICS_STAGE.md').open('a') as f:
  f.write('\n\n第二阶段固定非零修正采用修订接触开关，门口Rc按膨胀半径；只加安全投影保持原语义。配对参照为修订后的第一阶段。原语义非零第二阶段与集中式第二阶段均N/A，未额外运行。新分类、按实例类拆分增益与目标附近徘徊见OPTIONAL_RESULTS.json及INSTANCE_CLASSES_DYNAMICS_REVISED_20261021.json；controller级增益不计死锁解决。\n')
 paragraphs=[f"结束完整性审计通过：检查{audit['checked_files']}个受保护文件，差异{len(audit['differences'])}；v1与受保护源码保持只读。共享数据库结束审计通过：完成身份{db['unique_completed_identities']}，独立轨迹{db['unique_trace_files']}，本轮冲突隔离{db['own_quarantined_rows']}，未合并journal{db['unmerged_complete_journals']}。详见INTEGRITY_END.json、DB_POSTFLIGHT.json。v1原有两个非Python配置缺开工全文件哈希的严格覆盖限制仍保留。全部结论为开发证据。"]
 classification=[]
 for scene in NAMES:
  for layer in ('central','distributed'):
   d=next(r for r in classes['discrimination'] if (r['scene'],r['layer'])==(scene,layer));base=pick(results[scene]['fixed_full'],layer,'safety');rate=additional['original']['weighted_outcome_and_diagnosis_rates'][scene][layer+'|safety']['rates']
   classification.append(f"{NAMES[scene]}、{LAYERS[layer]}：基线成功{cell(base)}，逐机器人4a′局比例{pct(rate['deadlock_episode'])}；controller级失败{d['controller_level_failure_instances']}、state相关失败{d['state_related_failure_instances']}、安全层导致的全局失败{d['safety_layer_global_failure_instances']}个合格实例，墙前陷阱{d['wall_trap_instances']}个。")
 paragraphs.append(' '.join(classification)+' 窄通道分布式多数低成功实例归为安全层导致的全局失败，保留为死锁候选证据；门口两层各5个controller级失败另列。九格改变的实例与另一安全层旁证见INSTANCE_CLASSES_REVISED。timeout不等于死锁，有限窗口候选不能证明永久死锁。')
 gains=[]
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised'];items=[]
   for layer in ('central','distributed'):
    for method in ('fixed_eta10','fixed_eta13','human_progress'):
     row=pick(entry['fixed_full'],layer,method);items.append(LAYERS[layer]+'、'+METHODS[method]+'成功'+cell(row)+'，增益'+delta(row.get('delta_vs_safety') if row else None))
   gains.append(NAMES[scene]+'、'+SEM[sem]+'：'+'；'.join(items)+'。')
 paragraphs.append(' '.join(gains)+' 每种固定修正与重新学习增益按全部合格实例、controller级失败、其余实例及安全层全局失败分列于CORRECTION_GAIN_SPLITS_20261021.md。controller级增益称“修正推动controller自身停住的实例”，不计为解开死锁；死锁解决结论只从其余实例与安全层全局失败范围判断，不能从总体成功增益直接推断。目标附近徘徊的新旧两套率见FAILURE_DECOMPOSITION.md与JSON。')
 learned=[]
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised'];summary=entry['learned_evaluation'];row=pick(summary,'distributed','learned_scene')
   learned.append(NAMES[scene]+'、'+SEM[sem]+'：'+('场景内重新学习成功'+cell(row)+'，相对只加安全投影'+delta(row.get('delta_vs_safety')) if row else 'N/A：'+str(summary.get('reason'))))
 eligible_gains=[]
 for scene in NAMES:
  for sem in SEM:
   for layer,scopes in results[scene]['instance_class_gain_splits'][sem].items():
    for label in ('remaining_instances','safety_layer_global_failure'):
     summary=scopes[label]['fixed_full'];row=pick(summary,layer,'fixed_eta10')
     if row:eligible_gains.append(NAMES[scene]+'、'+SEM[sem]+'、'+LAYERS[layer]+'、'+label+'：全程η10相对基线'+delta(row.get('delta_vs_safety')))
 paragraphs.append('用于讨论死锁解决的实例范围：'+'；'.join(eligible_gains)+'。这些范围排除了controller自身停住的增益，仍需与逐机器人4a′、墙前4a″及徘徊诊断联合阅读；成功改善是有限样本操作性证据，不是永久死锁或全局可解证明。门口5个controller实例的保存轨迹与两种语义下η10三分量范数描述见DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.json；不把分量范数当分量消融的因果结论。')
 paragraphs.append('场景内重新学习：'+'；'.join(learned)+'。原菜单12动作，新菜单15动作补入两种全程固定η及人定双η；原结构、19维动作编码、留一类型、初始化种子17/23/41、2000训练步与checkpoint选择不变。选择在首条评估之前封存。修订集中式不重新学习。比较仅在相同独立评估state内，不能与全部主state混为同一分母；完整独立评估菜单regret未穷举，为N/A。')
 paragraphs.append('修订说明：看过预注册结果后，用户要求纠正窄通道安全层全局失败归类、门口controller自身停住的增益解释、学习菜单缺固定动作、目标附近徘徊四处问题。修订只给原Brel加接触开关，Rc=1.25×实际d_safe；三分量公式、η、两次安全投影、跟踪、到达停止与6/7回退不变。Rc在首条修订rollout前封存，不按结果调参。强制全开逐位一致、全部已到达时Brel为零、论文窄通道开关全开、编号置换数值误差门及每场景8条η0完整回放共32条通过；静态徘徊检查及实际例外见CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021。修订证据明确为事后开发证据，不能冒充预注册确认。全部已到达时Brel为零只消除了该分量；严格目标构型成为完整安全闭环平衡点仍需该构型满足安全约束，不能由开关单独推得。')
 notes=['v1按用户要求完成窄通道、二维拥挤随机、门口三场景收尾；圆形流程版为用户停止转入v2的缺口，不能写成v1四场景合并。论文版沿用v1的PAPER_REPRODUCTION.*，不重跑。', '旧语义圆形标签2044条保留并合并，4100条未落盘请求按用户决定停止、不恢复、不计unknown；旧圆形1536条评估未执行。修订圆形学习为独立新语义，不能补成旧语义完成。', '第二阶段与变体为非主线：实际动力学约束违反、采样速度违反、无解最小违反与门口非线性求解无全局最优性证书均保留；成功仍按原任务标准。没有额外集中式动力学试验，联合锥只是同动力学耦合诊断参照。', '冻结旧表示A3、冻结旧物理接口A0、无兼容冻结选择接口G_E、未要求专家参照为N/A；训练state上的菜单oracle与带符号regret仅作描述，不能当独立state性能。', '原语义和修订语义各自保留unknown识别上下界、配对伤害和10个百分点下降实例。基线按流程先于学习可见，选择文件outcomes_opened=False只指选动作阶段不打开所选评估输出，不代表全局盲测。']
 for gap in (read(HERE/'OPTIONAL_GAPS.json') if (HERE/'OPTIONAL_GAPS.json').exists() else []):notes.append('非主线缺口：'+canonical(gap))
 paragraphs.append('第二阶段各场景完成情况、动力学与修订第一阶段的配对差、类别拆分与目标附近徘徊见DYNAMICS_STAGE.md、OPTIONAL_RESULTS.json。所有图只读取封存仿真证据，不改变控制器、物理、事件或成功标准。')
 save('GAPS.md','\n\n'.join(notes));save('REPORT.md','\n\n'.join(paragraphs)+'\n\n'+'\n\n'.join(notes))
 save('HANDOFF.md',f"任务完成标志仅以V2_COMPLETE.json为准。工作目录{HERE}；v1只读。当前入口coordinator_correction_release_20261023.py，下游correction_after_distributed_fixed_20261021.py，optional_delivery_correction_revised_20261021_quota18_now_20261022.py。严禁恢复旧圆形语义或旧协调器。资源依据RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json：24CPU，2026-10-23 09:00 Asia/Singapore降18，GPU2份额。修订科学定义、门、执行身份与后验类别分别见CORRECTION_REVISION.receipt.json、PREREGISTRATION_ADDENDUM_20261021.receipt.json、INSTANCE_CLASSES_REVISED。全部原结果保留，combined主数值MIGRATION_NUMBERS、数据库DB_POSTFLIGHT、源码INTEGRITY_END、最终报告FINAL_REPORTS分别封存。无完成标志则继续STATUS/日志，不能写已完成。")
 outputs=['REPORT.md','AXIS_TABLE.md','AXIS_TABLE.csv','FAILURE_DECOMPOSITION.md','CONTROLLER_LEVEL_FAILURES.md','GEOMETRY_CENSUS.md','GEOMETRY_CENSUS_REVISED_20261021.md','DYNAMICS_STAGE.md','GAPS.md','HANDOFF.md','CORRECTION_GAIN_SPLITS_20261021.md']
 seal('FINAL_REPORTS.receipt.json',[HERE/n for n in [*outputs,'final_reports_correction_revised_20261021_quota18_now_20261022.py']],source_audit_passed=True,db_postflight_passed=True,main_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),posthoc_revision_explicit=True)
if __name__=='__main__':main()
