"""Final user-requested original/revised report from sealed evidence only."""
from experiment import *
from registry import verify_receipt
from final_reports_flat_static_allcpu_delivery import pct,cell,delta,table,METHODS,LAYERS,pick,geometry,dynamics,comparison
import csv,collections
METHODS=dict(METHODS,train_selected_fixed="训练集选择的最佳固定动作")
from training_selected_report_tables_20261022 import primary_tables
SEM={'original':'原修正语义（预注册）','revised':'修订修正语义（看过预注册结果后提出的事后修订，开发证据）'}

def save(name,text):
 path=HERE/name;assert not path.exists() or name in ('HANDOFF.md',),name;path.write_text(text.rstrip()+'\n')

def axis_reference(results,additional):
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
      for method in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene'):
       row=next((r for r in stats if (r['type'],r['layer'],r['method'])==(typ,layer,method)),None)
       values.append(cell(row));flat.append(dict(scene=scene,state_scope=scope,semantics=sem,type=typ,layer=layer,method=method,success_rate=None if row is None else row['success_rate'],ci95=None if row is None else row['ci95'],success_upper=None if row is None else row['success_upper']))
     rows.append([types.get(typ,'类型等权合并'),*values])
    chunks.append(NAMES[scene]+'；'+SEM[sem]+'；'+('全部主state' if scope=='fixed_full' else '独立学习评估state')+'\n\n'+table(['controller类型',*[LAYERS[l]+'：'+METHODS[m] for l in ('central','distributed') for m in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene')]],rows))
 for label,key in [('原语义四场景类型等权合并','four_scene_type_equal'),('修订语义四场景类型等权合并','four_scene_revised_type_equal')]:
  summary=results[key];rows=[[LAYERS[r['layer']],METHODS[r['method']],cell(r)] for r in summary.get('statistics',[]) if r['type']=='type_equal'];chunks.append(label+'\n\n'+table(['安全层','做法','成功率% [95% CI]'],rows))
 save('APPENDIX_REFERENCE_AXIS.md','全部为开发证据。固定主state与学习评估子集分别比较；类型→实例→可用state等权，seed成对，5000次bootstrap；成功下界将unknown记失败，上界另见CSV/JSON。N/A表示未要求或已明确停止。旧12动作与新15动作结果、原语义与事后修订分别保留。\n\n'+'\n\n'.join(chunks))
 with (HERE/'APPENDIX_REFERENCE_AXIS.csv').open('w') as f:
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

def handoff_text():
 entry=read(HERE/'ACTIVE_ENTRYPOINT.json');policy=read(HERE/'RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json')
 return f"""本轮为四个文献场景分布式安全层迁移v2，原修正语义预注册证据与看过结果后提出的修订开发证据分开。工作目录{HERE}；v1目录{V1}只读。唯有V2_COMPLETE.json表示实际完成；不存在时继续按STATUS与日志处理，不把后台仍跑写成完成。v1三场景收尾已冻结，圆形流程版缺口不恢复，论文版结果沿用，不冒充v1四场景合并。

当前唯一入口和进程信息以ACTIVE_ENTRYPOINT.json、COORDINATOR_PID.json为准：协调器{entry['coordinator']}，下游{entry['downstream']}，可选阶段{entry['optional']}，最终报告{entry['final_reports']}。启动前核对现存PID/命令及CORRECTION_COORDINATOR.lock，不能并发启动第二协调器；下游也有独立锁。已运行child期间不要更改下游配置。只选择本轮目录内源码哈希列在声明READY收据中的静态入口；不在运行时改模块或改封存文件。历史部署文字里的入口可能已过时。

流程Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B；最后画图才用.venv-c1-plots。禁止安装包、写v1/外部源码或做git commit。experiment.py将TMP/MPL/XDG/CUDA缓存限制于本轮且禁用bytecode。所有已封存raw、trace、checkpoint、配置和源文件不修改，修订用新文件并记录来源SHA。

分布式安全层在distributed_safety.LocalSafety：各机器人独立解二维投影，不输入其他机器人的名义或安全速度。配对行2(pi-pj)·ui>=-c*gamma*h，窄通道/拥挤随机/圆形主线c=1/2，门口c=1；墙全责，速度球和同问题数值重试不变。集中式保持原联合投影及原OrthoFlow3。构造snapshot和实际约束由projection_adapter.constraints/LocalSafety完成；runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022.run执行分布式与拥挤集中式，central_correction_revised_20261021.run执行其他集中式。

调用顺序仍为冻结名义控制器→原速度界和到达停止→第一次安全投影→三个原修正基→第二次同问题安全投影→原物理。correction_revision_20261021只把Brel在无接触冲突的机器人行置零；Brel启用时仍对全部其他机器人先bound再平均，不能改成仅近邻平均。接触为距离<=Rc且两者不同时在原到达容差内，Rc=1.25*d_safe，d_safe含分离buffer，第二阶段门口含前视膨胀半径；具体数值CORRECTION_REVISION.md在首条研究运行前固定，不据结果调整。到达位每步广播1bit，其他通信假设保持原样。双η主线使用平均进度及原6/7时限回退；逐机器人进度只作为非主线变体。

48个名义controller=8类型×6冻结实例，不训练这些controller；各场景能力门合格数44/44/45/45。旧12与新15指修正动作菜单，新菜单补η10、η13和人定双η。物理输入12维、动作编码19维、模型架构、36个source-type fit集合、三初始化17/23/41、2000训练步及嵌套留一类型选择不变。新模型路径critic_fits_revised_20261021/，选择LEARNED_EVAL_CHOICES_REVISED_20261021_<scene>在首条评估前封存。集中式修订学习不运行。原η0轨迹32条完整逐位回放通过后全部按精确执行身份复用；修订非零用新身份，同语义的三个固定训练动作和冻结后选到的现有执行直接复用。

顺序严格为D修订固定4场景→D修订学习窄通道/拥挤/门口→C修订固定4场景→圆形D修订学习→全部保存轨迹诊断和原/新合并主数值→第二阶段与变体→图、数据库和源码结束审计、报告。前一研究阶段全结束才进后一阶段；空核可做已完成轨迹的可让出诊断。统计27独立任务调用原数学内核，5000次bootstrap、原种子、类型→实例→state等权、seed配对和unknown识别界不变；source AST、5000抽样数组、串并行类别增益一致的证据在ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json。

每项新研究前先查HISTORY_CHECK及现存配置/数据/checkpoint/预测，遵守共享数据库README的官方preflight，只跑genuinely_missing。registry/revised_case_factory负责身份与注册；worker只通过官方cache_writer.append_journal写默认实验exp_baseline_literature_scenes_distributed_v2，不能写SQLite。唯一merger由db_merge.py包装官方程序并持锁。原圆形学习2044条保留且已合并，4100条停止不恢复，旧评估1536不运行；这项用户停止不产生unknown。数值求解失败的unknown独立保留，不能改成timeout/deadlock或默默重跑覆盖。

后续评估和第二阶段/变体另使用进程间低层A*精确查询缓存，完整键和SHA256校验通过才复用；未命中、碰撞或I/O失败执行原搜索。可替换的缓存槽只保存纯函数计算结果，最多65536槽、每槽16KiB，不是研究raw、轨迹或数据库。当前标签worker保留内存缓存版本，不为此中断正在跑的局；两版共享同一科学执行身份和预注册请求。证据见CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json。这不是全流程提速倍数保证。PA低层原生查询另缓存同一查询内同一网格单元/移动的静态墙谓词和已静止预约的碰撞谓词；连续起点首步与目标/保持边仍走原运算，最多65536单元，超过容量走原运算，搜索顺序、预算和轨迹不变。658项原生路径、含实际容量回退、四场景20查询及两条完整2000步轨迹逐位验证见PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json。单例计时不代表全流程倍率。PA另启用有界精确查询内存缓存：保留全部预约后缀与十个初始判定位，恢复真实连续起点，未命中仍执行原始A*。每进程最多8192项和32MiB估计内容；完整两个2000步保存episode回放与有限网格、四场景查询均须逐位通过。证据见PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json；单例计时不代表全流程倍数。当前资源按用户最新指令固定为18CPU（含1管理核）和2GPU份额。性能加速另含低层A*起点网格/完整初始边判定签名的精确缓存，证明及完整2000步逐位回放见CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json与CBS_START_SIGNATURE_THEORY_20261022.md。该优化只复用数学上相同的查询输出；路径第一点保留本次真实坐标，原700时域、CBS节点预算、物理与统计定义不变，不构成第三套修正语义。性能加速为已验证静态必要条件证书和原生精确bit-key几何缓存，科学语义/RNG/事件不变；CBS_FLAT_CACHE_EQUIVALENCE等收据含完整轨迹逐位回放。未采用的候选也保留证据，不能随意启用。实例类以INSTANCE_CLASSES_SUPPLEMENT_20261022为主：失败上界<=5%后用未到达机器人·局的（4a′+已到达者阻挡）/4a″多数判断，只去掉controller级失败，安全层全局失败保留，墙前陷阱单列。旧CONTROLLER_LEVEL_FAILURES.json保留作历史核对，不能替代新归类。timeout本身不是死锁。目标附近徘徊为超时且所有未到达机器人距目标<=3倍原容差，至少一台未到达；两种语义、两层和实际第二阶段分别报告。图只读封存仿真证据。

资源最新用户授权：立即释放六核后固定18CPU总额度（17调度+1本地管理核），GPU保持2份额。RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json为准；旧24→18日期安排已被立即释放决定替代。独立资源guard和watchdog持续校验，不能再次释放六核或恢复24核。本地管理优先绑定未分配核；若全机已分配满，仅与经用户/命令路径核验的本轮自有Slurm worker共享一核，绝不占其他用户核，有空核即迁回。当前guard以ACTIVE_ENTRYPOINT.local_cpu_guard为准，策略及核验见LOCAL_CPU_SHARED_OWN_FALLBACK_20261022与LOCAL_CPU_SHARED_OWN_FALLBACK_VERIFIED_20261022；18份额上限不变。

补充分析以ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json和INSTANCE_CLASSES_SUPPLEMENT_20261022为准，原INSTANCE_CLASSES_REVISED完全保留。安全层全局失败的多数计数合并滤波器平衡点死锁与已到达者阻挡；死锁证据仍只计滤波器平衡点死锁。分布式每场景按32个训练state、8类型等权在15菜单中选一个最佳固定动作，精确平局按菜单顺序。TRAIN_SELECTED_FIXED_20261022_<scene>先于对应评估preflight封存；评估严格复用相同执行身份，无结果才按官方preflight补跑。圆形训练标签齐全后才选。主比较在相同独立评估state、controller和seed上计算学习−训练选定固定动作，bootstrap不重选。三种旧固定修正只作附表参照，集中式保留三种固定修正。全部15动作训练排序见MENU_TRAINING_ACTION_RANKING_20261022.md。

主数值认MIGRATION_NUMBERS、后验指标认原/新MAIN_ADDITIONAL_METRICS、可选数值认OPTIONAL_NUMBERS、数据库认DB_POSTFLIGHT、源码只认INTEGRITY_END、交付认FINAL_REPORTS。全部实际作业结束后才做最终审计和完成标志；可选缺口独立披露，不混入主线或算unknown。v1原两个非Python配置缺开工全文件哈希的限制仍保留，不能倒补为严格覆盖。"""

def main():
 verify_receipt('FINAL_DELIVERY_CLARITY_20261022.receipt.json')
 verify_receipt('FINAL_DELIVERY_AUDIT_ORDER_20261022.receipt.json')
 verify_receipt('PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json');verify_receipt('PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json');verify_receipt('PA_QUERY_MEMO_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json')
 verify_receipt('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json')
 for proof in ('MIGRATION_NUMBERS.receipt.json','MAIN_ADDITIONAL_METRICS.receipt.json','MAIN_ADDITIONAL_METRICS_REVISED_20261021.receipt.json','OPTIONAL_NUMBERS.receipt.json','DB_POSTFLIGHT.receipt.json','INTEGRITY_END.receipt.json','CORRECTION_OPTIONAL_READY_20261021.receipt.json'):verify_receipt(proof)
 audit=read(HERE/'INTEGRITY_END.json');db=read(HERE/'DB_POSTFLIGHT.json');assert audit['passed'] and db['passed']
 data=read(HERE/'MIGRATION_RESULTS.json');results=data['results'];additional={'original':read(HERE/'MAIN_ADDITIONAL_METRICS.json'),'revised':read(HERE/'MAIN_ADDITIONAL_METRICS_REVISED_20261021.json')};classes=read(HERE/'INSTANCE_CLASSES_SUPPLEMENT_20261022.json');optional=read(HERE/'OPTIONAL_RESULTS.json')
 axis_reference(results,additional);primary_tables(results,save,cell,delta,table);decomposition(data,additional);class_gain_tables(results)
 geometry();dynamics(optional)
 with (HERE/'DYNAMICS_STAGE.md').open('a') as stream:stream.write('\n第二阶段低成功率类别若含碰撞、数值失败、约束/速度/加速度违反，不能直接称为controller自身停住，也不计死锁解决；具体逻辑局计数见INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.controller_failure_baseline_failure_modes及各类别gain_interpretation。分类与成功标准未改。\n')
 revised_geometry=read(HERE/'GEOMETRY_CENSUS_REVISED_20261021.json');geometry_groups=collections.defaultdict(list)
 for item in revised_geometry:geometry_groups[(item['scene'],item['layer'])].append(item)
 save('GEOMETRY_CENSUS_REVISED_20261021.md','修订语义，仅读保存仿真证据；候选几何不证明物理可解或无限时域死锁。完整约束法向、条件下界和逃逸量见JSON。\n\n'+table(['场景','层','候选轨迹','候选agent-step','最大安全速度'],[[NAMES[scene],LAYERS[layer],len(rr),sum(r['candidate_agent_steps'] for r in rr),max(r['maximum_safe_speed'] for r in rr)] for (scene,layer),rr in sorted(geometry_groups.items())]))
 # The legacy JSON is kept for historical equality checks; revised taxonomy is primary.
 if not (HERE/'CONTROLLER_LEVEL_FAILURES.md').exists():save('CONTROLLER_LEVEL_FAILURES.md','本轮最终口径以INSTANCE_CLASSES_SUPPLEMENT_20261022.md/.json为准。只加安全投影成功上界≤5%是候选条件，随后以失败局未到达机器人·局的滤波器平衡点死锁加已到达者阻挡的严格多数/墙前陷阱多数分类。只去掉controller级失败，安全层全局失败不排除；墙前陷阱单列。CONTROLLER_LEVEL_FAILURES.json保留原阈值口径用于核对已封存阶段报告，不是最终排除定义。\n\n'+table(['场景','层','合格数','controller级失败','state相关失败','安全层全局失败','墙前陷阱'],[[NAMES[r['scene']],LAYERS[r['layer']],r['qualified_instances'],r['controller_level_failure_instances'],r['state_related_failure_instances'],r['safety_layer_global_failure_instances'],r['wall_trap_instances']] for r in classes['discrimination']]))
 with (HERE/'DYNAMICS_STAGE.md').open('a') as f:
  f.write('\n\n第二阶段固定非零修正采用修订接触开关，门口Rc按膨胀半径；只加安全投影保持原语义。配对参照为修订后的第一阶段。原语义非零第二阶段与集中式第二阶段均N/A，未额外运行。新分类、按实例类拆分增益与目标附近徘徊见OPTIONAL_RESULTS.json及INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.json；controller级增益不计死锁解决。\n')
 paragraphs=[f"结束完整性审计通过：检查{audit['checked_files']}个受保护文件，差异{len(audit['differences'])}；v1与受保护源码保持只读。共享数据库结束审计通过：完成身份{db['unique_completed_identities']}，独立轨迹{db['unique_trace_files']}，本轮冲突隔离{db['own_quarantined_rows']}，未合并journal{db['unmerged_complete_journals']}。详见INTEGRITY_END.json、DB_POSTFLIGHT.json。v1原有两个非Python配置缺开工全文件哈希的严格覆盖限制仍保留。全部结论为开发证据。"]
 classification=[]
 for scene in NAMES:
  for layer in ('central','distributed'):
   d=next(r for r in classes['discrimination'] if (r['scene'],r['layer'])==(scene,layer));base=pick(results[scene]['fixed_full'],layer,'safety');rate=additional['original']['weighted_outcome_and_diagnosis_rates'][scene][layer+'|safety']['rates']
   classification.append(f"{NAMES[scene]}、{LAYERS[layer]}：基线成功{cell(base)}，逐机器人4a′局比例{pct(rate['deadlock_episode'])}；controller级失败{d['controller_level_failure_instances']}、state相关失败{d['state_related_failure_instances']}、安全层导致的全局失败{d['safety_layer_global_failure_instances']}个合格实例，墙前陷阱{d['wall_trap_instances']}个。")
 for scene in NAMES:
  contrast=comparison(results[scene]['fixed_full'],'safety').get('distributed_minus_central')
  classification.append(NAMES[scene]+'只加安全投影的分布式−集中式配对成功差：'+delta(contrast)+'。')
 paragraphs.append(' '.join(classification)+' 窄通道分布式低成功实例归为安全层导致的全局失败并保留在分类分析中；只有实际逐机器人滤波器平衡点死锁计死锁证据，被已到达者阻挡不计死锁证据；门口两层各5个controller级失败另列。九格改变的实例与另一安全层旁证见INSTANCE_CLASSES_SUPPLEMENT_20261022。timeout不等于死锁，有限窗口候选不能证明永久死锁。')
 gains=[]
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised'];items=[]
   for layer in ('central','distributed'):
    for method in ('fixed_eta10','fixed_eta13','human_progress'):
     row=pick(entry['fixed_full'],layer,method);items.append(LAYERS[layer]+'、'+METHODS[method]+'成功'+cell(row)+'，增益'+delta(row.get('delta_vs_safety') if row else None))
   gains.append(NAMES[scene]+'、'+SEM[sem]+'：'+'；'.join(items)+'。')
 paired_gains=[]
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised']
   for method in ('fixed_eta10','fixed_eta13','human_progress'):
    contrast=comparison(entry['fixed_full'],method).get('correction_gain_difference')
    paired_gains.append(NAMES[scene]+'、'+SEM[sem]+'、'+METHODS[method]+'的修正增益（分布式−集中式）配对差'+delta(contrast)+'。')
 paragraphs.append('三种历史固定修正的成功率、增益和两层配对差见附表 APPENDIX_REFERENCE_AXIS.md；集中式仍以这三种固定修正为对照。' +' 每种固定修正与重新学习增益按全部合格实例、controller级失败、其余实例及安全层全局失败分列于CORRECTION_GAIN_SPLITS_20261021.md。controller级增益称“修正推动controller自身停住的实例”，不计为解开死锁；死锁解决结论只从其余实例与安全层全局失败范围判断，不能从总体成功增益直接推断。目标附近徘徊的新旧两套率见FAILURE_DECOMPOSITION.md与JSON。')
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
 paragraphs.append('按新版实例类拆分的主固定与学习增益见 PRIMARY_CLASS_GAINS_20261022.md；历史三种固定修正拆分见附表。'+'。这些范围排除了controller自身停住的增益，仍需与逐机器人4a′、墙前4a″及徘徊诊断联合阅读；成功改善是有限样本操作性证据，不是永久死锁或全局可解证明。门口5个controller实例的保存轨迹与两种语义下η10三分量范数描述见DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.json；不把分量范数当分量消融的因果结论。')
 paragraphs.append('场景内重新学习：'+'；'.join(learned)+'。原菜单12动作，新菜单15动作补入两种全程固定η及人定双η；原结构、19维动作编码、留一类型、初始化种子17/23/41、2000训练步与checkpoint选择不变。选择在首条评估之前封存。修订集中式不重新学习。比较仅在相同独立评估state内，不能与全部主state混为同一分母；完整独立评估菜单regret未穷举，为N/A。')
 paragraphs.append('修订说明：看过预注册结果后，用户要求纠正窄通道安全层全局失败归类、门口controller自身停住的增益解释、学习菜单缺固定动作、目标附近徘徊四处问题。修订只给按全部其他机器人平均得到的原Brel加接触开关，Rc=1.25×实际d_safe；三分量公式、η、两次安全投影、跟踪、到达停止与6/7回退不变。Rc在首条修订rollout前封存，不按结果调参。强制全开逐位一致、全部已到达时Brel为零、论文窄通道开关全开、编号置换数值误差门及每场景8条η0完整回放共32条通过；静态徘徊检查及实际例外见CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021。修订证据明确为事后开发证据，不能冒充预注册确认。全部已到达时Brel为零只消除了该分量；严格目标构型成为完整安全闭环平衡点仍需该构型满足安全约束，不能由开关单独推得。')
 notes=['v1按用户要求完成窄通道、二维拥挤随机、门口三场景收尾；圆形流程版为用户停止转入v2的缺口，不能写成v1四场景合并。论文版沿用v1的PAPER_REPRODUCTION.*，不重跑。', '旧语义圆形标签2044条保留并合并，4100条未落盘请求按用户决定停止、不恢复、不计unknown；旧圆形1536条评估未执行。修订圆形学习为独立新语义，不能补成旧语义完成。', '第二阶段与变体为非主线：实际动力学约束违反、采样速度违反、无解最小违反与门口非线性求解无全局最优性证书均保留；成功仍按原任务标准。没有额外集中式动力学试验，联合锥只是同动力学耦合诊断参照。', '冻结旧表示A3、冻结旧物理接口A0、无兼容冻结选择接口G_E、未要求专家参照为N/A；训练state上的菜单oracle与带符号regret仅作描述，不能当独立state性能。', '原语义和修订语义各自保留unknown识别上下界、配对伤害和10个百分点下降实例。基线按流程先于学习可见，选择文件outcomes_opened=False只指选动作阶段不打开所选评估输出，不代表全局盲测。']
 notes.append('旧圆形闭合范围固定为用户停止时封存的2044条。修订阶段依指令补齐η0时，相同执行身份可能进入全局缓存；它们归新阶段授权，不扩大旧学习范围，不视为恢复旧批次。DB_POSTFLIGHT另外列出这些跨范围η0缓存命中，旧4100条仍按停止决定列示。')
 for gap in (read(HERE/'OPTIONAL_GAPS.json') if (HERE/'OPTIONAL_GAPS.json').exists() else []):notes.append('非主线缺口：'+canonical(gap))
 stage2=[]
 for scene in NAMES:
  entry=optional['results'].get('optional_dynamics_'+scene,{})
  estimates=[METHODS[row['method']]+' '+cell(row) for row in entry.get('statistics',{}).get('statistics',[]) if row['type']=='type_equal']
  stage2.append(NAMES[scene]+'：'+('; '.join(estimates) if estimates else 'N/A：'+str(entry.get('reason','未取得完整第二阶段证据')))+'，unknown '+str(entry.get('counts',{}).get('unknown','N/A')))
 paragraphs.insert(3,'第二阶段（非主线修订开发证据）：'+'；'.join(stage2)+'。配对参照是修订后的第一阶段，差异、约束和速度违反、新实例类、类别增益与徘徊率见DYNAMICS_STAGE.md、OPTIONAL_RESULTS.json及INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.json。没有额外集中式动力学执行，联合锥只是同动力学参照；不能把controller级增益计为死锁解决。')
 variant_rows=[]
 for stage,entry in optional['results'].items():
  if not stage.startswith(('optional_full_circle_','optional_local_progress_','optional_nominal_')):continue
  for row in entry.get('statistics',{}).get('statistics',[]):
   if row['type']=='type_equal':variant_rows.append([stage,METHODS.get(row['method'],row['method']),cell(row),entry.get('status')])
  if entry.get('background_only'):variant_rows.append([stage,'二维拥挤随机无安全层背景','collision='+str(entry['counts']['collision'])+' / completed='+str(entry['counts']['completed']),entry.get('status')])
 save('VARIANT_RESULTS_20261023.md','第二优先级非主线。圆形全责与个人进度采用修订Brel接触开关，名义无安全背景保持原语义。相对主线的配对差和无安全碰撞→安全结果转移的完整计数见OPTIONAL_RESULTS.json，不与四场景主线混合。\n\n'+table(['定位stage','做法','成功率或原始背景计数','状态'],variant_rows))
 paragraphs.append('第二优先级变体与无安全层背景单列于VARIANT_RESULTS_20261023.md；原始碰撞到安全结果的转移、个人进度−广播进度、全责−互惠责任的配对结果见OPTIONAL_RESULTS.json。图只读取封存证据，不改变控制器、物理、事件或成功标准。')
 from training_selected_report_tables_20261022 import report_lead
 paragraphs.insert(1,report_lead(results,cell,delta))
 save('GAPS.md','\n\n'.join(notes));save('REPORT.md','\n\n'.join(paragraphs)+'\n\n'+'\n\n'.join(notes))
 save('HANDOFF.md',handoff_text())
 outputs=['AXIS_TABLE.md','AXIS_TABLE.csv','MENU_TRAINING_ACTION_RANKING_20261022.md','PRIMARY_CLASS_GAINS_20261022.md','REPORT.md','APPENDIX_REFERENCE_AXIS.md','APPENDIX_REFERENCE_AXIS.csv','FAILURE_DECOMPOSITION.md','CONTROLLER_LEVEL_FAILURES.md','GEOMETRY_CENSUS.md','GEOMETRY_CENSUS_REVISED_20261021.md','DYNAMICS_STAGE.md','GAPS.md','HANDOFF.md','CORRECTION_GAIN_SPLITS_20261021.md','VARIANT_RESULTS_20261023.md']
 seal('FINAL_REPORTS.receipt.json',[HERE/n for n in [*outputs,'final_reports_audit_order_20261022.py','training_selected_report_tables_20261022.py']],source_audit_passed=True,db_postflight_passed=True,main_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),posthoc_revision_explicit=True)
if __name__=='__main__':main()
