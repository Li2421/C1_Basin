"""Finish paired comparisons, required paragraph order and standalone handoff."""
from experiment import *
from atomic_write import write
import ast,difflib
files=[Path(__file__)];sources=[]
def output(old,new,after):
 before=(HERE/old).read_text();ast.parse(after);target=HERE/new;assert not target.exists();target.write_text(after);diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)));files.extend([target,diff]);sources.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
old='final_reports_correction_revised_20261021.py';s=(HERE/old).read_text();s=s.replace('LAYERS,pick,geometry,dynamics','LAYERS,pick,geometry,dynamics,comparison')
helper='''def handoff_text():
 entry=read(HERE/'ACTIVE_ENTRYPOINT.json');policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json')
 return f"""本轮为四个文献场景分布式安全层迁移v2，原修正语义预注册证据与看过结果后提出的修订开发证据分开。工作目录{HERE}；v1目录{V1}只读。唯有V2_COMPLETE.json表示实际完成；不存在时继续按STATUS与日志处理，不把后台仍跑写成完成。v1三场景收尾已冻结，圆形流程版缺口不恢复，论文版结果沿用，不冒充v1四场景合并。

当前唯一入口和进程信息以ACTIVE_ENTRYPOINT.json、COORDINATOR_PID.json为准：协调器{entry['coordinator']}，下游{entry['downstream']}，可选阶段{entry['optional']}，最终报告{entry['final_reports']}。启动前核对现存PID/命令及CORRECTION_COORDINATOR.lock，不能并发启动第二协调器；下游也有独立锁。已运行child期间不要更改下游配置。只选择本轮目录内源码哈希列在声明READY收据中的静态入口；不在运行时改模块或改封存文件。历史部署文字里的入口可能已过时。

流程Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B；最后画图才用.venv-c1-plots。禁止安装包、写v1/外部源码或做git commit。experiment.py将TMP/MPL/XDG/CUDA缓存限制于本轮且禁用bytecode。所有已封存raw、trace、checkpoint、配置和源文件不修改，修订用新文件并记录来源SHA。

分布式安全层在distributed_safety.LocalSafety：各机器人独立解二维投影，不输入其他机器人的名义或安全速度。配对行2(pi-pj)·ui>=-c*gamma*h，窄通道/拥挤随机/圆形主线c=1/2，门口c=1；墙全责，速度球和同问题数值重试不变。集中式保持原联合投影及原OrthoFlow3。构造snapshot和实际约束由projection_adapter.constraints/LocalSafety完成；runtime_correction_revised_20261021.run执行分布式与拥挤集中式，central_correction_revised_20261021.run执行其他集中式。

调用顺序仍为冻结名义控制器→原速度界和到达停止→第一次安全投影→三个原修正基→第二次同问题安全投影→原物理。correction_revision_20261021只把Brel在无接触冲突的机器人行置零；Brel启用时仍对全部其他机器人先bound再平均，不能改成仅近邻平均。接触为距离<=Rc且两者不同时在原到达容差内，Rc=1.25*d_safe，d_safe含分离buffer，第二阶段门口含前视膨胀半径；具体数值CORRECTION_REVISION.md在首条研究运行前固定，不据结果调整。到达位每步广播1bit，其他通信假设保持原样。双η主线使用平均进度及原6/7时限回退；逐机器人进度只作为非主线变体。

48个名义controller=8类型×6冻结实例，不训练这些controller；各场景能力门合格数44/44/45/45。旧12与新15指修正动作菜单，新菜单补η10、η13和人定双η。物理输入12维、动作编码19维、模型架构、36个source-type fit集合、三初始化17/23/41、2000训练步及嵌套留一类型选择不变。新模型路径critic_fits_revised_20261021/，选择LEARNED_EVAL_CHOICES_REVISED_20261021_<scene>在首条评估前封存。集中式修订学习不运行。原η0轨迹32条完整逐位回放通过后全部按精确执行身份复用；修订非零用新身份，同语义的三个固定训练动作和冻结后选到的现有执行直接复用。

顺序严格为D修订固定4场景→D修订学习窄通道/拥挤/门口→C修订固定4场景→圆形D修订学习→全部保存轨迹诊断和原/新合并主数值→第二阶段与变体→图、数据库和源码结束审计、报告。前一研究阶段全结束才进后一阶段；空核可做已完成轨迹的可让出诊断。统计27独立任务调用原数学内核，5000次bootstrap、原种子、类型→实例→state等权、seed配对和unknown识别界不变；source AST、5000抽样数组、串并行类别增益一致的证据在CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json。

每项新研究前先查HISTORY_CHECK及现存配置/数据/checkpoint/预测，遵守共享数据库README的官方preflight，只跑genuinely_missing。registry/revised_case_factory负责身份与注册；worker只通过官方cache_writer.append_journal写默认实验exp_baseline_literature_scenes_distributed_v2，不能写SQLite。唯一merger由db_merge.py包装官方程序并持锁。原圆形学习2044条保留且已合并，4100条停止不恢复，旧评估1536不运行；这项用户停止不产生unknown。数值求解失败的unknown独立保留，不能改成timeout/deadlock或默默重跑覆盖。

性能加速为已验证静态必要条件证书和原生精确bit-key几何缓存，科学语义/RNG/事件不变；CBS_FLAT_CACHE_EQUIVALENCE等收据含完整轨迹逐位回放。未采用的候选也保留证据，不能随意启用。实例类以INSTANCE_CLASSES_REVISED为主：失败上界<=5%后用未到达机器人·局的4a′/4a″多数判断，只去掉controller级失败，安全层全局失败保留，墙前陷阱单列。旧CONTROLLER_LEVEL_FAILURES.json保留作历史核对，不能替代新归类。timeout本身不是死锁。目标附近徘徊为超时且所有未到达机器人距目标<=3倍原容差，至少一台未到达；两种语义、两层和实际第二阶段分别报告。图只读封存仿真证据。

资源最新用户授权：当前总24CPU（23调度+1本地），{policy['release_at_local']}降为18CPU（17调度+1本地），GPU保持2份额。实际限额读resources_corrections_release_20261023.limits与RESOURCE_RELEASE_STATUS，ACTIVE里的部署初值不能覆盖日期政策。07:30开始超额CPU worker逐局收尾，09:00准时释放；优先纯CPU任务，不打断GPU critic fits。独立CPU释放guard及watchdog持久运行。每次提交先squeue计全用户含pending；本地进程由local_cpu_guard_idle_cpu绑定未被Slurm分配的单核，不与worker抢同一物理CPU。

主数值认MIGRATION_NUMBERS、后验指标认原/新MAIN_ADDITIONAL_METRICS、可选数值认OPTIONAL_NUMBERS、数据库认DB_POSTFLIGHT、源码只认INTEGRITY_END、交付认FINAL_REPORTS。全部实际作业结束后才做最终审计和完成标志；可选缺口独立披露，不混入主线或算unknown。v1原两个非Python配置缺开工全文件哈希的限制仍保留，不能倒补为严格覆盖。"""

'''
a='def main():\n';assert a in s;s=s.replace(a,helper+a+" verify_receipt('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json')\n",1)
a=" paragraphs.append(' '.join(classification)"
b=""" for scene in NAMES:
  contrast=comparison(results[scene]['fixed_full'],'safety').get('distributed_minus_central')
  classification.append(NAMES[scene]+'只加安全投影的分布式−集中式配对成功差：'+delta(contrast)+'。')
 paragraphs.append(' '.join(classification)"""
assert a in s;s=s.replace(a,b)
a=" paragraphs.append(' '.join(gains)+"
b=""" paired_gains=[]
 for scene in NAMES:
  for sem in SEM:
   entry=results[scene] if sem=='original' else results[scene]['revised']
   for method in ('fixed_eta10','fixed_eta13','human_progress'):
    contrast=comparison(entry['fixed_full'],method).get('correction_gain_difference')
    paired_gains.append(NAMES[scene]+'、'+SEM[sem]+'、'+METHODS[method]+'的修正增益（分布式−集中式）配对差'+delta(contrast)+'。')
 paragraphs.append(' '.join(gains)+' '.join(paired_gains)+"""
assert a in s;s=s.replace(a,b)
s=s.replace('修订只给原Brel加接触开关','修订只给按全部其他机器人平均得到的原Brel加接触开关')
a=" paragraphs.append('第二阶段各场景完成情况";start=s.index(a);stop=s.index('\n',start)
replacement=''' stage2=[]
 for scene in NAMES:
  entry=optional['results'].get('optional_dynamics_'+scene,{})
  estimates=[METHODS[row['method']]+' '+cell(row) for row in entry.get('statistics',{}).get('statistics',[]) if row['type']=='type_equal']
  stage2.append(NAMES[scene]+'：'+('; '.join(estimates) if estimates else 'N/A：'+str(entry.get('reason','未取得完整第二阶段证据')))+'，unknown '+str(entry.get('counts',{}).get('unknown','N/A')))
 paragraphs.insert(3,'第二阶段（非主线修订开发证据）：'+'；'.join(stage2)+'。配对参照是修订后的第一阶段，差异、约束和速度违反、新实例类、类别增益与徘徊率见DYNAMICS_STAGE.md、OPTIONAL_RESULTS.json及INSTANCE_CLASSES_DYNAMICS_REVISED_20261021.json。没有额外集中式动力学执行，联合锥只是同动力学参照；不能把controller级增益计为死锁解决。')
 variant_rows=[]
 for stage,entry in optional['results'].items():
  if not stage.startswith(('optional_full_circle_','optional_local_progress_','optional_nominal_')):continue
  for row in entry.get('statistics',{}).get('statistics',[]):
   if row['type']=='type_equal':variant_rows.append([stage,METHODS.get(row['method'],row['method']),cell(row),entry.get('status')])
  if entry.get('background_only'):variant_rows.append([stage,'二维拥挤随机无安全层背景','collision='+str(entry['counts']['collision'])+' / completed='+str(entry['counts']['completed']),entry.get('status')])
 save('VARIANT_RESULTS_20261023.md','第二优先级非主线。圆形全责与个人进度采用修订Brel接触开关，名义无安全背景保持原语义。相对主线的配对差和无安全碰撞→安全结果转移的完整计数见OPTIONAL_RESULTS.json，不与四场景主线混合。\\n\\n'+table(['定位stage','做法','成功率或原始背景计数','状态'],variant_rows))
 paragraphs.append('第二优先级变体与无安全层背景单列于VARIANT_RESULTS_20261023.md；原始碰撞到安全结果的转移、个人进度−广播进度、全责−互惠责任的配对结果见OPTIONAL_RESULTS.json。图只读取封存证据，不改变控制器、物理、事件或成功标准。')'''
s=s[:start]+replacement+s[stop:]
start=s.index(" save('HANDOFF.md',f\"");stop=s.index('\n',start);s=s[:start]+" save('HANDOFF.md',handoff_text())"+s[stop:]
s=s.replace("'CORRECTION_GAIN_SPLITS_20261021.md']","'CORRECTION_GAIN_SPLITS_20261021.md','VARIANT_RESULTS_20261023.md']")
s=s.replace("'final_reports_correction_revised_20261021.py'","'final_reports_correction_delivery_20261023.py'")
output(old,'final_reports_correction_delivery_20261023.py',s)
old='optional_delivery_correction_revised_20261021.py';s=(HERE/old).read_text().replace('final_reports_correction_revised_20261021.py','final_reports_correction_delivery_20261023.py');s=s.replace('def main():\n',"def main():\n    verify_receipt('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json')\n",1)
output(old,'optional_delivery_correction_delivery_20261023.py',s)
old='correction_after_distributed_fixed_parallel_20261023.py';s=(HERE/old).read_text().replace('CORRECTION_OPTIONAL_READY_20261021.receipt.json','CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json').replace('optional_delivery_correction_revised_20261021.py','optional_delivery_correction_delivery_20261023.py');s=s.replace('def main():\n',"def main():\n verify_receipt('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json')\n",1)
output(old,'correction_after_distributed_fixed_delivery_20261023.py',s)
write(HERE/'FINAL_CORRECTION_DELIVERY_SOURCE_DERIVATIONS_20261023.json',dict(utc=utc(),changes=sources,scientific_and_statistical_computations_unchanged=True,report_fixes=['Explicit baselineD-C and correction gainD-C paired confidence intervals','Second-stage results in fourth paragraph','Standalone current-source handoff with construction,calling and reuse instructions','Supplemental variants separately reported'],same_native_numeric_sources=True))
files += [HERE/'FINAL_CORRECTION_DELIVERY_SOURCE_DERIVATIONS_20261023.json',HERE/'CORRECTION_OPTIONAL_READY_20261021.receipt.json',HERE/'CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json',HERE/'CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json']
seal('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json',files,reporting_and_entrypoint_only=True,no_scientific_worker_or_request_changes=True,no_new_rollouts=True,full_native_paired_contrasts_included=True,standalone_handoff=True)
print('Final reporting and standalone handoff corrections sealed',flush=True)
