from support import *
from analyze import ARM_FULL,TYPE_FULL
import collections
G=guard()
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:.2f}'
def main():
 import subprocess
 subprocess.run([sys.executable,'-B',str(HERE/'dependency_audit.py'),'END'],check=True)
 subprocess.run([sys.executable,'-B',str(HERE/'historical_teacher.py'),'compare'],check=True)
 r=read(HERE/'results.json');historical=read(HERE/'HISTORICAL_PROTOCOL_OLD_CRITIC_COMPARATOR.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');labels=np.load(HERE/'LABELS.npz');rates=(labels['labels']==1).mean((2,3));fam=labels['families'];best={f:[ARMS[i] for i in np.flatnonzero(np.isclose(rates[fam==f].mean(0),rates[fam==f].mean(0).max(),rtol=0,atol=1e-12))] for f in FAMILIES};besttext='；'.join(FULL[f]+'：'+'、'.join(ARM_FULL[a] for a in best[f]) for f in FAMILIES);ep=r['endpoints'];passed=[k for k in ['rule','critic'] if ep[k]['status']=='PASS'];fd=r['diagnostic_distributions'];dmmin=fd['DM']['freeze_proportion']['min'];othersmax=max(fd[f]['freeze_proportion']['max'] for f in FAMILIES if f!='DM');sep=dmmin>othersmax
 opening=f"源码完整性审计{'通过' if audit['passed'] else '未通过'}，官方共享数据库包与默认 journal 路径断言均通过。五个黑盒 controller 族的最佳固定修法分别为{besttext}。不看成败的诊断{'能够在本样本用冻结比例完全区分分布式模型预测控制与其余四族' if sep else '只能部分区分各族，冻结比例分布存在重叠'}，但原学生、联合教师和高斯行为克隆策略的特征有重叠。五族留一族评估中，规则选择器{'通过' if ep['rule']['status']=='PASS' else '未通过'}、诊断 critic 选择器{'通过' if ep['critic']['status']=='PASS' else '未通过'}“优于任何单一固定修法”的配对置信区间标准。本轮全部为 DEVELOPMENT。全仓审计未通过项为本轮未写入、未导入的multiscene_generalization_v1/test_observation.py发生外部变化；不得将本轮标为完整验收通过。"
 lines=[opening,'','完整性终检发现上述外部已有文件变化后，已停止实验流水线；未恢复、覆盖或删除该文件。全部仿真、拟合与主要统计在此之前已完成。之后仅封存已生成的证据、补充只读证据核对并生成报告。EXTERNAL_CHANGE_ATTRIBUTION.json记录起止sha256和非导入证明。实际复用目录、实际导入源码与checkpoint核对结果分别保留，不能用这些局部通过覆盖全仓一致性失败。','','本报告仅评价 controller 级六臂选择；不重训 controller，不改变物理、事件、成功标准或安全层。timeout 仅称超时；冻结与无进展是诊断，不能单独证明 deadlock。','','## 证据与复现','','46个冻结实例，32个既有评估状态，每状态两个成簇种子；六臂共17,664条。复用9,728条，补齐7,936条；未知结果共'+str(r['unknown'])+'条，主结果按成功下界计、另报上界，unknown 不称失败且未重跑。16条指定 round30e 复现门的动作、位置、速度 dtype/shape/bytes、步数、终点全部一致。736条诊断全部300步、关闭终止、不读取成败，特征整体封存在任何选择之前。','','核心执行文件逐字复制。起始全树清单覆盖PoC与本轮复用结果目录，全部既有Python源码另列哈希；实际checkpoint按已冻结的sha256在构造与交付时检查。直接依赖目录ring_exchange_stage1、controller_q_transfer_round3_v1及orthoflow3_tt_ff_matched_learnability_v1的全树清单是在运行中补充，并非原始BEGIN时采集；DEPENDENCY_BASELINE/END/COMPARISON.json明确记录这一时序，同时核查文件ctime早于本轮开始。该补充不被冒称为原始时刻的全树快照。`COPY_AUDIT.json` 和 `COPY_PATH_DIFFS.patch` 保存来源、sha256与允许的实验路径/审计目录差异。`HISTORY_AUDIT.json`、`HISTORY_REUSE.json`、`ANALYSIS_MANIFEST.json` 保存逐条复用依据与原UID。原学生采用 round30d 的有效 checkpoint 清单，未复用早期无效全局学生。高斯行为克隆旧右手规则的0.10m/s配置不冒充本轮右手规则丙的0.25m/s配置。','','## 六臂及选择结果','','成功率为controller等权，单元为成功下界。','', '| 黑盒controller族 | 只加安全层 | 全程η13 | 500步时间表 | 右手规则甲 | 右手规则乙 | 右手规则丙 | 留一族规则 | 留一族critic | 每controller事后最优 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 methods=ARMS+['rule','critic','controller_oracle']
 for f in FAMILIES:lines.append('|'+FULL[f]+'|'+ '|'.join(pct(r['policies'][k]['families'][f]['success_lower']) for k in methods)+'|')
 lines.extend(['','| 合并策略 | 成功下界—上界 | 层级95% CI | 对每controller事后最优的平均遗憾 |','|---|---:|---:|---:|'])
 policy_names={**ARM_FULL,'rule':'留一族三分支规则选择器','critic':'留一族诊断critic选择器','family_oracle':'每族事后最优固定修法','controller_oracle':'每个controller事后最优'}
 for k,p in r['policies'].items():lines.append(f"|{policy_names[k]}|{pct(p['success_lower'])}—{pct(p['success_upper'])}|[{pct(p['ci'][0])}, {pct(p['ci'][1])}]|{pp(p['controller_oracle_regret_lower'])}个百分点|")
 lines.extend(['',f"冻结通用critic的原历史协议对照覆盖46个controller：成功率{pct(historical['success_lower'])}—{pct(historical['success_upper'])}，95% CI [{pct(historical['ci'][0])}, {pct(historical['ci'][1])}]。其中14个教师沿用原联合流探测描述符，其他32个采用既有分布式描述符；这是原设计的历史协议对照，不是同一新诊断协议下的46实例评估。原菜单仍仅全程η10、全程η13、300步及500步时间表。详见HISTORICAL_PROTOCOL_OLD_CRITIC_COMPARATOR.json。",'','主要终点采用5000次配对controller→state分层成簇bootstrap，两个seed不拆开。每次重抽重新取六个固定臂的最大平均成功率，检验对任何单一固定臂的优势；不会把观察数据选出的固定臂当成先验给定。','', '| 终点 | 对最强固定臂差值（百分点） | 配对95% CI（百分点） | unknown保守包络（百分点） | 结论 |','|---|---:|---:|---:|---|'])
 for k in ['rule','critic']:
  e=ep[k];lines.append(f"|{'选择一：三分支规则选择器' if k=='rule' else '选择二：诊断critic选择器'}|{pp(e['difference'])}|[{pp(e['ci'][0])}, {pp(e['ci'][1])}]|[{pp(e['unknown_adversarial_ci'][0])}, {pp(e['unknown_adversarial_ci'][1])}]|{e['status']}|")
 lines.extend(['','最强合并单一固定臂为'+ARM_FULL[ep['rule']['best_pooled_fixed_arm']]+'。另报五族等权CI，见results.json；主终点按46个controller等权。bootstrap条件于已经冻结的选择器，不重拟合规则或critic。','','## 留一族与灾难诊断','','规则源族目标按四族等权；critic每折仅加载物理排除了留出标签的SOURCE_ONLY.npz。诊断均值标准化只拟合内部训练controller。内部留一源族四折 × 三fit seed，500到12000每500步，均值验证NLL最小的最早步重拟合；三seed平均概率选臂。每折先封存choices，再对留出标签评价。','','| 留出族 | 冻结阈值 | 第300步进展阈值（m） | 规则的右手组别 | critic选步 | 规则类型准确率 | critic类型准确率 | 规则灾难数 | critic灾难数 |','|---|---:|---:|---|---:|---:|---:|---:|---:|'])
 for f in FAMILIES:
  rule=r['folds'][f]['rule'];crit=r['folds'][f]['critic'];a=r['policies']['rule']['families'][f];b=r['policies']['critic']['families'][f];lines.append(f"|{FULL[f]}|{rule['A']}|{rule['B']}|{ARM_FULL[rule['right_hand_arm']]}|{crit['selected_step']}|{pct(a['type_accuracy'])}|{pct(b['type_accuracy'])}|{a['catastrophe_count']}|{b['catastrophe_count']}|")
 lines.extend(['','灾难定义为比该族最佳固定臂的族平均成功率低严格超过10个百分点；unknown可改变的计数范围在results.json中。类型按不修正、时间表类（含全程η13）、右手规则类划分；事后最优并列按预注册菜单顺序，允许任一并列类型的敏感性结果另报。类型准确率不能替代性能终点。','','逐controller留出选择：','', '| 留出族 | 规则选择的修法与实例数 | critic选择的修法与实例数 |','|---|---|---|'])
 for f in FAMILIES:
  row=[]
  for k in ['rule','critic']:
   cc=collections.Counter(r['folds'][f][k]['choices']);row.append('；'.join(ARM_FULL[a]+f'：{n}' for a,n in cc.items()))
  lines.append('|'+FULL[f]+'|'+'|'.join(row)+'|')
 lines.extend(['','## 无标签诊断的可分性','','下表为controller均值的范围，不能解释为任务成败标签或deadlock证明。','', '| 黑盒controller族 | 冻结比例范围 | 运动无进展机器人·秒范围 | 第300步进展范围（m） |','|---|---:|---:|---:|'])
 for f in FAMILIES:
  d=fd[f];lines.append(f"|{FULL[f]}|{pct(d['freeze_proportion']['min'])}–{pct(d['freeze_proportion']['max'])}|{d['moving_no_progress_robot_seconds']['min']:.3f}–{d['moving_no_progress_robot_seconds']['max']:.3f}|{d['progress300']['min']:.3f}–{d['progress300']['max']:.3f}|")
 lines.extend(['','分布式模型预测控制的冻结比例与其余族'+('在本样本无重叠' if sep else '有重叠')+'；运动无进展秒不是同样稳定的单一分隔特征，学习型策略内部异质性明显。较大第300步进展也不保证最终成功，不能把尚未到达目标的短期进展当成“无需修正”的充分证据。完整七特征分布和单一特征配对AUC见图与results.json。','','主运动无进展定义按prompt字面包含已到目标机器人；未到目标条件版本另保留，与round30g的定义对照。冻结主分母固定为1200机器人·步。混杂比例沿用累计角位移±0.15rad签名，在300个执行后时刻中取mixed比例。','','## 旧通用critic与缺口','','原通用critic只使用其原F10/F13/S300/S500菜单，不能选择不修正或任何右手规则。有效原学生14个与上一轮三个新族18个共32个保留原R5及预测；另报限制到全程η13与500步时间表的交集预测。联合教师在本分布式安全探测协议下没有兼容的既有R5/冻结预测，未补探测、未填值；旧DEVELOPMENT_TEACHER_CHOICES的探测执行身份不同，不能冒充兼容结果。协议兼容子集仅32个；另保留14个教师的原联合流R5与既有冻结选择（RETAINED_TEACHER_HISTORICAL_R5.npz及收据），构成46实例原历史协议描述性对照。该补充在任何新选择器之前封存，没有新探测或新预测，不输入新选择器，也不冒充精确D-local描述符。原菜单选择与逐实例成功上下界见FROZEN_OLD_CRITIC_REUSE.json和results.json。全程η10与前300步时间表的已有标签另存DESCRIPTIVE_LEGACY_ARMS.csv/json，仅描述；A*先前排除三实例的这些菜单外标签缺失不补。','','## 结论与局限',''])
 if passed:lines.append('至少一个开发终点通过，支持“短期无标签诊断可以为controller选择修法”作为候选主方法。下一轮应冻结本轮流程，在五族新实例上做前瞻确认。此处“零样本”只指留出族标签不参与本轮选择器拟合；右手规则菜单已用过族相关历史结果，且当前标签早已开放，不能称为完全未知族或前瞻确认。')
 else:lines.append('两个性能终点均未通过，当前证据不能支持选择器稳定优于最佳单一固定修法。即使冻结比例能揭示模型预测控制的特征，也不保证源族学出的阈值与critic在留出族上稳定迁移；应依据各折类型准确率、灾难与遗憾分析错误，增加族数或改善探测，然后重新预注册新实例验证。')
 lines.extend(['','右手规则甲、乙、丙的参数分别从历史模型预测控制、A*规划和学生调参中冻结；这使菜单携带族信息。本轮没有根据留出结果更改菜单、阈值网格、诊断或critic设置。原学生与联合教师一一对应，五族并非完全独立的模型谱系；留一族不会隔离这层历史关联。固定状态、Ring场景、46个实例与短诊断的范围限制不能外推到新地图、新机器人数量或任意黑盒。','','## 文件入口','','- PREREGISTRATION.md及receipt：参数与时序；REPLAY_GATE.json：16条逐位复现。','- DIAGNOSTIC_FEATURES.csv/npz、DIAGNOSTICS_FROZEN.receipt.json：诊断及冻结。','- CONTROLLER_SIX_ARM_TABLE.csv：每个controller×六臂成功上下界、来源轮次。','- ALL_CACHE_PREFLIGHT.json / ALL_CACHE_POSTFLIGHT.json、registration/、supplement_MERGE_RECEIPT.json：官方DB流程。','- folds/：每折源标签、网格、内部CV、checkpoint、预测与选择收据。','- results.json / BOOTSTRAP_DRAWS.npz：终点及统计证据。','- figures/：诊断分布、总体与留出族成功率，PNG/PDF/SVG。','- INTEGRITY_BEGIN.json / END.json / COMPARISON.json、COPY_AUDIT.json、IMPORT_SOURCE_AUDIT.json：完整性。'])
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 handoff='''# 分布式黑盒修法选择交接（DEVELOPMENT）

目标：冻结五族46个黑盒，在无标签300步安全诊断后，对六种固定修法做五折留一族controller级选择。用户规范为 docs/prompt_round30h_decentralized_fix_selection_20261017.md；全部新实现和产物仅在本目录。任何历史文件、PoC、官方MACFlow仓库与Python环境均只读；DB仅官方登记、默认journal与单进程merger追加。

'''+opening+'''

阅读顺序：REPORT.md → PREREGISTRATION.md → results.json → 各fold的RULE_SELECTION/STEP_SELECTION/CRITIC_SELECTION/CHOICES_FROZEN.receipt.json。CONTROLLER_SIX_ARM_TABLE.csv逐格注明历史来源，ANALYSIS_MANIFEST.json逐条含UID、raw与trace路径和sha256。历史学生使用本目录STUDENTS.json指定的有效原checkpoint；不使用round30f新学生。

复用9728条，补7936条，共17664条。诊断736条全部独立存储，不写任务DB。旧R5缺失不填，旧通用critic有32实例分布式探测兼容证据；14教师另保留原联合流R5和冻结选择作历史协议描述性对照。unknown保留且不重跑；timeout不是deadlock。三个右手规则参数曾用历史族结果选定，因此菜单携带族信息；这轮仅是DEVELOPMENT。

流水线脚本：support.py建立冻结身份；integrity.py BEGIN/END；design_new.py历史逐格审计、登记与preflight；replay_gate.py 16条复现；submit_new.py/run.py生成诊断与缺失任务；freeze_diagnostics.py封存全部特征；postflight.py调用官方merger；pack_labels.py生成各折SOURCE_ONLY文件；fit_selector.py仅拟合源族规则和critic；analyze.py在choice冻结后评价；plots.py只读结果画科学图；report.py整理文档；verify_delivery.py核验交付。复制的worker.py、pipeline.py及右手规则核心执行逐字不改，新的support.runtime仅预载原学生/教师。

本轮仿真与分析已完成，但全仓一致性审计因外部未导入文件变化未通过，不能视为完整验收通过；本轮已停止实验流水线。不应重新提交rollout或重跑unknown，也不得恢复外部文件来使审计表面通过。确需新实验先查共享DB并重新预注册。复核只需读取收据并运行verify_delivery.py（它不提交作业，不运行仿真，不修改历史或DB）。JAX精度False；所有入口先加载并断言官方DB路径。Slurm提交前记录squeue；本轮最多18 CPU shard、未使用GPU。

单进程DB合并的命令、ps/squeue审计保存在supplement_MERGER_PREFLIGHT.json和supplement_MERGE.log；登记前后旧行哈希及新登记行在registration/supplement/。INTEGRITY_COMPARISON.json列出并行任务外部新增项，IMPORT_SOURCE_AUDIT.json复核所有实际import源文件。最终交付检查见FINAL_VERIFICATION.json和DELIVERY.receipt.json。
'''
 (HERE/'HANDOFF.md').write_text(handoff);print('REPORT and HANDOFF written',flush=True)
if __name__=='__main__':main()
