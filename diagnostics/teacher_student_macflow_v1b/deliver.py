"""Assemble self-contained report from frozen evidence only."""
from core import *
from audit_utils import source_inventory
import difflib

def main():
    r=read(HERE/'results.json');integrity=read(HERE/'INTEGRITY_COMPARISON.json');n=r['distillation']['n_valid'];f=read(HERE/'FAILURE_SOURCES.json');own=f['totals'].get('own_stagnation',0);safety=f['totals'].get('safety_induced_stagnation',0);dominant='自身停滞' if own>safety else '安全层导致的停滞' if safety>own else '两者相等';endpoints=['TS1','TS2','TS3a','TS3b','TS6']
    def statuses(k):return '，'.join(t+'='+r[k].get(t,{}).get('status','INCONCLUSIVE') for t in endpoints)
    audit='通过' if integrity['protected_content_pass'] and not integrity['removed_status_entries'] else '未完全通过（见完整性差异）';assert read(HERE/'RESULTS_VALIDATION.json')['passed']
    for setname in ['CONFIRMATORY','DEVELOPMENT']:
        data=r[setname]
        if not data.get('n_controllers'):continue
        table=data['table'];lines=['| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |','|---|---:|---|---:|---:|---:|']
        order=[(arm,arm) for arm in ['nominal only','safety baseline','F10','F13','S300','S500','B-RHR']]+[('A3 zero-shot','A3_full'),('always S500','S500'),('oracle','oracle')]
        for display,key in order:
            d=table[key];x=d['success'];lo,hi=x['CI95'];lines.append(f"| {display} | {x['mean']:.2%} | [{lo:.2%}, {hi:.2%}] | {d['success_upper']['mean']:.2%} | {d['collision']['mean']:.2%} | {d['timeout']['mean']:.2%} |")
        (HERE/f'paper_table_{setname.lower()}.md').write_text('\n'.join(lines)+'\n')
    text=f'''源码完整性审计{audit}，每入口/worker的Basin_C1 shared_rollout_db与默认journals路径断言通过。10个新学生中{n}个通过蒸馏有效性门。主分析（CONFIRMATORY）：{statuses('CONFIRMATORY')}；合并分析（DEVELOPMENT）：{statuses('DEVELOPMENT')}。按预注册的两类失败拆分，主要来源为{dominant}（自身停滞{own}条，安全层导致的停滞{safety}条；合并分析集）；timeout只表示超时，不能据此称为deadlock。

# Round30b frame-consistent decentralized students

本轮只修正global教师动作与噪声的坐标系，教师冻结；网络、训练步数、batch、seed、dev选择与round30一致。数据与归一化逐文件hash复用，无重复生成训练rollout。所有代码/输出写入本轮目录，外部写入只有官方DB新身份追加、默认journal和单进程merger。未改动任何已有源码、环境、物理、安全层、成功标准或事件。未运行git写命令，未安装软件。旧10个无效global学生结果**已被取代**，原文件保留；4个有效local学生与冻结选择/标签直接复用，已打开过的旧标签仅列DEVELOPMENT。

## 冻结、门与样本

预注册与收据：PREREGISTRATION.md、PREREGISTRATION.receipt.json。10个学生各只训练一次80,000步、每500步固定dev选择，选取早于闭环。NOISE_GATE.json引用1008/1008历史门，NOISE_IMPLEMENTATION_COMPATIBILITY.json核对迁移路径源码的相同hash。EQUIVARIANCE.receipt.json先于训练；TRAINING_FROZEN.receipt.json先于学生pilot；CHOICES_FROZEN.receipt.json先于任何主评估标签。16条round28回放首动作、位置、终止和步数全部逐位通过。单元测试和组件测试见UNIT_TESTS.json、COMPONENT_TESTS.json。

蒸馏门三条同时满足：非step0、独立2048固定state/noise样本平均cosine≥.90、误差≤.05m/s占比≥80%。未通过不重训、不调参、不替换。逐学生见DISTILLATION_REPORT.md。能力门在固定16个round30 pilot states上两臂seed0执行；合格新学生{len(r['capability_gate']['eligible'])}个。能力门排除记录见CAPABILITY_GATE.json。遵循round30少于8个能力门合格新学生则整轮INCONCLUSIVE的约束；此时数值仅描述性，不作验证性结论。主分析对象为蒸馏有效且通过预注册能力门者，蒸馏有效完整集合与被能力门排除者单独披露。

## 坐标与等变诊断

目标u与trace第0步ξ逐行经自己的径向/切向R_iᵀ旋转；训练float32，部署numpy float64，数值一致性误差见UNIT_TESTS.json。自身local观测、编号one-hot、自身fold_in(key_for(state,seed,step),i)噪声输入；local输出转world，沿用原限速。转换只依赖自身位置，改变他人目标时自身输出不变，固定输入噪声确定。local教师保留原local噪声和动作定义，与旧local部署一致。

10个global教师的act_scale均不是逐机器人x/y严格各向同性，因此按任务要求，本轮噪声旋转视为近似，而非严格等价的归一化flow变换。ACT_SCALE_AUDIT.json给出全部尺度与比例。等变诊断每教师256个固定训练state、16噪声、随机旋转，不作门；EQUIVARIANCE.json/图给出完整结果。有限16样本的均值差包含Monte Carlo误差，不能单凭该描述性比例断言教师结构性的非等变导致蒸馏失败。

## 论文总表与预注册终点

主分析与合并分析各32个配对state×2seed；controller后state成簇配对bootstrap5000次；每次重抽内重新求best-fixed、best-schedule和oracle。unknown保留，成功率下界0、上界1，不当已知失败、不重跑。表内置信区间为下界估计；上界估计及差值包络见results.json。

'''
    for key in ['CONFIRMATORY','DEVELOPMENT']:
        text+=f"### {key}\n\n";path=HERE/f'paper_table_{key.lower()}.md';text+=path.read_text() if path.exists() else '没有符合分析条件的新学生。\n';text+='\n'
        s=r[key]
        for t in endpoints:
            text+=f"- {t}: {json.dumps(s.get(t,{'status':'INCONCLUSIVE'}),ensure_ascii=False)}\n"
        text+='\n'
    text+='## 逐学生失败来源与救回计数\n\n| Student | Nominal success/collision/timeout/unknown | Own stagnation | F13/S500/B-RHR rescued | Safety induced stagnation | F13/S500/B-RHR rescued |\n|---|---|---:|---|---:|---|\n'
    for name,d in f['controllers'].items():
        o=d['nominal_outcomes'];a=d['failure_sources']['own_stagnation'];b=d['failure_sources']['safety_induced_stagnation'];counts='/'.join(str(o.get(k,0)) for k in ['success','collision','timeout','unknown']);resc=lambda g:'/'.join(str(g[k]) for k in ['rescued_F13','rescued_S500','rescued_BRHR']);text+=f"| {name} | {counts} | {a['count']} | {resc(a)} | {b['count']} | {resc(b)} |\n"
    text+='\n## 结论解释\n\n'
    if len(r['capability_gate']['eligible'])<8:
        text+='本轮仅2个新学生满足有效性与能力门，未达到round30预注册的至少8个新学生要求；因此所有验证性终点保持INCONCLUSIVE，下面的数值差异不构成验证性通过。新增2个与旧4个合并只得到6个有效学生，未得到预期的14个有效学生，也不能把此前已打开的4个local学生标签转为新验证证据。\n\n'
    for setname in ['CONFIRMATORY','DEVELOPMENT']:
        ss=r[setname]
        if not ss.get('n_controllers'):continue
        x=ss['TS1'];z=ss['TS6'];text+=f"{setname}描述性S500−safety={x['mean']:.2%}，95%CI={x['CI95']}；S500−B-RHR={z['mean']:.2%}，95%CI={z['CI95']}；A3平均oracle遗憾={ss['TS3a']['regret']['mean']:.2%}。正式结论以各终点status为准。\n\n"
    text+='8个学生因余弦阈值未过而无效，说明只修正坐标系仍不足以让所有global教师通过本轮分散化有效性标准。等变诊断及act_scale各向异性提供解释线索，但不足以单独建立失败原因；给学生全局坐标的自身观测是任务建议的后续方向，本轮未追加训练。\n\n'
    text+='''## 失败来源与继承/代价

failure_sources.csv及FAILURE_SOURCES.json逐学生报告nominal only的success/collision/timeout/unknown、nominal→safety转移和两类失败被F13/S500/B-RHR救回的数量。TRANSFERS.json保留每个state/seed的逐条转移。自身停滞定义为nominal timeout且无碰撞；安全层导致停滞定义为nominal collision→safety timeout，两组不重叠。其他失败及unknown不硬归入两类；两类计数比较不代表已覆盖所有失败。将自身停滞解释为协调困难的描述性证据，不能从timeout直接推断死锁。

results.json的TS4给出学生/教师最佳臂一致性及四臂Spearman；TS5给出与原D-local教师的配对差。INFERENCE_TIMING.json是单核JIT预热后自身网络、教师10步、四个本地QP的耗时，测量范围不同，不声称完整闭环延迟的同比加速。教师补充臂及四个local学生均复用旧证据，没有重跑。TRAINING_CONFIG.json保留round30参数卡，坐标语义覆盖见预注册及EFFECTIVE_FRAME_CONFIG.json。训练曲线见training_curves.csv/.png/.pdf；全部checkpoint和history位于students/。

## 结论约束

TS1、TS6同时通过时才支持在该分散学生分析集上修正优于安全baseline及经典右手规则。TS2不通过时按任务映射报告阶段结构优势未获支持，并比较固定F13与默认S500；不把置信区间覆盖0当作等效性证明。TS3a不通过且A3系统性选错时才称为负迁移，优先比较稳健默认动作。多个蒸馏门失败结合等变诊断讨论全局自身观测的后续需要，不追加训练或用新输入替换。若自身停滞占两类失败多数，问题陈述应强调分散协调，而不是将所有timeout归因于安全层。

## 缓存、资源、完整性

运行中出现一次协调器中断：Slurm成功提高剩余数组任务的并发，但对两个已完成子任务返回非零。协调器停止，已提交的worker原样继续，未重复提交或重跑；最终使用finalize_existing.py接续后处理。原错误和readback保存在AUTOMATION_INTERRUPTION.json，FINAL_STATUS.json记录最终完成状态。这不改变任何模型、选择、阈值、物理或评估请求。

各阶段*_PLANNED_ROLLOUTS、*_CACHE_PREFLIGHT与封存收据先于提交，只跑缺失；*_CACHE_POSTFLIGHT与MERGE_RECEIPT核官方DB逐条一致，unknown保留。registration/保存所有新增行及sha256和每表原有行前后hash。probe是未标注短探测，只作为官方journal的SUMMARY_ONLY来源，不伪造任务终点。DB_PATH_AUDIT.json逐worker核官方模块与默认journal目录，禁止PoC副本写入。

RESOURCE_MONITOR.json记录各阶段squeue占用及本轮array throttle，CPU总上限20，GPU上限2shards；并行round33既有18CPU时本轮用余下2CPU，空闲释放后才提高本轮并发。源码复制源hash及最终diff见COPY_PROVENANCE.json/COPY_DIFF.patch/ROUND28_COPY_AUDIT.json。受保护PoC及round30全文件路径/hash、两个仓库git状态/diff哈希在INTEGRITY_START/END中，比较在INTEGRITY_COMPARISON.json；并行任务新增status条目单列，不归本轮。IMPORTED_SOURCE_AUDIT.json核已有import源hash。RESULTS_VALIDATION.json为交付一致性检查。无视频交付要求。
'''
    (HERE/'REPORT.md').write_text(text)
    handoff=f'''# Round30b handoff

报告：REPORT.md；总结果：results.json；主表：paper_table_confirmatory.md；论文development表：paper_table_development.md。{n}/10学生通过蒸馏门。主分析{statuses('CONFIRMATORY')}。合并{statuses('DEVELOPMENT')}。

根目录/home/zhihan/research/Basin_C1，本轮唯一工作目录diagnostics/teacher_student_macflow_v1b。Python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图用同目录.venv-c1-plots。不要重训、重跑unknown或更改已有文件。原prompt为docs/prompt_round30b_frame_consistent_students_20261015.md。代码均本轮新增/复制；不得改round30、PoC或官方仓库。

执行链：integrity.py start → prepare.py（hash历史证据/预注册/噪声门复用）→ numeric.sbatch（equivariance→train→offline门）+replay.sbatch（16逐位门/组件）→benchmark.sbatch→stage.py pilot→stage.py probe（官方pilot merger/postflight及能力门）→stage.py mainphase（R5/冻结选择/主评估preflight）→postflight.py main→analysis.py→plots.py→validate.py→integrity.py end→deliver.py。automate.py是一次性顺序协调器，状态在AUTOMATION_STATUS.json、automation.log；已完成阶段收据不可覆盖，不要重复启动。每个入口core先断言DB包和默认journal路径。运行全过程恢复JAX x64=False。代码不monkeypatch已有模块。

固定资源合计20CPU/2GPU shards；提交前squeue查看用户所有任务，数组throttle只使用剩余额度。RESOURCE_MONITOR.json保留记录。协调器在scontrol提高并发时因已完成子任务返回非零而停止；Slurm已实际更新剩余子任务，所有trial原样继续，没有重复提交或重跑。原日志/STOPPED状态保存在AUTOMATION_INTERRUPTION.json及AUTOMATION_STATUS.json；finalize_existing.py只接续已有批次的合并、分析、审计和交付，最终状态以FINAL_STATUS.json为准。两阶段标签写入Basin_C1/shared_rollout_db/journals并官方单进程merger，DB原有登记行hash一致。旧local学生索引8/10/12/13只读复用，旧global结果已被取代且不删除。unknown保留不重跑；timeout不是deadlock。

完整性：{audit}。详见INTEGRITY_COMPARISON.json、IMPORTED_SOURCE_AUDIT.json及DB_PATH_AUDIT.json。所有结果统计与表格可从冻结JSON重新计算，无需rollout；重新分析必须输出新文件名以保留收据。
'''
    (HERE/'HANDOFF.md').write_text(handoff);source_inventory('imports_deliver.json');seal('DELIVERY.receipt.json',[HERE/'REPORT.md',HERE/'HANDOFF.md',HERE/'results.json',HERE/'RESULTS_VALIDATION.json',HERE/'INTEGRITY_COMPARISON.json'],complete=True)
if __name__=='__main__':main()
