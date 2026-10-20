"""Refresh provisional delivery documents without opening locked arm outcomes."""
from pathlib import Path
import csv,datetime,hashlib,json

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (HERE/'CONFIRMATORY_RESULTS.json').exists(),'Use final reporting after confirmation'
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    local=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    phases={}
    for stage in ('old_probe','old_evaluation','old_tuning_v2','new_probe','new_evaluation','new_tuning'):
        phases[stage]=dict(new_requests=len(read(f'{stage}_MISSING_CASES.json')),
            raw_records=len(list((HERE/stage/'raw').glob('*.json'))),
            completed=(HERE/f'{stage}_COMPLETE.receipt.json').exists())
    cap={r['controller']:r for cohort in ('DEVELOPMENT','LOCKED') for r in read(f'{cohort}_CAPABILITY.json')['controllers']}
    with (HERE/'ZOO_MANIFEST.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        name=row['controller'];folder=HERE/'policies'/name
        if row['type_code']=='implicit':
            row.update(status='IMPLEMENTATION_GAP; UNCHANGED_TWO_MODE_GATE_NOT_MET',capability='NOT_RUN')
        elif name in cap:
            record=cap[name];row['capability']=record['status']
            if folder.exists() and (folder/'best.msgpack').exists():
                row.update(checkpoint=str(folder/'best.msgpack'),checkpoint_sha256=sha(folder/'best.msgpack'),status='TRAINED; CAPABILITY_'+record['status'])
    with (HERE/'ZOO_MANIFEST.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    status=dict(utc=utc,status='COLLECTING_DEVELOPMENT_EVIDENCE',complete=False,stage_integrity_passed=True,
        final_integrity_audit_pending=True,development_types=8,development_eligible_instances=56,
        new_development_trained_and_eligible=18,locked_ppo_trained_and_eligible=6,ppo_robot_steps_per_instance=20000000,
        historical_instances=38,probe_requests_complete=2688,unavailable_probe_feature_vectors=2,
        locked_eval_or_tuning_labels_opened=False,frozen_pipeline_exists=(HERE/'FROZEN_PIPELINE.json').exists(),
        development_targets='NOT_EVALUATED',scale_curve='NOT_EVALUATED',locked_endpoints='NOT_EVALUATED',
        old_type_new_physical_trajectory_cap=18846,development_extra_label_budget_used=0,
        cpu_shard_limit=18,gpu_shard_limit=5,resource_priority='Existing tasks first',phases=phases,
        implicit_status='Original and three smooth-activation implementations fail unchanged synthetic gate; no Ring training',
        supervisor='advance_evidence.py',supervisor_scope='Collect and assemble development evidence; cannot freeze pipeline or open locked evaluation')
    (HERE/'results.json').write_text(json.dumps(status,indent=2)+'\n')
    counts='\n'.join(f'- `{name}`：{v["raw_records"]:,}/{v["new_requests"]:,} 条需补证据已有原始记录，阶段完成={v["completed"]}。' for name,v in phases.items())
    (HERE/'REPORT.md').write_text(f'''# 跨 controller type 的实现：进行中

阶段完整性审计通过：8,165 个受保护 Python 文件、95 项开工导入来源、两个仓库的 HEAD/状态/差异哈希未变化；禁止与并行目录仅检查 metadata。整轮结束审计待全部工作完成后执行。开发目标、规模曲线及锁定终点均尚未评估，不能宣称跨类型选择已成功。参数共享独立近端策略优化强化学习策略（ring_ippo）六个实例训练及能力门已完成；隐式能量行为克隆策略（implicit）在保留原判据的多项诊断后仍未通过两峰合成门，记为实现缺口。

更新于 {local}（Asia/Singapore）。开发动物园已有八类、56 个合格实例：历史 38 个不重训，新增动作分块行为克隆策略（chunking）、循环网络行为克隆策略（recurrent）、条件变分自编码器行为克隆策略（cvae）各六个，均按既定配方训练并通过能力门。强化学习策略各完成 2,000 万机器人步，按 32 个训练留出状态的只执行动作成功率选择 checkpoint；未读取锁定修正臂成功率。

32/32 条历史复现逐位通过。六套坐标系、策略基础单元测试和训练环境等价性通过；float64 位置最大误差 3.11e-15、扫掠间距误差 4.44e-15，两种精度各 256 条终止类型与步数均一致。原 float32 附加观测检查混入累计轨迹漂移，保留原失败报告并用同一历史状态的无物理观测检查修正，见 `TEST_HARNESS_AMENDMENT_20261018.json`。

全部 2,688 条开发探测请求已完成，封存特征与图为 `DEVELOPMENT_FEATURES.npz`、`DEVELOPMENT_FEATURE_TABLE.csv`、`figures/development_feature_distributions.svg`。两个历史实例保留原探测数值失败，特征不可用时回退按比例切换时间表。旧探测复用了 1,216 个可验证前缀；最终精确 preflight 的旧类型缺口 18,944 条，扣除 98 条已证明的零物理终止前缀复用，新物理轨迹上限为 18,846，已按用户授权封存。开发追加标签的 6,000 条预算未动用。

{counts}

原冻结扩展选择模型、原不看 controller 的 critic、原 controller 级 critic 和规则选择均已在打开开发标签前生成选臂收据。原 controller 级 critic 部分选择当前菜单外的切换时刻，相关格需标 N/A，不伪造标签。完整 5,000 次配对层级 bootstrap、原 state 奇偶交叉拟合 oracle 重选及 unknown 上下界已实现并通过合成测试；真实留一类型拟合仍等待完整评估标签。

隐式能量策略的完整协方差采样在解析双峰上恢复 32/32，保存的学习能量却在动作边界比峰中心更低。平滑激活诊断保持网络宽深、InfoNCE 均匀负样本、训练步数、采样预算和原模式判据，三个候选仍均失败；完整证据见 `IMPLICIT_IMPLEMENTATION_GAP.json`。没有训练 Ring 隐式策略，不能据此断言该策略类型在 Ring 上无效。

用户最新资源上限为全局 18 CPU、5 GPU shard，现有任务优先。当前主要耗时为原环境 rollout。后台 `advance_evidence.py` 只推进开发证据收集与组装，不自动冻结 pipeline 或打开锁定评估。`AXIS_TABLE` 的结果仍为 PENDING/N/A，后续必须完成开发迭代、规模曲线、冻结、一次性锁定确认与最终审计。
''')
    (HERE/'GAPS.md').write_text('''# 当前缺口与待完成证据

- 隐式能量行为克隆策略：原两峰测试失败；完整协方差版本恢复 12/32，未达到原 26/32 门。解析能量下同一采样器通过，已定位学习能量边界外推问题；三个平滑激活候选仍失败，见 IMPLICIT_IMPLEMENTATION_GAP.json。补齐条件是有可验证的实现修正通过原门，再按原配方训练；锁定类型不替换，不放宽门。
- 开发评估/调参标签仍在采集，真实留一类型选择、模型迭代和规模曲线尚未执行。开发探测和 56 个合格实例已经齐全。
- 两个历史实例探测存在保留的数值失败；不重跑、不编造特征，零样本选择回退既定时间表。
- 规模曲线中 m≥3 的全实例设计至少有 18 个源实例，多于单一类型最多 14 个；完整等样本量单类型对照不可实现。保留所有可行匹配，并增加固定七实例的条件比较，详见 SCALE_FEASIBILITY_DESIGN.receipt.json。
- 原 controller 级 critic 的菜单外选择没有本轮标签；标记相关方法/类型不可用，不能改选当前菜单中的臂冒充原模型。
- 最终 pipeline 尚未冻结，锁定评估/调参标签仍未打开；确认性终点、典型轨迹图、最终 AXIS_TABLE 和整轮结束审计待完成。
''')
    (HERE/'HANDOFF.md').write_text(f'''# 跨 controller type 的实现：接续说明

更新于 {local}（Asia/Singapore），本轮未完成。主任务为 `docs/prompt_crosstype1_learned_zoo_selector_20261018.md`；所有新文件仅可写本目录，不作 git 写操作，不装包。禁止读取/写入 decentralized、teacher_student、multiscene 前缀目录内容；审计只可 stat。运行 Python 固定为 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`，CPU 程序显式 `JAX_PLATFORMS=cpu`，训练必须 Slurm GPU。

用户已授权：现有任务优先；最终精确 preflight 后调整旧类型预算；全局资源上限 18 CPU、5 GPU shard。旧类型执行 cap 已封存为 18,846 条新物理轨迹，见 `OLD_TYPE_FINAL_PREFLIGHT_BUDGET.json`，追加开发标签 6,000 条未用。旧 `old_tuning` 设计由未提交前修正的 `old_tuning_v2` 替代，复用 320 条具有原状态别名/RNG 身份的右手规则历史标签；不要提交旧设计。

32 条复现、六套坐标系、新策略基础测试、训练环境门已通过。18 个新增开发实例已训练且能力门全部通过，开发总数八类 56 个。六个强化学习实例各 20M 机器人步完成，1,920 条 nominal-only checkpoint 选择评估已合并，六个锁定能力门仅记录 PASS，不可读取其各臂率。隐式能量类型仍在合成诊断，尚无 Ring checkpoint。

当前证据作业为旧阶段 `old_evidence_SUBMISSION.json` 与新阶段 `new_evidence_SUBMISSION.json`；先查 squeue 和日志，不重复提交已有任务。`ADVANCE_EVIDENCE_PID.json` 为后台推进器 PID，`EVIDENCE_LIVE_STATUS.json` 为阶段；该推进器会完成 official merger、开发特征和完整 bundle 后停止，不能执行科学迭代或打开锁定评估。`advance_training.py` 已完成训练任务，不再需要启动。所有 worker/runtime 文件在各阶段执行收据中已封存，运行期间和后续核查均不得覆盖这些版本。

开发探测 2,688 条已完成；特征、逐条轨迹索引、表和 SVG 已封存。两个历史探测失败实例保留全 NaN 特征并按固定规则回退。比较方法的 outcome-blind 选择收据 `DEVELOPMENT_COMPARATOR_CHOICES.receipt.json` 已生成。资源/进度实时值见 `results.json` 与 Slurm，本文不是监控锁。

继续顺序：

1. 监控旧/新评估与调参阶段；`finish_phase.py` 只通过默认 journal 和官方单进程 merger 写 rollout，已有 unknown 不重跑。`merge_evidence.py` 合并 disjoint cohorts，`assemble.py` 生成 `DEVELOPMENT_BUNDLE.npz`。
2. 用 `develop_complete.py BUNDLE ITERATION` 做真实八折嵌套留一类型起点，保留每次迭代。`development.py` 的旧统计驱动不是最终统计入口。`analysis_statistics.py` 已包含 5,000 次配对 type/controller/state bootstrap，每次重选 oracle、保留原 state 奇偶与 unknown 上下界；测试在 `STATISTICS_UNIT_TESTS.json`。
3. 按开发结果继续改进，所有调参只看源类型内部留一；不得用外层留出标签拟合该折。四个预备候选为原岭回归、带增益回退的岭回归、最近源类型、浅层树；同结构R5-PCA2、序列化、`freeze_pipeline.py` 和 `scale_complete.py` 已支持这些候选，合成泄漏测试通过。真实比较仍待完整标签；最终选择原则见 DEVELOPMENT_ITERATION_SELECTION_DESIGN.json。
4. 固定最佳结构后跑 `scale_complete.py BUNDLE ITERATION`。主曲线 m=1/2/3/5/7，每档 20 组、m7 一组；全等实例对照不足按预先封存的可行性设计报告，不复制实例凑数。
5. 用全部开发类型拟合最终模型并冻结 `FROZEN_PIPELINE.json` 与完整代码/特征/参数/回退收据。然后才设计锁定探测、计算初始随机动作特征、生成 `LOCKED_FEATURES.npz`、`choose_locked.py` 选臂留收据。
6. 选臂收据必须早于第一个锁定 evaluation/tuning 提交。组装 locked bundle 后只运行一次 `confirmatory_analysis.py EVAL_STAGE TUNING_STAGE`；缺失类型保留缺口，不以单一锁定类型冒充完整确认性结果。
7. 补齐每类型/合并 AXIS_TABLE、每实例选择和遗憾、事件/转移表、训练曲线、规模图、锁定典型轨迹图；做最终完整性审计和完整报告。最终报告第一段先陈述审计与主要研究结论。

统计代码现已逐类型保留原 controller 级 critic 的可报告部分，菜单外选择不代换；“多数强化学习实例≥98%安全基线”的定义已在 LOCKED_ENDPOINT_DEFINITIONS.json 中提前封存，代表性轨迹选例规则在 TRAJECTORY_FIGURE_DESIGN.json。新的模型候选和分析代码必须在最终冻结收据内。下一轮跨type2的冻结pipeline/特征/清单目前尚未就绪。
''')
    print(json.dumps(dict(utc=utc,phases=phases)),flush=True)


if __name__=='__main__':main()
