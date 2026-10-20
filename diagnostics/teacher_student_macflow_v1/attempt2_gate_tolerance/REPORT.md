# Round30 续跑：噪声一致性门失败

源码完整性审计通过：两个仓库的 HEAD、git status、git diff SHA256 与本次开始完全一致，PoC `poc/` 下全部 Python 文件 hash 不变。TS1（学生是否安全但卡住）、TS2（时间表优势）、TS3a（零样本迁移性能）、TS3b（是否利用 controller 信息）均为 **NOT_RUN，无法判定**。本次已执行新版前置门，但首个样本触发原 runtime 内部断言失败；严格按第二节“任何一项不通过就停下报告”停止，未训练、未生成闭环标签，未完成全部实验交付。

旧版接口阻塞已由更新任务文档解除，本次没有沿用旧阻塞结论。第一次全部产物原样移入 `attempt1_blocked/`，内容 hash 验证见 ATTEMPT1_ARCHIVE_VERIFIED.json。重新建立 INTEGRITY_START.json，增加 PoC 源码清单；并检查14个教师 checkpoint、各自训练配置与实际数据 manifest，14/14精确匹配。wide 教师实际训练集为 `base_u_plus_w_v4_dataset`，并非 round28 适配用的 `base_u_dataset`；本次按其训练配置中的 hash 定位，没有混用。

预注册、TRAINING_CONFIG.json、TEACHERS.json 与噪声门代码在执行前封存，见 PREREGISTRATION.receipt.json。计划72样本/教师、合计1,008个，实际在第1个样本停止；不能声称完成或通过1,000样本门。

失败样本：

- 教师：`base_u_plus_w_v4_wide_macflow`，checkpoint hash 与 round28 相符。
- 数据：`diagnostics/ring_exchange_stage1/base_u_plus_w_v4_dataset/rollouts/train/train_nominal_000000.npz`，transition=0。
- key：`[2263789486, 2715912652]`，JAX x64=False，CPU单核，无仿真 step。
- 重新计算教师观测与存储观测最大差为0。
- `runtime.nominal(env,k,traced=True)` 在原 `poc/field_controller.py:212` 的 `np.testing.assert_allclose(decoded,flow,rtol=1e-10,atol=1e-11)` 失败。
- 8/8分量超容差；最大绝对差 `9.8882097e-09`，最大相对差 `4.19442745e-08`。
- 捕获失败后只保存同一样本的两项接口诊断：trace第0步与独立 `jax.random.normal` 逐位一致；native经原to_world和再限速后与非traced nominal逐位一致。两项通过仅针对该样本，不能替代失败门或推断其余1,007例。

失败发生在原实现对trace终点重新解码后与native动作比较的位置；差异量级与FP32/FP64解码舍入相容，但本轮没有进一步归因实验，不能确证唯一原因。未修改原断言、未放宽容差、未绕过断言、未改为其他runtime。原native动作生成链路本次已经实际调用，旧版“缺显式ξ接口”的理由不再适用。

NOISE_GATE.json 保存原始断言、完整traceback、样本及两项诊断；NOISE_GATE_FAILURE.npz 保存输入状态、key、观测、trace、raw与world动作；NOISE_GATE.receipt.json 封存失败证据。noise_gate.log 中另有非Slurm环境的CUDA插件探测警告；本次显式指定CPU，且已进入并执行原nominal数值断言，此警告不是本次停止原因。资源为1个CPU核，0 GPU；没有Slurm作业、训练、rollout、journal或共享DB写入。

历史审计见 HISTORY_AUDIT.json 及原样归档的 attempt1_blocked/HISTORY_AUDIT.md。round28四臂三管线、专家仍可作为后续DEVELOPMENT参照，但不能回答学生终点。专家本次未运行。pilot状态尚未生成或去重；没有能力门、R5、冻结选择、评估preflight/postflight、训练曲线/checkpoint、论文成功率/CI总表或推理耗时，未将这些缺项伪装为完成。

结束审计见 INTEGRITY_END.json、INTEGRITY_COMPARISON.json；PoC清单包含5个Python文件。GATE_IMPORTED_SOURCES.json记录此次全部导入Python源码的hash（包括依赖包），IMPORTED_SOURCES_END_CHECK.json复核其结束内容不变。ROUND28_FROZEN_AUDIT.json核对原冻结文件。只读审计及所有新产物均位于本目录，已有源码与环境保持原样。
