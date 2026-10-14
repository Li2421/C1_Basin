# Round13 handoff

# Round13：四场景时间表的前瞻检验

X1 **REJECTED**，X2 **REJECTED**；按预注册映射，四场景 LOSO 失败与 controller 级失败的动作压缩解释 尚未得到前瞻确认。X1 字面点门 未通过，X2 字面点门 未通过；unknown 边界影响另列。新40state/scene、3个原E8 Flow-reference算子、7臂、4seed共13440条实际rollout已完成；没有训练critic或新Flow。批次状态 **NUMERICAL_COMPLETE_VIDEO_INCOMPLETE**，不将视频缺口隐瞒为完整交付。

## 事实：X1

保持 Ring 原时间表 S13→10@t2；各场景成功率取3controller平均，与该场景3固定臂的最大均值比较。门为四场景各自差≥−5pp，未加新的CI显著性要求；CI为state成簇5000次、seed202610131301，保留3controller与4seed。主状态保守地检查unknown边界，字面点结果同时报告。

| 场景 | S13@t2 | 最佳固定 | 固定率 | 差 | 95% CI | unknown bounds | 点门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toy_give_way | 58.542pp | F10 | 73.958pp | -15.417pp | [-23.333pp, -7.500pp] | [-15.417pp, -15.417pp] | False |
| double_bottleneck | 67.917pp | F15 | 99.792pp | -31.875pp | [-35.208pp, -28.542pp] | [-32.083pp, -31.875pp] | False |
| four_way_intersection | 96.458pp | F15 | 99.792pp | -3.333pp | [-5.417pp, -1.458pp] | [-3.542pp, -2.292pp] | True |
| ring_exchange | 86.667pp | F13 | 87.083pp | -0.417pp | [-1.458pp, 0.417pp] | [-0.625pp, -0.208pp] | True |

逐controller（每controller160次，40state×4seed）：

| 场景 | controller | S13@t2 | 最佳固定 | 差 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| toy_give_way | flow_isotropic | 52.500pp | 81.250pp | -28.750pp | [-40.000pp, -18.125pp] |
| toy_give_way | flow_anisotropic | 69.375pp | 88.125pp | -18.750pp | [-35.625pp, -2.500pp] |
| toy_give_way | flow_coupled | 53.750pp | 66.875pp | -13.125pp | [-22.500pp, -5.000pp] |
| double_bottleneck | flow_isotropic | 33.750pp | 100.000pp | -66.250pp | [-72.500pp, -60.000pp] |
| double_bottleneck | flow_anisotropic | 71.250pp | 100.000pp | -28.750pp | [-35.625pp, -21.875pp] |
| double_bottleneck | flow_coupled | 98.750pp | 99.375pp | -0.625pp | [-3.125pp, 1.250pp] |
| four_way_intersection | flow_isotropic | 98.125pp | 100.000pp | -1.875pp | [-4.375pp, 0.000pp] |
| four_way_intersection | flow_anisotropic | 92.500pp | 99.375pp | -6.875pp | [-11.250pp, -3.125pp] |
| four_way_intersection | flow_coupled | 98.750pp | 100.000pp | -1.250pp | [-3.125pp, 0.000pp] |
| ring_exchange | flow_isotropic | 87.500pp | 92.500pp | -5.000pp | [-13.125pp, 0.000pp] |
| ring_exchange | flow_anisotropic | 85.000pp | 85.625pp | -0.625pp | [-1.875pp, 0.000pp] |
| ring_exchange | flow_coupled | 87.500pp | 97.500pp | -10.000pp | [-18.125pp, -3.125pp] |

## 事实：X2

源场景等权平均后argmax，留出场景不进入该fold选择。并列按固定菜单顺序取第一项；bootstrap每次重新算源选择和held七臂oracle，不把某个已选源动作当作已知常量。固定集F10/F13/F15；扩展集加4时间表。门仍为扩展最大遗憾≤5pp且平均≤固定均值的1/3。

| 留出场景 | 集合 | 源选动作 | held oracle | 遗憾 | 95% CI | unknown bounds | unknown可能动作 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toy_give_way | fixed | F15 | S13_t1 | 13.958pp | [7.500pp, 21.875pp] | [13.958pp, 13.958pp] | F15 |
| double_bottleneck | fixed | F10 | F15 | 72.292pp | [70.833pp, 73.750pp] | [72.292pp, 72.500pp] | F10 |
| four_way_intersection | fixed | F13 | F15 | 97.292pp | [96.042pp, 98.542pp] | [96.250pp, 97.500pp] | F13 |
| ring_exchange | fixed | F15 | S15_t1 | 88.958pp | [84.995pp, 92.708pp] | [86.250pp, 88.958pp] | F15 |
| toy_give_way | expanded | S15_t2 | S13_t1 | 14.167pp | [8.333pp, 22.083pp] | [14.167pp, 14.167pp] | S15_t2 |
| double_bottleneck | expanded | S13_t1 | F15 | 73.750pp | [71.458pp, 75.417pp] | [73.542pp, 73.958pp] | S13_t1 |
| four_way_intersection | expanded | S15_t2 | F15 | 0.208pp | [0.000pp, 0.833pp] | [0.000pp, 0.417pp] | S15_t2 |
| ring_exchange | expanded | F15 | S15_t1 | 88.958pp | [84.792pp, 92.708pp] | [86.250pp, 88.958pp] | F15 |

固定平均/最大遗憾：68.125pp/97.292pp；扩展平均/最大：44.271pp/88.958pp。平均比例=0.6498470948012233；配对平均改善95%CI [23.229pp, 24.583pp]。扩展平均/最大unknown bounds：[43.490pp, 44.375pp]/[86.250pp, 88.958pp]。

## 事实：X3（INCONCLUSIVE / 描述性）

X3没有另给二元判定阈值，故不制造SUPPORTED/REJECTED。E_a=S−F10，L_a=S−F_a，N_a=F_a−F10，E−L=N恒等式全部通过。两阶段均≥10pp的数量：{'toy_give_way': 0, 'double_bottleneck': 1, 'four_way_intersection': 0, 'ring_exchange': 1}，总2/24。

| 场景 | controller | 早期η | E | L | N | E 95%CI | L 95%CI | 两项≥10pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_give_way | flow_isotropic | 13 | -28.750pp | 0.000pp | -28.750pp | [-40.000pp, -18.125pp] | [0.000pp, 0.000pp] | False |
| toy_give_way | flow_isotropic | 15 | -28.750pp | 0.000pp | -28.750pp | [-40.000pp, -18.125pp] | [0.000pp, 0.000pp] | False |
| toy_give_way | flow_anisotropic | 13 | -18.750pp | -1.250pp | -17.500pp | [-35.625pp, -1.875pp] | [-3.125pp, 0.000pp] | False |
| toy_give_way | flow_anisotropic | 15 | -24.375pp | -0.625pp | -23.750pp | [-43.125pp, -5.625pp] | [-1.875pp, 0.000pp] | False |
| toy_give_way | flow_coupled | 13 | 1.250pp | 0.000pp | 1.250pp | [0.000pp, 3.125pp] | [0.000pp, 0.000pp] | False |
| toy_give_way | flow_coupled | 15 | 14.375pp | 0.000pp | 14.375pp | [5.625pp, 24.375pp] | [0.000pp, 0.000pp] | False |
| double_bottleneck | flow_isotropic | 13 | 33.750pp | 2.500pp | 31.250pp | [27.500pp, 40.000pp] | [0.625pp, 5.000pp] | False |
| double_bottleneck | flow_isotropic | 15 | 100.000pp | 0.000pp | 100.000pp | [100.000pp, 100.000pp] | [0.000pp, 0.000pp] | False |
| double_bottleneck | flow_anisotropic | 13 | 71.250pp | -19.375pp | 90.625pp | [64.375pp, 78.125pp] | [-25.625pp, -13.750pp] | False |
| double_bottleneck | flow_anisotropic | 15 | 96.250pp | -3.750pp | 100.000pp | [93.125pp, 98.750pp] | [-6.875pp, -1.250pp] | False |
| double_bottleneck | flow_coupled | 13 | 16.250pp | 10.625pp | 5.625pp | [11.250pp, 21.250pp] | [5.625pp, 16.250pp] | True |
| double_bottleneck | flow_coupled | 15 | 17.500pp | 0.625pp | 16.875pp | [12.500pp, 22.500pp] | [0.000pp, 1.875pp] | False |
| four_way_intersection | flow_isotropic | 13 | -1.250pp | 98.125pp | -99.375pp | [-3.750pp, 1.250pp] | [95.625pp, 100.000pp] | False |
| four_way_intersection | flow_isotropic | 15 | 0.625pp | 0.000pp | 0.625pp | [0.000pp, 1.875pp] | [0.000pp, 0.000pp] | False |
| four_way_intersection | flow_anisotropic | 13 | -6.250pp | 86.250pp | -92.500pp | [-10.625pp, -1.875pp] | [81.250pp, 91.250pp] | False |
| four_way_intersection | flow_anisotropic | 15 | 0.000pp | -0.625pp | 0.625pp | [-2.500pp, 2.500pp] | [-1.875pp, 0.000pp] | False |
| four_way_intersection | flow_coupled | 13 | 0.000pp | 97.500pp | -97.500pp | [-1.875pp, 1.875pp] | [95.000pp, 99.375pp] | False |
| four_way_intersection | flow_coupled | 15 | 1.250pp | 0.000pp | 1.250pp | [0.000pp, 3.125pp] | [0.000pp, 0.000pp] | False |
| ring_exchange | flow_isotropic | 13 | -5.000pp | -0.625pp | -4.375pp | [-13.125pp, 2.500pp] | [-2.500pp, 1.250pp] | False |
| ring_exchange | flow_isotropic | 15 | 3.125pp | 90.625pp | -87.500pp | [-3.750pp, 11.250pp] | [85.625pp, 95.000pp] | False |
| ring_exchange | flow_anisotropic | 13 | 53.125pp | -0.625pp | 53.750pp | [38.125pp, 67.500pp] | [-1.875pp, 0.000pp] | False |
| ring_exchange | flow_anisotropic | 15 | 63.125pp | 95.000pp | -31.875pp | [49.375pp, 75.625pp] | [90.625pp, 98.750pp] | True |
| ring_exchange | flow_coupled | 13 | -10.000pp | 0.000pp | -10.000pp | [-18.125pp, -3.125pp] | [0.000pp, 0.000pp] | False |
| ring_exchange | flow_coupled | 15 | -5.000pp | 73.750pp | -78.750pp | [-10.625pp, -0.625pp] | [63.750pp, 83.125pp] | False |

失败类型与实际发生步汇总：

| 场景 | 臂 | 事件数 | 事件发生步 min–max |
| --- | --- | --- | --- |
| toy_give_way | F10 | Counter({'deadlock': 112, 'timeout': 13}) | deadlock: 240–839; timeout: 850–850 |
| toy_give_way | F13 | Counter({'deadlock': 149, 'timeout': 48}) | deadlock: 242–840; timeout: 850–850 |
| toy_give_way | F15 | Counter({'deadlock': 186}) | deadlock: 249–794 |
| toy_give_way | S13_t1 | Counter({'deadlock': 106, 'timeout': 13}) | deadlock: 242–753; timeout: 850–850 |
| toy_give_way | S13_t2 | Counter({'deadlock': 139, 'timeout': 60}) | timeout: 850–850; deadlock: 242–624 |
| toy_give_way | S15_t1 | Counter({'deadlock': 128, 'timeout': 4}) | deadlock: 249–772; timeout: 850–850 |
| toy_give_way | S15_t2 | Counter({'deadlock': 180, 'timeout': 7}) | deadlock: 249–595; timeout: 850–850 |
| double_bottleneck | F10 | Counter({'timeout': 347, 'deadlock': 1}) | timeout: 850–850; deadlock: 346–346 |
| double_bottleneck | F13 | Counter({'timeout': 144}) | timeout: 850–850 |
| double_bottleneck | F15 | Counter({'numerical_failure': 1}) | numerical_failure: 660–660 |
| double_bottleneck | S13_t1 | Counter({'timeout': 354, 'numerical_failure': 1}) | timeout: 850–850; numerical_failure: 373–373 |
| double_bottleneck | S13_t2 | Counter({'timeout': 154}) | timeout: 850–850 |
| double_bottleneck | S15_t1 | Counter({'timeout': 352}) | timeout: 850–850 |
| double_bottleneck | S15_t2 | Counter({'timeout': 6}) | timeout: 850–850 |
| four_way_intersection | F10 | Counter({'numerical_failure': 5}) | numerical_failure: 41–235 |
| four_way_intersection | F13 | Counter({'timeout': 463, 'numerical_failure': 5}) | timeout: 1100–1100; numerical_failure: 29–256 |
| four_way_intersection | F15 | Counter({'numerical_failure': 1}) | numerical_failure: 209–209 |
| four_way_intersection | S13_t1 | Counter({'numerical_failure': 5}) | numerical_failure: 29–256 |
| four_way_intersection | S13_t2 | Counter({'timeout': 12, 'numerical_failure': 5}) | numerical_failure: 29–256; timeout: 1100–1100 |
| four_way_intersection | S15_t1 | Counter({'numerical_failure': 1}) | numerical_failure: 209–209 |
| four_way_intersection | S15_t2 | Counter({'timeout': 1, 'numerical_failure': 1}) | timeout: 1100–1100; numerical_failure: 209–209 |
| ring_exchange | F10 | Counter({'timeout': 124, 'numerical_failure': 1}) | timeout: 700–700; numerical_failure: 469–469 |
| ring_exchange | F13 | Counter({'timeout': 61, 'numerical_failure': 1}) | timeout: 700–700; numerical_failure: 303–303 |
| ring_exchange | F15 | Counter({'timeout': 429, 'numerical_failure': 13}) | timeout: 700–700; numerical_failure: 377–626 |
| ring_exchange | S13_t1 | Counter({'timeout': 22}) | timeout: 700–700 |
| ring_exchange | S13_t2 | Counter({'timeout': 63, 'numerical_failure': 1}) | timeout: 700–700; numerical_failure: 303–303 |
| ring_exchange | S15_t1 | Counter({'timeout': 15}) | timeout: 700–700 |
| ring_exchange | S15_t2 | Counter({'timeout': 18, 'numerical_failure': 9}) | timeout: 700–700; numerical_failure: 377–478 |

逐seed、原始UID、具体事件步见 FAILURE_RECORDS.json，完整步频分布见 FAILURE_TIMING_SUMMARY.json。真实deadlock仅来自Toy/DB原window_progress_v2终止事件；candidate单独记录，timeout始终为timeout。配对失效变化见 FAILURE_TRANSITIONS.json（每同state/controller/seed对照F10/Fa→时间表），不从动画或timeout推断新死锁。

## 解释与边界

Cross-scene regime elimination not prospectively confirmed; inspect descriptive within-scene phase effects and remaining LOSO regret. No post-hoc schedule/threshold tuning.

最好时间表相对最佳固定的场景差：{'toy_give_way': 0.012499999999999956, 'double_bottleneck': -0.01041666666666674, 'four_way_intersection': 0.0, 'ring_exchange': 0.09791666666666676}；每场景是否至少有一个双阶段≥10pp项：False。

本轮固定对照新增了历史E8 K4中没有的F13；X1是不劣检验，并不要求时间表严格优于每个固定动作。因此是否仅新增固定F13就改善LOSO，以及时间表是否有增量收益，必须看X2固定/扩展对照，不能直接拿本轮率与旧K4失败率作唯一因果证据。

新states上的比较为前瞻；历史E8、审计、Round9–12结论为DEVELOPMENT来源，不能把Round12真实checkpoint结论直接移给E8 affine算子。Flow checkpoint嵌套于场景，只覆盖四种固定几何；即使X1或X2通过，也不能证明H_compress是唯一根因。下一步critic/新几何LOSO仅为提示指定的研究方向，本轮没有实施。

## 执行、冻结与资源

预注册UTC 2026-10-13T14:40:15.420517+00:00，SHA f245770fdeb6bcd097a3857f0b05f087453f7d7572a8421e65c52a7f53a9b5db。原E8运行时/模型/物理/投影/事件未改，唯一动作干预是绝对物理动作索引t替换η；Toy JAXFP64，其他FP32，均在全部导入后恢复并每步断言。切换364/607、471/786、300/500，使用完整IEEE η值。旧Toy没有augmented_state接口，用已有snapshot和全部监测器内存只读记录；随后初态文件同hash临时名并发冲突改为每进程文件。两次纯IO修复与原代码/失败日志保留，已产生的gate记录复用、无unknown重跑。32例历史F10/F15的终止、长度、首动作float64字节全部精确一致，之后才启动主实验。

160状态原自然生成、expert接受规则与去重细节见states_part及STATE_DEDUP_AUDIT.json。DB原expert8模式可解性接受检查是初态生成的辅助仿真，不是主FF标签/Flow训练，抽样与拒绝计数完整记录。原18CPU资源上限后被用户明确改为24；最高23计算worker+1编排，0GPU。纯IO预留将尚未开始案例转交helper，原在途case不中断，标记无成功字段、只由owner换成真实记录；全部13440实际physical counter校验，零重复仿真。

全量到齐才打开终点，ENDPOINT_OPENING收据UTC 2026-10-13T15:23:36.513166+00:00。source-only argmax未知边界的并列处理由合成数据检查修正为预注册的最早菜单项，未读真实标签、没有更改参数或判定门。标准preflight无ambiguous、只缺失生成、worker-localjournal、官方单进程merger/本地完整seed身份adapter；postflight13440条trace/raw/DBflags一致、quarantine0、unknown50保留并给upper/lower边界。没有把unknown当作观测失败或重跑。

## 视频与交付

完整视频 12 个。统一四场景入口已实际调用，退出码2；缺口 {'toy_giveway': [], 'double_bottleneck': ['deadlock_resolution'], 'four_way_intersection': ['deadlock_show', 'deadlock_resolution'], 'ring_exchange': ['deadlock_show', 'deadlock_resolution']}。Ring/Four原runtime无deadlock判据，不可伪造；timeout-to-success如存在仅为完整诊断视频，不填正式deadlock角色。每个正式安全素材检查实际连续运动余量与全部过程。manifest含源trace/videoSHA、真实初态、controller、seed和event，所有正式pair同初态/seed/config。批次保持NUMERICAL_COMPLETE_VIDEO_INCOMPLETE；严格入口非零阻止COMPLETE标记。数值检验与视频交付状态分别报告。

results.json和LABELS.npz为机器结果；ARM_RATES.csv、X1_CONTROLLER/X2_SCENES/X3_EFFECTS及FAILURE*为逐项数值；图 four_scene_seven_arms、X2_loso_regrets、X3_phase_effects 各PNG/PDF；视频 videos/manifest.json；所有新资产仅在本目录，历史文件hash保持不变，没有gitcommit。

Reproduction: prereg receipt ->prepare_gate/run(gate)/gate ->states/dedup ->design/preflight ->run(main)/helper if resource transfer ->single merger ->analyze ->videos ->report. No target success analysis before all13440 actual raws. Unknowns neverrerun. No further rollout/video fabrication or critic training is authorized by this completed numerical stage. Fullbatch remains incomplete when required true-deadlock roles are missing.
