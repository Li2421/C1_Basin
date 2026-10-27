# 历史证据核查

开工条件：跨场景修正上一轮（`diagnostics/cross_scene_correction_v1/`）的 COMPLETE.json 存在，INTEGRITY_COMPARE.json 的 passed 为 true。两个仓库的原始状态、工作区及暂存区 diff 哈希见 INTEGRITY_START.json。近期其他 diagnostics 写入目录与作业记录见 OTHER_ACTIVITY_START.json；本轮不导入或写入这些历史目录。

| 来源 | 执行语义、兼容性与复用 | 仍缺证据 |
|---|---|---|
| `diagnostics/cross_scene_correction_v1/{LABEL_TABLES,SCENES,FEATURES}.json` | 复用全部 18 场景 × 49 候选标签，unknown 计失败；场景内控制器类型等权。场景特征及模型源码直接复用，不新增训练 rollout | 12 个指定新场景的能力门及冻结候选测试结果 |
| 同目录 `models.py`、LOSO_CHOICES.json、LOFO_CHOICES.json、STATISTICS.json、REPORT.md | 近邻选择器（knn）、岭回归选择器（ridge）源码逐字复制。上一轮岭回归减全局最佳为 +0.5 个百分点，区间 [−4.4,+5.9]；留一瓶颈族为 −32，区间 [−58,−12]，均未通过 | 旧留一分析不能作为新场景的前瞻检验 |
| 同目录 USER_STOP_AMENDMENT.md、EXTRAPOLATION_RESULTS.json | 缺口 10 机外推只完成 107/120；其余 13 局按用户指令停止，只作不完整描述 | 不补跑这些停止项，不并入本轮测试 |
| `diagnostics/cross_scene_homogeneity_audit_v1/` | 事后探索：全局最佳为 9/18 场景最优、12/18 差距不超过 5 个百分点；去重后留一近邻提升约 +8.0 [1.8,15.5]，与全 18 池固定全局最佳比较约 +4.9。仅读取，不能当作本轮结论 | 独立前瞻测试 |
| `diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_REVISED_20261021_lit_crowded_random.json` | 按文件顺序复用 training 中 N=14 的前 16 个状态，恰有 16 个；使用已有 8 类、每类 2 实例的固定参数池，不训练控制器 | 本场景逐实例单机器人能力门 |
| `bottleneck_family/scenario.py`、`bottleneck_family/__main__.py` | 直接只读使用 Config、Geometry、build_instance、preset；半径、动力学及指定布局不变 | 新布局五层证据：几何有效、单机器人路径、联合可解、控制器成功、死锁解决 |
| 上一轮 backend、native_resume、experiment.py、runtime_new.py、scene_environment.py | 所有有副作用的入口与执行依赖复制到本轮。原生冲突搜索缓存根改为本轮目录并重编译；其他数值源码逐字复制。来源及目标 sha256 见 BACKEND_BUILD.json；通过四局完整历史逐位重放才允许继续 | REPLAY_CHECK.json 通过前不得运行新测试 |
| `shared_rollout_db/README.md` | 官方 preflight，只执行真正缺失请求；数值失败保留 unknown、不重跑；官方默认 journal 与单进程 merger；新身份只允许单进程 INSERT 且已有行逐表哈希不变 | 每阶段精确预检与最终数据库核对 |

训练池去重、模型拟合及特征计算都不需要新 rollout。重放为用户明确要求的兼容性验证，不产生新的训练标签。新测试 rollout 只能在冻结完成以后运行；timeout 不等于死锁。源文件哈希的实际复用清单见 REUSED_EVIDENCE_SHA256_START.json，开工广域证据基线另存 EVIDENCE_SHA256_START.json。
