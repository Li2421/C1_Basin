# 历史检索与复用

本任务按用户明确切换，执行其指定路径的11场景版 pipeline prompt；原文逐字副本和来源哈希见 PROTOCOL.md、PROTOCOL_SOURCE.receipt.json。12新场景近邻前瞻任务已在原目录登记用户切换，未运行任何新测试rollout。本轮仅在 cross_scene_pipeline_v1 内写入代码和输出。

| 来源 | 执行语义、兼容性与复用范围 | 仍缺证据 |
|---|---|---|
| `diagnostics/cross_scene_correction_v1/{LABEL_TABLES,SCENES,SELECTED_CONTROLLERS,LOSO_CHOICES,STATISTICS}.json`、REPORT.md、models.py | 完整18场景×49候选种子0标签、unknown、状态、固定合格实例、模型选择源码全部复用；种子、物理、分布式安全投影及接触门控修正沿用。按场景岭回归旧结果+0.5个百分点[−4.4,+5.9]未通过 | 本轮生成器与critic对6维状态、12维环境、8类独热输入的逐折训练和新提案结果 |
| `diagnostics/cross_scene_homogeneity_audit_v1/` | 只读事后探索：去重11场景近邻留一提升+8.0[+1.8,+15.5]，岭回归−2.0[−10.4,+6.1]；使用更强全池固定基线后近邻约+4.9。不能当作本轮pipeline已有结论 | 预定去重池上pipeline的留一场景及留一族比较 |
| `diagnostics/orthoflow3_generator_critic_v1/train_evaluate.py` | 复用Generator结构、压缩高斯分布函数、stable_int、损失、优化、早停和提案抽样逻辑。原输入层、η盒及旧训练场景不同，旧checkpoint不能作为本轮无泄漏折模型 | 只训练真正缺失的预定折/种子，不补旧rollout标签 |
| `diagnostics/crosstype1b_generator_critic_pipeline_v1/critic_model.py`、REPORT.md | 原Critic类逐字复制；不看控制器和只看场景的消融用新子类。环形交换相对按进度切换基线+0.72[−1.14,+2.87]，未通过；旧checkpoint的输入、候选语义与划分不兼容本轮 | 6+12+8+3输入的预定折训练与消融 |
| `diagnostics/baseline_literature_scenes_distributed_v2/` | 文献线按状态critic的旧结果只作历史依据；更换critic编码不改变已有相同执行语义的49候选标签有效性 | 本轮不得重跑49候选，不增加能力门，不训练控制器 |
| 上一轮backend、native_resume、experiment.py、common.py、runtime_new.py | 有副作用的模块复制到本轮；原生冲突搜索只改查询缓存根并重编译；其他数值语句逐字不变。官方预检在逐位重放之前记录旧请求精确已有结果 | 六局完整重放：双瓶颈、对穿12机、拥挤随机各两局 |
| `shared_rollout_db/README.md` | 新身份只用官方connect/transaction单进程INSERT，旧行逐表哈希不变；官方preflight仅补真正缺失请求；默认journal和单一merger | 新提案精确身份预检和最终核对 |

本轮只有pipeline选中的生成器新提案和“仅生成器均值”变体需要新rollout。49候选内的pipeline选择、完整critic、两项消融、近邻、固定基线、η13及仅安全过滤全部直接用旧标签计分。原训练场景的物理状态、合格控制器、成功/碰撞/unknown判据不改；timeout不等于死锁。

源文件哈希见EVIDENCE_SHA256_START.json及REUSED_EVIDENCE_SHA256_START.json，隔离图见BACKEND_BUILD.json。近期24小时写入目录和全部用户作业目录见OTHER_ACTIVITY_START.json，两个仓库开工状态与diff哈希见INTEGRITY_START.json。复用的历史模型只阅读/复制源码，不导入其写入入口，不复用会造成留出泄漏的checkpoint。

## 停止与已复核成果

210次训练与31折冻结均已完成。pilot身份登记因本轮八机静态障碍请求版本/登记名称构造错误触发强制停止；事务回滚，旧行未覆盖，新rollout为0。六局重放、模型训练隔离/早停/种子选择、全部实际复用输入、冻结文件及两个仓库的结束审计通过。详见UID_CONFLICT_AUDIT.json、TRAINING_AUDIT.json、DB_POSTFLIGHT.json、INTEGRITY_COMPARE.json；停止不是科学主判定未通过。

## 用户授权修正后的完成核对

保持210次训练、31折模型/提案/选择全部有效且字节未变；没有重新训练。只新增按原请求逐字段构造的执行适配器，10例规范化JSON和rollout UID逐字节一致后继续。所有新请求先经官方预检，仅补缺失结果；最终数据库、轨迹、旧输入和两个仓库完整性核对通过。原失败/停止记录保留，详见REQUEST_CONSTRUCTION_CORRECTION.md与DB_POSTFLIGHT.json。
