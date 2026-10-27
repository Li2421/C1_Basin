# 停止原因与继续所需的明确修订

用户文档第零节第10行要求：“遇到与本说明矛盾、或要求的接口不存在：停下报告，不改源码。”本轮因此在历史核查阶段停止，没有自行放宽规则。

1. **共享数据库写路径冲突（首个停止原因）。** 文档第8行只允许 `cache_writer.append_journal` 与官方单进程 merger 写共享库。`shared_rollout_db/src/historical_ingest.py:137–151` 对 `registered_exact_uid_v1` 只检查已经存在的 scenario/state/controller，缺少任何一项就返回 ambiguous，不从 journal 注册新精确身份。`ingest_jsonl:188–205` 也没有 metadata-only 注册消息。通用控制器分支在110–112行只支持旧 modern OrthoFlow 或带完备指纹的 `MAC_ONLY` 达到 EXACT_PROFILE；本轮 MPC/CEM/规划控制器不能冒充 MAC_ONLY。旧 `ceiling_common.py:127–173` / `registry.py:17–46` 的注册函数则显式直接 SQL INSERT 元数据，超出本轮允许的两个写入口。`append_journal` 和 `merger` 都存在，缺的是在这些限制内登记本轮精确新身份的路径。

   最小可执行修订建议：明确准许本轮单进程使用官方 dbm.connect/transaction，只 INSERT 新 experiment/scenario/state/controller_config/eta 元数据；冲突即停、原有行逐表哈希不变；rollout 仍只经默认 journal 与官方 merger。此建议尚未执行。另一方案是提供已支持 metadata 注册的官方 merger 接口；本轮不修改共享库源码。

2. **Gap1 排序相关性没有共同候选。** 文档第32行规定本轮49点的历史16来自 controller_loco_real_flow_v1/protocol.json；第49行要求与旧 Gap1 标签做“历史16”新旧排序相关。逐候选 IEEE754 float64 比较显示，Gap v1/v2/v3 的16点、v4的全部128点，与这16点的精确交集全部为0。不能用相同编号或近邻代替同一 eta。

   继续需明确：该描述指标记 N/A（无共同候选），还是另行授权仅为此描述比较补充同一组 eta 的标签、指定在哪条执行语义下补跑及其状态/种子。不能偷偷扩展49点主菜单或混用不同 eta 的排序。

3. **旧场景 MPPI 选择冲突。** 文档第28行要求沿用上一轮，又列出7实例并称无合格 MPPI。但上一轮 SCREEN_SELECTION_PRIMARY_20261023.json 的窄通道、拥挤随机 training_controllers.centralized_mppi 均为 centralized_mppi0；圆形与三个八机为 null。HISTORY_FACTS.json 保存其训练/合格名单摘录。

   继续需明确：窄通道、拥挤随机按真实旧选择保留第8类，或本轮统一排除并登记相对旧选择的协议变化。现阶段未擅自选择。

4. **完整性原始 status 的字面要求。** 开工 status 在新目录创建前采集；新增本轮目录会使 Basin_C1 原始 status 多一条本轮目录记录。结束保留原始差异，不把它宣称为“原始 status 完全一致”；另核验剔除本轮新增目录后的原有状态、两个 git diff 哈希及全部审计文件哈希。没有为隐藏这一记录修改 .gitignore、Git index 或仓库配置。

待执行步骤：历史逐位重放、新场景五层验证、能力门、pilot、逐请求 preflight/标签、嵌套CV、模型冻结、外推、统计与完整DB后检均未执行。不生成 MODEL_FROZEN.receipt.json 冒充已冻结，不将未运行视为假设失败。
