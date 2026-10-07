# 顺序执行队列

1. 当前：K16 Critic Closure（Phase A），冻结和报告；随后 Physical Conditioning Canonicalization（Phase B），完成评估、报告和 artifact freezing。不得混淆两阶段结论。
2. 用户新增、尚未开始：`diagnostics/orthoflow3_closed_loop_contract_v1/`。以当前任务实际接受结果为起点；若 canonicalization 未接受，使用最后有效管线。先建立 continuation contract，最多约12 train/dev snapshots/主场景，4000新 continuation 初始预算，只有具体反例/修复可扩至12000。最多两次隔离修复；不改 MACFlow、安全、OrthoFlow3、域、horizon、K；不重复K sweep、ranking/LCB或泛化DB审计。可委派最多三个 Astra 只读推理任务。完成证据、最小修复、确认和冻结才结束。
3. 用户已补全的 red-team 审计：`diagnostics/orthoflow3_autonomous_redteam_v1/`，在以上产物冻结后进行。先 ALREADY_AUDITED ledger、RISK_REGISTER，再独立选最多5（优先2–4）高价值假设。先静态、后4–20状态小探针，预注册、反证控制；禁止重复K sweep/ranking/LCB/完整DB与物理对称审计。必须语义核对Ring oracle/critic候选相同、顺序、conditioning、安全hash与随机种子。可选1个高价值自发分支。每个确认问题最多2轮可逆修复；只回填失效证据/重训直接失效组件。不替换生成器family/critic架构，不改OrthoFlow3/维度/域/K>16/horizon/成功标准/MACFlow权重（除已证预处理使其无效）或安全语义（除已证bug）。初始<=3000新continuations；只有已复现重大问题验证修复才先记录理由扩至10000；最多1 GPU shard。若存在其他并发 contract 审计，独立 worktree，canonical只读、patch排队，先消化其结果。不扩大研究方向。

Red-team 交付需包括 REPORT、ALREADY_AUDITED、RISK_REGISTER、hypothesis manifests、witnesses、patches/tests、ARTIFACT_VALIDITY_MAP（VALID/VALID_AFTER_REENCODING/STALE_REQUIRES_RECOMPUTE/INVALID）、DB/cache、hash、decision。最终恰一状态 AUDIT_CLEAN_WITHIN_SCOPE / MINOR_REPAIRS_VALIDATED / MATERIAL_PIPELINE_REPAIR_REQUIRED / SCIENTIFIC_INTERPRETATION_AT_RISK；并给 generator、critic、RingK16、Four、v2 的 SUPPORTED/SUPPORTED_WITH_QUALIFICATION/NEEDS_REFRESH/NOT_SUPPORTED。无任务作业遗留。

后续两项不会并发修改本阶段文件，不预设当前阶段成功，不重复已完成实验。最终交付前检查所有本任务作业已退出。

4. 新授权：Phase B 与 Closed-Loop Contract 结束后执行 Unified Physical Representation。
   产物为 diagnostics/orthoflow3_unified_representation_v1/ 与 datasets/orthoflow3_basin_dataset_v3_unified_rep/。
   先读取实际接受的 contract 与模型；用共享 agent/pair/geometry entity 编码替换 learned scenario adapters，支持可变 N/M。
   保留 contract 因果字段、固定 MACFlow/安全/OrthoFlow3/eta域/K16/任务分布；不传模式标签。
   先结构变换与 aliasing 检验，后两个匹配训练种子 old-vs-unified、train/dev rollout、冻结后小新确认。
   物理证据精确缓存复用，仅重编码不重复 rollout；最终 PASS/PARTIAL/FAIL 后停止，不开启环境族实验。
