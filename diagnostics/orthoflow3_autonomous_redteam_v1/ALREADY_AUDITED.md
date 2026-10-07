# 已完成工作清单（红队审计启动时）

日期：2026-10-02。科学源仓库：`/home/zhihan/research/Basin_C1`。本审计位于独立 worktree/branch `Basin_C1_redteam_worktree` / `audit/orthoflow3-redteam-v1`。源仓库的大量科学文件未被 Git 跟踪，因此以下路径及内容哈希，而不是 worktree HEAD，标识科学版本。

当前可接受的冻结模型仍为 joint generator seed41 / critic seed23；K16 诊断定义是均值候选加16个随机候选，非“均值单独部署”。Phase-A critic acquisition 正运行；Phase-B 和 closed-loop contract audit 尚无完成报告，不能视为已验证新版本。此处只审计已冻结证据，不修改并行任务文件。

| 已回答的问题 | 最强现有证据 | 源仓库相对路径 | 当前适用性 / 本次处理 |
|---|---|---|---|
| Ring 安全投影为何仍越界？ | 14个旧碰撞均由漏掉外边界约束导致；48边约束修复及14个前兆重放通过 | `diagnostics/orthoflow3_ring_revision_v1/RING_SAFETY_AUDIT.md`, `REPORT.md` | 当前 safety `18419e…`；不重复修复 |
| 精确物理组件是否旋转/重排等变？ | Four 30、Ring 36案例全部通过，第二投影最大误差约2.72e-8 | `diagnostics/ring_fourway_symmetry_conditioning_audit/REPORT.md` | 仍适用；不重跑完整套件 |
| P0 与 OrthoFlow3 区别？ | 固定残差增益3.303687238760696、固定域，61/61 OrthoFlow3参考；非仅正交性单因素实验 | `diagnostics/double_bottleneck_eta_basis_redesign/REPORT.md` | 代表已冻结3D数学，不重新设计 |
| 新场景 robust basin 是否存在？ | 固定Sobol、Q16/后续局部与跨状态结果 | `diagnostics/four_way_intersection_safety_eta3/REPORT.md`, `diagnostics/ring_exchange_safety_eta3/REPORT.md` | Four沿用；Ring旧安全结果只能作为历史，当前安全由后续v2/K16证据支持 |
| seed robustness / 数值处理？ | Q16 B15标准、历史更强Q32/Q64、独立局部seed证据；数值失败应不作任务失败 | 上述报告及 `diagnostics/double_bottleneck_eta3_seed_robustness/REPORT.md` | 不重复鲁棒性扫描；检查聚合语义是新问题 |
| Ring K4是否太少？ | 同一60状态嵌套K16：oracle45→60，critic39+1未决→53，新增15命中；所有有效候选无碰撞 | `diagnostics/orthoflow3_ring_k16_diagnostic_v1/REPORT.md`, `frozen_proposals.json` | 已查看队列上的诊断；不再K扫描，只核对身份/聚合 |
| Ring critic 是否落后oracle？ | 同候选集报告7miss、4高分低Q；三次K4旧robust被新候选干扰 | 同上 `per_state_results.json` | 本次独立核对候选/Q/排名语义，不训练 |
| 冻结模型泛化？ | 旧/新测试报告分别区分泛化和诊断；Ring旧K4不佳，Double/Four强 | `diagnostics/orthoflow3_generator_critic_frozen_test_v1/REPORT.md`, `diagnostics/orthoflow3_ring_revision_v1/REPORT.md` | 保留具体队列与版本，不能被待完成Phase-A覆盖 |
| B1固定eta是否公平？ | 旧Double公平、旧Four/Ring test-informed；新fair版本train/dev选取，Ring57/60 | `diagnostics/orthoflow3_ring_revision_v1/B1_PROVENANCE_AUDIT.md`, `fair_fixed_eta.json` | 已解释，不重复选eta |
| Ring旧安全标签是否修复？ | 21,520条Ring标签重算；认证标签漂移79条/0.3674%；eta0 0/80改变 | `diagnostics/orthoflow3_data_hygiene_v2/REPORT.md` | v2已安全更新；当前旧模型仍部分旧数据 |
| dataset证据与结构完整？ | 245状态/45,169标签，33,770认证负、1弱负、38未认证；robust无欠证据 | 同上；`datasets/orthoflow3_basin_dataset_v2_audited/` | 不重做一般卫生/DB审计；Q解释可能需更窄核查 |
| split与归一化泄漏？ | 独立审计无父轨迹/测试泄漏，train-only归一化逐值重现 | 同上 `leakage_audit.json`, `normalization_audit.json` | 复用，不重跑全审计 |
| DB重放是否可重复？ | 75标签/695seed结果、终止、碰撞逐项一致；结构检查通过 | 同上 `reproducibility_replay_results.json` | 复用；不再一般DB完整性扫描 |
| Four/Ring表示捷径？ | Four旋转/槽位偏置，Ring local obs + world Flow混合；不是精确物理bug | `diagnostics/ring_fourway_symmetry_conditioning_audit/REPORT.md` | 已解释；Phase-B未完成，不假定修复 |
| critic ranking-loss是否优于当前NLL？ | `OBJECTIVE_MISMATCH_NOT_MAIN_BOTTLENECK` | `diagnostics/orthoflow3_ranking_aware_critic_v1/final_decision.json` | Toy/DB另一流水线的控制；不移植为Ring结论，不重试 |
| uncertainty/LCB是否修好critic？ | val选lambda0，无验证提升；Toy部分系统性高置信误差 | `diagnostics/orthoflow3_toy_critic_uncertainty_local_v1/final_decision.json` | 不重复；非当前joint Ring checkpoint |
| NLL权重/其他Success Basin工作？ | `CURRENT_NLL_WEIGHTING_NOT_MAIN_CAUSE`；Toy/DB当前性审计指出适用模型/数据必须分开 | `diagnostics/orthoflow3_nll_weighting_ablation_v1/final_decision.json`, `diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1/final_report.md` | 不复活旧目标；避免把另一路Toy critic当作Ring模型 |
| 当前Phase-A是否已修好critic？ | 只有冻结protocol、proposal manifests、进行中的Q采集，尚无accepted checkpoint/report | `diagnostics/orthoflow3_k16_critic_canonicalization_v1/protocol.json`, `RUNNING_NOTES.md` | 未完成，不引用预测结果；只读 |

本次不启动训练、环境族、K sweep、安全重审或一般DB审计。
