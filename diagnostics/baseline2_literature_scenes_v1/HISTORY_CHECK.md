# 历史证据检查

先按目录名前缀剪枝，未打开、哈希或使用禁止前缀目录的文件。目录清单、实际检索文件、逐行命中和数据集目录收据见 HISTORY_SEARCH.json。

既有窄通道审计 diagnostics/narrow_passage_part34_v1 是两机器人让行湾的历史轨迹审计，半径、目标、机器人数和执行语义与三机器人文献窄通道均不同，不能复用为本轮闭环标签。bottleneck_family 是文献启发的双向 Gap 母版，不是本轮四个参数实例。旧四路交叉口与对称对换也仅供机制参照。

本轮保留跨类型六的控制器实现、冻结权重、候选动作、训练结构，以及跨场景物理特征定义作为可复用组件。critic 输入变化本身不否定旧标签；本轮标签缺口来自新的几何、初始状态、控制周期、机器人数与动力学。尚未完成逐身份共享数据库 preflight，不据文本检索宣称数据库不存在兼容记录。

检索命中：

- docs/scalable_scene_audit_20261007.md:50：4. **LIVEPOINT，2025**：Doorway/Intersection 可作为小规模安全与活性对照，项目有视频；
- docs/drone_3d_deadlock_design_20261013.md:5：采用**低顶通道迎面交换＋侧上方避让舱**作为首个无人机导航场景。两机沿狭窄通道相向飞行，安全层使它们停在对方前方；左机必须后退、上升、侧移，让右机通过后再回到主通道。场景含真实三维几何和三轴运动，适合先检验 Basin C1 的安全投影、干预可达性和 Q 标签语义。
- docs/drone_3d_deadlock_design_20261013.md:15：| [Chen 等，障碍密集环境中的轨迹规划与死锁解决](https://arxiv.org/abs/2210.04231)，2023 修订版 | 在线安全走廊保证避障；动态优先级配合右手规则处理静态障碍造成的死锁；包含最多 8 架微型四旋翼实验 | 窄通道及让行空间是有依据的无人机任务；单靠局部安全约束并不能完成协调 |
- docs/gap_flow_scaling_20261007.md:23：| N | train/dev/test 独立专家初态 | train+dev 门口恢复轨迹 | dev 原始 Flow 全队成功 | dev 安全投影全队成功 | dev 组合控制器全队成功 | 组合控制器直接 Flow 步占比 |
- docs/gap_flow_scaling_20261007.md:53：相向机器人，其余机器人在两侧排队，明确展示对称放行造成的门口互锁。
- diagnostics/axis4a_geometry_horizon_v1/REPORT.md:22：- Basin: status: {"before": " M AGENTS.md\n M docs/controller_generalization_ring_summary_20261014.md\n?? diagnostics/axis1_state_v1/\n?? diagnostics/axis2_controller_v1/\n?? diagnostics/axis3_cross_scene_v1/\n?? diagnostics/controller_behavior_repr_v1/PROMPT_EXTERNAL_CHANGE.diff\n?? diagnostics/controller_correction_mediation_local_v1/fits/\n?? diagnostics/controller_correction_mediation_local_v1/targets\n?? diagnostics/controller_crossed_coverage_loso_v1/fits/\n?? diagnostics/controller_early_eta_localization_v1/fits/\n?? diagnostics/controller_early_eta_localization_v1/targets\n?? diagnostics/controller_eta_switch_time_v1/pipcache/\n?? diagnostics/controller_eta_switch_time_v1/vendor/\n?? diagnostics/controller_ff_crossed_cache_full_v1/features\n?? diagnostics/controller_ff_slot_repair_v1/fits/\n?? diagnostics/controller_late_phase_probe_v1/journals/\n?? diagnostics/controller_local_r
- diagnostics/narrow_passage_part34_v1/REPORT.md:1：# C1 Narrow Passage Part 3–4 read-only audit
