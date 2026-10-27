# 历史检索

- `diagnostics/critic_pool_supplement_v1/`：COMPLETE、最终核验、全部 36 训练版本、22 个新场景的完整 49 标签、原始轨迹、checkpoint、预测、报告已存在。标签与当前物理语义兼容，全部可用于探索；本任务没有模型训练问题，不重复拟合或推理。
- `diagnostics/hetero_scene_pipeline_v1/`：8 个验证场景完整标签和指定 16 状态；全局缺口 11.4921875 pp，固定场景区间 [6.9494791667,18.3959635417] pp；可复用为验证侧，明确它们已被使用。
- `diagnostics/cross_scene_correction_v1/`：完整原请求及入口可复用，文献场景逐字段保留，尤其修正八机版本及 geometry、moving_policy、scene_revision 字段。
- 只读检查已发现：上一轮 22 个新增训练场景多数偏好 sE29，直接保留 16 个不能证明组内异质。允许补充真正不同参数的场景，新增配置与历史 UID 比较后由官方预检确定是否需计算。
- 其他 agent 活跃目录 `/home/zhihan/.claude/jobs/8061f3d8/tmp/hetset` 仅只读观察，不执行或修改；本任务采用独立路径、实验 UID、种子和参数，保留可交叉复核产物。未向外部发送消息。
- 加速规范完整读取 `docs/exact_engineering_acceleration_20261026.md`。未发现已有 ACCELERATION_CERTIFICATE.json；不能跳过本轮完整认证。
- `diagnostics/cross_scene_prospective_v1` 保持 stat-only。
