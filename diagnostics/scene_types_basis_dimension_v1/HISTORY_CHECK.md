# 历史证据检查

上一轮收尾提交：ab0c481，分支 research/paper-baselines-20261028；本轮独立分支 research/scene-types-basis-dimension-v1-20261028。从收尾后起，旧目录只读。

| 路径 | 复用范围 | 不可替代的新证据 |
|---|---|---|
| diagnostics/b5_oracle_20261027/report.md | 原 B3 等价、失败分类、原参数菜单 | 新的 17 类型、状态与路线引导未测试 |
| diagnostics/basis_discovery_20261027/report.md | 18 列定义、支持集覆盖算法、Oracle 数据格式 | 新地图分布上的低维覆盖 |
| diagnostics/paper_baselines_20261028/report.md | 已完成 4,128 条新 rollout、888 条精确复用；旧参数冻结和论文适配 | 五篇论文方法在新 17 类型上的成功率和成功条件平均完成时间 |
| diagnostics/paper_baselines_20261028/basis/FINAL_FROZEN.json | 冻结的朝目标 + 局部净空梯度族，纳入待比较族 | 旧固定族跨域失败，不能直接宣称新场景有效 |
| diagnostics/paper_baselines_20261028/CROSS_DOMAIN_FROZEN.json | 旧见证支持最小覆盖 [G,P,R,C2] | 仅有限池支持集结果，未获本轮独立确认 |
| diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027 | 冻结名义控制器、原物理 step、两次安全投影、原 G/P/R | 新场景配置、墙半径一致、路线引导等价与适用性门 |

没有训练。新状态种子及场景、安全墙半径、路线输入语义改变，旧 rollout 不能作为精确标签搬来；旧代码、配置、候选定义与报告可直接只读复用。所有新请求仍须完整身份登记与官方 preflight，不能以此文字替代逐请求缓存检查。

参考压缩包 C1_B5_validation_v0_20261027.zip 在旧轮检查时缺失；没有声称读取或运行该包 run_checks.py。
