本轮开工只读证据清单（尚未进入结果计算）。

- `diagnostics/axis4b_agent_count_v1/runtime.py`、`controllers.py`、`CONSISTENCY_GATE.json`：机器人数量泛化轮的原执行链与旧复现证据。旧复现检查通过不代替本轮十六条逐位复现；原 `runtime.py` 经 `common.py` 导入有输出目录副作用，必须按共同协议复制并记录差异。当前本轮未运行该代码。
- `bottleneck_family/scenario.py`、`environment.py`、`expert.py`：物理及路径算法。四机器人几何接口可以继承，但 `Config.__post_init__` 拒绝少于两个机器人，`BottleneckEnv.step` 对空机器人对执行最小值归约。真正单机器人运行存在接口冲突，详见 `INTERFACE_CONFLICT.md`。
- `diagnostics/axis3_cross_scene_v1/REPORT.md`、`features.py`、`train_v2.py`、`fewshot.py`、`statistics_core.py`、`DESIGN_CONFIG.json`：已有留一场景负结果与方法设置可复用；新的几何族、射线观测、分散执行策略与通用持续右手规则身份不同，不能用旧策略标签冒充新族标签。目标族评估标签尚未生成或打开。
- `diagnostics/blackbox_families_v1/REPORT.md`、`train_policies.py`、`learned.py`、`TRAINING_CONFIGS.json`：确定性与混合密度行为克隆的架构、优化器、固定开发集检查点选择可参考；旧策略是四机器人联合输出且观测不同，不能直接作为本轮共享参数、分散执行的新策略。旧环形交换标签仍保留其原有证据等级和执行语义，不能自动补齐通用持续右手规则、按比例切换或新修正参数的缺臂。
- `diagnostics/blackbox_addendum_v1/trigger.py`、`continuing_rhr.py`：触发器可逐字复制，绕场景中心的旧切向方向不等于本轮目标方向顺时针旋转，旧持续右手规则标签不兼容新动作身份。
- `diagnostics/axis4a_geometry_horizon_v1/progress.py`：按进度切换函数可以只读复用。
- `docs/gap_flow_competence_training_20261007.md`：非对抗导航训练与能力门的历史依据；双机器人冻结检查点不支持本轮新观测和四机器人任意几何。

尚缺逐条旧标签兼容性及共享数据库物理状态去重清单；不会在缺少这些检查时生成新闭环证据。没有读取被禁止的并行轮次目录。
