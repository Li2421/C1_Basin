本轮 prompt 第二节要求 `GeneratedEnv(BottleneckEnv)` 只重写 `__init__`，`reset`、`snapshot`、`step`、碰撞和终止判断全部继承。

同一 prompt 第三节要求训练示范来自真正单机器人 episode（场景里只有一个机器人），并在各族十六个单机器人 episode 上做导航能力门。

现有只读接口有两处阻塞：

1. `bottleneck_family/scenario.py` 的 `Config.__post_init__` 要求 `num_agents >= 2`，否则抛出 `ValueError('num_agents must be an integer >= 2')`。
2. 即使仅在新初始化中构造一个机器人，继承的 `bottleneck_family/environment.py` 的 `BottleneckEnv.step` 对空的 `_i, _j` 机器人对调用 `point_segment_distance(...).min()`，空数组归约没有恒等元，不能运行一步。

此问题不是能力门失败，不产生任何成功、碰撞、超时或 unknown 标签。四机器人等价性单元测试尚未运行，不把此接口检查称为等价性测试失败。

待用户明确的最小方案：仅在本轮目录另写单机器人专用环境，为单机器人配置和空机器人对的无碰撞距离提供定义；四机器人 `GeneratedEnv` 仍只重写初始化、完全继承原 `step`。在用户选择之前，不实现依赖此例外的代码，不以停车的额外机器人代替真正单机器人 episode。
