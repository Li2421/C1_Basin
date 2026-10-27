# 场景构造与证据层级

论文版与流程版参数逐行差异见PARAMETERS.md；所有结论均为开发证据。几何有效、单机器人有路、联合可解、控制器成功、死锁解除是五层不同证据，本轮不以栅格单机有路冒充联合可解。原始状态、排除原因与路径检查均有封存收据。图只读取保存的场景与轨迹，最后生成。

## 窄通道场景（lit_narrow_passage）

原始64个状态；有效64个，剔除0个。有效数量分布：3机64个。

半径0.15米，周期0.2秒，速度界1米/秒，最大250步。容量给出显式安全点阵的可容纳数量下界；下界达到N仅证明能放置，不证明任意起终点联合可解。全部eligible状态通过膨胀墙栅格的逐机器人路径检查；联合可解未形式认证。

构造调用：`states.config(scene,N)`、`states.walls(scene)`、`flow_scene.LiteratureEnv(state,config)`；原始状态在`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_lit_narrow_passage.json`，逐项检查在`GEOMETRY_VALIDATION_lit_narrow_passage.json`。

![窄通道场景](figures/lit_narrow_passage.png)

无墙三机。两台已到达机器人保留为真实机器人，图8第一帧的穿行机器人位置声明为reset假设；不能恢复未公开的真正t0。零初始目标距离的进度扩展只解决原函数未定义处；非零距离仍用原进度定义。

## 二维拥挤随机场景（lit_crowded_random）

原始700个状态；有效28个，剔除672个。有效数量分布：2机21个、4机7个。

半径0.15米，周期0.15秒，速度界1米/秒，最大333步。容量给出显式安全点阵的可容纳数量下界；下界达到N仅证明能放置，不证明任意起终点联合可解。全部eligible状态通过膨胀墙栅格的逐机器人路径检查；联合可解未形式认证。

构造调用：`states.config(scene,N)`、`states.walls(scene)`、`flow_scene.LiteratureEnv(state,config)`；原始状态在`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_lit_crowded_random.json`，逐项检查在`GEOMETRY_VALIDATION_lit_crowded_random.json`。

![二维拥挤随机场景](figures/lit_crowded_random.png)

官方GUI采样器原函数体保持；硬工作区边界使672组原样本不满足流程版几何条件。没有补抽、缩小半径或把无效状态算成死锁。有效状态仅2机、4机，不能对8机至14机作控制成功或学习迁移结论。工作区能放置的容量证据与这批起终点是否有效分别报告。

## 圆形围场五机随机场景（lit_circle_five）

原始100个状态；有效100个，剔除0个。有效数量分布：5机100个。

半径2.0米，周期0.05秒，速度界0.373651095854342米/秒，最大2000步。容量给出显式安全点阵的可容纳数量下界；下界达到N仅证明能放置，不证明任意起终点联合可解。全部eligible状态通过膨胀墙栅格的逐机器人路径检查；联合可解未形式认证。

构造调用：`states.config(scene,N)`、`states.walls(scene)`、`flow_scene.LiteratureEnv(state,config)`；原始状态在`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_lit_circle_five.json`，逐项检查在`GEOMETRY_VALIDATION_lit_circle_five.json`。

![圆形围场五机随机场景](figures/lit_circle_five.png)

外圆半径11米，流程版用外切正128边形硬线段墙，最大向外偏差0.00331401345米；原论文外圆为软约束。机器人中心采样拒绝半径>9的提案。流程版速度界由同一100组名义采样保持LQR代数预测的全100秒速度95分位确定，不另生成优化rollout。实际目标有路而粗规划网格终端cell贴墙时，只算对应controller的能力问题。

## 门口场景（lit_doorway）

原始64个状态；有效64个，剔除0个。有效数量分布：2机64个。

半径0.1米，周期0.1秒，速度界0.3米/秒，最大404步。容量给出显式安全点阵的可容纳数量下界；下界达到N仅证明能放置，不证明任意起终点联合可解。全部eligible状态通过膨胀墙栅格的逐机器人路径检查；联合可解未形式认证。

构造调用：`states.config(scene,N)`、`states.walls(scene)`、`flow_scene.LiteratureEnv(state,config)`；原始状态在`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_lit_doorway.json`，逐项检查在`GEOMETRY_VALIDATION_lit_doorway.json`。

![门口场景](figures/lit_doorway.png)

参与计算的墙圆由两个紧包矩形包围：[.9,1.1]×[.3,1.1]与[.9,1.1]×[-1,-.2]；矩形填充相邻圆外缘凹口，门宽.5米。上下墙有限长、无外边界，外侧绕行宽度无上界。两个目标恰好相切；只在reset目标数据校验允许相切，真实占据状态和原swept碰撞判据未放宽。时限40.4秒。


## 用户决定的最终统计范围

v1按用户决定收尾并转入v2。窄通道、二维拥挤随机、门口三个完成的流程版场景进入迁移统计；圆形流程版保留682条raw及完成的48000条单机筛选，8918条未落盘安全基线请求丢弃、不记unknown、不恢复。圆形几何图只表示已构建的几何，不表示迁移实验完成。不得重启v1协调器、恢复圆形claims或提交学习型试点。
