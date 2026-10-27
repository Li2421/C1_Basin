# 论文方法身份与接入边界

2026-10-27，根据用户补充“记得有篇论文做强化右手规则”检索；尚不能确认用户记忆中具体是哪一篇。

1. **Chen、Guo、Li：Deadlock Resolution and Recursive Feasibility in MPC-based Multi-robot Trajectory Generation**。[论文](https://arxiv.org/abs/2202.06071)。MBVC warning band、MPC内的检测与自适应消解；2023后续论文明确称该前作为 adaptive right-hand rule。
2. **Chen、Wang、Guo、Li，2023 RA-L：Multi-Robot Trajectory Planning with Feasibility Guarantee and Deadlock Resolution: An Obstacle-Dense Environment**。[作者论文](https://mengguo.github.io/personal_site/papers/pdf/chen2023multi.pdf)、[作者代码 IMPC-OB](https://github.com/PKU-MACDLab/IMPC-OB)。在自适应右手规则遇障碍失效时加入动态优先级；还使用安全走廊和轨迹优化。它是“强化右手规则”的较可信候选，但不是只给当前safe action添加一个侧向向量。完整移植会引入新的规划和安全域，不能在冻结nominal/CBF的Correction实验中冒充纯修正基线。**本轮未实现或运行IMPC/IMPC-OB；调参RHR成绩不代表它们。**
3. **Lee等，2025 IROS：Merry-Go-Round**。[论文](https://arxiv.org/html/2503.05848v1)、[作者代码](https://github.com/Aiden-wjLee/merry-go-round)。机制为临时环岛、加入/协调旋转/退出；论文另用右手规则绕障碍。它并非简单放大右偏置。

## 本轮MGR成绩的严格含义

`Merry_paper_SI` 内部键沿用已登记UID，但展示应读作 **MGR环岛部分的SI适配**。现有 `baseline_comparison_v1/code/roundabout.py` 的环岛机制被逐字节复制后接入原过滤器；未运行作者unicycle plant/CLF-CBF，也未完整移植作者 `cbf.py` 的 obstacle goal rotation、邻居传播和临时目标逻辑。

作者 [cbf.py](https://raw.githubusercontent.com/Aiden-wjLee/merry-go-round/main/cbf.py) 第116–183行还包含预测碰撞后障碍旋转目标、远视野无碰撞退出、邻居传播和距离相关旋转角；本轮原生产nominal以及原安全模型均保持冻结，不能将缺少这些机制的适配成绩称为原论文性能。报告中3/16、1/16仅评价这一部分适配。

两独立集该部分适配各有4次静态墙碰撞，共8次；原CBF violation计数均0，符合已知物理墙几何与CBF间距不一致这一风险，但未通过新对照确证每次碰撞唯一原因。其余失败为timeout；两组分别17次创建环岛、67/105次无法找到合格中心的尝试，这些次数不是互斥失败类别。未据此否定原论文。

独立的 `Merry_heuristic` 是局部冲突组绕当前中心的右切向启发式，没有论文环岛半径和退出控制；15/16、15/16也不能归给论文MGR。

## CBF–CLF

[Ames等原框架](https://arxiv.org/abs/1609.06408)支持将软性能目标和硬安全约束合入优化。本任务采用本系统原CBF行、原速度球、原raw reference，添加软距离CLF再通过原filter。它是本系统的CLF–CBF基线，不是作者汽车系统的原样复现。p=1和100事先固定，均报告，不按测试结果只展示较好者。
