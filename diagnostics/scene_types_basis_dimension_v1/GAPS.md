# 证据缺口

- 散布障碍森林·随机起终点：Geometrically impossible requested forest: disjoint obstacle dilations area 7.03619449019 > enclosure 6.25; 0.3m blocks, 0.5m wall/pair gaps, 3m box. No rollout.
- 方岛环岛·四向：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- T 形路口·三向：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 开阔·圆周对穿：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 开阔·两股垂直交叉流：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 宽走廊·同向换道：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 带障碍的圆圈对穿：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 窄门·两组对向互换：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 单向挤门（疏散）：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 单车道长走廊·两组对向：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 单车道走廊 + 中间让行湾：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- L 形盲角·对向：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 窄十字路口·四向对穿：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 四房间连通（串联瓶颈）：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 货架仓库·窄巷：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.
- 集结到小区域：No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.

某控制器不适用时，留出分母仍按预登记 8 个案例显示，缺失项不会用成功填补；足够类型的 6/8 门槛不改。

开阔垂直交叉流：纵列场景实现错误，当前标为缺口，原横列结果全部留档但不用于选族或验证。6×6m/半径0.1m与3×3m/半径0.05m两项几何提案已生成且通过纯几何检查，等待用户决定。其余15类维持原冻结定义，原有兼容结果复用。

科学比较边界：论文端口保留各自controller/safety，未接入C1外部路线引导，也没有两个冻结MPC同等的全队列单机能力门；大量unknown及原生动力学适配限制，使其结果不足以用于原论文死锁能力排名。原始异常及不重跑的原因见PAPER_FAILURE_ANALYSIS.md。搜索中87个已解决案例的并集未找到单一合法参数盒全覆盖族，最大合法覆盖85；不将这解释成连续空间不可能。固定参数选取使用不均匀开发子集，仍缺同预算统一校准后的部署确认。

Slurm成本附表不可用：CalledProcessError(1, ['sacct', '-j', '29428,29452,29476,29500,29524,29548,29572,29740,29765,29789,29813,29837,29861,29870,29894,29919,29943,29968,29992', '--noheader', '--parsable2', '--format=JobIDRaw,JobName,State,AllocCPUS,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,Start,End,AllocTRES'])

Slurm成本附表不可用：CalledProcessError(1, ['sacct', '-j', '29428,29452,29476,29500,29524,29548,29572,29740,29765,29789,29813,29837,29861,29870,29894,29919,29943,29968,29992', '--noheader', '--parsable2', '--format=JobIDRaw,JobName,State,AllocCPUS,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,Start,End,AllocTRES'])

测量覆盖：论文端口未采集原C1约束违反指标，统计值为null。原C1重试链没有逐事件计数，因此其内部重试总量不可恢复；最终unknown、恢复分支及原约束残差已保留，不把未测量指标写0。

Slurm成本附表不可用：CalledProcessError(1, ['sacct', '-j', '29428,29452,29476,29500,29524,29548,29572,29740,29765,29789,29813,29837,29861,29870,29894,29919,29943,29968,29992', '--noheader', '--parsable2', '--format=JobIDRaw,JobName,State,AllocCPUS,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,Start,End,AllocTRES'])
