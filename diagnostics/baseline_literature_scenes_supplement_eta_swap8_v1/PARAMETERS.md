# 八机对径交换参数来源

来源优先级：论文 v2正文→官方代码环境默认→图/视频→声明假设。论文 [arXiv 2506.22117v2](https://arxiv.org/pdf/2506.22117v2)，官方[项目页](https://nus-core.github.io/assets/standalone/HJB-GNN/index.html)及[代码](https://github.com/hublan24/HJB-GNN)只作参数来源，未运行、导入或安装。论文页码以下为PDF物理页。

| 参数 | 值 | 来源与解释 |
|---|---|---|
| 机器人/目标 | 8台等角圆周分布、对径目标 | 论文p8 V.B及图7(a/k)；图中编号的旋转/反射不改变任务；代码从x正轴逆时针编号 |
| 起点圆 | free/moving 1.0m；static 1.4m | 论文p8 V.B |
| 机器人半径 | 0.05m | 用户明确将论文r作为半径；代码env/crazyflie.py:55的drone_radius同值 |
| 感知 | 0.5m | 论文p15 Appendix F及p12 Appendix E；优先于代码三维comm_radius=1.0 |
| 周期 | 0.05s | 论文p15 Appendix F硬件周期；优先于三维仿真代码0.03s |
| 平面速度范数界 | 0.5m/s | 声明假设。官方仅CrazyFlie三维环境，crazyflie.py:95的vel_targets_scale=(2,2,.5,.1)是高层目标缩放，:733–744为机体坐标各分量±.3，均不是本轮二维单积分器的平面速度范数上限，不能直接冒充该参数 |
| 到达容差 | 0.08m | v2原LiteratureConfig |
| 时限 | free/moving16s=320步；static22.4s=448步 | 声明规则4×原论文构型直线起终距离/vmax；所有扰动state保持同一场景时限，参照v1门口固定配置 |
| 工作区 | free/moving[-1.5,1.5]^2；static[-2,2]^2 | 声明采样区域，无边界墙；机器人可离开，不人为裁剪 |
| 非凸障碍 | 4个L，各两条0.5m线段，corner半径.7m、22.5/112.5/202.5/292.5度，凹口向心 | 用户指定的后备声明布局。论文p8图7(k–o)为实景I形物体，无轴、无平面标尺/相机标定；420dpi渲染page8.png已读取，不能可靠恢复米制坐标，因此不声称图像几何复现；不是从结果调节布局 |
| L厚度 | 零宽线段；wall_radius=0；碰撞净空半径+.005m | 按用户线段表示和v2墙参数；未擅自加入未指定实体厚度 |
| 直线路径 | 部分L臂穿过直线机器人路径 | corner不在对径线上不意味着整个L避开直线。GEOMETRY_VALIDATION逐条报告；所有state通过静态单机器人栅格连通及可见边验证，不将绕行误标死锁 |
| 移动圆盘 | 半径.3m，速度.25m/s | 用户指定，半径对应论文三维球障碍半径上限；代码obs_len_range=[.1,.6]且采样除2，env/crazyflie.py:58,146–149 |
| 障碍运动 | 区域内均匀随机waypoint，匀速、到点重抽，不受机器人影响 | 用户声明；初始离每台机器人≥.45m，state决定路径，所有controller/seed/solo共享父state路径，MOVING_PATHS已封存；跨step内waypoint转折精确分段 |
| 状态 | 各32，0原构型，其余起点±.1m均匀扰动，重叠重抽，目标不变 | 用户规则；2026102311/12生成free/static，2026102313生成moving障碍；moving机器人state逐位复用free。前16训练、后16评估 |
| 安全层 | 第一阶段分布式互惠pair、墙/移动障碍全责；.5m以内 | 用户规则；HardProjectionConfig默认gamma/gamma_wall及separation_buffer；d_safe=.1001,Rc=.125125 |
| 移动约束 | 当前位置圆盘，半径额外+.25×.05=.0125m | 用户指定，不用障碍速度、目标或未来路径；物理碰撞用真实.3m，分别连续扫掠 |
| 修正通信 | Brel按v2全部其他机器人平均+接触门；同时广播到达位/平均进度 | 不把安全层局部感知误称完整controller局部通信。修正保留v2全局位置假设 |
| controller | 8类×6，原参数不改 | CONTROLLER_MANIFEST；私有副本仅加新墙几何成本分支，移动障碍不进入名义输入 |

该场景是用户定义的二维迁移开发证据，不是HJB-GNN策略或三维Crazyflie硬件结果复现。
