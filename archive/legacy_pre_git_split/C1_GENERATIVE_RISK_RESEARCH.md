# C1 直接生成与未来死锁风险：文献及候选推导

2026-09-18状态纠正：本文下方“未接入实验/仍暂停”等文字为2026-09-17早期记录。有界上界已经作为`reachability_union`在`results/c1_gradient_closed_loop_v1/`完成360回放局部诊断；不是未测的新候选。旧参数0.005尺度上死锁20→10、成功16→27，但包含新增死锁3，没有完成当前独立预测/匹配随机梯度/共享训练门槛。按用户停止重复旧风险的要求，该形式归档，不重测。当前状态以`C1_RISK_CANDIDATE_STATUS.md`为准；保留下面原推导和历史说明供追溯。

本次用户澄清优先于此前“生成候选再选路径”的建议：动作由 G_phi 在原 C1 控制链内直接生成。执行时不增加候选路径排序器、规则脱困控制器或规划专家。GPU 实验仍暂停。本文是新候选研究记录，不替换既有实验、风险定义、来源哈希或结果。

## 1. 文献能支持什么

- Zhang et al., *Deadlock-Aware Control for Multirobot Coordination With Multiple Safety Constraints*, TRO 2025, https://doi.org/10.1109/TRO.2025.3600159 。项目有全文 `papers/tro_deadlock.pdf`。任务力与活跃 CBF 安全力的平衡提供局部阻塞机理；Remark 3 包括无 CLF 的检测情形。CLF 重塑的条件性结论不能继承为本项目神经残差收敛定理。
- Grover, Liu, Sycara, *Deadlock Analysis and Resolution in Multi-Robot Systems*, https://arxiv.org/abs/1911.09146 。用对偶/KKT 分析多机器人平衡，是相同机理的另一原始来源。
- Tan, Dimarogonas, *On the Undesired Equilibria Induced by Control Barrier Function Based Quadratic Programs*, Automatica 159 (2024), https://doi.org/10.1016/j.automatica.2023.111359 ，预印本 https://arxiv.org/abs/2104.14895 。说明需明确 QP、CLF 和名义控制条件；不能把任意停滞均归因于 CBF，也不能照搬其修改 QP 的方法到当前固定双投影协议。
- Ganai et al., *Iterative Reachability Estimation for Safe Reinforcement Learning*, NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/file/dca63f2650fe9e88956c1b68440b8ee9-Paper-Conference.pdf 。未来进入目标事件集合的概率及 Bellman 递推可作为风险定义。本文只借用可达性语义，不引入其 actor-critic 架构或宣称其算法收敛结论适用。
- Welikala, Lin, Antsaklis, *Smooth Robustness Measures for Symbolic Control Via Signal Temporal Logic*, https://arxiv.org/abs/2305.09116 。平滑时序逻辑的误差与梯度是连接事件定义和数值训练的工具，不能证明所有非凸优化均成功。
- Meng, Fan, *Signal Temporal Logic Neural Predictive Control*, https://arxiv.org/abs/2309.05131 ，RA-L/ICRA 2024。通过预测轨迹及逻辑鲁棒度训练神经控制器支持“直接生成”这一技术路线；不采用其备用控制器，不将其架构称为当前 C1。

## 2. 原 C1 保持不变

u_nom = bounded(FlowBC_theta(o,xi)); u_safe = Pi_U(x)(u_nom)

v_phi = u_safe + G_phi(o,u_safe); u_phi = Pi_U(x)(v_phi)

x_next = x + dt*u_phi

theta 冻结。风险对上述同一个 G_phi 产生的未来闭环计算，包含未来状态对冻结 FlowBC 的影响。训练更新 phi；执行时直接用 phi 输出动作。随机训练轨迹用于估计期望，不是在执行时从多条路径中选一条。

原优化仍为 min J_def subject to E[R]<=epsilon，使用同一非负对偶更新。它要求低风险可行且干预最小，不等于 argmin 死锁概率；固定 epsilon 下基线已可行时选零残差是预期行为。若要求全局最低风险，则是不同的优化目标，不能默默替换。

## 3. 机理层：精确投影平衡

设 U(x) 非空闭凸，0 属于 U，采用当前欧氏投影。定义标准外法锥 N_U(u)={n: n^T(w-u)<=0, 对所有 w 属于 U}。凸投影的充要条件是

    v_phi-u_phi 属于 N_U(u_phi).
    u_phi=0 当且仅当 v_phi 属于 N_U(0).

对 U={u:Au>=b, ||u_i||<=s_i}, s_i>0，在 0 处速度约束不活跃。若 b<=0，则 N_U(0) 由满足 b_j=0 的 -a_j 生成，因此

    u_phi=0 <=> 存在 lambda>=0:
                 v_phi + A^T lambda=0, lambda_j*b_j=0.

这是冻结状态、冻结本次名义输入下的精确局部等价关系，不需要 FlowBC 是 CLF 稳定控制器。若声称闭环平衡还需未来控制保持该状态；随机策略的一次零动作不能推出永久停滞。若 v_phi=0，可能是策略自身静止，不能声称一定由安全力抵消造成。若存在非零漂移、不同投影度量或非凸约束，需重新推导，不能直接套用。

当前环境死锁允许小速度并要求持续时间；因此瞬时法锥条件既不能替代原标签，也不直接给出未来概率。这里的正确结合是：用真实投影闭环描述该机理，再计算其进入持续死锁事件的可达性。没有额外相加的几何启发式惩罚。

## 4. 概率层：有限时域进入死锁的条件概率

Z_t 包含联合位置、控制所需观测、检测器的进展历史、持续计时、终止状态和剩余时限，使其 Markov 化。检测记忆只用于训练/评分，不暗中新增 actor 输入。D 为原严格死锁或原终局停滞死锁；成功吸收，普通 timeout 是另一事件。定义

    q_t^phi(z)=P_phi(在剩余时限内先于成功触发 D | Z_t=z).

终端标签先按原协议处理。非终端状态满足

    q_t^phi(z) = E_xi[q_(t+1)^phi(F(z,Pi_U(u_safe+G_phi)))],

死锁终端值 1，成功/普通 timeout 终端值 0。这是给定实际 phi 的未来风险，不是“假设未来能换成最优控制”的可达性值。所有机器人与未来动作均按同一闭环展开，不引入动作 argmin 或候选路径选择。递推只定义语义，不意味着要在高维网格求解 HJB，也不要求新增 critic。

## 5. 可实现候选：有上下误差控制的平滑事件上界

对实际预测闭环轨迹 tau，令 rho(tau) 为死锁事件的时序鲁棒度：正数表示满足死锁谓词，负数表示未满足，零是边界。严格项沿用原速度/进展/未完成谓词及持有窗口；终局停滞项沿用原终端标签语义。两项用 max 形成联合事件鲁棒度，而非把两项 ReLU 分数相加。严格阈值下 rho=0 单列为边界，不能将其随意当成确切事件。

例如严格项 rho_D=max_w min_(t,j in w) m_tj。选择可证明的上近似 rho_tilde，满足

    rho <= rho_tilde <= rho+e.

对单个 L*M 原子窗口和 K 个有效窗口，可取此前经检验的 log-mean-exp soft-min 及 log-sum-exp soft-max，e=tau_temp*log(L*M*K)。若进一步合并不同事件分支，上近似误差还须包括分支 soft-max 的 tau_temp*log(B)，不能漏记。没有有效事件分支时直接赋零风险。原事件终止/门控仍有分段及离散边界。

取 0<eta<1、delta>0，k=log(1/eta)，提出

    R_risk(tau)=(1+eta)*sigmoid(k*(1+2*rho_tilde(tau)/delta)).

eta 控制概率尺度上的尾部预算，delta 控制无量纲谓词边界过渡宽度。它们不是新增物理死锁阈值。本文没有根据已揭盲测试选定生产参数。

逐轨迹证明：若 D 发生，则 rho_tilde>=0，故 R>= (1+eta)*sigmoid(k)=1；始终 0<=R<=1+eta。若 rho_tilde<=-delta，则 R<= (1+eta)*sigmoid(-k)=eta。因此

    1_D <= R_risk <= 1+eta.

令 p_D=P(D)，B={非D 且 rho>-(delta+e)}（将零边界包含在内），有

    p_D <= E[R_risk] <= p_D + P(B) + eta.

证明：在非D且非B上 rho_tilde<=-delta，R<=eta；在 D 或 B 上 R<=1+eta，分解期望即可。若 e 依赖回合，用逐回合 e 定义 B，或使用统一上界。给定初始状态时同样成立，故预测风险的条件期望上界 q_t^phi。

这比单侧上界增加了可量化的保守性说明，但只有边界概率 P(B) 小才接近真实概率，不能直接宣称 Pearson 相关性或任意两策略的排序一致。若两个策略均有误差预算 <=Delta，则较低 E[R] 只能推出其真实概率不超过另一策略真实概率加 Delta；两者真实排序仍可能反转。若 eta,delta,e 趋零且事件边界无概率质量，则 E[R] 趋于 p_D；边界有原子质量时该结论不成立。

## 6. 数值与目标边界

新文件 `single_integrator/c1/risk/reachability_margin.py` 只实现上述标量变换和通用单事件时间窗口接口。没有将其接入任何 trainer，没有完成终局联合事件的全链路新版本；原风险仍为现有实验使用的 R_D+R_S。测试只能支持公式及谓词空间求导，不能称为控制效果。

2026-09-17 CPU 检查：`JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-c1/bin/python -m unittest single_integrator.tests.test_c1_reachability_margin`，4 项通过。覆盖逐事件上界/安全裕量尾部界、含平滑误差的有限分布上下界、时间窗口方向有限差分、无有效窗口与非法参数。JAX 插件发现阶段报告 CUDA 初始化警告，测试仍在指定 CPU 后端完成；没有恢复 GPU 训练。

风险对 rho_tilde 的导数为 (1+eta)*(2k/delta)*sigmoid(z)*(1-sigmoid(z))，有限点数学上为正。但很大的正/负 z 会饱和，浮点中可为零；从 rho 到 G_phi 的完整梯度仍会被对称静止点、硬投影平坦区域、活动集切换或终止分支截断。概率近似更紧不必然更好优化。这是本候选的重要未解决项，不能只换激活函数就声称修复机理方向。

所有界仅针对指定时限及模型闭环。没有证明 20 秒包含全部风险；模型预测不准确、训练均值达标、测试均值达标与总体概率保证是不同问题。

纯死锁风险不能保证任务完成：若基线一直运动但失败，p_D=0，最小干预可能保留它。普通 timeout 继续单独报告。若后续必须在目标函数中排除这类解，需要明确增加完成性条件/约束或经论证的次要项；不能宣称已由死锁风险定理得到，也不能擅自破坏原 C1 目标。

下一步研究顺序（未启动 GPU）：在训练/开发数据上验证机理归因及完整 G_phi 梯度、检查可行且成功的低风险策略是否存在，再冻结小规模多种子独立实验。当前已知两组训练不达标和大场景零残差结果均必须保留。新公式未证明最终死锁改善；整体目标尚未完成。
