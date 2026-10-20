# C1 — Toy Give-Way 局部区域 K 的定量认证

日期：2026-10-17。冻结实现根目录 F=`/home/zhihan/research/Basin_C1_flow_field_poc_20261004`；困难状态证据根目录 H=`/home/zhihan/research/Basin_C1_hard_field_20261006`。

**最终判断：存在有限尺寸、正体积、无需初始正横向偏移的几何及完整 CBF 参考可达区域；尚未认证原固定 η 的 TT/FF 对整个区域 guaranteed escape。** 入口子域占本文明确定义的主通道冲突几何域约 **0.200576%**，不能宣称广泛覆盖所有冲突配置。原 η* 在保守通过域内的 raw progress 可以全域严格认证，但安全投影后的持续进度、横向保持和完整任务成功仍有独立缺口。

本报告保留原 controller、Flow checkpoint、三列 B、候选 η 盒、CBF、投影器、TT/FF placement、环境及 evaluator。所有参考运动只用于证明原安全动作集是否允许某条路径，**没有替换实际策略、实施新控制器或生成新 rollout**。

## 1. 执行顺序、证据等级与版本审计

依用户要求使用四个独立 subagent：A 完成实际几何契约后，B/C 独立并行；D 在取得三者结果后整合。原始结果保存在 [A](agent_a/report.md)、[B](agent_b/report.md)、[C](agent_c/report.md)、[D](agent_d/report.md)。主协调器另检查了截面边界归属、域外反例不能否证域内、体积的严格区间算术和源文件不变性。

证据等级统一为：**PROVEN**（列明假设下的解析证明）、**NUMERICALLY CERTIFIED**（指定数值问题的证书）、**EMPIRICALLY OBSERVED**（有限状态/种子/轨迹观察）、**FAILED**（具体反例或违反）、**UNKNOWN**（不足以建立）。有限采样不会被标成整个域的证明。

先复用 `diagnostics/tt_ff_correction_realization_20261017_v1/` 的配置、68 条动作、340 子步及轨迹审计，并核对冻结源哈希。之后仅新增 24 个无物理步进的零偏移动作探针；用途、状态/种子数量和成本上限在执行前已告知用户。D 只读重算 768 条已有轨迹。没有新 rollout 请求，故没有创建 rollout preflight、写共享数据库或重新登记标签；旧 AMBIGUOUS exact-η* 缓存未作为兼容结果复用。

实际 checkpoint：`F/validation_v2/assets/toy_ckpt.pkl`。

一次新诊断元数据修订保留完整追溯：根协调器复核时纠正了 A 截面字符串在 y=h 的边界归属。B 曾读取修订前契约，已恢复其字节及哈希完全相同的输入快照；运行配置、八墙、常数、源码均未改变，B 的有限段逐约束证明不依赖该错误字符串。最终几何定义以修订后的 A 契约为准，差异记录在 `geometry_contract_revision.json`。没有为“配平哈希”重跑动作或改写 B 的历史输入身份。

SHA256：`8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`。

`validation_v2/adapters.py` 的 `OtherRuntime('toy_give_way')` 通过 `single_integrator.evaluate.load_policy` 读取 checkpoint 的 evaluation_environment；A 实际重新加载，并与旧审计一致。关键差别是 **corridor_half_length 实际 1.3 m，源码默认 2.5 m**。不能混用默认地图及其目标。

Toy 主投影是 `single_integrator/cbf.py` 的 SLSQP 凸 QP 加速度球外切平面，接受前检查可行性/KKT；原重试链可调用相同问题的 tighter Clarabel SOCP，再调用相同非线性约束 polish。FF 的球投影快捷分支仅在全部线性行也满足时返回。`shared_control/hard_projection.py` 是不同的通用 Clarabel 实现，不能与本次 Toy 主后端混为同一历史运行版本。完整配置、源码及旧证据哈希见 [input_audit.json](input_audit.json) 和最终 manifest。

## 2. 实际参数、八条有限墙段及完整自由空间

**PROVEN / 实际加载配置核对：**

| 参数 | 实际数值 | 意义 |
|---|---:|---|
| wall_layout_scale | 1 | 世界坐标不再缩放 |
| corridor_half_length L | 1.3 m | 源码默认 2.5 m |
| corridor_width | 0.4 m | 主通道墙面 y=±0.2 |
| bay_top | 0.68 m | 侧袋墙面顶部 |
| bay wall x boundaries | ±0.2 m | 墙面位置，不是机器人中心边界 |
| robot radius r_b | 0.16 m | 不改变机器人大小 |
| wall radius r_w | 4/600=0.00666666666667 m | 墙膨胀半径 |
| m_wall, m_pair | 0.005 m, 0 | evaluator 碰撞 margin |
| CBF buffer β | 0.0001 m | 原数值 separation buffer |
| effective R_w | 0.171766666666667 m | r_b+r_w+m_wall+β |
| D_s | 0.3201 m | 2r_b+m_pair+β |
| V, dt | 0.5 m/s, 0.05 s | 直接单积分器 |
| γ, γ_w | 1/s, 1/s | γ dt=0.05≤1 |
| horizon | 850 步=42.5 s | timeout 与 deadlock 分开 |
| goals | (1.09,0), (-1.09,0) m | 两机器人交换方向 |

八条墙段端点来自 `environment.py:114–124`，精确采用运行配置生成：

| 索引 / 名称 | 起点 | 终点 |
|---|---|---|
| 0 left_end | (-1.3,-0.2) | (-1.3,0.2) |
| 1 right_end | (1.3,-0.2) | (1.3,0.2) |
| 2 floor | (-1.3,-0.2) | (1.3,-0.2) |
| 3 left_ceiling | (-1.3,0.2) | (-0.2,0.2) |
| 4 right_ceiling | (0.2,0.2) | (1.3,0.2) |
| 5 bay_left | (-0.2,0.2) | (-0.2,0.68) |
| 6 bay_right | (0.2,0.2) | (0.2,0.68) |
| 7 bay_top | (-0.2,0.68) | (0.2,0.68) |

定义几何常数

\[
w=.2,\quad a=L-R_w=1.12823333333333,\quad
h=w-R_w=.0282333333333333,\quad H=.68-R_w=.508233333333333.
\]

在环境的**内部合法连通分支**上，所有有限墙段的中心距离约束等价于：

\[
\Omega=\{(x,y):-h\le y\le H,\ |x|\le g(y)\},\qquad
g(y)=\begin{cases}
a,&-h\le y\le h,\\
w-\sqrt{R_w^2-(w-y)^2},&h<y<w,\\
h,&w\le y\le H.
\end{cases}
\]

中间截面来自两个口部端点的圆形排除区，不能替换为无限直线。**y=h 的合法截面仍是整个主通道 |x|≤a；中段只能使用 h<y。** 在这条二维面积为零的边界线上截面跳变，不能用连续插值删除合法分支。环境内部条件也不能省略：仅要求远离有限墙段会错误接纳地图外的远处点。

联合几何自由空间是

\[
\mathcal F=\{p_1,p_2\in\Omega:\ s^2+z^2\ge D_s^2\},\quad
s=x_1-x_2,\quad z=y_2-y_1.
\]

因此单独坐标极值均为 `x_i∈[-a,a]`、`y_i∈[-h,H]`，但不能把这些极值组成安全 Cartesian box。还必须保留各机器人 `|x_i|≤g(y_i)` 及 pair 圆外约束。选 y1≤0 是理论角色分支，不是环境本身只允许机器人 1 在下方。

完整闭自由空间的最小墙 CBF h 为 0，出现在 floor、bay side、top 等膨胀边界；相对 evaluator 的阈值仍多原 buffer。以下严格内部子域才有正墙净空。环境与 CBF 的 pair 约束不同于线性距离墙 barrier：实际 pair barrier 是 `s²+z²-D_s²`。

## 3. 建议用于论文的分阶段 K 定义

**PROVEN：** 不将一个未加耦合的矩形当作全过程不变域。采用下面有限尺寸的入口、连接管道与通过域；全部坐标单位为 m。

\[
\mathcal K_{\rm entry}=\left\{
\begin{aligned}
-.65\le x_1\le-.35,&\quad -.025\le x_2\le.025,\\
-.02\le y_1\le0,&\quad -.02\le y_2\le.02
\end{aligned}\right\}.
\]

这一区域**不要求 y2 初始为正，也不要求 z 初始为正**。它本身保证 pair 距离至少 .325 m。连接 Yield 管道保持同样的 x1、x2、y1 范围，将 y2 范围扩展为 `[-.02,.4]`；只对下文给定的参考方向证明可达性，不称为实际 TT/FF 的不变管道。

\[
\mathcal K_{\rm pass}=\left\{
\begin{aligned}
-.025\le x_2\le.025,&\quad -.65\le x_1\le x_2+.34,\\
-.02\le y_1\le0,&\quad .38\le y_2\le.46
\end{aligned}\right\}.
\]

通过域中有

\[
s\in[-.675,.34],\qquad z\in[.38,.48],\qquad
D_{\rm exit}=.34,\quad Z_{\min}=.38,\quad D_s=.3201.
\]

这里 **D_exit 是纵向出口，Z 是横向分离，D_s 是 pair 安全距离**。三者不能写成一个 D。`s≥-.65-x2` 是隐含但必须保留的耦合下界；仅列 s∈[-.675,.34] 会得到一个稍大的不同集合。

| 域 | 配置体积 m⁴ | 全域最小墙 CBF 净空 | 最小 pair 距离 |
|---|---:|---:|---:|
| K_entry | 1.2×10^-5 | .00823333333 m，y1=-.02 的 floor | .325 m |
| Yield 参考管道 | 1.26×10^-4 | .00323333333 m，x2=±.025 且 y2≥.2 的 bay side | .325 m |
| K_pass | 7.92×10^-5 | .00323333333 m，同侧袋边界 | .38 m |

K_pass 的纵向区间随 x2 有 `.965–1.015 m` 长度，x2 宽 .05 m，y1 宽 .02 m，y2 宽 .08 m。x2 范围占高侧袋全部中心可用宽度 .0564667 m 的约 88.55%。这些不是无限小邻域；但体积相对整体配置域仍可能很小，见第 7 节。

h、H、a 和高侧袋最大公共半宽 ±h 来自实际几何。`.025,.02,.38,.46,.65,.35,.34` 是本次明确选择并认证的保守内边界，**不是几何唯一值，也未证明最优或最大**。完整几何外容器 F 已解析给出；最大的实际 TT/FF 可认证逃逸域仍 UNKNOWN。原式中 S,D_exit 不由墙面唯一决定；若强求 x2∈[-h,h] 与 s∈[-S,D_exit] 独立，端墙给统一 S,D_exit≤a-h=1.1 m，但这也没有消除 pair 耦合。

## 4. Yield：删除初始正偏移的能力及其限度

### 4.1 完整安全动作集允许零偏移 Yield：PROVEN

对 K_entry 中所有状态，令机器人 1 保持，机器人 2 以 `u2=(0,.03)` 上升到 y2=.4。沿路 s 不变、|s|≥.325、z≥-.02。实际 pair CBF slack 满足

\[
2zv_Y+\gamma(s^2+z^2-D_s^2)
\ge .325^2-D_s^2-2(.02)(.03)
=.00196099>0.
\]

口角水平距离至少 A=.175；下面的证明保留 ceiling 与 bay_side 的重复端点行。写 t=.2-y2≥0、d=√(A²+t²)。朝上墙 CBF 需要 `vY≤d(d-R_w)/t`，而

\[
{d(d-R_w)\over t}\ge {A^2-R_w^2+t^2\over2t}
\ge\sqrt{A^2-R_w^2}\ge .0334845072>.03.
\]

上侧袋内点的侧墙法向水平；top 最小 slack 为 `.68-.4-R_w-.03=.0782333333`。floor、end walls、其他端点及速度球的完整分支检查见 B 的逐约束证明。由凸 barrier 的支持不等式及 γdt≤1，允许的 held 动作在整个物理步内安全，非仅检查端点。

最多 `(.4-(-.02))/.03=14 s / 280 steps`；最后参考步可缩短以落在 y2=.4。完整移动步速度 .03>.025，不满足实际 deadlock 的低速 guard。一次缩短尾步不足以累计 5 s hold。该结论要求开始时尚未终止，并保留相应剩余 horizon；普通 true-t0 初态由 reset 初始化监测历史。

### 4.2 实际 TT/FF 能从零偏移启动：EMPIRICALLY OBSERVED

新探针的三种位置是 `((- .4,0),(0,0))`、`((-D_s,0),(0,0))`、`((-D_s/2,0),(D_s/2,0))`；previous velocity 均零。每个位置使用两个固定种子，配对 zero/η* 与 TT/FF，共24条动作。实际 key 构造、合成 UID 和求解状态保存在 [zero_offset_actions.csv](agent_b/zero_offset_actions.csv)。

所有 6 个 state-key 对、两 placement、两 η 的 u2y 均正：TT 范围 `.0052853–.0453486 m/s`，FF `.0035249–.0478628 m/s`。这证明冻结实现存在零偏移的真实瞬时启动样本，**不证明持续进入 bay**。

在零偏移时 goal 与 rel 列的 y 分量为零，η* 的 perp 系数也为零，所以 **δ2y=0 精确成立**；TT 的 yielding 与 zero baseline 完全相同。启动来自原 Flow/anchor，而非 raw witness 提供的侧向激励。两种 pair 接触位置的纵向 passing gap 只有约10^-17，精确 pair 行禁止严格正 gap。正 Yield 首动作不意味着立即 Pass。

因此：把“η* 在所有零偏移状态具有严格正 raw yielding”写成命题是 **FAILED**；把“零偏移物理上或原安全集下不可 Yield”写成必要条件同样是 **FAILED**。冻结策略从这些状态必然持续让行和逃逸仍为 **UNKNOWN**。

### 4.3 z_seed 的量化及不可推广边界

对固定位置族 `p1=(-.4,0),p2=(0,z)` 与 η*，原始 bound_rows 精确给出

\[
\delta_{2y}(z)=\tfrac12 z\left(1-{.5\over\sqrt{1.09^2+z^2}}\right),\quad 0\le z\le.2.
\]

其导数严格正；解该单调标量方程得到以下 **NUMERICALLY CERTIFIED** 的 raw 阈值：

| 要求 raw δ2y | 最小 z_seed | 占全部主通道中心高度 2h | 占本族 y1=0 的向上范围 h |
|---:|---:|---:|---:|
| .001 m/s | .003694897264 m | 6.5435% | 13.0870% |
| .005 m/s | .018472328736 m | 32.7137% | 65.4274% |
| .010 m/s | .036931203484 m | 65.4035% | 130.8071%，超出窄通道矩形中心带 |

在上述主通道矩形比较带中，两中心最大相对偏移为 2h=.0564667；若另固定 y1=0，则该带内最大正 z 仅 h。**h 不是 x2=0 处全部合法初始 y2 的上界**：口部截面允许更高的 y2，因此第三行的点本身仍可在口部合法，只是不在用于宽度比较的矩形中心带。不能把这个比较带选择冒充真实碰撞边界。这些数值**不是实际 TT/FF 输出或闭环 reachability 阈值**。在24个真实初态上，仅按 z 大小比较，上述阈值分别有11/24、4/24、0/24满足；由于位置族不同，不能宣称它们通过了对应 raw-action 条件。

没有位置、η、先前速度、Flow key、placement 无关的通用正 z_seed。合法 η=(.5,0,0) 在正 y2 时仅有朝下 goal 修正，已经排除所有合法 η 共用的正 raw 保证。实际零偏移启动样本又排除了“必需一个统一正种子才能首步启动”的说法。能替代 y_e 的一般条件是实际总动作下界及安全管道不变性，但本次没有认证后者；不能把这一新证明义务写成已经满足的假设。

## 5. Pass：全域参考速度、原基表达性及执行缺口

对 `u1=(vP,0),u2=0`，实际平方 pair CBF 为

\[
2sv_P+\gamma(s^2+Z^2-D_s^2)\ge0.
\]

配方给出对 s 的统一充分限

\[
v_P\le\gamma\sqrt{Z^2-D_s^2},\qquad
Z_{\rm req}=\sqrt{D_s^2+(v_P/\gamma)^2}.
\]

**PROVEN：** K_pass 中 Z≥.38，pair 允许同一参考速度至 .204782787 m/s；但墙的约束不能省略。

| vP (m/s) | Z_req (m) | 本次全墙证书是否还允许该速度 |
|---:|---:|---|
| .025 | .321074773 | 是 |
| .05 | .323981496 | 是 |
| .06 | .325674700 | 是 |
| .10 | .335356542 | 是 |
| .20 | .377444049 | 此墙证书不支持；不能仅用 pair 式认证 |
| .30 | .438707203 | 不支持 |
| .40 | .512312414 | 不支持；接近最大几何分离 |
| .50 | .593686795 | 连几何最大分离也不足 |

在 y1≤0 分支，最大几何分离 `Zmax=H+h=.536466667 m`，但该极值触及零墙 CBF 净空边界；其 pair-only 极限速度 .430502584 不是实际全约束最大速度。

对机器人 1 向右运动，口角在其上方距离 t≥.2，水平前距 q>0。墙行的分子满足

\[
d^2-R_w d-v_Pq\ge
\left(1-{R_w\over.4}\right)q^2-v_Pq+.2(.2-R_w).
\]

在 vP=.06 时右式全实数最小值 `.004069333528 m²>0`。该界统一允许 `vP≤.113523458 m/s`；它是保守充分速度界，不是最大速度的优化解。机器人 2 的 hold 动作满足八条墙行；机器人 1 floor、end walls、其他端点和两速度球逐项证明见 C。

取 vP=.06，即在整个 K_pass 中有允许参考 `dot s=.06`，保持 z 不变，最多339步到 D_exit=.34。与 B 的 Yield 接合共≤619步=30.95s。若另选已被同一证书覆盖的参考 vP=.10，从最坏 s=-.675 到出口最多203步；速度改变仅用于存在性证书，不改变实际策略。

原三列和 η* 在**整个** K_pass 上还可解析证明

\[
\delta_{2y}\ge .02351227\ \mathrm{m/s},\qquad
\delta_{1x}-\delta_{2x}\ge .04371005\ \mathrm{m/s}.
\]

两 goal 列在该域均饱和；rel 未饱和/饱和分支分别处理。C 给出单调界及推导；不是用几点差分外推全域。

**UNKNOWN：** 原固定 η 的真实 TT/FF 是否整个域保持 z、产生持续 `dot s≥νP>0`、在剩余时间内退出。参考动作可行仅说明投影输入若恰等于参考动作则不改变它。实际输入并非参考动作；η* 正的 raw y 也不是 bay hold 的不变性条件。上一轮68动作没有任何点位于此 K_pass，域外 t215 的 FF 负 gap 不能拿来否证这里，域外正点也不能拿来认证这里。

到出口后 x1=x2+.34∈[.315,.365]，真实端墙仍很远；保持机器人 1、让机器人 2 沿固定 x2 返回 y2=0，水平距离始终 .34>D_s，存在几何安全路径。**实际 Resume、到达两个目标和剩余任务 horizon 仍 UNKNOWN**，不并入 Pass。

## 6. 从普通冲突状态开始的正体积参考可达性

**PROVEN（安全集参考存在性）：** 更早的普通初态外包域

\[
\mathcal E:\quad x_1\in[-.705,-.590],\quad x_2\in[.590,.705],\quad y_i\in[-.02,.02]
\]

具有 `2.116×10^-5 m⁴` 体积，包含本次24个冻结困难初态。它是明确选定的外包域，不宣称是完整采样分布的精确支持。

整个 E 可依次选择如下原安全动作集允许的参考段：

| 段 | 参考运动 | 最多步数 |
|---|---|---:|
| 对齐 lower 角色 | 若 y1>0，以 .026 m/s 下降到0 | 16 |
| 对齐 x1 | 若 x1<-.65，以 .05 m/s 右移到-.65 | 22 |
| 对齐入口 | x2 以 .05 m/s 左移到0 | 282 |
| Yield | 按 B 以 .03 m/s 到 y2=.4 | 280 |
| Hold–Pass | 按 C 已允许的 .10 m/s 到 s=.34 | 198 |
| 合计 | 每段最后一步允许缩短 | **798=39.9s<850步** |

D 对整个外包域逐项验证墙角、pair、floor、速度球及 deadlock guard：横向对齐最难口角竖距≥.18，统一允许速度至少 `sqrt(.18²-R_w²)>.05`；lower 机器人向下段 floor slack 至少 h-.026>0；相遇前的 pair 水平距离≥.59；之后转入 B/C 的已证域。完整运动速度均>.025，阶段尾部短时低速不会累计5s hold。24个冻结实值的上界为695–775步。

这给出从普通冲突初态开始、无需初始 y_e>0 的**条件性几何/CBF参考 Yield–Hold–Pass 存在命题**。它没有证明原 Flow+η 策略会选择这些动作，不能称为其 guaranteed escape theorem。

## 7. 非退化性及覆盖：定量结果与诚实分母

### 7.1 区域相对大小

选一个可复核的比较域，包含两个机器人在主通道窄中心带、先后顺序尚未交换的全部 buffered-safe 配置：

\[
\Omega_{\rm conflict}=\{|x_i|\le a,\ |y_i|\le h,\ x_1<x_2,\ s^2+z^2\ge D_s^2\}.
\]

这是几何分母，不是采样概率分布，也不是全部可能的 bay 冲突情形。其体积精确积分为

\[
\int_{-2h}^{2h}(2h-|z|)\,{(2a-\sqrt{D_s^2-z^2})^2\over2}\,dz.
\]

根协调器用解析圆积分、有理数、带余项界的 asin/Machin π 级数及整数平方根区间独立认证。以实际加载 binary64 参数作为精确实输入，得到

`V_conflict ∈ [0.00598278019616768260552945468595134000271121, 0.00598278019616768260552945468595134000432684] m⁴`。

因此 `V_entry/V_conflict≈0.00200575645545`，即 **0.2005756455%**。这是 **NUMERICALLY CERTIFIED** 的几何体积比；不是 rollout 成功概率。K_pass 位于 bay，并非 Ω_conflict 的子集，不称其体积比为该初始域的覆盖率。

另对全部联合几何自由空间 F，有 `2 V_conflict≤Vol(F)≤Area(Ω)²`，`Area(Ω)≈.1671822852 m²`。对应 K_entry 的全域体积占比界约 **0.04293%–0.10029%**，K_pass 约 **0.28336%–0.66190%**。这些界仍不意味着实际冲突状态按体积均匀采样。详细严格区间见 [arithmetic_certificate.json](arithmetic_certificate.json)。

判断应为：**有限尺寸、正体积且参考可达，但相对全配置域范围较小；真实策略对全域可达性未建立。** 不能只凭“体积>0”回应审稿人的实际意义质疑。

### 7.2 实际旧数据覆盖

**EMPIRICALLY OBSERVED：** H/confirmation_manifest 的24个 Toy 初态，H/confirm_q16_tasks 的16个固定候选，每个候选只读 seed index 0，TT/FF 各384条，共768条完整旧轨迹。每条 episode 的 η 固定；不同候选没有当作一次 episode 的切换协议。

读取333,491条位置记录，768对JSON/NPZ全部存在，无截断/缺失/数值异常/碰撞；384对 state×η 的 TT/FF 共享噪声前缀逐键一致。未重新生成 PRNG keys 来重新认证整个历史生成器；源代码和 manifest 的噪声语义已核对。全部源/索引/结果文件哈希在 D 的 results.json。最大 saved dynamics residual 为1.11×10^-16。

| 统计项目 | TT | FF | 分母及含义 |
|---|---:|---:|---|
| 初态已在 K_entry | 0/24 | 0/24 | 独立物理初态，非384个独立状态 |
| 初态已在 K_pass | 0/24 | 0/24 | 相同24个初态 |
| 可按本报告安全参考进入并通过 | 24/24 | 24/24 | 与 placement 无关的参考存在性，不是控制观察 |
| episode 曾访问 K_entry | 83/384（21.61%） | 95/384（24.74%） | 有限候选×状态，seed0 |
| episode 曾访问 K_pass | 0/384 | 27/384（7.03%） | 保守角色/净空域的访问，不是所有 safe pass |
| K_pass 后保持 z>D_s 到 s≥.34，且无碰撞 | 0/384 | 27/384 | 这27条均最终success |
| 先 K_entry 再 K_pass 并保持到出口 | 0/384 | 13/384（3.39%） | 明确完整指定区域次序 |
| 全任务 success | 220/384 | 172/384 | 原 evaluator 完整任务标签 |
| 原 detector deadlock | 112/384 | 192/384 | 不由 timeout 推断 |
| timeout | 52/384 | 20/384 | 保持独立结局 |
| collision / numerical / missing / truncated | 各0/384 | 各0/384 | 不丢弃异常改变分母 |

每一个固定 η 的24状态统计在 [D报告](agent_d/report.md) 的逐候选表及 [coverage.csv](agent_d/coverage.csv)，不能用总数推断每个 η 都可达或 B15 可靠性。只是一个已打开、非前瞻留出、单seed有限面板。

TT 的220条success不经过所选 K_pass，说明该保守域**不是安全成功通过的必要条件**，也说明其覆盖能力有限。FF 在该域访问更多同时全任务成功更少，不能推出 FF 全局更好。

**η* 完整 episode 的 Coverage_TT/FF：UNKNOWN，不是0%。** η* 不在这16个历史候选中；不能用候选4或其他固定候选替代。对该 witness 的已知新增证据仅为24条动作探针。

## 8. 三项核心判定与原始假设处理

| 核查项目 | 数学结果 | 实际 Toy 结果 | 缺口 |
|---|---|---|---|
| Yield 可达性 | PROVEN：K_entry及E允许从零偏移进入bay，全17行CBF+球 | EMPIRICALLY OBSERVED：冻结Flow的有限零偏移首动作可启动 | 原固定η持续进入、统一seed条件、管道不变性 UNKNOWN |
| Pass 可执行性 | PROVEN：K_pass有统一安全参考速度；η* raw两正有严格下界 | EMPIRICALLY OBSERVED：27条历史FF候选轨迹访问该域后保持到出口 | η*实际进度、hold、uniform TT/FF保证 UNKNOWN |
| 区域非退化性 | PROVEN：有限宽、正体积、E参考可达；NUMERICALLY CERTIFIED体积比 | 有限面板的region coverage已明确，0初始membership但非不可达 | 全冲突人口覆盖未知；保守入口占主通道比较域约0.2%，不能称广泛 |

原假设逐项处理：

| 原假设/推断 | 处理 |
|---|---|
| 几何/安全参考可达性必须初始 y2≥y_e>0 | **成功删除，PROVEN**：允许负、零和正初始偏移 |
| 原式各坐标极值能构成完整安全矩形 | **FAILED**：必须保留 Ω 分支及 pair 耦合 |
| 只检查 pair 或两个轴向墙面即可 | **FAILED**：口角实际限速；本次保留八墙分支 |
| 始终用 y2 当作 clearance | **减弱为实际 z=y2-y1**；y1负值可帮助安全通过 |
| y_e 可统一替换成一个 universal 正 z_seed | **UNKNOWN/不能认证**：仅有固定族 raw 数值，不是总动作阈值 |
| η* 在零偏移靠原三列产生严格正 raw yielding | **FAILED**：横向raw为0；实际启动依赖Flow |
| 原控制在整个域持续正yield、hold及pass | **仍是未证明的执行假设**，不能靠参考可行性删除 |
| 足够剩余时间、未提前终止 | **仍必要**；本次只认证特定参考速度和监测条件 |
| 全候选η、全noise都有同样符号保证 | **FAILED于raw全候选说法；实际全noise UNKNOWN** |
| Pass自动蕴含Resume及完整任务success | **UNKNOWN/无此认证**，必须拆开 |

因此，不再需要 y_e>0 作为真实几何或参考可达性的前提；但本次也没有得到一个可以直接替换现有 TT/FF guaranteed-escape 证明的、已经验证的通用执行条件。不能据此反向声称正 y_e 必然是控制器逃逸的必要条件。

## 9. 对 Proposition 5、式 (15)、Part 3/4 的修改建议

**CONDITIONAL：论文适用性说明。** 用户确认 Proposition 5 名为 *Lateral Reconfiguration Necessity*，式 (15) 为本任务给出的 K 定义；并提供文件名 `C1_Narrow_Passage_Part1_to_Part6_v0_2.md`。在本仓库及两个指定冻结/历史根按具体文件名搜索未找到。ChatGPT 文件库中的全文未取得，故以下是对所述内容的结构性建议，**不是论文全文逐字审计**。CONDITIONAL 不代替前述五种证据等级。

### 9.1 Proposition 5：保留必要性，明确其不是逃逸充分性

可由真实几何 **PROVEN** 的定量版本：连续 buffered-safe 换序若从 s<0 到 s>0，必有 s=0，此时 `|z|≥D_s`。若两者都留在主通道中心带，则 `|z|≤2h=.0564667<D_s`，不可能。因此至少一台机器人必须离开这条中心带，并在换序时达到

\[
y_{\rm upper}\ge D_s-h=.291866666666667>.2\ \mathrm m.
\]

换序时上方机器人必在高侧袋，其 x 必在 ±h 范围内。该命题不指定一定由机器人2让行，不要求起始 y_e>0，也不声称 Flow 会完成所需重构。若原 Proposition 5 只陈述这种必要性，其核心不需要否定，建议补真实常数、角色/分支和安全阈值；若它额外暗示原策略会实现，则须拆分。仅针对 evaluator collision-free 而非 buffer-safe 集合时，应改用对应无buffer阈值，不能无说明复用 D_s。

### 9.2 式 (15)：不宜继续单独当作完整认证区域

建议将原式作为带角色选择的外容器，显式加 `p_i∈Ω`、`s²+z²≥D_s²`；定量结论使用第3节的 K_entry、Yield 管道和 K_pass。h/H/高bay宽可由几何给出；S、D_exit、余量及entry选择必须承认是选择值。将入口 y_e>0 与“已完成部分让行”的通过区域前提分开；禁止用通过域的高度下界倒推普通初态已经可达。

### 9.3 Part 3/4：哪些可保留，哪些要重新证明

| 证明内容 | 建议（均为对原文的 CONDITIONAL 适用性判断） |
|---|---|
| 窄通道无法在原主通道带内换序的必要性 | 保留，加入上述真实定量阈值 |
| 参考几何路径存在、完整CBF允许路径 | 可使用本报告全域证书，明确标reference existence |
| 通过域 η* 原始Bη正进度 | 有全域新证据，可替换人工例子；不可推广所有η |
| Safety QP realization | 只能保留附有真实完整活跃约束及总动作条件的命题；可行动作存在不代表QP选择它 |
| FF recovery、十子步正进度传播 | 仍需真实Flow及约束上的单独条件/认证，有限首动作不能补齐 |
| Yield后保持横向净空、K不变性 | 当前 UNKNOWN；必须新增证明义务，不能由δ2y>0自动推出 |
| 从普通冲突状态完成Yield–Hold–Pass | 可建立原安全集下的参考存在命题；不能识别为原TT/FF闭环定理 |
| 从Pass出口到双目标完整成功 | 必须独立证明Resume、事件和时间，不自动继承 |

可以形式化“若原实际动作满足域不变性、正进度下界和剩余时间，则有限步通过”的条件命题；本次没有认证它的原控制器前提。不要把这些更强响应条件写成已验证结果来保住原充分性结论。

## 10. 审计文件、复现与实际成本

| 证据/实现 | 关键位置 |
|---|---|
| checkpoint环境恢复 | F/validation_v2/adapters.py:55–64 |
| 墙、目标及workspace分支 | F/source/single_integrator/environment.py:114–138 |
| 观测包含previous velocity | 同文件:128–133；K的体积仅配置空间，不是全8维状态盆地 |
| 有限墙最近点与所有CBF行 | F/source/single_integrator/cbf.py:51–93 |
| Toy切平面主投影 | 同文件:96–173；retry helper实际路径见manifest |
| 原三列、radial bound及perp | F/source/shared_control/basis_families.py:129–152；diagnostic_corrector.py:10–40 |
| 共享anchor及FF递推 | F/poc/field_controller.py:215–264 |
| TT/FF dispatch | F/validation_stage2/factorial_controller.py:23–29 |
| 真实碰撞、deadlock及终止 | F/source/single_integrator/environment.py:183–228 |
| 基础初态采样 | F/source/scripts/giveway_initial_state.py:4–14 |
| 困难初态分布、任务身份及noise | H/design.py:29–40,91–94；H/runner.py:20–36 |

基础采样约 `x1∈[-.88,-.82],x2∈[.82,.88],yi∈[-.002,.002]`，先float32存储。困难tier1两x共用scale∈[.72,.80]，yi再加[-.018,.018]；坐标相关，不能当独立均匀盒。真实初态以冻结manifest为准。

成功需两goal误差都≤.08；碰撞检查含swept轨迹；deadlock需2s窗口净进展绝对值均<.01且最大速度<.025，持续候选5s；优先级collision→success→deadlock→timeout。不存在把timeout改成deadlock的重标注。

从 `/home/zhihan/research/Basin_C1` 执行（B脚本将重新做已声明的24动作，其余主要为只读算术）：

```sh
C1_K_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/audit_inputs.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/agent_a/certify_geometry.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/agent_b/compute.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/agent_c/certify_pass.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/agent_d/compute_coverage.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/certify_arithmetic.py
"$C1_K_PY" diagnostics/toy_K_quantitative_certification_20261017_v1/verify_and_manifest.py
```

原始各agent报告独立保存；最终跨文件一致性和冻结源不变性见 [verification.json](verification.json)。

本次新增成本：A一次CPU runtime配置/checkpoint加载，模块导入外约.823s；B24动作，6 native samples（60内部网络步）、12 FF积分（120场调用）、138 projector requests，约1.166s measured wall/1.638s CPU；C静态算术约.0013s；D重算768旧轨迹、1559唯一读文件，约1.744s wall/1.744s CPU。这些是各脚本测量段，不冒充整个对话的端到端耗时。主协调器另执行有理区间及最终校验，无额外Flow/投影。**新增训练0、rollout0、物理step0、GPU0、数值失败0；未修改任何已有源、checkpoint或结果。**
