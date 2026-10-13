# C1 Part 2：闭环灵敏度与 basin reshaping 数值核验

日期：2026-10-12。范围：冻结的 Toy Give-Way、Ring Exchange 控制器；固定初态、整段固定 η、逐物理步重算 B、固定 Flow key/未来噪声。**总判定 PARTIAL。** Ring 的正规固定噪声局部边界链有实证支持；Toy 在被测切换处为 NONSMOOTH。现有局部导数不能预测有限幅度 Q16/B15 成员变化，不能把完成度代理盆称作真实鲁棒盆。

## 1. 旧证据与新增成本

|证据|路径与兼容性|本轮复用及缺口|
|---|---|---|
|四链代码/权重|`/home/zhihan/research/Basin_C1_flow_field_poc_20261004` 的 `validation_stage2`、`poc`，冻结 checkpoint|直接复用；没有训练|
|H 确认评估|`/home/zhihan/research/Basin_C1_hard_field_20261006/results`、`rollout.sqlite`、`confirmation_manifest.json`|48 初态×16 η×16 seed×4 链 = 49,152 条，逐文件核验；用于同样本 gain/loss 与 seed0 完整轨迹|
|旧 M/G、S、边界|`diagnostics/placement_boundary_mechanism_audit_v1/`|复用 TT/FF H80 数组、held-out 预测、H4 边界根；旧诊断 ρ 包含碰撞净空，**不等于本轮完成度 ρ**|
|H1–H4|`diagnostics/placement_strong_hypotheses_v1/`|复用 H3 Sobol64/独立 Sobol128、Q16/B15，H4 Ring 法向和曲率及 Toy 非光滑反例|
|缺口|TF/FT 的零点 H80 灵敏度、四链 K 与 Q、按本轮定义的 ρ|先查共享缓存：56 genuinely missing、0 ambiguous、0 exact reuse；成本审计后 CPU 跑 56 条 H80 短 continuation，最多 4,480 物理步；**新增完整 continuation = 0/320**|

审计前置文件为 [`cost_audit.json`](cost_audit.json)、[`cache_preflight.json`](cache_preflight.json)、[`trajectory_protocol.json`](trajectory_protocol.json)。这些短轨迹全部于第 80 步截断，未被计为完整 success rollout。`cache_index.json` 列明旧证据和本轮文件 SHA256。研究目录中的其他未提交实验文件没有修改。

## 2. 实际公式与解析链核对

代码入口是 [`factorial_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py:23)，`Runtime.prepare` 先得到原始 Flow nominal `a_flow`、投影 `a_safe=Π_x(a_flow)`、`δ=B(x,a_safe)η`（[`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:198)）。B 为三个固定家族列；在每个状态对 η 线性，但含状态与 `a_safe` 依赖（[`basis_families.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/shared_control/basis_families.py:46)、[同文件](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/shared_control/basis_families.py:146)）。四链实际为

\[
u_{TT}=\Pi_x(a_{safe}+\delta),\quad
u_{TF}=\Pi_x(H_{0,x,k}(\delta)),\quad
u_{FT}=\Pi_x(H_{F,x,k}(0)+\delta),\quad
u_{FF}=H_{F,x,k}(\delta).
\]

`H0` 每次从真实 Flow 初始噪声 `h0` 积分，不作 field 子步投影；`HF` 每个子步投影 `h_j+(1-s_j)(f(h_j,x,s_j)+δ)`，再以凸组合更新 `h_{j+1}`；这是 [`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:96) 和 [同文件](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:222) 的真实递推。TF/FT 有末端安全投影（[`factorial_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py:30)）；FF 末端只有安全断言。`HF(0)` 已在安全集内，FT 的末端投影在精确算术中幂等。TT/FT 以及 TF/FF 的 η=0 动作一般不相同；旧 `baseline_offsets.json` 在 Ring 920013 的 TT/FF 首步差为 0.003399，因此 placement 比较必须包含零点偏移。Ring 的局部↔世界坐标变换依赖位置（[`local_frame.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/ring_exchange/local_frame.py:28)），不得从 K 漏掉。原始 Flow 的输出经过 clip、speed bound 和 world-frame 变换（[`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:198)），使 nominal 的状态导数分段。

安全算子是欧氏目标投影到 CBF 半空间与逐机器人速度球的交集，没有 slack；Toy 用 cutting-plane QP，Ring 用 SOCP（[`cbf.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/cbf.py:96)、[`hard_projection.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/shared_control/hard_projection.py:1)）。固定状态下安全集不依赖 η，但状态扰动会改变 CBF 行和右端（[`hard_projection.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/shared_control/hard_projection.py:72)）。求解失败显式报数值失败，不做静默 fallback；`FixedSafetySet` 有同问题更紧求解/打磨（[`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:69)）。

令 `Π_x(z)` 在所测正规分支的输入导数为 `P`、固定 z 的状态导数为 `Q`，`C_x=∂_x(Bη)`（包含 `a_safe` 的导数），`D_xH` 为固定 δ 的积分导数，则**核对后**

|链|`M=∂u/∂δ`|`G=MB`|`K=∂u/∂x`（固定 η）|
|---|---|---|---|
|TT|`P_TT`|`P_TT B`|`Q_TT+P_TT(∂_x a_safe+C_x)`|
|TF|`P_TF D_δH0`|`M_TF B`|`Q_TF+P_TF(D_xH0+D_δH0 C_x)`|
|FT|`P_FT`|`P_FT B`|`Q_FT+P_FT(D_xHF(0)+C_x)`|
|FF|`D_δHF`|`M_FF B`|`D_xHF+D_δHF C_x`|

TF 的 `D_xH0`、FT/FF 的 `D_xHF` 必须包含 `h0(x)`、Ring frame、Flow 输入、子步反馈与投影 Q；`∂_x a_safe=Q_safe+P_safe ∂_x a_flow`。固定噪声、分支正规时，物理状态 `x_t=(positions,velocities)`、`F(x,u)=(positions+dt·u,u)`，故 `S_0=0`、`S_{t+1}=(F_x+F_uK_t)S_t+F_uG_t`。如求解/clip/最近墙/监视器/终止分支切换，则须改用方向导数或分段差分；单一双侧 Jacobian 不成立。

## 3. Validation A：P/Q/M/G/K/S

[`jacobian_validation.csv`](jacobian_validation.csv) 记录 Toy 820001/820013、Ring 920004/920005 的四链三个尺度、活跃约束、KKT 残差、绝对/相对误差及余弦；S 行含旧 TT/FF 和新 TF/FT held-out。**P/Q 的独立扰动只在 TT 的零点与 η[4] 投影目标执行**；其余链的末端 P 通过 M 的完整动作差分间接检查，TF/FT 末端 Q 未分别独立核验。P 采用活跃 CBF 半空间加速度球曲率的 KKT 微分；Q 对固定目标的 KKT 状态微分，CBF 几何导数另以中心差分计算。**K 是乘法法则分解后以中心差分求 frozen-policy 与 `B_x` 项的半解析交叉检查，不是独立的完整符号/AD K。** M 的无场 Flow JVP 与 FF 子步递推复用旧 AD/KKT 分支；所有中心差分固定相同 key。

|层与状态|尺度、误差和判定|
|---|---|
|正规候选 η[4] 的 P/Q|状态 ε=`10^-3,3×10^-4,10^-4`。Toy 活跃 2 条 CBF 时 P 误差约机器精度、Q 相对误差最大约 `2×10^-6`；Ring 活跃 4 个速度球时 P 最大相对 `3.86×10^-4`（最小 ε，求解底噪）。Ring Q 理论为零，FD 绝对 `2×10^-6`–`5×10^-5`，不能以“相对误差 1”判失败。**PARTIAL**（有限状态与几何导数方法）。|
|零点弱活跃 P/Q|Toy 820013 的 TT/FT P 相对约 `0.277`、Q 有跃变；Ring 920004 TT P 约 `0.186`。活跃集边界上两侧 Jacobian 不同；**NONSMOOTH**，不是 KKT 公式在正规点的反例。|
|M/G|ε=`0.005,0.001,0.0001` 的 ambient δ 与固定 B 方向；正规 Toy 820001 误差接近机器精度，Ring TF/FF 常见 `0.2–0.3%` 数值底噪；弱活跃零点不能记通过。**PARTIAL**。|
|K|状态 ε=`0.001,0.0003,0.0001`，每场景预定首初态、η[4]；Toy 12 行相对误差中位 `4.8×10^-5`、最大 `0.00112`；Ring 中位 `0.000749`、最大 `0.00214`。包含 B 与 CBF 几何；**PARTIAL**，因缺少独立完整 AD K 和更多 held-out 状态。|
|S 与闭环递推|新 TF/FT 的 H80 ε=`0.01,0.003` 中心差分、held-out 对角 ε=`0.001`；旧 TT/FF 同协议结果复用。Ring TF/FT 第 80 步 held-out 相对误差 `0.00398/0.01019`；Toy `0.396/0.612`，对应绝对误差仅 `5.44×10^-7/9.14×10^-6`，响应范数很小且分支切换。四链在 t=`0,10,40,60` 重算 K/G 的递推 spot check：Ring action 相对误差最大 `0.01097`、Toy 最大 `0.1966`；所有 32 行原动作 replay 为 0、key 全匹配。**Ring PARTIAL；Toy NONSMOOTH**。|

详细见 [`trajectory_validation.csv`](trajectory_validation.csv)、[`recurrence_validation.csv`](recurrence_validation.csv)。这些误差随尺度的稳定区间及 `active_switch` 均逐行保存；没有从三个尺度中事后挑选最小误差作为结论。模拟器若发生事件切换，`F` 的简单线性化仅在分段内适用。

## 4. Validation B：完成度裕度

本轮统一使用 `ρ=ε_goal−min_{1≤t≤H} max_i ||p_i(t)−g_i||`，Toy `ε_goal=.08,H=850`，Ring `.09,H=700`。既有完整轨迹从初态后第 1 步起计，合法提前终止时用末状态补齐；数值失败单列，不赋连续值。H80 导数试验用**同一公式但 H=80 的前缀裕度**，不能解释为完整 success。真实 success 另取仿真 evaluator 的碰撞/目标/事件标签（Toy [`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:194)，Ring [`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/ring_exchange/environment.py:308)）。

若唯一最佳时刻 `t*` 和唯一最差机器人 `i*`，且目标误差非零，则 `∇_ηρ=−(p_i−g_i)^T S^p_{t*,i}/||p_i−g_i||`。并列且各轨迹分支可微时，方向 `d` 的导数为 `−min_{t∈argmin f} max_{i∈argmax d_{t,i}} Dd_{t,i}[d]`，其中 `f_t=max_i d_{t,i}`；事件分支另论。正裕度保持的充分界为 `ρ(η+Δ)>0` 若 `sup_{t,i}||p_i(t;η+Δ)−p_i(t;η)||<ρ(η)`；观测的 `|Δρ|≤sup` 在全部 3,817 有效 seed0 配对上无违反，但正裕度证书仅触发 1 条，不能据此推广。完整 seed0 的 3,061 条有效轨迹中 `sign(ρ)` 与真实 success 一致；另有 Ring FT 10、TT 1 条数值失败。该一致性依赖成功时目标判据及实际安全执行，**不是** ρ 为死锁证书的证据。

H80 TF 的 Ring `ε=.003` 梯度直接 FD 与 S 链差范数 `2.95×10^-5`，held-out `ε=.001` ρ 变化预测 `0.000708066`、实测 `0.000709511`；FT 对应梯度差 `0.01046`、预测 `0.0058950`、实测 `0.0059630`，有末端/活动分支影响。Toy TF 两尺度梯度差约 `0.038` 且活动切换；FT 达 `0.815–1.128`，不能宣称稳定 `∇ρ`，虽然 held-out ρ 变化绝对误差均约 `3×10^-7`（小响应）。见 [`trajectory_margin_validation.csv`](trajectory_margin_validation.csv)。**Ring 局部 PARTIAL，Toy 切换点 NONSMOOTH。**

## 5. Validation C：boundary response

只用 H4 已登记 Ring 920001/920013、H1/H4 已登记 920004/920005 线段与 key。[`boundary_validation.csv`](boundary_validation.csv) 共 24 条三尺度（920001/13 两尺度）局部法向对照；正规点直接 ρ FD 与 S 链梯度余弦接近 1、最大梯度相对误差 `1.67×10^-4`。旧 H4 的 TT/FF 法向夹角分别为 `7.87°,3.65°,3.90°,23.36°`；曲率仅 920013/920004 的小半径拟合较稳，不能泛化。Toy 820001/820013 的 deadlock/timeout 与 success 跳变约 1.3，**不存在由这些数据支持的光滑 ρ=0 边界法向**。

本轮完成度 ρ 与旧 H4 含碰撞净空的诊断 ρ 不同。旧 920004 TT/FF 根处新 ρ 为 `+0.005903/−0.009589`，920005 为 `+0.001002/−0.001558`；因此 [`boundary_response.csv`](boundary_response.csv) 的四条跨 placement 线性化是**在旧已登记根附近的响应检查**，不是新 ρ 的精确根位移。920004 TT→FF 分数坐标位移实测 `0.30740`、局部预测 `0.17956`（误差 `0.12783`）；反向误差 `0.01284`。920005 正向误差 `5.73×10^-5`、反向 `5.39×10^-4`。四条符号相同，但大位移的正向误差说明局部梯度不能外推成有限 TT→FF 精确位移。图见 [`boundary_response.svg`](figures/boundary_response.svg)。若书写 `ρ_λ=(1−λ)ρ_p+λρ_q`，其中 λ 只是数学裕度插值，**不是控制器**；本轮没有把中间 λ 当 rollout。

## 6. Validation D：同样本 gain/loss 与机制强度

[`gain_loss_audit.csv`](gain_loss_audit.csv) 对每个 scene、state、η、seed、五种有向 pair 做同样本对齐，共 61,440 行；未知数值结果单列。下表为**旧 Q16 候选集的 B15**（每 scene 24×16=384 候选，固定 16 seeds、≥15 success），不是有界盒体积分数：

|Scene|TT→TF|FT→FF|TT→FT|TF→FF|TT→FF|
|---|---:|---:|---:|---:|---:|
|Ring gain/loss/unknown|48/1/0|43/5/10|8/3/10|2/4/0|46/1/0|
|Toy gain/loss/unknown|6/111/0|10/178/0|92/9/0|20/0/0|16/101/0|

每 pair 的对称差为 gain+loss：Ring TT→FF `47/384`、Toy `117/384`；Ring 含 FT 的 10 个未决 pair **不**纳入 gain/loss。seed 级别各 pair 的分母为每 scene `24×16×16=6,144`，不能与 B15 混算；完整计数见 [`reuse_summary.json`](reuse_summary.json)。独立 H3 Sobol64+扰乱 Sobol128 的**有界盒**每状态 192 点、6 状态/scene，TT→FF 平均分数差 Ring `+0.18142`、Toy `−0.12847`、Combined `+0.02648`；最后一个描述性区间跨零。B15 的 Q16 候选分数与 Sobol 盒体积不是同一测度；图见 [`b15_gain_loss.svg`](figures/b15_gain_loss.svg)。

[`gain_loss_mechanism_seed0.csv`](gain_loss_mechanism_seed0.csv) 对 3,817 个有效配对读取真实完整轨迹，比较首步动作、共同前 80 步动作与位置、完整轨迹目标裕度。所有配对 Flow key 匹配、`|Δρ|≤sup_{t,i}||Δp_i(t)||` 无违例；11 个数值失败相关配对不被悄悄过滤为成功。TT→FF seed0：Ring gain 62/loss 11，首动作差中位分别 `0.0739/0.0545`、前 80 步最大位置差 `1.880/2.194`；Toy gain 62/loss 110，首动作差 `0.2428/0.2359`、位置差 `0.372/0.551`。动作幅度重叠，不能仅凭第一步动作预测翻转。完整 ρ 符号在有效 seed0 上复述目标达成，因此翻转时 `Δρ` 符号与标签一致是**事后诊断**，不是由局部 M/G/K 对有限 η 或 16-seed B15 的先验预测。四链有显著不同 gain/loss，但零点动作偏移、状态反馈和非线性长程事件均可能贡献。

机制解释强度：**action level** 有正规分支的 P/M/G/JVP 与活跃集证据；**trajectory level** 在 Ring H80、小扰动部分通过，Toy 切换处不稳；**margin level** Ring 正规局部通过且保持界成立，Toy 不光滑；**actual binary membership** 仅有同样本关联和局部 Ring 边界支持，尚无跨全部 Q16 的局部导数预测 B15 gain/loss 准确率，故为 `INSUFFICIENT_EVIDENCE`。这也与旧有限候选零点 M 预测误差中位 Toy `69.5%`、Ring `48.3%` 一致。旧 H4 成功不能自动补上这一级。

## 7. 可复现性、限制与论文建议

固定状态、η、key、权重与代码源路径见 [`trajectory_protocol.json`](trajectory_protocol.json)；执行顺序见 [`reproduce.sh`](reproduce.sh)。旧实验的 source/checkpoint 哈希由冻结 controller descriptor 保存（[`factorial_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py:10)）。本轮源码版本为 `a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e` 加此独立诊断目录；无模型、控制器、安全标准改动。投影可行容差 `1e-9`、最优性 `2e-6`、速度 `1e-10`，Toy SLSQP `ftol=1e-12`（[`hard_projection.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/shared_control/hard_projection.py:20)）。相对误差在理论零导数附近无解释力，应看绝对误差及事件标志。

可以用于 RSS 正文：四条真实控制链、P/Q/M/G/K/S 的**分段局部公式**、Ring 正规点法向变化与局部响应、同样本 gain/loss 及 Ring/Toy 相反的有限样本方向。必须明确固定噪声、正规活跃集、短时尺度及所用测度。完成度 ρ 仅写作代理；Toy 切换写方向/事件响应；B15 仅写实际统计，不写局部 Jacobian 的预测成功。不得声称闭环解析链已统一解释完整 basin reshaping 或死锁逃逸保证。

**六问结论：**(1) 四链 M/G 在正规分支可信；Q 与 K 的当前核验为部分，弱活跃点无统一双侧导数。(2) S 能预测 Ring 小扰动 H80，Toy 分支处不能稳定外推。(3) Ring 正规点 ρ 局部响应有支持；Toy 被测切换点不光滑。(4) Ring 规则边界的法向/短位移可预测，较大位移误差显著。(5) 真实 gain/loss 获得动作、局部轨迹、局部裕度的机制解释，尚无全候选 B15 先验分类解释。(6) RSS 只能采用上述局部、代理、场景限定的结论。

|最终 verdict|判定|
|---|---|
|Ring|**PARTIAL**：正规局部链成立；大位移与 B15 全局预测不足。|
|Toy|**NONSMOOTH**：动作传递可测，切换附近的 S/ρ/边界单一梯度无效。|
|Combined|**PARTIAL**：placement 重塑有限样本 basin 且有 gain/loss；完整解析链并未封闭到真实 B15 成员预测。|

机器可读判定见 [`verification.json`](verification.json)。
