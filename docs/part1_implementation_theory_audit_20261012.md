# C1 Part 1 — Implementation–Theory Consistency Audit (2026-10-12)

## A. Executive Verdict

**结论：三层命题作为有条件的数学陈述自洽；作为当前 C1 全部 placement 的统一执行公式则不成立。** TT 的单步动作映射直接符合欧氏投影模型（固定物理状态、Flow 噪声及安全参考时）；FT 可把基准换成其自身的安全场积分终点后使用同一定理。TF、FF 的修正进入 Flow 递推，不能直接使用单次投影公式。Part 1 可以作为**终端修正机制**的论文说明，不能无范围限定地称为整个 OrthoFlow3/四 placement 的理论。

| 层/结果 | 数学命题 | 当前实现适用性 |
| --- | --- | --- |
| Layer 1：完全消除、切锥方向导数、非扩张界 | PASS，针对精确欧氏投影及其条件 | **CONDITIONAL**：TT；FT 以自身基准重写后。TF/FF **NOT APPLICABLE** |
| Layer 2：有效方向与双向完整保留方向 | PASS，二者必须区分 | **CONDITIONAL**：TT/改写后的 FT；一般 SOCP 不可整体按多面体处理 |
| Layer 2：`rank(B)-rank(A_I B)` | PASS，仅多面体切锥及完整活跃线性行的 lineality | **MISMATCH** 若用它计数当前含速度球的全部有效方向；速度球活跃时还需其法向，且公式只计 `S` |
| Layer 3：局部进展展开及正对齐充分条件 | PASS，给定可微 `P_x` 与正系数 | **CONDITIONAL / NOT VERIFIED**：动作映射仅在上述范围匹配；未知 `g`、正对齐未验证 |

这里的 PASS 评价**定理自身**，右列评价**对代码的应用**。候选集实际为 `H=[0.5,1.25]×[-0.5,0.5]×[0,0.75]`、实验为其中 K16，`0∉H`；代码函数允许传入零与双侧小扰动，所以零点命题是合法的**诊断性扩展**，不是已评估候选的内点定理 [历史四链审计](../diagnostics/placement_action_family_audit_v1/final_report.txt) 第 125–133 行。

## B. Actual Control Pipeline

### 证据来源与复用

本次先检查了 [四链动作审计](../diagnostics/placement_action_family_audit_v1/final_report.txt)、[边界机制代码审计](../diagnostics/placement_boundary_mechanism_audit_v1/code_audit.txt)、[性质扫描](../diagnostics/placement_theory_property_scan_v1/final_report.txt)、`/home/zhihan/research/Basin_C1_hard_field_20261006/confirmation_manifest.json`（Toy/Ring 各 24 个冻结状态）、同根目录 `transfer_mechanism_v2/state_dependence_control/*.npz`（`safe_nominal`、`basis`、四链动作与局部差分），以及 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_v2/assets/{toy_ckpt.pkl,db_ckpt.pkl,db_manifest.json}` 和 `poc/assets/diagnostics/*_stage1` 的既有 checkpoint/manifest。旧证据与冻结 source、placement、固定噪声语义兼容；本次直接计算 SHA256，外部 source 与当前仓库的 `basis_families.py`、`hard_projection.py`、`single_integrator/cbf.py` 逐字节相同（分别为 `f78c1b37…`、`847f7045…`、`f127d09c…`）。这些证据覆盖真实执行链与 Toy/Ring 局部响应；缺口是 Part 1 的切锥及范数命题未按明确前提汇总，故仅复算四个缓存状态的投影，无新训练、rollout 或视频 batch。DB 的执行链由代码核验；本轮未补跑 DB 的模型前向。

固定 `x`、checkpoint、Flow key 及其高斯样本 `ξ`，原生 Flow 先从 `z_0=ξ` 做 `K=10` 次 Euler；解码后逐分量 clip、逐机器人速度球径向限幅，Ring 再转世界坐标并以 float64 重新限速，得 `a_ref` [场控制器](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py):139–158,198–213；Toy/DB adapter 的对应分支在同一外部目录 `validation_v2/adapters.py:75–108,125–149`。此处的 clip 发生在**首次安全投影之前**，不作用于最终执行动作。

安全集在固定物理快照下为

`U_x={u∈R^{2N}: A_x vec(u)≥b_x, ||u_i||₂≤v_max (每个 i)}`。

`A_x,b_x` 由两两 CBF 与有限墙段最近点 CBF 给出；只读位置、墙、半径、速度上限和固定 CBF 参数，不直接依赖 nominal、`ξ`、`η` 或上一速度 [通用投影](../shared_control/hard_projection.py):72–151,178–235；[Toy 投影](../single_integrator/cbf.py):51–104。Ring 用 48 边形安全代理而物理碰撞评价用真圆 [Ring 安全代理](../ring_exchange/safety.py):19–60、[Ring 环境](../ring_exchange/environment.py):294–306。优化目标为 `min_{u∈U_x} 1/2||u-target||₂²`，单位 Hessian；无加权范数、slack 或 penalty。通用路径用 Clarabel SOCP，失败抛错；Toy 用 SLSQP 外切速度球并验 KKT，重试保持同一优化目标 [通用投影](../shared_control/hard_projection.py):198–267、[Toy 投影](../single_integrator/cbf.py):96–163、外部 `poc/field_controller.py:69–93` 与 `validation_v2/adapters.py:16–47`。`project_field_target` 的速度球快捷路径在所有线性行也可行时恰是交集的精确投影；否则回到原投影器。

共同准备为 `a_safe=Π_{U_x}(a_ref)`，`δ=B(x,ξ)η`。`B` 三列为限速目标方向、`3.303687...` 倍的 `a_safe` 对目标方向的平滑正交残差、平均限速相对方向 [基](../shared_control/basis_families.py):114–152、[限速基](../shared_control/diagnostic_corrector.py):10–40；外部 `poc/field_controller.py:215–220`。固定一次物理动作的 `x,ξ,a_safe` 时 `B` **严格线性**作用于 `η`，没有需线性化的 correction 余项；跨物理步则重新计算 `B`。因此写 `B(x)` 时必须将 Flow key/安全参考作为条件变量，或记 `B(x,ξ)`。

令 `H_x^0(δ)` 为从未裁剪的 Flow 高斯端点 `h_0` 出发，十步积分 `f_x(h_k,s_k)+δ` 的终点；`H_x^F(δ)` 为每步令 `q_k=h_k+(1-s_k)(f_x(h_k,s_k)+δ)`、`c_k=Π_{U_x}(q_k)`、`h_{k+1}=(1-ρ_k)h_k+ρ_kc_k` 的终点，`ρ_k=min(1,(1/K)/(1-s_k))`。内部 `h` 是**动作空间的伪时间变量**，物理状态保持冻结 [场控制器外部源码](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py):222–246。四分派的真实公式 [factorial_controller 外部源码](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py):23–40 为：

| placement | 最终执行动作 | 零点基准 |
| --- | --- | --- |
| TT | `Π_U(a_safe+Bη)` | `a_safe` |
| TF | `Π_U(H_x^0(Bη))` | `Π_U(H_x^0(0))` |
| FT | `Π_U(H_x^F(0)+Bη)` | `H_x^F(0)∈U`（精确算术） |
| FF | `H_x^F(Bη)` | `H_x^F(0)∈U`（精确算术） |

首字母是安全层位置，第二字母是修正位置。FT **确有终端 closure**；FF 最后一步 `ρ_{K-1}=1`，故末端安全，仅再 `assert_safe`，没有隐含终端投影。TF 的末端虽有投影，其输入 `H_x^0(Bη)` 对 `η` 一般非线性，且零点不必等于 `a_safe`；FF 是投影与非线性 Flow 的复合递推。真实 continuation 每步调用上述动作，随后 `env.step(u)` 直接用 `p⁺=p+dt·u, v⁺=u`；超速/非有限报错，无后续 clip [四链 runner 外部源码](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/run_rollouts.py):23–49、[Toy](../single_integrator/environment.py):170–181、[DB](../double_bottleneck/environment.py):346–358、[Ring](../ring_exchange/environment.py):281–293。生产终端 OrthoFlow3 也明确是首次投影、基修正、最终投影、`env.step` [四场景运行器](../new_benchmark_common/safety_eta3.py):310–380、[DB 历史入口](../diagnostics/double_bottleneck_eta3_basin/tools/run_jobs.py):102–120。

上述四链动作路径未见 at-goal 动作 guard、终端额外裁剪或以伪动作代替失败求解的 fallback；终止后 runner 停止下一步，求解异常记录 `numerical_failure` [四链 runner 外部源码](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/run_rollouts.py):27–47。此结论限于所审计的冻结 placement 入口，不代称仓库全部实验脚本。

## C. Assumption Audit

| 假设/结论 | 判定 | 代码证据与数学影响 |
| --- | --- | --- |
| 固定 `x` 的闭凸 `U_x` | **PASS** 对已定义几何：线性半空间与闭速度球交；**CONDITIONAL** 非空 | `hard_projection.py:137–151,211–235`、`cbf.py:82–104`。安全初态且 CBF `h≥0` 时 `u=0` 可行；任意异常/不安全快照不保证非空。若无法认证可行，求解器报错而非返回动作。 |
| `u_0∈U_x` | **PASS** TT；**CONDITIONAL** FT/FF；**MISMATCH** 若把共同 `a_safe` 称为 TF/FT/FF 的零点动作 | `field_controller.py:215–256`、`factorial_controller.py:23–40`。FT/FF 终点有安全断言；数值容差内可行，不是数学上的精确成员声明。 |
| 固定同一安全集 | **PASS** 单物理步 | 投影约束由快照位置计算，`field_controller.py:190–196,215–220`；伪时间内安全对象不重建。物理步间 `U_x` 变化。 |
| 欧氏投影且无软化 | **PASS** 优化结构；**CONDITIONAL** 数值精度 | 单位二次目标、硬 CBF/球约束、无 slack：`hard_projection.py:178–267`、`cbf.py:96–163`。重试为同一问题，非加权投影；不可按多面体整体处理。 |
| `Bη` 固定线性 | **PASS** 固定 `(x,ξ)` 的一次动作 | `basis_families.py:40–51,114–152`；`field_controller.py:215–220`。`B` 依赖首次安全动作，后者依赖 nominal 和 `ξ`；`B(x)` 若仅指物理状态则 **MISMATCH**。 |
| 零点是参数内点 | **MISMATCH** 对实际 K16/H；**PASS** 对代码的诊断延拓 | `H` 与零点见历史四链审计第 125–133 行；correction 接口可接任意有限三维 `η` [基](../shared_control/basis_families.py):46–51。论文要明确局部命题不直接断言已部署候选盒的零点邻域。 |
| 完全消除 `Π_U(u_0+Bη)=u_0 ⇔ Bη∈N_U(u_0)` | **PASS** 定理；**CONDITIONAL** TT/重基准 FT；TF/FF **NOT APPLICABLE** | 采用外法锥 `N_U(u_0)={v:vᵀ(z-u_0)≤0,∀z∈U}`。这是欧氏投影最优性条件；有限求解容差使数值“等于”须给阈值。 |
| 方向导数 `Π_T(Bd)` | **PASS** 定理；**CONDITIONAL** TT/重基准 FT；TF/FF **NOT APPLICABLE** | 闭凸集上从集合内 `u_0` 出发的 Hadamard/单侧方向导数为切锥投影。它一般是**非线性锥投影**，边界双侧 Jacobian 未必存在；`code_audit.txt:67–71` 的 TT 折点实证与此一致。 |
| `||u_η-u_0||≤||Bη||` | **PASS** 定理；**CONDITIONAL** TT/重基准 FT；TF/FF **NOT APPLICABLE** | 同一 `U` 的欧氏投影非扩张。TF/FF 的 Flow 反馈没有此 1-Lipschitz 保证。 |
| `V_eff=Π_T(Im B)`、`S=Im B∩T∩(-T)` | **PASS** 作为 TT/FT 零点**方向集合**；**CONDITIONAL** 应用 | `V_eff` 是投影所得集合，未必线性空间；`S` 只计双向完整保留的输入方向。最小反例：`U={u:u₁≥0},u_0=0,B=I`，`V_eff=T` 为二维半空间，而 `S={u:u₁=0}` 仅一维。 |
| `dim S=rank B-rank(A_I B)` | **PASS** 在多面体切锥 `T={v:A_Iv≥0}`；当前 SOCP **CONDITIONAL** | 公式来自 `S=ImB∩ker A_I`，需将 `A_I` 定义为**全部活跃线性面**。当速度球活跃，加入其切平面法向才能计对应一阶 lineality；一般弯曲球面不能因此变成多面体。不能把该秩称为 `V_eff` 维数。 |
| `P_x` 可微及 `gᵀΠ_T(Bd)>0` | **NOT VERIFIED** | 代码只有目标、碰撞、终止等评价，没有用户假设的可微 `P_x` 或 `g`。成功率/二值标签不验证梯度；`Proj_S g≠0` 只是可选择双向 `d` 的充分条件。 |

Layer 3 的严格逻辑是：若 `P_x` 在 `u_0` 可微且相同固定状态动作映射具上述方向导数，则 `P_x(u_{εd})-P_x(u_0)=εgᵀΠ_T(Bd)+o(ε)`；严格正系数才推出足够小 `ε>0` 的改善。该陈述没有闭环死锁逃逸或有限候选成功保证。

## D. Numerical Verification

**方法与适用范围。** 从上述历史缓存读取 Toy/Ring 各 24 个真实共态初始状态的 `safe_nominal` 和 `basis`；选择每场景最接近活跃、以及裕度最大的各一例。冻结场景配置与已有 checkpoint（运行时仅用于重建环境和安全快照），CPU 调用原投影器，在 `η=0` 及固定单位方向上 `ε=0.01,0.003,0.001`；活跃点另测 `0.0003`。无新 Flow 前向、无物理步/rollout、无数据覆盖。此检验针对 TT 的**后段投影**；旧 `independent_directional_probes.json` 和 [边界机制报告](../diagnostics/placement_boundary_mechanism_audit_v1/final_report.txt):64–92 已覆盖真实模型的共态 TT/FF 局部调用。DB 的轻量数字未新增，故其 Part 1 数字证据为 **NOT VERIFIED**，其同构硬投影代码结构已核验。

| 场景/UID | 基准最小裕度 | `Π_U(u_safe)` 与 `η=0` 差 | 选定方向 `ε=.01/.003/.001` 的 `||Δu||/(ε||Bd||)` | 约束残差/速度 |
| --- | ---: | ---: | --- | --- |
| Toy `820012`，活跃线性行 13 | `−6.94e−18` | `0` | 另选外向方向：`0.631069/0.631069/0.631069` | 最小线性残差 `0`，速度余量约 `0.362` |
| Toy `820016`，内部 | `0.02457` | `0` | `1/1/1` | 最小线性残差 `≥0.02439`，速度余量 `≥0.4318` |
| Ring `920004`，活跃速度球 2 | `4.92e−10` | `0` | 另选外向方向：`0.872230/0.872264/0.872274` | 最小线性残差 `≥0.35818`，最大速度超量 `≤0`，约 `1e−11` 数值级 |
| Ring `920016`，内部 | `0.06344` | `0` | `1/1/1` | 最小线性残差 `≥0.38359`，速度余量 `≥0.06344` |

对 Toy 活跃行取外法向 `n=-A_13`、`d∝Bᵀn`，真实切锥在该点为 `nᵀv≤0`，`Π_T(Bd)=Bd-max(0,nᵀBd)n/||n||²`。`nᵀBd=0.122018`，在四尺度有限差分与此预测的绝对误差依次为 `1.22e−15,2.94e−15,7.19e−15,3.09e−14`。对 Ring 活跃球取其外单位法向，`nᵀBd=0.650086`；同一切半空间预测误差为 `9.69e−4,2.93e−4,9.77e−5,2.51e−5`，相对误差 `8.35e−4→2.16e−5`，符合弯曲球面的一阶收敛。内部点切锥为全空间，测试方向有限差分等于 `Bd` 至浮点级。对这些样本未观测到范数界违反；此为必要一致性检查，不代替结构证明。

容差取冻结配置：`solver_ftol=1e−12`、`feasibility_tol=1e−9`、`speed_tol=1e−10`、`optimality_tol=2e−6`、`max_iterations=200` [配置](../shared_control/hard_projection.py):20–30、[Toy 配置](../single_integrator/cbf.py):16–29。Ring 最小 `0.0003` 尺度误差可能混有约 `1e−11/ε` 的求解底噪；不能把有限差分偏差当成解析定理错误。完全消除的双向数值判据未在这四个样本找到 `ImB∩N_U(u_0)` 的非零证人，故其**数值实例**为 NOT VERIFIED，结构性最优性证明仍 PASS。接近活跃集切换的状态由旧 [性质扫描](../diagnostics/placement_theory_property_scan_v1/final_report.txt):12–38 覆盖：`72` 个动作中心在 `ε=.0003` 有 `71` 个活跃签名稳定，而 Toy `820006` 的 TT 双侧导数相对差约 `1.958`；这说明不能用固定 Jacobian 代替单侧切锥投影。旧 [边界机制报告](../diagnostics/placement_boundary_mechanism_audit_v1/final_report.txt):64–100 中 `η=0` 的 TT/FF 基线差中位为 Toy `0.000176`、Ring `0.00367`，直接否定四链共用一个零点动作。

## E. Required Mathematical Corrections

1. 将主式限定为固定 `(x,ξ)` 的 **TT**：`u_TT(η)=Π_{U_x}(a_safe+B(x,ξ)η)`。**FT** 可用 `u_0=H_x^F(0)∈U_x` 改写为同式；不要把 `a_safe` 当作其 `u_0`。
2. 给 TF/FF 单列真实复合映射 `Π_U∘H_x^0∘B` 和 `H_x^F∘B`。在正规分支可讨论各自方向导数/链式导数，但不宣称单次投影 normal-cone 消除、切锥公式或 `||Bη||` 非扩张界。场修正本身严格线性，**最终映射**一般非线性。
3. 把 `x` 的条件写明包含固定的 Flow key/nominal 内部变量，或明确使用 `B(x,ξ)`；把 `η=0` 指定为代码允许的诊断延拓，实际盒 `H` 无零点邻域。
4. 将 `rank B-rank(A_I B)` 限于多面体模型下 `S` 的维数；对实际 CBF+速度球安全集，需要所有活跃线性 CBF 法向与活跃非零速度球法向描述一阶 lineality，且不得以此代替全部 `V_eff`。方向响应保持锥投影，不能无条件写常 Jacobian。
5. Layer 3 保留为对未知可微 `P_x`、未知 `g` 及正对齐的**条件命题**；论文不得从成功率反推这些前提。

## F. Final Recommendation

1. **数学推导本身自洽**：在非空闭凸集、欧氏投影、可行基准、局部参数扰动及 Layer 3 的可微/正对齐前提下成立。
2. **可作为当前 C1 的严格机制说明，但范围是 TT 和重基准后的 FT 的局部动作映射**，以及四链共有的“修正被安全过滤”的定性背景；不是 TF/FF 的完整公式。
3. 论文可直接写欧氏投影的法锥消除、切锥单侧方向导数和非扩张定理，须在同一处标注 TT/FT 的执行条件；可写 `V_eff` 与 `S` 的区别及充分进展命题的条件形式。
4. 秩式需多面体/活跃面限制；数值误差、实际候选盒与固定噪声条件须声明。Layer 3 的 `g`、正向对齐与可微进展均标 **NOT VERIFIED**。
5. **存在必须修复的理论适用范围冲突**：若当前文本把 `u_η=Π_U(a_safe+Bη)` 称为四链共同公式，或把 SOCP 的 `rank(A_I B)` 当全部有效干预维数，该说法与执行代码明确冲突；无需改控制器。
6. **可在完成 E 的最小文字/公式限定后冻结 Part 1 的终端机制版本**。未经这些限定，不宜以“C1 当前全部 placement 的理论”正式冻结。

