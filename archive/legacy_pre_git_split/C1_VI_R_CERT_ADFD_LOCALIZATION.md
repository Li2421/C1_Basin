# FIXED VI-augmented R_CERT：AD/FD 失配定位报告

日期：2026-09-18。范围仅限诊断；未改生产实现、风险定义、控制器、monitor、证书、训练或干预流程。

## 结论与最小复现

失配首先出现在**第一个干预动作 action 100 的第二次安全投影**，不是 Flow-BC、动力学、monitor history、direct/VI margin、witness 投影或 aggregation。

当前冻结的 `G_phi` 是零初始化输出，所以在零扰动处 `w=u_safe`。action 100 的 `u_safe` 正好位于 CBF row 0 边界：

- `w = [0.0103925664381, 0.000238623121819, -0.00485121202029, -0.00223375146843]`；
- `Pi_U(w)=w`；最小线性 slack 为 `6.94e-18`，row 0 活动，速度上界不活动；
- forward solver 按 `min(Aw-b)>=0` 返回 `nominal_feasible`；
- custom VJP 按 `slack<1e-8` 将 row 0 纳入活动等式系统。

对归档干预方向的第一个四维块，`+h` 进入严格可行的恒等分支，`-h` 进入 row 0 活动的投影分支。两侧导数从 `h=3e-2` 到 `1e-7` 收敛到不同常数，因此该点不存在一个能同时表示两侧方向导数的普通一阶梯度。

使用 action 100 后状态的固定 cotangent，最小复现为：

| 量 | 数值 |
|---|---:|
| AD | -0.00918101508933 |
| `h=1e-3` 正向单边 FD | -0.00894161854670 |
| `h=1e-3` 反向单边 FD | -0.00918101508932 |
| `h=1e-3` 中心 FD | -0.00906131681801 |
| 中心相对误差 | 0.0130376 |

AD 等于活动投影侧的单边导数；它不是该拐点处不存在的双侧导数。换一个固定输出 cotangent 后，两侧极限分别为 `0.0518187945542` 和 `0.0673775617854`，AD 为后者，中心 FD 为 `0.0595981781698`，相对误差稳定在 `0.1155`。重复相同 forward 五次的数值范围为 0。

相同投影和 backward 在两个显式平滑对照点上正常：将输入移到 row 0 严格可行侧后，AD/FD 最佳相对误差约 `1.34e-16`；移到固定活动集侧后为 0（其余步长误差约 `1e-15–1e-10`）。这排除了稳定区内的 custom backward 公式错误，也排除了 solver 抖动造成上述平台误差。

相关实现位置：零初始化 residual 在 `single_integrator/c1/models/residual.py:18-19,48-51`；`w` 和第二投影在 `single_integrator/c1/rollout_vi_r_cert.py:55-65`；forward 的 `nominal_feasible` 判定在 `single_integrator/cbf.py:112-114`；custom VJP 的活动集和 KKT adjoint 在 `single_integrator/c1/risk/joint_frozen.py:20-35,41-54`。

## 冻结协议与 provenance

- 初始状态：`[[-0.4, 0.005], [0.4, -0.005]]`。
- Flow noise：`noise_seed=84001`，`rollout_id=73000`；每次 `+h/-h` 都从同一初始状态重算完整 850-step forward，未冻结状态依赖 policy 输出。
- 干预变量：actions 100:119，共 `20x4`；方向固定为 NumPy RNG seed `2026091895` 生成后整体单位化的向量。
- 层级 cotangent seed：`2026091903`。
- checkpoint：`baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl`，SHA256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`。
- 精度：JAX x64 开启；控制器、状态、约束与风险为 float64；冻结 Flow-BC 按生产实现内部使用 float32（`differentiable_rollout.py:38-47`）。
- 环境：`dt=0.05`，`max_steps=850`，`max_speed=0.5`，三个 `terminate_on_*` 均为 true。
- solver：`ftol=1e-12`，feasibility `1e-9`，optimality `2e-6`，speed `1e-10`，最多 200 iterations、32 speed cuts；没有 fallback。
- FD 步长：`1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6`；投影拐点另检查到 `3e-7,1e-7`。
- 相对误差分母：`max(|AD|,|FD|,1e-10)`；符号容差 `1e-8`。
- 软件：JAX 0.6.2、NumPy 2.4.6、SciPy 1.17.1、Python 3.11.16；Slurm GPU backend。

生产源 SHA256：

| 文件 | SHA256 |
|---|---|
| `single_integrator/c1/risk/joint_frozen.py` | `f0f8a7a273fe0daa2a2a41b2d2e4dbbd50daaedea09fdb2cea78d02835eb5dbc` |
| `single_integrator/c1/differentiable_rollout.py` | `e7a6042359d02de1329fc36ba922809146b75309a8edcb6aecf094012e43ad75` |
| `single_integrator/c1/risk/vi_r_cert.py` | `d2aa104ec7e433571a4299f8dec4b06b53bea6ac4a65bdc893e847ab361327ce` |
| `single_integrator/c1/rollout_vi_r_cert.py` | `d29b6a54356adc83c41fcd1787387d10091eba500aa5ea6fbfa14da20037fdf6` |
| `single_integrator/c1/termination.py` | `3be4e701227e42509b2e8514d2d9e26b94037b47c7f53a10a6281f1284404eec` |
| `single_integrator/cbf.py` | `f127d09cddb490c9367630fee0d2a8a8434fcb2579ba363ac312ee8ca1159eda` |
| `single_integrator/environment.py` | `427b1c0db1d68698e095a95e50a4404bae6c806a0f87351bfac431f6eeefd49b` |
| `flowbc/giveway_flowbc_agent.py` | `8b94fb64b7448cdcfb3c7bc882d10266427339a22a155defa6d71aee449f5adc` |

工作目录的 `.git` 为空，无法记录可信 commit；以上 checkpoint/source/result hashes 是本次 provenance。

## 同一完整函数的复现

完整 AD 和 FD 都调用生产 `rollout(...)` 与同一个 `direct_R_CERT`/`augmented_R_CERT` scalar，完整重算两次投影、八个 witness、动力学、monitor history、historical guards、termination 和 outer max。没有 direction 再归一化、energy matching、clipping、cache 或 warm start。

基线为 251 个真实动作，`first_deadlock_step=251`，terminal code 为 safe_deadlock；direct 与 augmented 的 outer-max winner 都是 index 111，即 action 250。五次完全相同 forward 的 scalar range 均为 0。

| 风险 | AD | h | 中心 FD | 绝对误差 | 相对误差 | 符号 |
|---|---:|---:|---:|---:|---:|---|
| direct | 0.0350397792 | 1e-3 | 0.0622050918 | 0.0271653126 | 0.436706 | +/+ |
| direct | 0.0350397792 | 3e-4 | 0.0622531120 | 0.0272133329 | 0.437140 | +/+ |
| direct | 0.0350397792 | 1e-4 | 0.0625286398 | 0.0274888607 | 0.439620 | +/+ |
| direct | 0.0350397792 | 3e-5 | 0.0615160028 | 0.0264762236 | 0.430396 | +/+ |
| augmented | 0.00375528528 | 1e-3 | 0.00523874966 | 0.00148346439 | 0.283171 | +/+ |
| augmented | 0.00375528528 | 3e-4 | 0.00524440559 | 0.00148912032 | 0.283945 | +/+ |
| augmented | 0.00375528528 | 1e-4 | 0.00526433161 | 0.00150904633 | 0.286655 | +/+ |
| augmented | 0.00375528528 | 3e-5 | 0.00518972548 | 0.00143444020 | 0.276400 | +/+ |

到 `h=1e-6` 时 float32 Flow 与 host solver 的差分消减误差开始明显，不能把该步长当作更可信参考；但 `1e-3` 到 `3e-5` 的平台已足够证明 mismatch 不随步长收敛。

## 深度与时间定位

从 action 100 的真实 prefix state 出发，对同一个显式 segment function 检查 action 100 后第 1、2、4、8、16、32、64、128、151 个状态。重建 segment 与生产 trace 的最大 baseline 差为：state `2.12e-9`，control `2.66e-8`，A `5.60e-9`，b `2.49e-9`；这是 host solver 重放尺度，segment 内部的 AD/FD 始终比较同一个函数。

在 `h=1e-3` 时，相对误差依次为 `0.0130, 0.0578, 1.749, 0.0413, 0.228, 0.0928, 0.0853, 0.0616, 0.0577`。第 4 步的 AD/FD 已异号。故首个分歧就在 action 100 的 one-step 链，而不是长 BPTT、progress anchor 或 action 250 的证书计算才首次出现。

这一区分了两种梯度：局部根因是 action 100 的同刻 preprojection-to-execution kink；完整风险 winner 在 action 250，因此完整 AD 的确经过 130–150 步闭环传播到更早动作。它不是仅靠同刻 `w` 进入 VI margin 的短路梯度。

## 逐层排查

下表给出每个固定 scalarization 在其稳定步长中的最佳相对误差。固定 margin/aggregation 分支的结果仅用于定位，不算原始完整函数 PASS。

| 层/依赖 | 最佳相对误差 | 结果 |
|---|---:|---|
| Flow-BC 对状态 | `8.26e-6` | 中间步长收敛；过小 h 出现 float32 差分底噪 |
| 第一次投影对状态（含 U(x)） | `8.55e-6` | 稳定区通过 |
| `A(x),b(x)` | `4.30e-11` | 通过 |
| 第二投影对输入，严格可行点 | `1.34e-16` | 通过 |
| 第二投影对输入，固定活动集点 | `0` | 通过 |
| witness 投影对状态/U(x) | `1.02e-10` | 通过；重复约束虽秩亏但本 probe 未失配 |
| direct margin 对状态 | `7.00e-12` | 通过 |
| direct progress anchor | `1.25e-12` | 通过；anchor index 161 未 detach |
| direct speed margin 对执行控制 | `0` | AD=FD=0；处于坏集内部的合法局部平坦区 |
| VI margin 对状态 | `5.72e-13` | 通过 |
| VI progress anchor | `1.21e-13` | 通过 |
| VI margin 对 preprojection `w` | `3.22e-11` | 通过 |
| VI margin 对 witness 输入 | `3.73e-10` | 通过 |
| direct 固定分支 softmin | `9.70e-11` | 通过 |
| augmented 固定分支 softmin | `1.43e-9` | 通过 |

生产路径的依赖位置为：Flow 与第一投影 `differentiable_rollout.py:23-29,36-47`；状态依赖约束 `differentiable_rollout.py:50-72`；动力学和 history update `rollout_vi_r_cert.py:73-89`；VI/direct margins `vi_r_cert.py:78-110`；rollout-generated anchors `vi_r_cert.py:164-194`；softplus/logsumexp 与 outer max `vi_r_cert.py:133-139,195-233`。monitor 的离散输出按设计 stop-gradient，位置和控制仍留在周边可微链上（`termination.py:82-105`）。

## 分支、solver 与数值证据

在完整函数的 `h=1e-3,1e-4,1e-5` 两侧：

- action count、termination time、candidate timer、alive、latch、event code 都不变；
- direct/augmented winner 始终为 index 111，outer max 没有 tie；top gap 约 `2.27e-4`；
- 第一次投影没有 active-set change；
- 第二投影相对基线分别有 12（`+h`）与 8（`-h`）个 step 改变 active set；
- witness 在 `h<=1e-4` 没有 active-set change；
- 基线和 `±1e-3` 的 winning-window margin sign counts 完全相同：0 positive、2626 negative、101 exact-zero、0 nonfinite。

因此小步长完整 mismatch 发生在 monitor、margin 与 outer-max 分支固定时，唯一持续切换的是第二投影。`h=1e-2` 的负侧另使 termination 从 251 提前到 250、winner 从 111 变成 110；这些大步长结果明确归为额外分支切换，不作为平滑 FD 证据。

action 100 的投影两侧进一步显示：对所有 `h=3e-2...1e-7`，正侧为 `nominal_feasible`，负侧为 `solved` 且 row 0 活动。抽查 actions 100、119、120、150、250 的第一、第二与八个 witness 投影：重放误差均为 0；KKT stationarity 最大 `1.67e-16`，complementarity 最大 `6.74e-17`；无 NaN/Inf、solver failure 或 fallback。故这不是 solver 未收敛。

margin action 200 的 27 个值中有 26 negative、1 exact-zero；softplus cost 范围 `[1.0,25.1112]`，对应 sigmoid derivative factor `[0.5,0.999999972]`。没有 overflow/underflow 或数值归零。完整 direct/augmented gradient norms 分别为 `0.822785` 和 `0.0577353`；本次证据没有把较小的 augmented norm 解释为 witness cancellation。

现象在当前 BPTT 点不是偶发：冻结 residual 的最终层使 `G_phi=0`，第二投影反复接收第一次投影产生的边界点；归档 trace 的 251 个真实动作中有 230 个 near-CBF-active entries，且上述五个跨时刻样本都含 row 0 活动、速度球不活动。一个普通梯度无法同时描述朝可行内侧和不可行外侧的扰动。

## 根因分类与修复判断

未发现 smooth-region implementation mismatch、错误的 margin/anchor/witness 依赖、AD/FD 比较不同函数、mutable state 污染、precision-only failure 或 solver differentiation failure。根因是当前零 residual 闭环把第二次欧氏投影放在其边界不可微点。custom VJP 选择活动侧的 KKT 单边导数；中心 FD 取两个不同单边极限的平均，二者没有应当相等的数学前提。

本证据不支持一个“保持现有语义且恢复普通一阶梯度”的最小实现修复。改选另一侧/广义导数仍不能产生唯一双侧梯度；移离边界、平滑投影、删除第二投影或改变 controller 都会改变当前指定语义。本任务未应用任何修复。

## 可复现命令与产物

```bash
cd /home/zhihan/research/02_C1_Toy_GiveWay
sbatch scripts/run_c1_vi_adfd_localize.sbatch   # 成功作业 73
sbatch scripts/run_c1_vi_adfd_layers.sbatch     # 成功作业 74
sha256sum results/c1_vi_adfd_localize_v1/report.json \
              results/c1_vi_adfd_layers_v1/report.json
```

- 完整同函数、逐深度与 branch trace：`results/c1_vi_adfd_localize_v1/report.json`，SHA256 `5b4af42191d3e30637ea72d60f84a4a1283537fa049bfafb90d96ed412f7f063`。
- 投影单边导数与 margin 层：`results/c1_vi_adfd_layers_v1/report.json`，SHA256 `14383a3b934d37e1849274ee5c1c2a3797f90879515a28846766487013731e8a`。
- 只新增两个隔离 probe 及对应 sbatch；生产文件 hash 未变化。诊断 probe SHA256 分别为 `c4315447f01b44e625d1ce78cdffc36d9b450a7d4c0720a995e2e569df40ab30` 和 `b314216b6fbdbaea366b1511a3811524f244520538f5c4b2014ca0890a82b58c`。

**3. fixed candidate lacks reliable first-order gradients for current BPTT use**
