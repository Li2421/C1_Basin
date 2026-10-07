# CL-FHCB 冻结规范

方法名：**CL-FHCB — Closed-Loop Full-Horizon Continuation Barrier**  
冻结日期：2026-09-22（Asia/Singapore）  
构造状态：**CONSTRUCTION_PASS_EMPIRICAL_CERTIFICATE_ONLY**

本文件冻结的是后续独立 P/Q/G/PROJ qualification 将要评估的方法。后续
qualification 不得修改控制器、`G_phi`、特征/分箱、经验转移核、Bellman 递推、
`R_K`、K、数据或任何常数。本轮没有训练最终 `G_phi`，也没有执行最终
qualification。

## 1. 冻结系统与闭环控制器

完整系统重建见 `system_reconstruction.md`。两台二维单积分器的物理状态为当前位置
和上一执行速度；`dt=0.05 s`，联合动作维数为 4，每台机器人速度上界为
`0.5 m/s`，物理时域为 850 步（42.5 s）。精确增广状态为

\[
z_k=(p_k,v^{\rm last}_k,k,E_k,c_k,\ell_k),
\]

其中 `E_k` 是严格死锁监测器所需的有序 41 个双机器人目标误差样本，`c_k` 是
`candidate_since`（或等价的 active flag 与 age），`ell_k` 是 first-event 终止
latch。固定目标、墙、环境常数和策略参数不是逐状态变量。

每个物理步严格执行

\[
\begin{aligned}
\xi_k&\sim {\cal N}(0,I_4),\\
u_{{\rm Flow},k}&=\operatorname{speedbound}
  (\operatorname{FlowBC}_\theta(o(z_k),\xi_k)),\\
u_{{\rm safe},k}&=\Pi_{\cal U}(p_k)(u_{{\rm Flow},k}),\\
g_k&=G_\phi(o(z_k),u_{{\rm safe},k}),\\
w_k&=u_{{\rm safe},k}+g_k,\\
u_{{\rm exec},k}&=\Pi_{\cal U}(p_k)(w_k),\\
z_{k+1}&=F_{\rm plant+monitor}(z_k,u_{{\rm exec},k}).
\end{aligned}
\]

`Pi_U` 是对 17 个 CBF 半空间（1 个机器人间约束、每台机器人各 8 个有限墙段
约束）与两个逐机器人二范数速度球交集的欧氏硬投影。两个投影都保留，参数为
`gamma=gamma_wall=1`、`separation_buffer=1e-4`。第二投影绝不是裁剪或软惩罚。

Flow-BC 每步重新采样四维标准高斯，进行 10 个 flow Euler 步，反归一化，逐分量
裁剪至 `[-1,1]`，再进行逐机器人速度 bound。构造 RNG 为

```text
fold_in(fold_in(PRNGKey(7319), pair_id), physical_step)
```

不同 `phi` 对同一 pair 使用相同的逐步 key，直到其 first event 终止。所有
`P_phi`、`Q_D` 和 `R_K` 的期望都包括这些逐物理步 Flow 随机性。

### 当前/预期最终 `G_phi` 接口

活动代码中的 `ResidualCorrection` 是确定性网络：输入为当前 20 维扁平化并按冻结
Flow 统计量归一化的 observation、当前 4 维 `u_safe`、固定标量零，环境条件维数
为 0；输出为恰好 4 个联合速度 residual。它不输出 mean、gate/logit、scale 或
任何概率分布。因此部署时没有独立的 `G_phi` sampling，也不存在“用均值代替
随机 residual”的问题；随机性来自 Flow。

仓库没有可用于本构造的已训练最终 residual 参数，所以本轮冻结如下小型确定性
诊断族。它保持相同的 4D residual 语义，但直接使用原始当前 observation：

\[
G_\phi(o,u_{\rm safe})=
 \phi_g\,b_{\rm goal}(o)+
 \phi_s\,u_{\rm safe}+
 \phi_r\,b_{\rm rel}(o).
\]

对每台机器人，`b_goal` 是其二维 goal displacement 以二范数投影到半径 0.5
的球；`b_rel` 是 self-minus-other displacement 以同样方式投影到半径 0.5 的球。
三个标量 gain 由两台机器人共享，所以该族对机器人交换等变，没有编码“A 让行、
B 通过”或左右通过语义。参数维数为 3，输出维数为 4。

本次冻结只定义以下四个 policy-conditioned certificate；不得在 qualification 中
对 `phi` 插值或外推：

| ID | `(phi_g, phi_s, phi_r)` |
|---|---:|
| `zero` | `(0, 0, 0)` |
| `goal025` | `(0.25, 0, 0)` |
| `damp035` | `(0, -0.35, 0)` |
| `relative025` | `(0, 0, 0.25)` |

`zero` 只是闭环基线成员；非零成员的 residual 随当前状态/`u_safe` 改变。对所有
成员，同一 `phi` 在完整 episode 中不变，但 `G_phi` 在每一个物理步重新计算，
包括 `K` 之后的所有 continuation 步；不存在固定 `alpha`、K 步后关闭 correction
或恢复 baseline 的分支。

## 2. 事件、H 与独立真值

严格死锁候选在完整 2.0 s history 可用后定义为：尚未成功，两个机器人
`max(abs(window_progress)) < 0.01 m`，且当前机器人最大速度严格小于
`0.025 m/s`。候选连续 5.0 s 后触发 strict deadlock；在 `dt=0.05` 下是 101
个候选样本跨 100 个间隔。候选为假即重置 timer。旧 `stalled_deadlock` 分支不属于
DEADLOCK。

每个动作后的 first-event precedence 为

```text
collision > success > strict deadlock > timeout
```

碰撞使用 swept 与 endpoint/outside 几何。`H` 是第 850 次转移完成所有前三类事件
检查之后的 timeout boundary；因此末步触发的 deadlock 在词典式事件时间上严格
早于 H。独立 full-horizon 真值固定为

\[
Q_D(z_t;\phi)=
\Pr_\phi\{\tau_D<\min(\tau_S,H)\mid z_t\},
\]

其中 collision 是独立吸收结果并赋 deadlock value 0；timeout 不是 deadlock。
`Q_D` 不是 critic、训练 surrogate 或本构造表的别名。后续 qualification 必须用
未用于下述构造的数据独立估计它。

## 3. 闭环轨迹对象

对 `k=t,...,t+K-1`，实现记录

\[
\tau_\phi[t:t+K]=(z_t,g_t,w_t,u_{{\rm exec},t},z_{t+1},\ldots,z_{t+K}),
\]

并另外保存 `u_Flow`、`u_safe`、两个投影状态、Flow key 和 monitor 字段以便审计。
K 只限制 `R_K` 观察的 prefix；如果 prefix 存活，完整 rollout 继续在每步使用同一
`phi` 的闭环 `G_phi`，直到 first event 或 H。

## 4. continuation 特征与分箱

`B` 使用增广状态中以下精确特征：

```text
candidate_active
candidate_age = (k - candidate_since) * 0.05
goal_errors[2]
positions[2,2]
window_ready
recent_progress[2] = error[k-40] - error[k]
remaining physical steps n
phi ID
```

上一执行速度和完整 error history 已通过 `candidate_active/age` 与精确 recent
progress 进入风险状态；`G_phi` 和下一步 monitor 仍使用完整增广状态。`B` 的
离散特征映射按以下先后顺序取第一个满足的 cell：

1. `candidate_late`：candidate active 且 age `>=2.5 s`；
2. `candidate_early`：candidate active 且 age `<2.5 s`；
3. `progress_starved`：window ready，`max(abs(recent_progress))<0.02 m`，且
   `max(goal_errors)>0.08 m`；
4. `resolved_progress`：`p0_x-p1_x>=0.10 m`，且
   `sum(recent_progress)>0.01 m` 或 `max(goal_errors)<=0.24 m`；
5. `bay_progress`：window ready，`max(p_i,y)>=0.36 m`，且
   `sum(recent_progress)>0.005 m`；
6. `advancing`：window ready 且 `sum(recent_progress)>0.01 m`；
7. `unresolved`：其余状态。

这里 `progress_starved` 不读取 timer；因此 jitter 或一次 timer reset 不会把
DELAYED_DEADLOCK 错当成 escape。`resolved_progress` 与 `bay_progress` 提供了与
TRUE_ESCAPE 相关、但不依赖 timer 的类别。类别函数是最小的可解释分段常数族，
time-to-go 由 `n` 的有限时域递推显式编码。

## 5. 经验 continuation certificate B

对每个冻结 `phi`，计数矩阵的列固定为

```text
D, S, C,
next:candidate_late, next:candidate_early, next:progress_starved,
next:resolved_progress, next:bay_progress, next:advancing,
next:unresolved
```

timeout 边界上的无前三类事件转移仍按其物理 next cell 计数；递推中的 `B(.,0)=0`
使它在零剩余步时成为 timeout value 0。逐行除以行总数得到经验核 `P_hat_phi`。
没有观测的行固定为 `p_hat_D=1`，其余概率为 0，不做乐观外推。

### 冻结计数（它们与公式共同给出精确系数）

#### `zero`

| cell | D | S | C | late | early | starved | resolved | bay | advancing | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_late | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate_early | 0 | 0 | 0 | 0 | 66 | 63 | 0 | 0 | 0 | 0 |
| progress_starved | 0 | 0 | 0 | 0 | 63 | 481 | 15 | 1 | 0 | 8 |
| resolved_progress | 0 | 8 | 0 | 0 | 0 | 15 | 1736 | 0 | 0 | 1 |
| bay_progress | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 581 | 0 | 0 |
| advancing | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 842 | 0 |
| unresolved | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 7 | 8 | 665 |

#### `goal025`

| cell | D | S | C | late | early | starved | resolved | bay | advancing | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_late | 3 | 0 | 0 | 194 | 0 | 1 | 0 | 0 | 0 | 0 |
| candidate_early | 0 | 0 | 0 | 4 | 196 | 1 | 0 | 0 | 0 | 0 |
| progress_starved | 0 | 0 | 0 | 0 | 5 | 46 | 0 | 0 | 0 | 0 |
| resolved_progress | 0 | 5 | 0 | 0 | 0 | 0 | 935 | 0 | 0 | 0 |
| bay_progress | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| advancing | 0 | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 1087 | 0 |
| unresolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 312 |

#### `damp035`

| cell | D | S | C | late | early | starved | resolved | bay | advancing | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_late | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate_early | 0 | 0 | 0 | 0 | 589 | 231 | 0 | 0 | 0 | 0 |
| progress_starved | 0 | 0 | 0 | 0 | 234 | 1186 | 14 | 4 | 0 | 10 |
| resolved_progress | 0 | 0 | 0 | 0 | 0 | 21 | 1439 | 0 | 0 | 0 |
| bay_progress | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 952 | 0 | 0 |
| advancing | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 1284 | 0 |
| unresolved | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 4 | 8 | 802 |

#### `relative025`

| cell | D | S | C | late | early | starved | resolved | bay | advancing | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_late | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate_early | 0 | 0 | 0 | 0 | 2202 | 696 | 0 | 0 | 0 | 0 |
| progress_starved | 0 | 0 | 0 | 0 | 703 | 406 | 0 | 0 | 2 | 0 |
| resolved_progress | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bay_progress | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| advancing | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 2269 | 0 |
| unresolved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 504 |

令 `a(z)` 是上述 cell，`G={candidate_late,candidate_early,progress_starved}`。
经验 certificate 的精确递推为

\[
\begin{aligned}
B(D,n;\phi)&=1,\\
B(S,n;\phi)=B(C,n;\phi)=B(T,n;\phi)&=0,\\
B(z,0;\phi)&=0\quad\text{(nonterminal)},\\
\widetilde B_s(n;\phi)&=\widehat p_D(s;\phi)
 +\sum_{s'}\widehat P_{\rm run}(s,s';\phi)B_{s'}(n-1;\phi),\\
B_s(n;\phi)&=
\begin{cases}
1,&s\in G,\\
\widetilde B_s(n;\phi),&s\notin G.
\end{cases}
\end{aligned}
\]

最后只作 `[0,1]` 数值 clip；冻结数据上所有递推值本来就在该区间。精确的
`n=0,...,850` 数值表位于 `data_v1/certificate.json`，其 SHA256 为
`eb334ca92a8aa1f07291f6a8df926384d6b948bcb3650cfb46270e8a3c708be7`。
公式、上述整数计数、行归一化和 guard 已唯一决定该表；JSON 不是另一个拟合模型。

对每个经验 cell 和 `n>=1`，构造保证

\[
B_s(n;\phi)\ge \widehat p_D(s;\phi)+
\sum_{s'}\widehat P_{\rm run}(s,s';\phi)B_{s'}(n-1;\phi).
\]

`n=0` 的非终止值为 0。假设 `n-1` 时 `B` 上界经验核下的 eventual-deadlock
概率，对 first transition 按 D、S/C/timeout、running 分解并应用上式，即得到
`n` 时的上界；从 `n=0` 归纳至 850。因此它是经验抽象核上的 stochastic
Bellman supersolution。

这**不是**连续状态域的形式证明：同一 cell 内的状态可能有不同条件转移分布，
同一 rollout 的逐步样本也相关。每行另外报告 immediate-deadlock 概率的单边 95%
Jeffreys quantile

\[
\operatorname{Beta}^{-1}(0.95;d+0.5,N-d+0.5),
\]

但不把它塞入递推（这样做会在 850 步上人为饱和为 1）。该不确定性只是构造诊断，
不升级为连续域保证。这正是最终状态含 `EMPIRICAL_CERTIFICATE_ONLY` 的原因。

## 6. 冻结的 R_K

令 `m=min(K,H-t)`，在 prefix 内按权威 precedence 取第一个 terminal event。
轨迹级分数精确定义为

\[
R_K(\tau_\phi[t:t+m])=
\begin{cases}
1,&\text{prefix 内 first event 是 strict deadlock},\\
0,&\text{prefix 内 first event 是 success、collision 或 timeout},\\
B(z_{t+m},H-t-m;\phi),&\text{prefix 存活}.
\end{cases}
\]

所以在随机 prefix 上

\[
\mathbb E_\phi[R_K\mid z_t]
=\Pr_\phi(D\text{ first in prefix}\mid z_t)
+\mathbb E_\phi[\mathbf 1\{\text{prefix survives}\}
B(z_{t+m},H-t-m;\phi)\mid z_t].
\]

如果 `B` 对真实 closed-loop kernel 是 supersolution，对 survival state 使用其上界，
再按 prefix first event 分解，即得

\[
Q_D(z_t;\phi)\le\mathbb E_\phi[R_K\mid z_t].
\]

对本轮的经验抽象 `B`，相同证明严格成立于 `P_hat_phi`；对真实连续系统则是待独立
qualification 的经验主张。collision 分支始终为 0 且单独报告；success 在 prefix
立即清零；末步前三类事件先于 timeout；无事件到 `n=0` 时 certificate 为 0。

主 K 值冻结为

```text
K = 20
K = 100
```

`K=1` 不属于主方法。`K=full remaining horizon` 只在每个 `phi` 的一条已缓存轨迹
上做事件 bookkeeping sanity；它不是调参或 qualification。无论 K 为何，实际
controller 在 K 后继续使用同一 `G_phi`。

## 7. 构造数据、预算与 sanity 结果

构造数据只使用训练 split 的 pair IDs `0,...,7`、Flow seed `7319`，四个冻结
`phi` 各 8 条，共 32 条新 rollout。预告上限为 27,200 个物理步，实际为 20,983
步，全程 `TFRT_CPU_0`。这些 ID、seed 和轨迹不得进入之后的独立 qualification。
每条压缩 trace 的 ID 与 SHA256 在 `data_v1/manifest.json` 中冻结。

构造 outcome（只用于构造和 sanity，不是最终 Q 测试）为：

| phi | collision | success | strict deadlock | timeout |
|---|---:|---:|---:|---:|
| zero | 0 | 8 | 0 | 0 |
| goal025 | 0 | 5 | 3 | 0 |
| damp035 | 0 | 0 | 0 | 8 |
| relative025 | 0 | 0 | 0 | 8 |

通过的构造检查：每步重算 `G_phi`；K=100 后仍调用 correction；每步两个硬投影
都存在且满足冻结 feasibility tolerance；Flow step keys 在 episode 内唯一；
deterministic `G_phi` 没有错误采样分支；四类 terminal value 正确；所有 delayed
timer/timer-reset guard cell 对正剩余时域取 1；所有构造 deadlock 的 full-horizon
score 为 1；四个 `phi` 的 certificate 均不恒等于 1；最小经验 Bellman slack 均为
0。新增确定性测试 5/5、既有回归 58/58 通过。

在 `n=850`，按 cell 顺序 `[late,early,starved,resolved,bay,advancing,unresolved]`
的值为：

```text
zero        [1,1,1,0.659935,0.659746,0.999674,0.839571]
goal025     [1,1,1,0,1,0.374258,0.373952]
damp035     [1,1,1,0.999995,0.998054,0.994873,0.996131]
relative025 [1,1,1,1,1,0.976091,0.966756]
```

因此 certificate 不是常数 1；`progress_starved` 的 timer-independent guard 又保证
单纯推迟或重置 strict timer 不会获得较低 continuation score。这里不声称这些数值
已经通过早期预测、P/Q/G/PROJ、校准或因果干预测试。

## 8. 来源、哈希与冻结边界

Git worktree 有效；构造前的 base HEAD 是
`aba7d6d284e479cb0678b6270cc79f9b617dc654`。新增构造文件尚未由本任务提交，因而
以下 SHA256 是冻结内容的权威标识，不把 base HEAD 冒充这些新增文件的 commit。

| 文件/资产 | SHA256 |
|---|---|
| `diagnostics/cl_fhcb/__init__.py` | `76fd638945e20b7b87e843dce3598bcc6eba28d9920f4f5adc8b2298f1045b6e` |
| `diagnostics/cl_fhcb/closed_loop.py` | `aeca60cc1968733ca2dde4535c0de11a5431e2adcbac01027005a2ffc6694fb0` |
| `diagnostics/cl_fhcb/certificate.py` | `1c09b3997ce41461dffab5674bab9a69d88c239a0e253761b4ca36ee85be9529` |
| `diagnostics/cl_fhcb/construct.py` | `651881a8ae6c43281e3e162679ac431fdf4652a2485324076c4d237fbf743bd1` |
| `diagnostics/cl_fhcb/system_reconstruction.md` | `613c654cf6df5fa0863ece88334b5d859e011d82e70f8480971b4f81a09a4c72` |
| `tests/test_cl_fhcb.py` | `2dba8d686a5859e54c7efce309cab51d7aaa3bbb3d8255fd72cb06f856b7183a` |
| `single_integrator/environment.py` | `427b1c0db1d68698e095a95e50a4404bae6c806a0f87351bfac431f6eeefd49b` |
| `single_integrator/cbf.py` | `f127d09cddb490c9367630fee0d2a8a8434fcb2579ba363ac312ee8ca1159eda` |
| `single_integrator/evaluate.py` | `294d233ff6f7a3137730100fbfbaf0a5e2902d31311642cb98a84515821de4ae` |
| `single_integrator/outcomes.py` | `26b41cdaafccd0eda0e00de2fef9e551be753873716c4e96b7e9008658e19e82` |
| `single_integrator/c1/differentiable_rollout.py` | `e7a6042359d02de1329fc36ba922809146b75309a8edcb6aecf094012e43ad75` |
| `single_integrator/c1/models/residual.py` | `99e5def6e1bfb517a8bbb38efb0bd7b0cb80e7ca25e1472534524db343e4e377` |
| `flowbc/giveway_flowbc_agent.py` | `8b94fb64b7448cdcfb3c7bc882d10266427339a22a155defa6d71aee449f5adc` |
| `data_v1/manifest.json` | `d80bdd1c879d7b8953f2d0c1ce25deeb661983de398c01e34d1356c2220a85b5` |
| `data_v1/certificate.json` | `eb334ca92a8aa1f07291f6a8df926384d6b948bcb3650cfb46270e8a3c708be7` |
| `data_v1/prefix_scores.json` | `dee7f787691d4602c873638c8b32da476ea73aecae08fb542c026f49233799de` |
| `data_v1/sanity.json` | `c7716cd061fc96ec6e2dd7efc528adc5361964ee6cce0c7ba1e4fc200700efc5` |
| approved Flow checkpoint | `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32` |
| Flow checkpoint config | `fb4c7193fe7bdb36030397e79c6d7edb86aa9a312c3f8fe4fb9bb8b29d92ed2d` |
| Flow benchmark metadata | `f060bb78374555bc16b109a911003828be07d47e4184c62235f656a7eeda96a6` |
| dataset environment config | `39f819fbd380b889ee6db44a261a381f1fb78dd8a1f8e5475349ae6ee6ad9147` |

冻结后允许的 qualification 操作只有：读取这些对象；在完全独立的数据/seed 上
估计 `P_phi` 与 `Q_D`；计算上述固定 `R_20/R_100`；检查梯度、投影和统计性质；
报告通过或失败。任何改变 `phi` 集合、basis、分箱阈值、guard、转移计数、递推、
K、H、事件 precedence 或 sampling 的操作，都构成新方法而不是本冻结方法。
