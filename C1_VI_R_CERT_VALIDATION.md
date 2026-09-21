# 冻结 VI-augmented R_CERT 验证结项

日期：2026-09-18（Asia/Singapore）。结论：**FAIL（训练前验证门槛）**。未启动完整 `G_phi` 训练，未打开最终测试集。

## 1. 冻结语义与数学规格

动作 `u_k` 把状态 `x_k` 推到 `x_(k+1)`，`k=0,...,849`，`dt=0.05 s`。本次实际配置为：

```text
terminate_on_collision = true
terminate_on_success   = true
terminate_on_deadlock  = true
max_steps              = 850
```

监控和终止优先级保留为启用事件的
`agent_collision > wall_collision > task_success > safe_deadlock > other_timeout`。
`alive_k` 与 `latch_k` 都是动作前状态；终止动作计入轨迹。raw deadlock 先更新历史
`first_deadlock_step`，与 `terminate_on_deadlock` 开关独立。

主事件为

```text
D = historical_strict_deadlock OR stalled_deadlock.
```

对 `t=139,...,849`：

```text
H_strict,t = alive_t AND NOT latch_t
Q_strict,t = AND_{k=t-100,...,t} (G_k AND P_k AND V_k)
```

这里保留101个 candidate 样本和100个间隔的原语义：

```text
G_k: max_i ||x_i,k+1-g_i|| > 0.08 m
P_k: max_i |d_i,k-39-d_i,k+1| < 0.01 m
V_k: max_i ||u_i,k|| < 0.025 m/s.
```

stalled 只接受原始、重分类前 `other_timeout`，且使用动作 `810:849` 和
`d_811,d_850`：40个速度原子 `<0.05 m/s`，两个进展原子
`|d_i,850-d_i,811|<0.02 m`。不因既有非终止 raw latch 排除 stalled。

令每台机器人的控制块为 `q_i`，
`a_i=(g_i-x_i)/dt`，`r_i(q)=||q_i-a_i||`。冻结 bad set 及精确距离为：

```text
B_goal = UNION_i {q: r_i(q) >= 0.08/dt}
dist²(q,B_goal) = min_i [0.08/dt-r_i(q)]_+²

B_prog(anchor) = PRODUCT_i {q: r_i(q) in
                    [max(anchor_i-epsilon,0)/dt,
                     (anchor_i+epsilon)/dt]}
dist²(q,B_prog) = sum_i dist(r_i(q), interval_i)²

B_stall_prog,i = the corresponding single-agent annulus, other block free

B_speed(rho) = PRODUCT_i {q: ||q_i|| <= rho}
dist²(q,B_speed) = sum_i [||q_i||-rho]_+².
```

直接 margins 保持为：

```text
c_goal   = (0.08-max_i d_i,k+1)/0.08
c_prog   = (max_i |anchor_i-d_i,k+1|-0.01)/0.01
c_speed  = dist²(u_k,B_speed(0.025))/0.25
c_sprog,i= (|d_i,850-d_i,811|-0.02)/0.02
c_sspeed = dist²(u_k,B_speed(0.05))/0.25.
```

八个 witness 全部保留，包括投影后重合者：

```text
v_j(x_k) = Pi_U(x_k)(+/- 0.5 e_r), r=1,...,4.
```

对每个上述原子的相应 bad set：

```text
c_VI = [dist²((w+v_j)/2,B)-||w-v_j||²/4]/0.25,
w = u_safe + G_phi + diagnostic_offset.
```

实际投影是欧氏投影。若事件原子给出 `u=Pi_U(w) in B`，且
`v_j in U(x)`，则

```text
dist²((w+v_j)/2,B) - ||w-v_j||²/4
 <= (w-u)^T(v_j-u) <= 0.
```

所以 `c_VI>0` 排除对应原子。目标/进展的 bad set 由
`x_(k+1)=x_k+dt*u_k` 推出，没有用低控制替代。

每个原子一个 direct margin 加八个 VI margin，故：

```text
direct-only: M_strict=101*3=303, M_stalled=2+40=42
augmented:   M_strict=303*9=2727, M_stalled=42*9=378
L=1, sigma=1, kappa=1, s_u=0.5 m/s.
```

对每个有效事件分支精确计算

```text
h(z)=softplus(z)/log(2)
S_ej=h(-c_ej)
R_e=-log[(1/M_e) sum_j exp(-S_ej)]
R_CERT=max_e R_e.
```

strict outer max 只用当时保存的 `H_strict,t`；stalled 分支只在原始
`other_timeout` 且有850个真实动作时启用。无有效分支返回显式 undefined。
终止后不调用控制器或 witness 投影，也不评分固定形状数组中的无效槽位。

## 2. 实现正确性：PASS

19项相关单元测试通过。隔离测试覆盖计数、闭式距离、归一化 soft-min、
`D=>R_CERT>=1`、启用/禁用终止开关、同刻优先级、最后一步 raw deadlock、
deadlock 后恢复、历史 guard、insufficient/empty window 和稳定距离分支差分。

单条完整 GPU 对照中：

- 真实动作数均为251；与权威 operational executor 的最大动作误差
  `6.25e-17`；candidate/raw-deadlock 序列逐项一致；
- witness 最小 CBF 余量 `-9.09e-16`，最大速度超限 `0`；实际控制最小
  CBF 余量 `-2.78e-17`；求解失败/回退为0；
- 该真实 deadlock 上 direct/augmented `R_CERT=1.68135/1.08568`；
- 预测批次中28次真实主事件，direct/augmented 均有0次 `R<1-2e-6`；
- 终止后控制/witness 槽位为不可用值，未被任何有效窗口读取。

## 3. 数值声音性：FAIL

闭式距离在稳定分支的多步长中心差分通过。完整闭环在动作100--119的
同一随机方向上，active outer-max 分支位于动作250，因而测试的是到达
更早动作的梯度，而非同刻 margin 梯度。四个步长 `1e-3,3e-4,1e-4,3e-5`
下：

- direct AD方向导数 `0.03504`，FD约 `0.0615--0.0625`，相对误差
  `75.6%--78.5%`；
- augmented AD方向导数 `0.003755`，FD约 `0.00519--0.00526`，相对误差
  `38.2%--40.2%`。

该轨迹有230个近活跃 CBF 条目；这与已知 projection active-set/Flow
长链不光滑相符，但当前证据没有把误差唯一归因于某个开关，因此不能把
它误报为稳定区数值一致。真实轨迹 margin 无 NaN/Inf、无 penalty 下溢到
`<1e-12`、无 solver failure。对称全 tie 合成例产生3244个非有限梯度分量，
明确暴露 outer-max/distance tie 的梯度未定义；没有加平滑或直通梯度。

离散 guard 来自权威 callback 并 stop-gradient；隔离语义和历史快照通过，
但 guard 切换处本来就没有连续导数，本次完整FD也没有足够信息把误差从
projection switch中单独分离。outer max 在审计轨迹只把动作250分支的梯度
传回，其他分支被精确遮蔽。该点 direct/augmented 梯度范数为
`0.823/0.0577`、余弦为`0.994`；25个 checkpoint 的中位数为
`0.492/0.0594`，说明增广归一化后有显著幅值衰减，不能排除 witness 项间
抵消。没有把这种衰减解释为有效导航信号。

## 4. 独立提前预测：PASS（小样本开发诊断）

25个独立初态在动作100（5秒）形成 pre-deadlock checkpoint；Flow 前缀、
上一执行速度和 monitor history 一致。每个 checkpoint 用2个风险 suffix
求均值（共50条 continuation），另用2个互不重合的 suffix 估计未来事件
频率（共50个配对场景）。future noise 都是逐动作独立标准高斯 Flow latent；
只共享动作0--99的条件前缀。

| 风险 | Spearman（scenario bootstrap 95% CI） | AUC（95% CI） | undefined |
|---|---:|---:|---:|
| direct | 0.860 `[0.743,0.870]` | 1.000 `[1.000,1.000]` | 0/25 |
| augmented | 0.860 `[0.743,0.870]` | 1.000 `[1.000,1.000]` | 0/25 |

Augmented 四个风险分箱的均值风险/独立 deadlock 频率为：

```text
0.775 -> 0/6
0.921 -> 1/6
1.084 -> 6/6
1.085 -> 7/7
```

这些是排序证据，不是概率校准。

## 5. 提前干预：FAIL

冻结 `G_phi`。只在动作100--119加入 residual-action perturbation，之后恢复
原控制器。direct 与 augmented 各自比较 baseline、`-gradient`、
`+gradient` 和2条各向同性随机方向；随机结果取均值，从未择优。

共100次标量反传、50个独立配对场景、450次主回放。25/25个 checkpoint
两种梯度均非零；direct 与 augmented 各100/100个方向通过 outcome-blind
投影后幅值校准。Augmented 实际 RMS：负 `0.0024992`、正 `0.0024981`、
随机 `0.0025158/0.0025116`。无零执行效果或被丢弃样本。

| 风险/组 | historical strict | terminal safe-deadlock | stalled | success | ordinary timeout | collision |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 28 | 28 | 0 | 22 | 0 | 0 |
| direct -grad | 28 | 28 | 0 | 22 | 0 | 0 |
| direct +grad | 28 | 28 | 0 | 22 | 0 | 0 |
| direct random mean | 28 | 28 | 0 | 22 | 0 | 0 |
| augmented -grad | 28 | 28 | 0 | 22 | 0 | 0 |
| augmented +grad | 28 | 28 | 0 | 22 | 0 | 0 |
| augmented random mean | 28 | 28 | 0 | 22 | 0 | 0 |

两种风险的 baseline-minus-negative、positive-minus-negative、
random-mean-minus-negative、成功变化和 timeout 变化均为 `0`，聚类 bootstrap
95% CI 均 `[0,0]`。Augmented 的平均 `J_def`：baseline约 `2.0e-34`，
负/正/随机约 `1.486e-6/1.517e-6/1.526e-6`。动作确实改变，但没有改变事件。

225次50秒扩展回放（每个 scenario 一个 suffix、全部九组）有0个短时域
timeout、0个扩展 timeout、0个延迟事件转换。安全审计所有 collision、
outside、CBF 和 speed violation 计数均为0。

## 6. 反例与适用边界

- 正常 yielding：记录轨迹 `reference_1_84220.npz` 在458步成功；至少一台
  机器人低于0.025 m/s的样本有374个，但 candidate 只有8个，历史 strict
  为假。低速本身不等于 deadlock。
- 零控制阻塞：首次 strict 在动作139，direct/augmented 风险
  `1.756/1.188`。
- 非零 jitter：两台以0.03 m/s对称抖动，strict 速度原子失败，但原 timeout
  的尾窗触发 stalled；两种风险仍大于1。
- 一台停止、另一台0.1 m/s抖动：strict 与 stalled 均不成立。这是 frozen
  全局事件不覆盖的 local-deadlock 语义，没有擅自加入。
- 持续 retreat：速度0.1 m/s、明显远离目标，不属于 frozen deadlock；这是
  task abandonment 反例。
- horizon delay：前830步0.1 m/s抖动、随后停止；H=850时只有20个候选样本，
  非 stalled，D为假；延长到1000步后动作930首次 strict。上界声明仍只限H。
- 对称 tie：合成完全对称轨迹在 exact outer-max/distance ties 上产生非有限
  梯度，未静默平滑。
- projection plateau：`U=[-2,2]x{0}, w=(2,10), B=unit ball` 时 exact margin
  为4，四个二维 signed witness 的未归一化 VI margin 为
  `[-8.3078,-10.1119,-9.1980,-9.1980]`；VI 可完全漏检。

## 7. 判定与下一步

| 项目 | 判定 |
|---|---|
| 实现正确性 | PASS |
| 数值声音性 | FAIL |
| 独立提前预测 | PASS |
| 提前负梯度 outcome 改善 | FAIL |
| 总判定 | **FAIL** |

本结论表示固定候选没有通过进入完整训练所需的因果门槛；它不证明投影最优性
证书在所有系统中无效。当前样本没有 operational stalled/timeout，stalled
分支只有合成语义覆盖，也不能外推该分支的机器人预测效果。

训练仍缺两项必要证据：完整闭环更早动作梯度在明确稳定分支上的可信数值一致性；
以及匹配投影后预算下，`-gradient` 相对 baseline、随机和 `+gradient` 降低
独立主事件且不损失成功、不增加 timeout。两项都未满足，因此禁止启动完整
`G_phi` 训练，也不修改该失败候选再筛选。

## 8. 复现

当前目录不是Git工作树，故没有可报告 commit。冻结 baseline SHA256：
`8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`；
精确源码哈希在 `results/c1_vi_r_cert_probe_v1/protocol.json`。

```bash
JAX_PLATFORMS=cpu .venv-c1/bin/python -m unittest \
  tests.test_c1_r_cert tests.test_c1_vi_r_cert -v

sbatch scripts/run_c1_vi_r_cert_check.sbatch
sbatch scripts/run_c1_vi_r_cert_probe.sbatch

JAX_PLATFORMS=cpu .venv-c1/bin/python \
  scripts/check_c1_vi_r_cert_counterexamples.py \
  --out results/c1_vi_r_cert_counterexamples_repro.json
```

主要产物：

- `results/c1_vi_r_cert_check_v1/report.json`
- `results/c1_vi_r_cert_probe_v1/protocol.json`
- `results/c1_vi_r_cert_probe_v1/analysis.json`
- `results/c1_vi_r_cert_probe_v1/records.json`
- `results/c1_vi_r_cert_probe_v1/extended_records.json`
- `results/c1_vi_r_cert_counterexamples_v1.json`
