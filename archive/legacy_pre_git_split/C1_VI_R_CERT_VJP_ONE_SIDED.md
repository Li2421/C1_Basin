# CURRENT VJP 对原 nonsmooth VI-augmented R_CERT 的单边局部测试

日期：2026-09-18。范围仅为冻结候选的局部 surrogate 测试；没有修改生产代码、风险、VJP、控制器或 solver，没有运行 outcome study 或训练。

## 判定

结果为 **C / INCONCLUSIVE**。`d_minus` 在全部六个步长上都使 `F` 下降且符号已解析，`d_plus` 的三个最小步长稳定为正；但按预注册规则，`d_minus` 的三个最小已解析 slope 没有稳定：

| 方向 | 三个最小已解析 h | slopes | spread | 5% 阈值 | 稳定性 |
|---|---|---|---:|---:|---|
| `d_minus` | `1e-7,1e-6,1e-5` | `-0.0723369,-0.0585117,-0.0579672` | `0.0143697` | `0.00292558` | FAIL |
| `d_plus` | `1e-7,1e-6,1e-5` | `0.0578404,0.0603309,0.0578852` | `0.00249045` | `0.00289426` | PASS，positive |

不能丢弃 `h=1e-7` 后把结果改判 A。精确阻碍是：重复输入完全确定，但在固定的混合精度/host projection forward 中，`d_minus` 的小步长响应未达到规定的尺度稳定性。特别是 common-prefix 执行动作变化除以 `h` 从 `0.9077`、`0.9183` 跳到 `1.8256`。规定的 repeat-based function floor 无法覆盖这种确定性小步长偏差。

## 冻结函数与方向

`F(a)` 是归档 AD/FD 失败脚本的原生产 `augmented_R_CERT`：

- 初始状态 `[[-0.4,0.005],[0.4,-0.005]]`；
- `noise_seed=84001`，`rollout_id=73000`；归档标量只有一条固定 continuation，因此原样保留 `K=1`，没有新增或删除平均项；
- residual perturbation 为 actions 100:119 的 `20x4` 向量；
- 每个输入从初始状态重算完整闭环、两次投影、八个 witnesses、monitor history、historical guards 与 termination；
- `F(0)=1.085680175108144`，三次重复完全相同；
- 当前 VJP 只计算一次，`||g_hat||_2=0.05773525660376328`；`||d_minus||=||d_plus||=1`，且 `d_plus=-d_minus`；
- `d_minus` 的 float64 byte SHA256 为 `2666ef8cd03ee23dd1d5b64957760a5b06acf133828dc7ee21d9274acec62014`，`d_plus` 为 `d9e1e9fd232e41f51ac18e202215a24fe190e70a5992034f474e693af7361707`；完整 `20x4` 数组保存在 machine-readable report 的 `gradient.directions`。

方向的首/末 timestep 为：

| 方向 | action 100 block | action 119 block |
|---|---|---|
| minus | `[-0.1342455992,-0.0241091322,-0.1327681211,-0.00108888443]` | `[-0.0888734232,-0.00350759050,-0.0999503100,-0.1548347167]` |
| plus | 上行取反 | 上行取反 |

精度与 solver 保持归档设置：JAX x64；控制/状态/风险 float64；Flow-BC 内部 float32；`ftol=1e-12`、feasibility `1e-9`、optimality `2e-6`、speed `1e-10`、200 iterations、32 speed cuts。三个 `terminate_on_*` 均为 true。

## 全部步长结果

所有 slope uncertainty 都通过指定 resolved 规则，符号区间也完全落在 `±tau_s` 之外。表中 `Lwin` 是 actions 100:119 的 post-projection action-change L2；`Lprefix` 是共同真实执行前缀的 L2；括号内为除以 h 的值。

| dir | h | F(hd) | delta_F | slope | uncertainty | sign | Lwin (/h) | Lprefix (/h) |
|---|---:|---:|---:|---:|---:|---|---|---|
| minus | 1e-2 | 1.085082072055 | -5.98103e-4 | -0.0598103 | 4.82e-12 | negative | 7.71665e-3 (0.7717) | 9.09763e-3 (0.9098) |
| minus | 1e-3 | 1.085622278507 | -5.78966e-5 | -0.0578966 | 4.82e-11 | negative | 7.71927e-4 (0.7719) | 9.09691e-4 (0.9097) |
| minus | 1e-4 | 1.085674400814 | -5.77429e-6 | -0.0577429 | 4.82e-10 | negative | 7.71845e-5 (0.7718) | 9.09547e-5 (0.9095) |
| minus | 1e-5 | 1.085679595436 | -5.79672e-7 | -0.0579672 | 4.82e-9 | negative | 7.69353e-6 (0.7694) | 9.07699e-6 (0.9077) |
| minus | 1e-6 | 1.085680116596 | -5.85117e-8 | -0.0585117 | 4.82e-8 | negative | 7.41599e-7 (0.7416) | 9.18331e-7 (0.9183) |
| minus | 1e-7 | 1.085680167874 | -7.23369e-9 | -0.0723369 | 4.82e-7 | negative | 9.83610e-8 (0.9836) | 1.82556e-7 (1.8256) |
| plus | 1e-2 | 1.085793785478 | 1.13610e-4 | 0.0113610 | 4.82e-12 | positive | 7.72349e-3 (0.7723) | 9.09712e-3 (0.9097) |
| plus | 1e-3 | 1.085511933402 | -1.68242e-4 | -0.168242 | 4.82e-11 | negative | 7.72060e-4 (0.7721) | 9.09753e-4 (0.9098) |
| plus | 1e-4 | 1.085685947940 | 5.77283e-6 | 0.0577283 | 4.82e-10 | positive | 7.72098e-5 (0.7721) | 9.09840e-5 (0.9098) |
| plus | 1e-5 | 1.085680753960 | 5.78852e-7 | 0.0578852 | 4.82e-9 | positive | 7.76082e-6 (0.7761) | 9.14066e-6 (0.9141) |
| plus | 1e-6 | 1.085680235439 | 6.03309e-8 | 0.0603309 | 4.82e-8 | positive | 7.96245e-7 (0.7962) | 9.62540e-7 (0.9625) |
| plus | 1e-7 | 1.085680180892 | 5.78404e-9 | 0.0578404 | 4.82e-7 | positive | 1.20855e-7 (1.2085) | 2.16192e-7 (2.1619) |

没有 zero-executed-effect 或 surrogate-only/preprojection-only case。

## 数值 floor、重复与分支

所有 13 个输入各重复三次，函数值和真实 trace hash 在输入内完全相同，最大 repeat spread 为 0。`scale=1.08579378548`，float64 machine epsilon 为 `2.22044604925e-16`，故

`e_F = 2.41094652126e-14`，`slope_uncertainty=2e_F/h`。

这个 floor 只反映 repeat variability 与 machine epsilon。它没有证明 `h=1e-7` 的混合精度闭环和 host solver 没有确定性尺度偏差；动作变化的 `/h` 跳变正是 C 的 blocker。

小步长 `h<=1e-4` 的两方向均保持：251 actions、first_deadlock_step 251、winner 111/action 250、单一 outer winner、guard/latch/candidate/raw-deadlock 序列不变，第一次/执行/witness active sets 均与 baseline 相同，winning-window margin signs 和 direct/VI piecewise branch signatures也不变。outer gap 保持约 `2.274e-4`。因此三个最小步长的不稳定不能归因于已记录的离散 branch switch。

较大步长的变化完整保留：

| 方向/h | actions | winner | execution active changed steps | witness changed steps | 其他分支变化 |
|---|---:|---:|---:|---:|---|
| minus/1e-2 | 251 | 111 | 13 | 16 | progress branches changed，monitor不变 |
| minus/1e-3 | 251 | 111 | 3 | 0 | margin/monitor不变 |
| plus/1e-2 | 249 | 109 | 13 | 11 | termination、guards、candidate、raw deadlock均变化 |
| plus/1e-3 | 250 | 110 | 3 | 1 | termination、guards、candidate、raw deadlock均变化；slope为负 |

所有方向/步长都重新计算真实闭环，没有 post-terminal branch 被当作有效轨迹比较。不同 termination length 已在上表和 JSON 中保留。

## Solver 与异常

每条 251-step trace 检查执行投影和八个 witness 投影，共 2259 个 KKT 点（提前终止的 trace 相应减少）：

- production solver failures：0；诊断 NNLS/KKT failures：0；fallback：0；
- 最坏线性 primal slack：`-1.069e-15`；最大 speed excess：0；
- 最大 stationarity：`4.927e-16`；最大 complementarity：`6.689e-16`；
- action 100 replay 均与保存执行动作完全一致；
- 梯度和所有真实前缀无 NaN/Inf。

这些残差没有显示求解失败，但也不能把 `h=1e-7` 的确定性尺度偏差纳入 repeat-based uncertainty，因此仍按规则判 C。

## Provenance 与复现

checkpoint SHA256：`8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`。

主要生产源 SHA256：

| 文件 | SHA256 |
|---|---|
| `single_integrator/c1/risk/joint_frozen.py` | `f0f8a7a273fe0daa2a2a41b2d2e4dbbd50daaedea09fdb2cea78d02835eb5dbc` |
| `single_integrator/c1/differentiable_rollout.py` | `e7a6042359d02de1329fc36ba922809146b75309a8edcb6aecf094012e43ad75` |
| `single_integrator/c1/risk/vi_r_cert.py` | `d2aa104ec7e433571a4299f8dec4b06b53bea6ac4a65bdc893e847ab361327ce` |
| `single_integrator/c1/rollout_vi_r_cert.py` | `d29b6a54356adc83c41fcd1787387d10091eba500aa5ea6fbfa14da20037fdf6` |
| `single_integrator/c1/termination.py` | `3be4e701227e42509b2e8514d2d9e26b94037b47c7f53a10a6281f1284404eec` |
| `single_integrator/cbf.py` | `f127d09cddb490c9367630fee0d2a8a8434fcb2579ba363ac312ee8ca1159eda` |
| `flowbc/giveway_flowbc_agent.py` | `8b94fb64b7448cdcfb3c7bc882d10266427339a22a155defa6d71aee449f5adc` |

可复现命令：

```bash
cd /home/zhihan/research/02_C1_Toy_GiveWay
sbatch scripts/run_c1_vi_vjp_one_sided.sbatch  # 成功作业 76
```

产物：

- `results/c1_vi_vjp_one_sided_v2/report.json`：完整方向、三次 repeats、逐步长值、分支与 solver summaries；SHA256 `f76d4857b591c550aaa135fbe172b6a96eb0154cbcf28fe5a7ae6de8c8b0458e`。
- `results/c1_vi_vjp_one_sided_v2/traces.npz`：baseline 与全部 12 个扰动的真实执行前缀、witness、约束、guards、margin branches；SHA256 `7c4d688dba53a80494553436f1cf923a02d45409321ec2052dd11cb0d1fbdba8`。
- 独立 probe `scripts/diagnose_c1_vi_vjp_one_sided.py`：SHA256 `7bee5c62bf9b789389d46482a7c99811390fcb323806f1e29d76f6c3b0f426b0`。

此前独立配对 outcome 实验的负证据保持不变：50 个 augmented 场景中 reference、negative、positive、random 都是 28 个 primary deadlock、22 个 success，差值及 CI 均为 0。本次未重跑 outcome。

**C: slopes cannot be stabilized under the frozen numerical configuration; INCONCLUSIVE**
