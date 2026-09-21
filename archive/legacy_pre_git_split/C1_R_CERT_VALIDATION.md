# R_CERT 训练前验证结论（2026-09-18）

## 判定

总判定：**INCONCLUSIVE，禁止开始完整 G_phi 训练。**

| 层次 | 判定 | 证据 |
|---|---|---|
| 通用聚合实现 | PASS | 公式逐项实现；7项新测试通过；稳定区多步长有限差分一致 |
| 证书有效性 | INCONCLUSIVE / semantic blocker | 仓库没有经批准的具体 `c_ejℓ`、索引集、尺度或 `C_ej ⊆ complement(D_e)` 证明 |
| 提前预测 | NOT RUN | 无法计算 R_CERT 或覆盖失败 F_C；未冻结 K/M、检查点及条件未来分布 |
| 梯度因果用途 | NOT RUN | 无轨迹证书可反传；未构造或执行干预方向 |
| 完整训练 | NOT AUTHORIZED | 前两项机器人验证均未发生 |

这不是对 R_CERT 机制的实证失败，也不是 PASS。缺少的对象正是风险定义的一部分；用经验相关性、历史风险或进展势补上都会改变候选。

## 冻结的当前系统语义

控制器代码路径为：

```text
u_nominal = speed_bound(FlowBC_theta(observation, xi))
u_safe    = Pi_U(x)(u_nominal)
correction = G_phi(observation, u_safe)
u_phi     = Pi_U(x)(u_safe + correction)
```

`FlowBC_theta` 和 `G_phi` 参数在本验证中应冻结。`ResidualFlowField.prepare` 没有对状态、Flow 输出、第一投影或 correction 做 `stop_gradient`；控制器投影使用自定义 VJP。因此以后冻结参数时仍须保留状态/输入通过 Flow-BC 和两次投影的导数。只有原 first-event 离散决定在 `termination.event_step` 中停止求导。本轮因语义阻断，没有新 rollout 接口，也没有改变这些路径。

当前权威死锁并集的有限时域为 `H=850×0.05=42.5 s`：

1. `D_strict`（环境 `safe_deadlock`）：每个动作后的联合状态先检查成功。成功要求两机器人目标误差均不超过 `0.08 m`。在完整 `2.0 s` 历史可用时，候选死锁要求两机器人目标误差变化的最大绝对值严格小于 `0.01 m`，两机器人当前速度的最大值严格小于 `0.5×0.05=0.025 m/s`，且尚未成功。候选必须连续维持 `5.0 s`。`dt=0.05 s` 时首次候选样本计时为0，因此触发包含101个候选样本、跨100个时间间隔。碰撞优先于成功，成功优先于死锁，死锁优先于期限 timeout。
2. `D_stalled`（独立六分类监测器 `stalled_deadlock`）：只细分 first-event 为 `other_timeout` 的轨迹。取最后 `round(2.0/0.05)=40` 个动作后样本；所有样本的系统最大速度严格小于 `0.05 m/s`，且每个机器人末端目标误差与该窗口首样本的差的绝对值严格小于 `0.02 m`。这40个样本跨39个时间间隔。成功、碰撞和 strict deadlock 均不进入这一分支。
3. `D = D_strict OR D_stalled`。严格死锁首次触发步保存在环境历史中；若用不终止的扩展轨迹，后来恢复不能清除已经发生的 `D_strict`。条件续演必须携带当前位置、上一执行速度、目标距离历史、`candidate_since`/stuck timer、已发生 first-event 标志和剩余时限；从检查点重新初始化监测器不是同一个条件状态。

当前实现只有两个机器人。题目要求的“moving outsiders 掩盖 local deadlock”涉及更多机器人或不同局部事件语义，不能用当前二机器人全局最大速度检测器冒充已经定义的子事件。

## 已实现的固定聚合

新模块只接收调用者提供的、按事件分组的二维 margin 张量 `[M_e,L_e]` 和同形固定尺度。它实现：

```text
h(z)       = softplus(z) / log(2)
S_ej       = sum_l h(-c_ejl / sigma_ejl)
R_e        = -kappa log[(1/M_e) sum_j exp(-S_ej/kappa)]
R_CERT     = max_e R_e
```

实现使用稳定 `jax.nn.softplus` 和 `logsumexp`。没有原子平均、裁剪、替代平滑、detached gate、直通梯度或附加项。外层是精确 `max`，并列点保留非光滑性。空事件集、空证书集、无原子证书、形状不匹配、非正/非有限 `sigma` 和非正/非有限 `kappa` 都抛出错误。

对任一事件，令 `s*=min_j S_ej`，则

```text
s* <= R_e <= s* + kappa log(M_e).
```

若已证明 `C_ej={all_l c_ejℓ>0} ⊆ complement(D_e)`，则 `D_e` 发生时每个 `j` 至少有一个 `c_ejℓ<=0`。因为 `h(0)=1` 且 h 单调，所有 `S_ej>=1`，所以 `R_e>=1`；任一 `D_e` 发生即有 `R_CERT>=1`。该证明在 `c=0` 边界仍成立，因为证书使用严格 `>0`。

单元测试还包括一个必要性反例：若在一条标记为 `D_e` 的轨迹上允许某个证书所有 margins 均为大正数，则该行 `S_ej` 接近0，`R_CERT<1`。这直接说明聚合公式不能代替 inclusion 证明。

## 语义阻断

本次提案及当前仓库没有冻结以下必要内容：

- 对 `D_strict` 和 `D_stalled` 分别有哪些 `j`，每个证书有哪些原子 `ℓ`；
- 每个 `c_ejℓ(τ)` 的公式、正负方向、物理单位、时间窗口和轨迹索引；
- 每个固定 `sigma_ejℓ` 及固定 `kappa` 的数值；
- 每个证书集合对相应事件补集的解析证明，包括严格阈值、first-event 顺序、历史状态和42.5秒时域；
- 证书是否以及如何处理已发生死锁的历史锁存，避免后来恢复擦除事件；
- valid coordination 未被证书覆盖时的预期行为和可接受覆盖失败率。

历史 `completion_certificate.py`、`temporal_certificate.py` 和 `progress_debt.py` 不能填这个空缺。它们使用 max/min/ReLU、期限失败门控或欧氏进展欠账，既不是此次固定聚合，也没有声明为本次 `C_ej`。把 detector occurrence margins 改写成排除证书同样需要新的语义选择和 inclusion 证明，属于设计候选，超出此次“只验证、不改进”的授权。

因为 `C_ej` 未定义，`F_C={exists e: every C_ej fails}` 也未定义。历史轨迹上没有上界违例，只能提供经验覆盖，不能证明集合包含关系。

## 通用数值与梯度审计

在没有 event tie、certificate tie 或投影的纯聚合测试点，自动微分方向导数为 `-0.4258055973`。中心有限差分步长从 `1e-2` 到 `1e-7`；相对误差依次约为 `6.97e-6, 6.97e-8, 6.95e-10, 2.49e-11, 7.55e-10, 4.93e-9`。

显式 outer-max tie 反例取两个相同事件值和交换方向。JAX 在并列点给出的所选次梯度方向值为0，而两侧差商为 `+0.7213477` 与 `-0.7213477`；普通导数不存在。实现没有平滑这个断点。

原子在大正 certificate margin 上饱和：`c/sigma=10` 时导数约 `-6.55e-5`，为100时约 `-5.37e-44`。在大负 margin 上惩罚线性增长，导数趋于 `-1/log(2)`。`c/sigma=±10^6` 的测试没有 NaN/Inf；正侧允许稳定地下溢到精确0。通用测试没有求解器调用，因此不能报告 closed-loop projection attenuation、动作梯度覆盖、证书间抵消或 solver failure。

## 未执行的验证及反例

按提案要求，以下均未执行：held-out early prediction、风险分箱、AUC/排序及不确定性、`P(F_C and not D)`、早期 `-g/random/+g` 配对回放、J_def、投影后能量匹配、扩展时域、机器人有限差分和代表性失败轨迹。没有 Slurm/GPU 作业，也没有开始训练。

指定的 normal slow yielding、long nonblocking timeout、local deadlock masked by moving outsiders、jitter/livelock、retreat-and-stop abandonment、deadlock delayed beyond H、deadlock followed by recovery、symmetric escape modes、projection plateaus、valid coordination missing from certificates，都依赖具体 `c_ejℓ` 才能成为 R_CERT 反例。现在强行构造会等价于发明证书。当前只保留上面的逻辑 inclusion 反例；其余必须在证书获批后按冻结定义实现。

## 训练前仍缺的证据

先提供并批准完整证书表：`e,j,ℓ,c_ejℓ,unit,index/window,sigma`，固定 `kappa`，以及逐证书 inclusion 证明。随后才能冻结早期 checkpoints、K/M、独立 seeds、未来随机分布、干预窗口、随机方向数量、投影后能量目标、场景聚类和扩展时域。之后依次执行 coverage/counterexamples、独立预测、机器人梯度审计和配对因果干预。只有证书有效性、预测用途、负梯度实际用途都通过，才有理由开始完整 G_phi 训练。

## 可复现命令与文件

```bash
cd /home/zhihan/research/02_C1_Toy_GiveWay
JAX_PLATFORMS=cpu .venv-c1/bin/python -m unittest tests.test_c1_r_cert
JAX_PLATFORMS=cpu .venv-c1/bin/python -m unittest \
  single_integrator.tests.test_deadlock \
  single_integrator.tests.test_c1_deadlock_union \
  single_integrator.tests.test_c1_completion_certificate \
  single_integrator.tests.test_c1_temporal_certificate
audit_dir=$(mktemp -d)
JAX_PLATFORMS=cpu .venv-c1/bin/python scripts/audit_c1_r_cert_aggregate.py \
  --out "$audit_dir/audit.json"
```

变更文件：

- `single_integrator/c1/risk/r_cert.py`：通用精确聚合；不含机器人证书。
- `tests/test_c1_r_cert.py`：7项公式、上界、条件性事件关系、反例、JIT梯度和非法输入测试。
- `scripts/audit_c1_r_cert_aggregate.py`：稳定区多步长有限差分、outer-max tie 与原子饱和审计。
- `results/c1_r_cert_aggregate_audit_v1.json`：机器可读审计结果。
- 本文件：冻结语义、阻断与判定。

测试结果：新测试 `7/7 PASS`；相关现有监测/历史证书回归 `17/17 PASS`。这些 PASS 只证明通用聚合和原有代码未被破坏，不证明机器人 R_CERT 已定义或有效。
