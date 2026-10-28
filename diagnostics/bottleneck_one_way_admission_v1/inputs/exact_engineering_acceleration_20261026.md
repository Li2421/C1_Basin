# 仿真执行的精确工程加速：规范与逐位认证（2026-10-26）

**用途：** 之后每份 Basin_C1 prompt 都包含"执行加速"一节并引用本文件（用户 2026-10-26 要求）。本文件给出四项已做原型、已逐位核对的改动，以及每轮必须完成的认证流程。

## 一、原则

- 只改计算机工程实现，不改执行语义：同样的输入必须得到逐位相同的每一步数值与结局。
- **不减少**场景、候选、状态、控制器或种子；不换求解器、不改容差、不改精度（`jax_enable_x64` 保持 False）。
- 改动只放在本轮复制的执行代码里；`shared_control/`、`single_integrator/`、`bottleneck_family/` 等共享源码只读。
- 每项改动在主线 rollout 前必须通过第四节的逐位认证；某项不过就关掉该项、用原代码继续，记入 `ITERATION_LOG.md`。加速或认证本身出问题属于非主线失败，**不得停止整轮**。

## 二、时间花在哪（2026-10-26 剖析，3 个快控制器，单核每局 1.2–3 秒，带障碍对穿 3–6 秒）

| 部分 | 占比 | 说明 |
|---|---:|---|
| 安全投影 `project_velocity`（Clarabel） | 37–51% | 每台机器人每步两次、每次只有 2 个变量；其中约一半是 scipy 构造稀疏矩阵的开销 |
| 名义控制器 | 13–33% | 方形场地上"轨迹到各墙段距离"的代价 `wall_penalties` 占 24% |
| 安全层局部约束 `local_problem` | 12–18% | 对每台机器人逐段遍历全部墙段 |
| JAX 随机数键 `jax.random.fold_in` | 4–20% | 每次约 88 微秒的调度开销；分布式模型预测控制每步每台机器人各一次 |
| 修正场、诊断、存轨迹 | 约 6%、4%、1–4% | 不在本文件的改动范围 |

不同 η 之间没有可共享的计算：修正的三个分量中只有"互斥"分量受接触门控，"朝目标"与"侧移"分量每一步都起作用，轨迹从第一步就不同。

## 三、四项改动（规范与参考写法）

### 1. 投影求解器：直接构造同样的稀疏矩阵

在本轮复制的 `swap_safety.py` 中，把 `from shared_control.hard_projection import project_velocity` 换成从本轮新模块导入同签名的 `project_velocity`。新函数与原函数逐行相同，只有两处不同：约束矩阵按列直接拼出 CSC 数组（每列先放线性约束块中非零的行，行号递增，再放该机器人速度锥块的那一行，值为 −1.0；行索引与列指针用 int32），代替逐块 `sparse.csc_matrix` 再 `sparse.vstack`；二次项矩阵（单位阵）按维数缓存数组、每次用缓存数组的拷贝新建。这样交给 Clarabel 的 `data`、`indices`、`indptr`、维数和 dtype 与原写法完全相同，求解结果因此逐位相同。早返回（名义速度可行）分支、诊断字段、认证判据、异常类型都不改。

```python
_P = {}
def project_velocity(nominal, matrix, lower, max_speed, config, check=False):
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 2: raise ValueError("nominal must have shape [N,2]")
    num_agents = len(nominal); target = nominal.reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64); lower = np.asarray(lower, dtype=np.float64)
    if not all(np.isfinite(v).all() for v in (target, matrix, lower)): raise ValueError("nonfinite projection input")
    if matrix.shape != (len(lower), 2 * num_agents) or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("invalid projection dimensions or speed limit")
    if matrix.size and np.min(matrix @ target - lower) >= 0 and np.max(np.linalg.norm(nominal, axis=-1)) <= max_speed:
        d = HP._diagnostics(target, target, matrix, lower, max_speed)
        d.update(backend="clarabel_exact_socp", solver_status="nominal_feasible", accepted=True, conic_iterations=0, projection_input=target.tolist())
        return nominal.copy(), "nominal_feasible", d
    m = len(lower); n = 2 * num_agents; neg = -matrix; data = []; idx = []; ptr = [0]
    for j in range(n):
        r = np.flatnonzero(neg[:, j] != 0); data.append(neg[r, j]); idx.append(r)
        agent, k = divmod(j, 2); data.append(np.array([-1.0])); idx.append(np.array([m + 3 * agent + 1 + k]))
        ptr.append(ptr[-1] + len(r) + 1)
    cone_matrix = sparse.csc_matrix((np.concatenate(data), np.concatenate(idx).astype(np.int32), np.asarray(ptr, np.int32)), shape=(m + 3 * num_agents, n))
    if check:   # 认证用：与原写法的数组逐一比较
        blocks = [sparse.csc_matrix(-matrix)]
        for agent in range(num_agents):
            b = np.zeros((3, n)); b[1:, 2 * agent:2 * agent + 2] = -np.eye(2); blocks.append(sparse.csc_matrix(b))
        ref = sparse.vstack(blocks, format="csc")
        assert ref.shape == cone_matrix.shape and all(np.array_equal(getattr(ref, a), getattr(cone_matrix, a)) and getattr(ref, a).dtype == getattr(cone_matrix, a).dtype for a in ("indptr", "indices", "data"))
    cones = [clarabel.NonnegativeConeT(m)] + [clarabel.SecondOrderConeT(3) for _ in range(num_agents)]
    cone_rhs = np.concatenate([-lower] + [np.asarray([max_speed, 0.0, 0.0])] * num_agents)
    if n not in _P: e = sparse.csc_matrix(np.eye(n)); _P[n] = (e.data.copy(), e.indices.copy(), e.indptr.copy())
    P = sparse.csc_matrix((_P[n][0].copy(), _P[n][1].copy(), _P[n][2].copy()), shape=(n, n))
    s = clarabel.DefaultSettings(); s.verbose = False; s.max_iter = config.max_iterations
    s.tol_feas = min(config.feasibility_tol, config.speed_tol); s.tol_gap_abs = config.solver_ftol; s.tol_gap_rel = config.solver_ftol
    sol = clarabel.DefaultSolver(P, -target, cone_matrix, cone_rhs, cones, s).solve()
    # 以下与原函数逐行相同：candidate、_diagnostics、dual、slack、stationarity（用 cone_matrix.T @ dual）、complementarity、
    # diagnostics.update(...)、certified 判据、CBFSolverError、返回值。
```

### 2. 随机数键：用 NumPy 复现 JAX 的键折叠

控制器用到的随机性只是把 JAX 键的字节做 sha256 后交给 `numpy.random.default_rng`（`controllers_new.rng_for`），因此只需键的 32 位整数值完全相同。用下面的 NumPy Threefry-2x32（20 轮）替换本轮复制代码中的调用点：`runtime_new.run` 每步的 `jax.random.fold_in(root, t)`，`controllers_new` 中分布式随机采样与分布式交叉熵的每机器人 `jax.random.fold_in(key, i)`；每局开头的 `root` 可仍用 JAX 计算，再转成 `numpy.asarray(root, numpy.uint32)`。只适用于旧式 `uint32[2]` 键（不启用自定义随机数实现）。原型核对：3,000 组随机键与数据、以及 `PRNGKey(2026102004) → 哈希 → 种子 → 步 → 机器人` 整条链（300 步 × 10 台）全部与 JAX 相同；单次 88 微秒降到 10 微秒。

```python
M = np.uint64(0xFFFFFFFF); R = ((13, 15, 26, 6), (17, 29, 16, 24))
def _rotl(x, d): return ((x << np.uint64(d)) | (x >> np.uint64(32 - d))) & M
def threefry2x32(k0, k1, x0, x1):
    k0, k1, x0, x1 = (np.uint64(v) for v in (k0, k1, x0, x1))
    ks = (k0, k1, (k0 ^ k1 ^ np.uint64(0x1BD11BDA)) & M)
    x0 = (x0 + ks[0]) & M; x1 = (x1 + ks[1]) & M
    for i in range(5):
        for r in R[i % 2]:
            x0 = (x0 + x1) & M; x1 = _rotl(x1, r); x1 = x1 ^ x0
        x0 = (x0 + ks[(i + 1) % 3]) & M; x1 = (x1 + ks[(i + 2) % 3] + np.uint64(i + 1)) & M
    return np.array([x0, x1], dtype=np.uint32)
def fold_in(key, data):
    k = np.asarray(key, dtype=np.uint32).reshape(2)
    return threefry2x32(int(k[0]), int(k[1]), 0, int(data) & 0xFFFFFFFF)
```

### 3. 控制器的墙代价：剔除不可能进入安全边距的墙段

只改 `controllers_new.wall_penalties` 中 `xsc_` 场景、NumPy 路径（梯度下降模型预测控制的 JAX 路径不改，原函数改名保留并在其余情况下调用）。对每台机器人 j：取它全部采样点的外接矩形，中心 c、半对角线 ρ；墙段到 c 的距离减 ρ 大于 `margin + 1e-9` 的墙段剔除；对保留的墙段用原公式逐元素算距离、取最小值；没有保留墙段时该机器人的代价为 0。代价是 `max(margin − 最近距离, 0)²`：被剔除的墙段到所有采样点都远于边距，最近距离小于边距时取到的最小值必然来自保留墙段（逐位相同），否则两边都是 0。

```python
def wall_penalties(p, env, xp=np):
    c = env.config
    if xp is not np or not c.scene.startswith('xsc_'): return _wall_penalties_original(p, env, xp)
    margin = c.agent_radius + c.wall_collision_margin + c.wall_radius; zero = np.zeros(p.shape[:-1])
    walls = np.asarray(env.walls)
    if not len(walls): return zero, zero
    a_all = walls[:, 0]; ab_all = walls[:, 1] - walls[:, 0]; inner = np.zeros(p.shape[:-1])
    for j in range(p.shape[-2]):
        q = p[..., j, :]; lo = q.reshape(-1, 2).min(0); hi = q.reshape(-1, 2).max(0); ctr = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
        tt = np.clip(((ctr - a_all) * ab_all).sum(-1) / (ab_all * ab_all).sum(-1), 0, 1)
        keep = np.flatnonzero(np.linalg.norm(ctr - (a_all + tt[:, None] * ab_all), axis=-1) - rad <= margin + 1e-9)
        if not len(keep): continue
        a = a_all[keep]; ab = ab_all[keep]; d = q[..., None, :] - a
        t = np.clip(np.sum(d * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0, 1); delta = d - t[..., None] * ab
        distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 1e-30))
        inner[..., j] = np.maximum(margin - np.min(distance, axis=-1), 0) ** 2
    return inner, zero
```

### 4. 安全层局部约束：向量化预筛远墙

在本轮复制的 `swap_safety.py` 中把原 `local_problem` 改名保留，新 `local_problem` 先向量化算机器人 i 到全部墙段的距离，只保留距离 ≤ 0.5 + 1e-9 的墙段，再对保留墙段调用原函数（原标量代码、原顺序），最后把标签里的墙段编号映射回原编号。原函数对距离 > 0.5 的墙段本来就跳过；余量 1e-9 保证原函数会收下的墙段不会被预筛掉，因此约束行、下界、标签与原写法完全相同。`swap_safety_user_fix_20261023` 以名字导入 `local_problem`，导入的就是新函数。

```python
def local_problem(snapshot, i):
    walls = np.asarray(snapshot['walls'])
    if len(walls) == 0: return _local_problem_original(snapshot, i)
    p = np.asarray(snapshot['positions']); a = walls[:, 0]; ab = walls[:, 1] - a
    t = np.clip(((p[i] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    keep = np.flatnonzero(np.linalg.norm(p[i] - (a + t[:, None] * ab), axis=-1) <= .5 + 1e-9)
    A, lower, labels = _local_problem_original(dict(snapshot, walls=walls[keep]), i)
    return A, lower, [['wall', l[1], int(keep[l[2]])] if l[0] == 'wall' else l for l in labels]
```

### 原型实测（2026-10-26，90 局，原代码与改后代码逐局对比，全部逐位相同）

| 场景 | 集中式随机采样 | 梯度下降模型预测 | 分布式模型预测 |
|---|---:|---:|---:|
| 带 8 个障碍的对穿 | −67% | −32% | −56% |
| 十字路口半宽 0.25 | −54% | −18% | −42% |
| 方形场地 8 机 | −45% | −17% | −29% |
| 6 机对穿（时间紧） | −11% | −3% | −23% |
| 无墙对穿 10 机 | −4% | −5% | −21% |
| 合计 | | | −38%（快 1.6 倍） |

## 四、每轮的逐位认证（主线 rollout 之前完成）

1. **单元检查：** 随机数键折叠在 ≥ 3,000 组随机输入与本轮实际用到的整条链上与 JAX 完全相同；认证重放期间对每次投影求解开 `check=True`，至少 10,000 次求解的稀疏矩阵数组与原写法相同。
2. **重放已存记录：** 从同一执行语义的已有完整记录中分层抽样（`raw/*.json` 里有完整请求 `case`，`traces/*.npz` 里有逐步轨迹）：`hetero_scene_pipeline_v1/main`（15,288 局）、`cross_scene_correction_v1/main`（34,064 局，含文献场景、瓶颈场景与全部控制器）、`eta_diverse_scene_test_v1/main`（1,800 局），以及之后同语义轮次的记录。按 (场景类型, 控制器, 结局) 分层，合计 ≥ 1,000 局，覆盖本轮要跑的每种场景类型和每个控制器，可用时每层 ≥ 30 局，必须包含墙段多的场景。用改后代码重跑，与存档比较：`positions`、`velocities`、`nominal`、`safe`、`actions`、`projection_rewrite`、`swept_clearance`、`clearance_types` 按字节比较（形状、dtype、`tobytes()` 都相同）；`success`、`collision`、`timeout`、`numerical_failure`、`outcome`、`episode_length`、`constraint_violation_steps`、`sampled_speed_violation_steps`、`restoration_steps`、`minimum_swept_clearance`、`final_goal_distances` 完全相同。
3. **本轮新场景类型：** 没有存档的新场景类型，每类取 2 个状态 × 本轮全部控制器 × 2 个候选，原代码与改后代码各跑一次逐字节比较（这些原代码结果可直接作为正式标签入库，不浪费）。
4. **判定：** 100% 相同才启用。有不同：逐项单独开关找出责任改动，关掉该项（用原代码）后对其余改动重新认证；记入 `ITERATION_LOG.md` 与 `GAPS.md`，主线照常。
5. **复用：** 若前一轮已有 `ACCELERATION_CERTIFICATE.json` 通过、且改后模块与被替换原模块的 sha256 都与本轮一致，可复制模块并做缩减认证（≥ 200 局，含第 3 步的新场景类型）。
6. **记录：** 写 `ACCELERATION_CERTIFICATE.json`：启用的改动、改后模块与原模块 sha256、重放的 rollout 编号与分层计数、比较结果、原 `elapsed_seconds` 与改后耗时。改后模块进入 `execution_sources`，数据库按新的执行身份登记；与旧记录能否精确复用由官方预检决定；报告写明"已逐位认证等价于原代码（列出原模块 sha256）"。

## 五、尚未做原型的候选（可选，同样必须通过第四节认证）

- 机器人对的距离代价：相距远到不可能接近的机器人对跳过距离计算，但在原位置填 0，保持数组形状，使后续求平均逐位相同（不能直接删掉零项，否则求和顺序会变）。
- 按预计时长"最长的对局先跑"：只影响调度，不改数值，可直接采用。
- 精简轨迹文件中的逐步诊断记录：不影响结局但减少留档，须用户同意才做。
- CPU 份额上限由用户决定，prompt 中不自行提高。
