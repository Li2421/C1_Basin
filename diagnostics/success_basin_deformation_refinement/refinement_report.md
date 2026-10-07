# Success-basin deformation refinement

## 结论

三组状态都在旧网格之外找到了更低的、满足当前成功判据的 `J_def` 区域，且相对旧最低点的 32-seed 配对 CRN 差异均为正。可是复核后的最低候选仍位于本次搜索边界，D1/D4 还紧邻 SUCCESS/UNKNOWN 转换；与近邻候选的差异也不足以确立唯一局部最低点。因此不能称为全局或已解析的局部最优。

**最终建议：EXPAND_FURTHER**

## 固定量与实验量

实现的量严格为

```text
delta_u[k] = u_exec[k] - u_safe[k]
J_def = 0.05 * sum_k ||delta_u[k]||_2^2
```

其中 `u_safe` 与 `u_exec` 来自同一个 corrected state 和同一次 Flow sample，`u_exec` 是第二次 hard projection 的输出。没有折扣、时长归一化、进度项、平滑项或额外正则。

第一阶段新增 1,024 条 rollout，第二阶段新增 1,376 条；预声明复核再新增 192 条（每状态 4 个固定候选 × 16 个新种子）。新运行总计 2,592 条、673,769 physical steps；复核 192/192 全部成功。搜索阶段每个 cell 用 16 个共同随机种子，入围候选与旧最低点用额外 16 个共同随机种子复核，合并为 32 seeds。

## 各状态结果

| state | 旧 eta | 旧 16-seed mean J | refined eta | success | mean ± std J | mean steps | 配对 old-new mean [95% CI] | 位置 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D1 | (0.75, 0.0, 0.0) | 0.284445 | (0.59375, -0.3125, -0.125) | 32/32 | 0.243668 ± 0.009890 | 213.7 | 0.036520 [0.031983, 0.041056] | ADJACENT_TO_SUCCESS_FAILURE_BOUNDARY_AND_STILL_ON_SEARCH_BOUNDARY |
| D2 | (0.5, -0.5, 0.0) | 0.221029 | (0.4375, -0.5625, 0.0) | 32/32 | 0.210658 ± 0.011096 | 273.7 | 0.010072 [0.004848, 0.015296] | STILL_ON_SEARCH_BOUNDARY |
| D4 | (0.5, -0.5, 0.0) | 0.217819 | (0.375, -0.4375, -0.0625) | 32/32 | 0.197764 ± 0.013116 | 325.8 | 0.022251 [0.016075, 0.028427] | ADJACENT_TO_SUCCESS_FAILURE_BOUNDARY_AND_STILL_ON_SEARCH_BOUNDARY |

这里配对差异定义为 `J_old - J_refined`，正值表示 refined 候选更低。`refined eta` 的均值/std/success/episode length 均基于 32 seeds。

## 唯一性与边界

- D1：次优 eta=(0.59375, -0.3125, -0.1875)；次优减最低的配对均值=0.000635，95% CI [-0.000608, 0.001878]；唯一最低点未得到支持。
- D2：次优 eta=(0.5, -0.5625, -0.0625)；次优减最低的配对均值=0.000141，95% CI [-0.004905, 0.005186]；唯一最低点未得到支持。
- D4：次优 eta=(0.40625, -0.4375, -0.0625)；次优减最低的配对均值=0.001167，95% CI [-0.005780, 0.008113]；唯一最低点未得到支持。

所以当前 `eta*_emp` 是“已测试候选中的 refined empirical success-constrained minimum”，不是已解析的连续局部最优。D2 与 D4 不再共享完全相同的最低 eta；两者参数距离为 0.15309，仍落在相近的 moderate-goal / negative-safe / small-relative 区域。

最近的 failure/unknown 点：

- D1：eta=(0.5625, -0.3125, -0.125), UNKNOWN_CELL, S=13/16, d=0.03125; eta=(0.59375, -0.25, -0.125), UNKNOWN_CELL, S=14/16, d=0.06250; eta=(0.5625, -0.3125, -0.1875), UNKNOWN_CELL, S=13/16, d=0.06988。
- D2：eta=(0.375, -0.5, 0.0), UNKNOWN_CELL, S=14/16, d=0.08839; eta=(0.375, -0.5, -0.125), UNKNOWN_CELL, S=14/16, d=0.15309; eta=(0.25, -0.5, 0.0), UNKNOWN_CELL, S=3/16, d=0.19764。
- D4：eta=(0.34375, -0.4375, -0.0625), UNKNOWN_CELL, S=12/16, d=0.03125; eta=(0.34375, -0.4375, 0.0), UNKNOWN_CELL, S=12/16, d=0.06988; eta=(0.3125, -0.5, 0.0), UNKNOWN_CELL, S=15/16, d=0.10825。

## 完整性检查

- 检查了全部 2,592 条新轨迹；NPZ 哈希、terminal outcome、动作维数与冻结源码哈希均通过。
- `J_def` 重构最大误差为 0.000e+00；逐步 `||delta_u||^2` 重构最大误差为 0.000e+00。
- `w-u_safe-g` 最大误差为 5.551e-17；位置积分最大误差为 1.110e-16。
- `delta_u` 在第二次 hard projection 后计算；`u_safe` 来自同一 corrected state、同一 Flow sample。logger 只读这些量，冻结源码哈希未变化，输出仅写入本目录。
- 数据中没有全程 `u_exec == u_safe` 的实际新 rollout；正定平方和给出 `J_def=0 iff 每一步 delta_u=0`，并通过零数组单元检查。未观察到与该等价关系冲突的轨迹。

## 尚未执行的实验

**尚未运行实验 B：拆解 `J_def` 和 episode length 的关系。** 本轮按要求只报告 episode length 作为伴随描述量；没有做按时长归一化、固定时域反事实、每步强度/持续时间分解、相关性归因或因果解释。因此当前较低 `J_def` 可能同时包含动作偏差幅度与 episode duration 的共同影响，这一点尚未被拆开。

所有 cell、置信区间与最近边界点见 `refined_cells.json`、`refined_minima.json` 和 `paired_comparisons.json`。
