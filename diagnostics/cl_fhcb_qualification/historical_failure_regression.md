# 历史失败回归

| 历史问题 | 结论 | 本次证据 |
|---|---|---|
| 1. one-step flatness | NOT_APPLICABLE | 冻结主方法仅测试 K=20/100；没有把 K=1 旧证据冒充当前结论。 |
| 2. old ACTION_RANKING_FAIL | NOT_SOLVED | Stage-1 初态中 `goal025` 的 R 最低但 strict-deadlock rate 最高；正式 Q 因证书 gate 停止。 |
| 3. old `-grad_w R` failure | NOT_APPLICABLE | 正确对象应为 `grad_phi R_K`；冻结 B 仅定义四个离散 phi，故没有合法连续导数。 |
| 4. projection aliasing | PARTIALLY_SOLVED | 32 条 Stage-1 轨迹显示非零执行/状态分离且无数值 collapse；没有通过预注册 Stage-2 分布审计。 |
| 5. active-set AD/FD instability | NOT_APPLICABLE | G 未进入；没有软化投影或伪造 AD/FD。 |
| 6. cone/controller mismatch | PARTIALLY_SOLVED | 新对象使用实际双投影闭环轨迹，但 continuation B 在独立 Bellman 检验失败。 |
| 7. short-timer delayed-deadlock failure | PARTIALLY_SOLVED | timer-independent guard 在接近 deadlock 时给出预测信号，但初态/长时域排序反向。 |
| 8. timeout substitution | NOT_APPLICABLE | 本次低-R `goal025` 不是用 timeout 替代 deadlock；高-R damping/relative 才产生 timeout。 |
| 9. predictive-but-not-control-relevant risk | NOT_SOLVED | P 仅在临近终点通过，而低初态 R 对应更高最终 deadlock；OAS/LHC 失败。 |
