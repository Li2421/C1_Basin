# Four-Way Stage-I iteration log

This is a data/protocol log, not a post-hoc optimization of frozen-test cases.

| Version | Train/dev-only diagnosis and action | Development outcome | Decision |
|---|---|---|---|
| v4 | Reset clearance audit found incidental outer-wall exposure; expert/observation/action alignment was numerically clean. | 0% success; 95% wall. | Expand outer buffer only; do not alter central crossing. |
| v5 Base-U / U+W | Broad nominal + uniform data; wall precursor re-query allowed from dev. | Base-U: 0/20, 80% wall. Dense U+W: 0/20, 80% wall. | Wall data alone was insufficient. |
| v6 | Uniform re-query from valid development policy trajectories. | 0% success; 70% wall, 10% agent, 20% timeout. | Audit rollout physics. |
| v7/v8 | Lane-local representation and goal-focused recovery were tested. | v8: 0% success; 93.3% wall. | Negative representation ablation; return to world frame. |
| v9 | World-frame goal audit found 42 agents reached a goal and 41 later departed; 23 terminally overshot, 19 still forward. Expert goal-hold plus generic/goal recovery. | 15% success; wall removed; 10% agent. | Goal-hold is a physical recovery fix. |
| v10 | Dense timeout-tail/goal-settling recovery. | 45% success, 0% wall, 10% agent, 45% timeout. | Add separately audited agent precursors. |
| v11 | Timeout generic recovery plus 9 valid, pre-contact `targeted_agent` continuations. | 60% success, 0% collision, 40% timeout on dev20. | Mature provisional baseline. |
| v12 | One bounded final timeout-only recovery ablation. | 45% success, 5% agent, 50% timeout. | Negative; freeze v11 rather than iterate indefinitely. |
| v13 | First v11 frozen test had 73.3% t0 OOD. Per protocol, did not adapt to those 30 states: made independent broad train120/dev30/test60 global draws, retained only prior train recovery, and sealed new test. | New dev30: 63.3% success, 0% wall, 3.3% agent, 33.3% timeout, 6.7% t0 OOD. | Master opened new test once; no subsequent adaptation. |

Frozen v13 test: 48.33% success, 1.67% wall, 8.33% agent, 41.67% timeout.
The final decision is `PASS_STAGE1`; this slightly-below-50% result is not
used to reopen the data-collection loop.
