门的执行正确，N=2 Flow 在相向任务上能力不足。本部分只读，未重判门，未运行新rollout。

历史能力依据：dev navigation 12/12 in all four basic categories; dev temporal release 12/12 even-first and 11/12 odd-first; no opposing rollouts inspected。本轮为同时相向交换；导航和有序释放证据不能替代此任务的成功证据。

| 方法 | 成功state索引 | 成功率 | 终止类型 | 失败分类 |
|---|---|---:|---|---|
| 只加安全投影、不加修正 | [11] | 6.25% | {'timeout': 15, 'success': 1} | {'仍在移动的超时': 15} |
| 全程固定用η10的修正 | [0, 10] | 12.50% | {'success': 2, 'timeout': 14} | {'停滞候选': 14} |
| 前round(5/7·H)步用η13、之后用η10的时间表 | [0, 7, 10] | 18.75% | {'success': 3, 'timeout': 13} | {'仍在移动的超时': 12, '停滞候选': 1} |

逐state明细见 N2_GATE_AUDIT.json。动作记录在专家数据中原为float32；修复检查器后按原记录dtype比较，原生动作与位置执行仍为float64。原有Flow/safety/native轨迹及专家完整位置、终止、步数均逐位一致；本轮再次核对32条既有复现产物，未重跑。

对应的“只执行controller动作”终止标签在此能力门中缺失，本部分禁止补跑，已逐条列为证据缺口；它不阻止基于现有动作路径与原门阈值的审计结论。
