# N2 历史选择、验收与相向测试分开列示

冻结元数据的选择依据：dev navigation 12/12 in all four basic categories; dev temporal release 12/12 even-first and 11/12 odd-first; no opposing rollouts inspected。这是dev选择依据，不能把它替换为后续fresh acceptance的分母。

| 来源 | 模式 | 记录 | 成功 | 终止分布 |
|---|---|---|---|---|
| gap_flow_competence_v3_recovery/dev_50k | None | 48 | 48 | {'success': 48} |
| gap_flow_competence_v3_recovery/acceptance_basic | None | 48 | 48 | {'success': 48} |
| gap_flow_competence_v3_recovery/acceptance_release_even_v2 | temporal_release_even_first | 12 | 11 | {'success': 11, 'timeout': 1} |
| gap_flow_competence_v3_recovery/acceptance_release_odd_v2 | temporal_release_odd_first | 12 | 11 | {'success': 11, 'timeout': 1} |
| frozen_eval/opposing | opposing | 12 | 0 | {'timeout': 12} |
| replication_n2/opposing | opposing | 12 | 1 | {'timeout': 11, 'success': 1} |

acceptance_release_even_v2与odd_v2均为11/12；与freeze selection_basis中dev even12/12、odd11/12不是同一cohort。冻结后opposing测试0/12，后续opposing replication1/12；均独立于R36新pilot且不是同菜单/同RNG的逐条重跑。所有源路径和SHA见N2_GATE_AUDIT.json。C的成功state索引从0开始，C无新增仿真、不修改原20%门。
