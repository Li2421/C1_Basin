# 工件有效性映射

本审计没有改物理状态、控制器、eta、模型、K或rollout结果。旧工件保留；修正的是证据类型和聚合输出。原始源仓库路径均相对 `/home/zhihan/research/Basin_C1`。

| 工件/具体用途 | 状态 | 适用范围与处理 |
|---|---|---|
| `datasets/orthoflow3_basin_dataset_v2_audited/states.parquet` | VALID | 本次未发现新的状态身份/标签绑定缺陷；并非对尚未进行的完整continuation contract作保证 |
| v2 `eta_labels.parquet` 的seed计数、B15、原始结果引用 | VALID | 保留停止/数值语义；不用几何近邻代替tuple证据 |
| v2早停比例被当作完整Q16的派生表述 | INVALID | 原数据本身记录count；只有完整16有效seed是exact Q16，其他保留计数、上下界或似然用途 |
| 冻结generator seed41 | VALID | 历史版本，部分v1 Ring安全训练标签；当前K16输入/输出逐值重现。不是新Phase-B模型 |
| 冻结critic seed23 | VALID | 历史模型和实测排名证据有效；不能把sigmoid直接宣称校准后的真实成功概率。无需为报告修复重训 |
| Ring K16 `frozen_proposals.json` | VALID | 60/60嵌套及内容hash通过，17候选皆不重复；critic/oracle同集合 |
| Ring K16 frozen evaluation physical manifests | VALID | 这批测试已被查看，仍仅诊断，不恢复untouched资格 |
| Ring K16 seed-level DB rows | VALID | 16,320条均存在；16,214有效、106数值未认证；后者不作普通失败 |
| Ring K16 candidate中87条含数值失败的标量 `Q16` | INVALID | 作为exact值无效；原值作为lower bound仍有效。替代：本目录 `H2_k16_corrected_per_state.json` |
| Ring K16平均regret=0.080208…作为精确值 | INVALID | 替代严格区间[0.0791667,0.0802083]；不是新模型性能 |
| Ring K16 oracle60、critic53、miss7、exploitation4 | VALID | 用上下界仍不变；B15是经验资格，非总体成功概率保证 |
| Ring revision最近eta“饱和/排名”发展证据 | VALID（proxy用途）；INVALID（精确候选证明用途） | 文件本来注明proxy；不应据其宣布部署候选精确覆盖/排名已经验证。详见H1证据注释 |
| Four/Ring旧对称性结论 | VALID | 本轮未重跑完整对称性套件，也未替代其表示风险结论 |
| 历史Four的sigmoid-argmax选择结果 | VALID | 真实执行过的旧selector结果有效；切换到logit-argmax是versioned tie-handling变更，不能偷偷归因于权重变化 |
| 正在运行的Phase-A / 未完成Phase-B工件 | 不作有效性验收 | 尚未接受，不能借本次审计宣布成功；Phase-A代码已有logit排名和完整Q限制 |

本次无 `STALE_REQUIRES_RECOMPUTE` 的物理rollout，无需重编码h，无需重训generator/critic，无需重放训练数据。`VALID_AFTER_REENCODING`未使用：此次没有改conditioning表示。
