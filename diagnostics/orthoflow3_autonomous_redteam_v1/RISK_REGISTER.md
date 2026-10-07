# 检验前风险排序

只选择以下四项。排序依据已读的可执行代码，而非提示中的示例列表。所有模型/数据只读；新rollout初始预算3000，优先0新增。已冻结测试候选只允许身份与报告重算，不作训练/选择。

| 优先级 | 假设 / 影响范围 | 若成立的影响 | 可信度与已有约束 | 最便宜的判别实验 |
|---|---|---|---|---|
| H1 | Ring train/dev“饱和/排名完美”依据是邻近eta标签，不是部署候选的真实Q | 高：可能错误解释为只在测试上发生的泛化失败；影响critic诊断与发展决策 | 高：`analyze_train_dev.nearest`直接复制最近label；报告虽称proxy，但据此跳过R3/R4 | 保留同state/eta/safety语义，用现有DB中精确proposal证据核对；只对齐现成候选，不新扫K、不重训 |
| H2 | K16聚合把 numerical-uncertified seed按0加入名为Q16的标量 | 中：可能误报exact regret/概率，甚至在边界影响oracle选择；B15分类可能仍正确 | 高：`run_k16.evidence`直接successes/16；早期frozen evaluator反而保留Q16=None+lower | 对16320现成seed查询，逐candidate计算Q上下界/B15，检验headline与exploitation是否在区间内不变；完全解析证据，无新rollout |
| H3 | K16候选身份/输入/随机数或critic与oracle候选列表不一致 | 高：若存在可直接颠覆60/60 vs53/60解释 | 中低：exact nesting gate已有，但未独立复算全链 | 静态callgraph；60冻结manifest全量hash/顺序/去重及DB对应；4个train+4个dev Ring/Four每场景按hash抽样重建网络输出、置换候选顺序控制；不评新结局 |
| H4 | 早停B15证据被无条件称为校准Q，或被误当full16，且经验失败率选择偏差改变critic解释 | 中高：标签学习目标/校准而非安全 | 中：原loader用success/observed count并乘count；该计数似然在已知停止规则下可能本来正确，不能直接判bug；v2报告READY范围过宽 | 数学停止似然控制、只读数据各seedcount/early-stop语义统计；与已有完整Q匹配比较及新Phase-A已做的限定核对，不生成标签 |

## 明确不重复

外边界、全物理对称性、一般DB完整性、K4→16性能扫描、ranking-loss/LCB、完整continuation contract均不在本轮。后者已有独立授权任务，除非本轮身份测试发现具体新冲突，否则只作为未完成限制标注。

## 预注册共同判据

- 候选/数据身份：原始float64 eta逐值和canonical eta ID必须一致，不以近邻容差替代。
- 网络重算：固定旧归一化和checkpoint；同dtype误差<=2e-6，top1须一致（真实tie单列）。
- Q：只有16个有效canonical seed才能称exact Q16；数值未认证保留[s/16,(s+u)/16]。
- B15：s>=15肯定positive；有效失败>=2肯定negative；其余unknown。
- H1的proxy偏差不是physical rollout失效；必须同tuple的精确证据才能否定proxy数值。安全哈希不同不能偷换因果解释。
- H4停止规则允许的Bernoulli计数似然与“将停止比例作为无偏Q估计”明确分开；不因直觉宣称训练bug。
- 不用单个14/16→15/16翻转证明模型/物理语义问题。

## 自主追加唯一分支 H5（首次推理探针失败后，追加结果查询前）

H3探针首次失败于train/dev的argmax检查。静态追踪显示：历史K16在float32 sigmoid概率上argmax；正在运行的Phase-A则在logits上argmax。有限精度sigmoid可能把不同高logit变成相同1.0，使旧流程按候选先后打破人为tie。这不是新模型、连续优化或K实验。

H5：**概率饱和造成有限候选选择的非语义tie/顺序依赖**。影响可能高，但未假定会改变Ring的7miss。

最小测试：同一16个train/dev输入和60个已冻结artifact，记录raw logits、float32概率、两种argmax、真实tie/饱和tie，并以float64 sigmoid和候选置换作为控制。随后仅查询这些已有候选的缓存Q，定量是否影响历史headline。无新rollout，无模型/校准变更；若历史结果不变则按潜在或已在Phase-A修正的数值限制报告，不当作Ring失败解释。

H4静态补充：原 `critic_batch` 第391–394行按观测B15类别进行50/50重采样；这与停止本身不同。用精确停止树加固定正例采样权重1/3/3.5作为校准控制，检验“count-weighted NLL天然保证概率校准”是否成立。即使该控制偏移，也不能把真实网络的4个错误全部归因于此；新Phase-A已经使用proposal/state均衡，不再按B15类别均衡，故不另开重训。
