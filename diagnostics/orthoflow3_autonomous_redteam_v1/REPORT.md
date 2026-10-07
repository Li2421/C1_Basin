# 当前 OrthoFlow3 / Success Basin 红队审计

## 结论

**`MINOR_REPAIRS_VALIDATED`**。

找到了报告/证据使用层面的局部问题，并在独立工件中完成修正与回归；没有找到能推翻当前 Ring K16 “oracle已有好候选、critic仍有排名缺口”结论的候选错配或实现错误。没有新增rollout、训练、K扫描或方法变更。

| 当前证据 | 分类 | 说明 |
|---|---|---|
| Generator | `SUPPORTED_WITH_QUALIFICATION` | K16有限集合在这60状态有B15候选；不等于分布总体保证或新untouched结果 |
| Critic | `SUPPORTED_WITH_QUALIFICATION` | 实测53/60、7miss成立；原始score并非自动校准的成功概率 |
| Ring K16结论 | `SUPPORTED` | 60/60、53/60、7miss、4exploitation逐条DB核对保持不变 |
| Four-Way结论 | `SUPPORTED_WITH_QUALIFICATION` | 原对称性风险仍成立；另有高分饱和tie，未在探针中改变B15结果 |
| Audited dataset v2 | `SUPPORTED_WITH_QUALIFICATION` | 物理seed证据/B15标签保持有效；不是45,169条完整精确Q16 |

### 当前版本边界

审计对象是joint generator seed41 `a7bb0476…`、critic seed23 `f0a2dd35…`、旧归一化、修复外边界后的Ring safety `18419e7a…`以及mean+16 stochastic的冻结有限集合。均值只是已有候选，不新增均值单独部署。

审计启动和结束时，前项K16 critic closure仍在采集，未产生已接受的新critic或Phase-B报告。closed-loop contract任务也没有完成工件。因此不假定那些阶段成功，不拿另一路Toy/DB的模型替代当前joint模型。本审计使用独立worktree，未修改并行阶段代码。完整版本/hash见 `hypothesis_test_manifest.json`、`hash_verification.json`。

## A. 检验的最强现有主张

- Ring：同一已查看的60状态，K16 oracle60/60，critic53/60，7个漏选、4个高分低Q；有效rollout无碰撞。
- Four：候选覆盖好、critic接近oracle，但世界坐标/槽位表示存在偏置；精确物理链已通过对称性检查。
- v2：安全语义更新，B15证据足够，数值异常单列；不能因此推断所有派生概率报告都正确。

已完成研究范围及不重复理由见 `ALREADY_AUDITED.md`。未重复外边界修复、完整对称性、basin存在性、K sweep、ranking-loss、LCB或一般DB完整性审计。

## B. 测试前风险排序

先登记H1最近eta代理被过度用于发展结论、H2数值Q聚合、H3候选/输入/随机数身份、H4停止证据与校准。执行H3时发现概率/logit的argmax差别后，仅追加H5饱和tie这一项自主分支。总计五项，详见 `RISK_REGISTER.md`。

## C. 测试及为什么足够判别

1. **静态callgraph**：逐行追踪 `run_k16.freeze → candidates/tasks → lookup → evidence/finalize`、`analyze_train_dev.nearest`、joint模型输入/eta变换和critic sampler。没有通过Runtime构造函数暗中写DB或生成测试状态。
2. **冻结候选重建**：从物理manifest恢复60个Ring conditioning，仅采一次用于h的Flow参考动作，不推进环境；再加固定hash选择的Four/Ring各4train+4validation，共76输入。提议eta和score全部逐值一致，top1 76/76一致；候选置换后logit排名76/76一致；每组17个不同eta，未发现proposal seed或noise cloud重复。
3. **精确缓存核对**：读取1020个Ring(state,eta)×16个canonical seed。全部16320记录存在，16,214有效、106未认证。controller的MACFlow、OrthoFlow3和当前安全hash逐项匹配；critic和oracle使用同一17候选。没有重跑K16。
4. **train/dev代理反证**：只复用并行采集已写入的精确tuple，在同一当前controller下比较nearest-v2标签和实际候选Q。读取快照中完整状态为Ring20、Four39；这不是随机性能队列，也不报告为泛化指标。不同eta本就允许不同Q，检验的是“代理不能冒充证书”，不是物理矛盾。
5. **解析/数值控制**：穷举15成功或2失败停止树；float64 sigmoid、raw-logit排序与float32饱和对照。不产生科学continuation，不做模型训练。

## D. 确认的问题

### H2：数值未认证的Q被写成精确标量

分类：`CONFIRMED_PROTOCOL_DEFECT`，局部聚合/报告问题。

`diagnostics/orthoflow3_ring_k16_diagnostic_v1/run_k16.py:267`中`evidence`虽然正确单列numerical且B15判定正确，仍无条件返回`Q16=successes/16`；344行继续把它作为精确regret相减。87/1020个候选含106个未认证seed，不能把这些未知结果当作已知任务失败。

修正后：

| 项目 | 修正结果 |
|---|---:|
| Oracle B15 | 60/60，不变 |
| Critic B15 | 53/60，不变 |
| Oracle有B15而critic漏选 | 7，不变 |
| 高分且Q上界<0.5 | 4，不变 |
| Oracle mean Q16 | 1.000000，当前每状态均有完整16/16候选 |
| Critic mean Q16 | **[0.919792, 0.920833]** |
| Mean oracle-critic regret | **[0.079167, 0.080208]** |
| 有效seed碰撞 | 0/16,214 |
| 数值未认证seed | 106，未填充 |

被选中的候选只有1状态受数值缺失影响，已有15成功，因此仍明确B15。其它未认证候选不改变本队列oracle资格。不能据此把16/16解释为总体成功概率1。

### H1：发展结论的证据范围越界

分类：`CONFIRMED_PROTOCOL_DEFECT`（发展证据范围），不是DB标签或训练数据损坏。

`diagnostics/orthoflow3_ring_revision_v1/analyze_train_dev.py:26`取最近**任意**标签；67–89行将其robust/Q拷到不同生成器eta上，作为饱和代理。141行后的critic top1/regret在已有标签池上计算，不是部署的17提议真Q。文件本身写了proxy，本审计不指控伪造；问题是它不足以证明部署proposal已精确饱和/排名完美，也不足以独立关闭R3/R4诊断。

同controller、当前v2语义的完整缓存快照里，Ring340候选有7个B15分类与nearest不同；Four663候选有29个不同。最近距离中位约0.15–0.17（按半域宽归一化），并非exact match。存在最近标签0/2而实际候选16/16的非阈值小翻转例子，见 `confirmed_witnesses.json`。原物理标签都有效；不能把它们搬到邻近eta。

主动反证结果：这些完整状态上旧critic实际选中候选仍全部B15，oracle也全部B15；因此**没有证据证明这项代理问题导致Ring全部7miss**。本次修复是将proxy隔离为探索指标、以精确tuple或UNKNOWN决定证据资格，而非宣称模型被修好。

## E. 被证伪/没有复现的风险

- H3的候选错配、oracle额外候选、eta squashing/归一化错位、重复随机噪声、critic输入与冻结manifest不一致：未复现。76/76逐值重建，60/60嵌套hash及DB匹配通过。
- “早停本身使Bernoulli计数NLL数学上无效”：被解析控制否定。保留停止规则后，计数似然在真p的期望score为0（误差<1e-10）。不能因为stopped fraction有偏就凭空重设训练目标。
- “Four高分饱和解释Ring当前7miss”：被Ring68个输入控制否定，无一选择改变。
- 新的模式输入、连续eta优化、额外oracle候选：未发现。

## F. 非实现bug的统计/数值限制

### H4：停止比例和校准不能混为一谈

v2中完整且无数值异常的Q16+条数分别为Double1177、Four192、Ring257。Four 21,328条、Ring21,263条seed数少于16。这些B15 positive/negative可以完全有效，但`successes/observed_count`不是完整Q16，也不是无偏概率估计。解析例子真p=.5时该停止比例期望约.3863。

原 `train_evaluate.py:386–395`还按观测B15类别50/50取样，不能把“count-weighted BCE”自动称作校准保证。独立解析控制中，正例按3.5倍取样时，真p=.9的计数似然最优点约.937585；这是采样改变统计目标的例子，**不是对当前网络误差大小的估计**。因此原critic的0.9375是既定诊断阈值，不是校准保证。实际网络失误仍应依据精确Q证据衡量。Phase-A已经改为proposal/state采样且只用完整Q校准，本次不再启动重训。

### H5：Four的float32 sigmoid饱和tie

分类：`NUMERICAL_LIMITATION`，已由在写的Phase-A logit排名覆盖，未修改canonical。

8/8预注册Four train/dev输入中，不同高logit经float32 sigmoid变成相同1.0，历史概率argmax与logit argmax选择不同。float64 sigmoid控制8/8支持logit顺序，候选置换不改变logit选中者。缓存显示8/8两种选择都B15，因此没有性能提升结论。Ring60冻结+8train/dev输入没有此差异。未来报告需区分selector tie规则变更与权重变化，不能把两者混为纯critic学习收益。

## G. 已做修复

仅修复独立分析工件，不覆盖旧结果/并行代码：

- `scientific_evidence.py`：canonical16认证/区间、exact-vs-proxy类型保护、有限logit排序控制。
- `H2_k16_corrected_*`：完整seed引用和修正Q/regret区间。
- H1证据注释与witness：保留旧proxy数值但撤销其exact-certificate用途。
- 13项新回归通过；旧代码的反例被测试直接复现，修复消除。

详见 `REPAIR_AND_PATCH_QUEUE.md`。本次无新训练、无标签重算、无canonical源码修改。历史数值/报告修复只需1轮，不进行性能追逐。

## H. 哪些历史结果仍有效

Ring oracle60、critic53、rescue23/25、break5/35、7miss和4exploitation保持原经验解释；有效rollout无碰撞保持。原始状态/eta/seed和当前安全DB记录保持有效。Four精确物理链通过、学习表示存在风险的旧结论保持。v2不需作废或重放。

必须限定的是：含数值异常的“精确Q16/regret”、以nearest代理当作生成器候选真实覆盖、以及未验证校准的critic概率解释。详细映射见 `ARTIFACT_VALIDITY_MAP.md`。

## I. 剩余未解决风险

- Ring finite-ranking的实测7miss和4高分低Q未被本轮小修复消除；这是当前统计学习瓶颈，不应伪装成已修好。
- Four主动旋转场景下的上游MACFlow行为等价、h的因果充分性、continuation horizon/动作恢复完整合同尚属另一个已授权任务；本审计不凭物理对称性通过就宣布它们正确。
- 此60状态已经用于K诊断，不能再升级为untouched泛化证据。
- 并行Phase-A和未完成Phase-B不在本轮验收范围。

## J. 唯一最高价值下一项

**完成已经运行的proposal-aligned Phase-A精确train/dev评估及预注册冻结确认**，并显式区分旧概率argmax与logit tie处理。它直接检验已经确认的Ring排名瓶颈；不再开K sweep、另一个loss实验或环境族。其后任务按既定顺序处理，而非由本审计自动启动。

## 资源、持久化、回归

- 新continuation=0；新训练标签=0；DB写入=0；无需journal/merger新增条目。
- 标准cache preflight及16,320个引用逐条存在检查完成。标准planner把数值未认证条目计为missing；本报告明确106条实际存在、只是未认证，不重复执行已耗尽重试。
- 13项本审计测试 + 3项既有basis回归：16/16通过。
- 23个直接冻结文件和8个runtime附加hash：全部一致。
- 本审计CPU单shard短时推理，GPU=0；推理作业已退出。先前任务的Slurm作业不是本审计新建的实验，未被取消；资源上限调整仅保留数据库已完成工作并为实验室留出约定空闲份额。
- 本审计结束，不训练、不改变K、不启动环境族。
