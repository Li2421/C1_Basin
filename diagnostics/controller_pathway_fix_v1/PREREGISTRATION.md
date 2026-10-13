# Round6 Controller pathway fix — PREREGISTRATION

证据等级全部 DEVELOPMENT：所有旧终点已经打开；这不是前瞻确认。本协议在本轮首次模型重放/数值重分析之前冻结。用户最新版资源上限：CPU总计15核（GPU任务CPU亦计入），GPU3 shards，通过Slurm；零新增rollout，零共享DB写入，历史资产只读，无git commit。

## 目标、数据与复现门

按 docs/prompt_round6_controller_pathway_fix_20261013.md 执行。先核查旧协议/报告，冻结此文件，再对 round5 4LOPO×DENSE/SPARSE的R0/R3/R_oracle共24任务，以及 round4 Stage A 7LOCO的R0/R3共14任务，逐3seed从原checkpoint重新推理。输入/归一化/精度逐值沿用旧代码，比较保存logits（absolute tolerance 2e-5）、argmax要求全相同，success要求1e-10内相同。若不一致先诊断，不能直接跳过。CPU静态推理不得调用任何env.step；不重新查询Flow。新数据只从已有标签/静态特征取得。

round5 TRAIN原Flow128state、融合96state，验证仅源原Flow的P1 24state；终点32state seed2–3。round4源6Flow128state，验证源P1 24state，终点64state seed4–7。所有训练masked binary NLL仅valid标签，unknown不训练、不重标失败。必要controller/folds沿用旧发现门，round5各臂3个，round4 fold1/4，不重选门。源先验各controller有效state均值后等权平均，eta ties lowest index。

## 固定方法设置

F1-PCA-1/2/3：controller-level R3均值，每controller在共享源训练state分布上取均值，源controller等权；坐标逐维source zscore（sd<=1e-6设1），源均值矩阵SVD，无目标参与，component最大绝对loading定正号。输入是该controller固定均值的PCA坐标，不是逐state descriptor。目标均值只取已有无标签静态查询，绝不用目标标签；round5用共同前96state，round4用共同前128state。PCA explained variance保留。网络physical23+controller padded513，即536输入，64/64/16 SiLU。

F1-oracle-low：[lambda,regime_A_onehot4,regime_B_onehot4]（9维），类别顺序[10,13,2,4]。锚点经验regime只从该折源TRAIN标签算，按此四候选最高有效均值选；全K16经验best另报告。原Flow[0,ownregime,ownregime]。round5全部锚点都是源，因此目标融合成分坐标不泄漏目标标签。真实LOCO留出原Flow无可获得自身regime：该arm及其F2版本如入选，标INAPPLICABLE FOR ZERO-SHOT REAL FLOW；禁止读留出S_train标签来补其onehot。不能换成其他预测regime而仍声称同一arm。

F1 optimizer沿用round5：AdamW lr.0008 wd.0001，clip5，batch128标签head，每step源controller等概率，max12000，fitseeds17/23/41，checkpoint schedules50,100,200..12000每200；3fit平均源验证NLL共同选step。源训练physical/controller归一化，不用目标输入统计。SPARSE源相同，完全相同模型每arm只优化一次、跨4折SHA复用（不是删arm/预算）；DENSE每折独立优化。每task保存3seed checkpoint/prediction。

F2-PCA-2/F2-oracle-low：分别冻结同折同seed B0(R0)源训练与验证/部署logits，仅训练16/16/16 SiLU残差MLP，最后层kernel/bias零初始化；输出1.0*tanh(raw)，wd.01，其他优化/步数/batch/选步同F1。明确F2是小网络+有界残差+强weight decay组合，单独因果贡献不能拆开断言。

F3a-k1及k0.5敏感性：sigmoid逐fit logits，mean(Q)-k*population_std(Q)，取argmax；无重训。F3b：3fits top1一致，且mean(Q_candidate-Q_source_prior)>=.10则偏离源prior；否则prior。应用B1/B2及所有F1/F2。原argmax每fit各自评价并平均，F3是单个ensemble选择，重复到3fit维便于与B0同seed配对；不同部署规则明确报告。不得按终点结果调整k/threshold。F4：source PCA2→four-class logistic，固定L2=1，类缺失不补目标/不重采样，max2000 iterations，最大概率>=.6选相应eta，否则srcprior；无逐state residual。

## E1/E2/E3统计与分支

E1在所有12/7留出controller等权，平均3fit，same physical state cluster resample joint across controllers，5000 bootstrap，95%max-T simultaneous CI across all actually evaluated methods and DENSE/SPARSE settings in fusion；LOCO另成family。unknown：不同eta的差lo=Ya-Yb-Ub、hi=Ya+Ua-Yb；相同eta/seed unknown精确对消。CI下界用lo、上界用hi联合max-T；E1 pass lower>-0.02。

E2仅原必要controller：vs源prior和B0配对保守95%CI（描述性，不用于调参）。target-oracle-fixed为用旧discovery seeds选的目标固定eta，不是终点hindsight；gap closed = (method-src)/(target-fixed-src)，分母<=0时NA；CI由同state bootstrap对ratio重算、分母不正报告不稳定/NA，不任意clip0..1。正迁移候选要求vs prior和vsB0两个描述CI下界均>0，仍属DEVELOPMENT。

E3每留出controller平均fit success比srcprior低>=.10即灾难；保守unknown给确定/可能灾难数，合格要求possible E3=0，另报告point与certain E3。所有留出controller照报，不删不必要/困难目标。

先完成融合全部6个重训arms及F4、F3和基线，再冻结 Stage2 eligibility.json：任一融合coverage setting上E1 pass且possible E3=0的method可进入realLOCO；同architecture多policy/coverage资格只训一次，各合格policy全部报告。真实LOCO所有B1-F3无条件执行；原B2不存在，不伪造。oracle-low输入不可部署故明确INAPPLICABLE而非用目标标签。其他入选F1/F2/F4对全部7折源-only重训。未合格方法不做LOCO新增fit，记录gate理由。

## D1–D3

D1：在相应源-only normalized controller均值空间（B1-R3、B2-oracle513、PCA/oracle-low）计算nearest距离及simplex constrained least-squares到source凸包距离(SLSQP固定tol1e-10 max5000)，几何只用静态输入，报告solver失败而非0；与success relative B0 loss Pearson/Spearman相关均为描述，n<=12、共享state/多fold并非独立因果证据。
D2：selected mean-fit预测Q与同选择的valid endpoint observed success，unknown给差值上下界；原per-fit选择和F3 ensemble分别计算，按state配对 bootstrap。D3：selected head源请求/有效/unknown数、有效经验success及controller覆盖；同一source权重定义，eta1/8单列。不使用未来轨迹、测试Q作输入。

## 判定及边界

按用户四条映射：存在修复E1pass且possible E3=0而B1/B2确有灾难→SUPPORTED H_method as an operational explanation of harm; 不能单凭回退到prior证明高维过参数化的具体机制，降维/收缩/uncertainty/F4分别标证据。E2两比较正才正迁移候选；只消伤害无收益→方法缺陷解释伤害，正迁移仍未解决。全部修复不合格→REJECTED as sufficient repair/explanation under this design；不证明覆盖/表示必为因果根因。未知过多无法判断→INCONCLUSIVE。所有结论Ring固定geometry/K16/旧DEVELOPMENT，非普遍方法定理。未运行/不适用不能写性能0。

资源及成本：最多3GPU/15CPU，所有GPU占用CPU计总额。先重放，再做1个F1-PCA2 task实测（属于正式训练，缓存后不重训），根据其耗时估最大144融合fits+条件LOCO成本；若预计>8h先报告，不删arm。纯Q无视频。只在新目录写；历史资料hash与shared DB只读meta/hash前后证据保存。图表必须输出E1/E2/E3及D1/D2/D3 standalone PNG/PDF。最终REPORT/HANDOFF独立可读，含实现、例外、证据索引、命令、版本、0rollout与0DB写。
