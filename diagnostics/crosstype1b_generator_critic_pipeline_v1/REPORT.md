完整性审计通过。训练state上按state样本内事后最佳候选相对一律按进度切换时间表（Sprog）的类型等权空间为3.78pp，按controller样本内空间为1.79pp；训练完整菜单交叉拟合空间不可识别，不能把样本内数值当作交叉拟合证据。训练生成器覆盖的不差比例为100.00%，严格更好比例为3.78%。开发主提升CI未取得正下界，训练交叉拟合目标缺标签，开发目标未全部达到，按预算内当前源类型嵌套选择的版本冻结。新state终点1未通过：真正的生成器与看state和controller的critic（pipeline）相对一律按进度切换为+0.72pp [95% CI -1.14, +2.87]；终点2通过，交叉拟合平均遗憾0.39pp，可能伤害controller数0。终点3相对不看controller的critic：-0.79pp [95% CI -2.15, +0.55]，相对不看state的critic：-0.02pp [95% CI -1.41, +1.70]；终点4相对原生成器与原critic：+22.13pp [95% CI +14.61, +30.60]。

结论是：真正的生成器与critic流程已完整跑通，在未参与本轮拟合的controller类型和新state上，相对只加安全投影大幅增加成功率；但本轮没有证实它胜过一律按进度切换时间表。主终点按原先规定判为未通过，保守版本和全部类型参照模型也没有达到对该时间表的正CI下界。

新state上，按state交叉拟合事后最佳候选相对进度切换的空间为1.11pp，CI [-0.33, 2.55]；按controller交叉拟合空间为0.64pp。样本内按state空间为3.25pp，明显更乐观。当前Ring动物园中，进度切换已接近这一有限seed候选参照，额外按state选择的可确认收益有限；该结果不能排除更小收益或其他类型上的收益。

主方法相对不看controller的critic为-0.79pp，CI跨0；不看controller的点估计反而更高。相对匹配的不看state消融为-0.02pp，CI也跨0。本轮没有建立这两组输入各自的额外成功率贡献；去state消融仍保留由state生成的候选η，因此这一比较不等于移除生成器携带的state信息。

冻结生成器的十个动作在99.85%的controller/state组合上不差于进度切换，主方法86.63%的选择来自生成器，覆盖不足并非本轮的主要迹象。只用生成器均值并按进度切换已有97.33%成功，主方法仅高0.15pp（描述性差值）。相对原始仅选整局固定η的pipeline提升22.13pp，不能把这整个差值归因于controller输入，候选的时间表扩展也是方法变化的一部分。

后续跨类型研究可优先加入最佳修法确实不同的controller类型，继续检查controller表示和critic排序的跨类型泛化。这里没有读取或使用跨type2目录；可直接复用的代码、候选规则及冻结模型均列在HANDOFF。

本轮修正参数由冻结生成器按每个state产生五个提议，分别构成整局固定和按进度切换两个动作；新的critic预测十三候选在当前state、controller下的成功概率，按三seed集成均值取最大。仅安全投影、按第500步切换时间表和按进度切换时间表作为三个锚点。生成器不看被测controller，controller、物理、安全投影、事件与成功标准均沿用旧执行链。

新state类型等权成功率为97.48% [95% CI 95.93, 99.50]。相对只加安全投影的配对提升为+81.65pp [95% CI +59.63, +96.68]。碰撞0.00%，超时1.58%，unknown 0.94%。timeout仅作为事件，不直接称为死锁。

动物园包含九类型、62个合格controller。环形交换场景流匹配策略（Flow）使用FF位置，其余八类型使用TT位置；Ring、H700、x64=False。冻结生成器为seed41第1900步，原critic为seed23第2700步。16条历史执行记录首动作、全位置轨迹、终止与步数逐位一致；16个Ring验证state的生成器提议与原critic分数逐位一致，最大误差0。

48个训练state由16个旧调参state与32个旧评估state组成。24个新评估state的物理生成种子在开工预注册，候选与输入提前冻结，闭环仿真只在模型冻结和选择收据之后提交。旧锚点标签、探测特征和controller checkpoint直接复用。全部请求先官方登记和preflight，只跑genuinely_missing；unknown保留且不重跑，通过默认journal和单进程官方merger入库。

资源最初为全局18CPU、2GPU shard；后按用户最新授权，在全队列没有其他任务时提高到24CPU，GPU仍2shards。每次提交重新检查其他任务；条件不满足时提交上限回到18CPU。没有训练新controller，没有安装包，没有git写操作。历史数据与源码边界见INTEGRITY_START/END及PREIMPORT_SOURCES。

critic采用100维state分支96、33维环境分支32、10维动作分支32、24维controller分支32，共享128/64输出logit。源类型内填补与标准化；state归一化只用48个训练state。二项似然排除unknown，实际观测seed数上限16。AdamW学习率1e-3、权重衰减1e-4、裁剪5，seed17/23/41。新critic使用float32最高矩阵乘法精度，x64仍False；这不改变仿真或controller的精度。

推理显式恢复h/c归一化参数为float32、controller归一化参数为float64；27组留出类型与模式的checkpoint恢复预测逐位一致，最大误差0。排查过程及原失败日志保留在ITERATION_LOG与PREPROCESSING_DTYPE_DIAGNOSIS。

开发共两个版本。初版为单个完整critic；第二版将完整分支与不看controller的分支按概率加权，保留controller的权重仅从0.1、0.25、0.5、0.75、1中按源内NLL选择。权重1包含初版；各外折使用源内最优权重，而没有在外层结果之间挑最好的版本。每个分支沿用三个对应seed，组合后仍按集成均值argmax执行。主方法同时看state、controller和候选动作。保守选择规则额外报告，不替换主终点。

为匹配第二版的结构，去state对照由看controller和动作的分支、只看动作的分支组合；后者新增92组三seed拟合，仍全部复用已有标签。不看controller时两个分支按相同源内规则选择同一个不看controller模型，因此合并为该模型。两版全部配置、结果与收据保留；初版见DEVELOPMENT_RESULTS.json，最终匹配版本见DEVELOPMENT_RESULTS_V2B.json。

九次外层留一类型中，初始化与checkpoint只按其余八类内部的留一类型集成负对数似然选择；缓存完全相同的七源类型训练避免重复计算。每100步保存到2000步，外层曲线不用于选参。同state候选排序仅对经验成功差至少0.25的可辨识候选对计算，预测平局记0.5。从头训练、旧critic初始化和去controller/去state消融均单独报告。去state分支保留候选动作特征；候选本身随state改变，因此不能把它严格解释为每个controller始终执行同一候选。

留出曲线显示后期拟合存在泛化风险：环形交换场景流匹配策略在源内选中的第300步，完整分支的外层NLL为0.589、同state排序准确率91.32%；到第2000步NLL为0.913；参数共享独立近端策略优化强化学习策略在源内选中的第1000步，完整分支的外层NLL为0.374、同state排序准确率85.10%；到第2000步NLL为0.577。这些外层数值仅用于报告，checkpoint仍按源内规则选择，没有改选各外层NLL最低点。

新state使用没有参与本轮该模型训练和选参的留出类型模型；全部九类型模型仅作已见类型、未见state参照。旧critic初始化此前在原生Flow训练过，这一预训练范围不因外层排除而消失；从头训练对照用于呈现此限制。

每个新state所有十三候选和nominal only均执行seed0–1。类型等权为主，另报controller等权。5000次配对bootstrap依次重抽类型、controller、state，两seed保留同簇，oracle在每次抽样中重选；unknown同时给上下界，同UID候选的不确定性在配对差抵消。oracle先按unknown下界规则选取，其上下界条件于这一选取规则，不是对所有潜在unknown取值重新最优化的界。交叉拟合按相反seed选择与评分，另报样本内和按controller的oracle。

按controller的事后最佳以跨state固定候选编号定义；编号对应生成器动作时，η仍由各state的冻结生成器产生。交叉拟合只用一个seed选臂，估计有噪声，其他方法可能高于其点估计，表内相应遗憾保留负值。样本内oracle另列，帮助判断有限seed选择的乐观程度。

新state生成器十动作覆盖的不差比例为99.85%，严格更好比例为4.99%；主方法选中生成器动作比例86.63%，锚点比例13.37%。有100.00%的controller在不同state选择不同动作。

同一controller随机取两个不同state，选中候选编号不同的比例为82.62%；按实际完整精度η及切换规则比较，动作身份不同的比例为97.90%。同一个生成器候选编号在不同state可对应不同η，逐state精确值已列入CONTROLLER_STATE_DETAILS.csv。

每类型及两种合并权重的结果见AXIS_TABLE.md/CSV；事件、平均/最大遗憾与伤害数见AXIS_METRICS.json。CONTROLLER_STATE_DETAILS.csv逐候选列出每个controller/state的选择、两seed标签、成功数、完整精度η和遗憾。figures/含36张静态SVG：每类型选中动作分布、完整分支留出曲线、源内权重选择和同state三条执行轨迹；轨迹示例按观察到的相对安全投影增益排序，仅用于描述，不改变控制与评估标准。图例、unknown说明与预览核查见FIGURES_README.md及DELIVERY_CHECKS.json。

DB规模（请求数/精确缺失数）：pilot 12480/12480；remaining 17280/17280；new_evaluation 41664/41664。

训练交叉拟合、历史坐标去重、隐式能量策略和旧默认参数灾难数的限制见GAPS.md。完整结果与收据分别为DEVELOPMENT_RESULTS_V2B、CONFIRMATORY_RESULTS、FROZEN_PIPELINE及NEW_EVAL_CHOICES。
