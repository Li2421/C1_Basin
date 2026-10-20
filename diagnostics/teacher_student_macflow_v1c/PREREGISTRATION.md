# Round30c: regate and student-tuned clockwise right-hand baseline

本规范在任何新dev指标前封存，完整继承docs/prompt_round30c_regate_and_tuned_rhr_20261015.md。只写本目录，外部写入仅官方DB追加流程；两个git仓库只读，既有代码、PoC、round30、round30b及环境不改。所有入口先断言Basin_C1 DB包、cache_writer、默认journals路径，再写文件。每个模块全部import完成后恢复JAX x64=False。资源用户所有任务合计20CPU/2GPU shards；每次提交squeue核占用，只用余量。无视频。

修正门仅判定8个原未通过者：教师indices[0,2,3,4,5,6,9,11]。全部使用round30b已冻结best.pkl及COMPLETE哈希，不重训、不换checkpoint。独立dev索引default_rng(2026103033).integers(0,len(dev),2048)，keys=split(PRNGKey(2026103034),2048)，与round30b完全一致；world target及trace第0步噪声逐行to_local_jax，global位置取teacher observation前2维。三个条件同时满足：best_step!=0；仅teacher动作norm>=.10m/s的逐机器人样本mean cosine>=.90；全部样本error norm<=.05m/s占比>=.80。筛选后没有样本则门不通过，报告undefined。已有效的indices[1,7,8,10,12,13]保持有效，不重新判定。对全部14个报告未筛选cosine及norm阈值.05/.10/.20的cosine、样本数/覆盖率、RMSE、误差占比；.05/.20只作敏感性。原离线未保存预测，只能用旧checkpoint/dev重算；本轮保存完整predictions/targets/keys/indices，无新训练/环境step。

新通过者先在round30固定16 pilot states上跑safety baseline/S500、seed0。两臂均<15%或均>98%排除，否则能力门合格；unknown保留上下界，门以预注册下界判定。不调门、不替换、不重跑unknown。R5原16probe×F10/F13×200steps，D-local，固定原预处理和A3/A0/G_E算法。探测与选择必须在任何新学生主评估标签前封存。已有6个学生的全部7臂标签、probe、冻结选择和教师补充臂直接复用，先核当前DB/旧收据。原学生nominal输出、D-local、episode及nominal-only/R28 RHR实现从round30b复制，除checkpoint路径定位不改执行；复制core/pipeline/episode_frozen/episode_extended/actions文件若字节相同则引用round30b的16条逐位门，否则先重放16条门。新tuned RHR仅写新函数/子类，不改已有模块函数或状态。

RHR网格：speed=[.10,.15,.25,.35]m/s × consecutive slow steps=[20,40,80] (1/2/4秒，dt=.05)。slow<.01m/s，goal_distance>goal_tolerance，顺时针单位切向、1.5m D-local安全层，不加OrthoFlow3。网格代码固定；(.15,40)必须与R28原RHR动作、触发/计数逐位一致（单元测试）。调参pool=已有6个+本轮修正门通过者（即使能力门排除，仍属有效学生，调参/新tuned臂照任务要求覆盖；标准7臂分析只纳能力门合格者）。新调参state候选seed=[2026101503000,2026101503999]，按升序选前16个，与DB state_uid/content hash/physical bit signatures及round28 eval、round30 pilot去重，不读取结果选择。generate使用既有Ring真实场景生成器，仅做新独立state，不生成rollout选state。

所有有效学生在全部12组×16调参state×seed0上评估。一组全局参数按每学生成功率的等权均值最高选出（每学生均16样本，等价总成功数最大）；unknown以下界0参与选择但不算已知失败，另报上界。平局按speed小、trigger时长短依次选。不按学生分别选参数。选定收据先于任何tuned主评估标签。既有R28参数若最终被选，精确复用已有B-RHR标签；否则tuned主评估对所有有效学生跑32 eval×seed0/1。所有阶段design→官方preflight封存→只跑missing→默认append_journal→单进程merger→postflight；新登记官方connect/transaction仅INSERT OR IGNORE，uid内容冲突停止，原登记表每行hash不变。调参/旧臂之间若execution semantics精确相同直接复用UID，不以改编码等理由重复rollout。

主分析CONFIRMATORY=round30b已有2名+本轮修正门及能力门合格者；敏感性SENSITIVITY=仅本轮新增能力门合格者；DEVELOPMENT=主分析+round30旧4名local。如主分析<8，全部终点INCONCLUSIVE；达到8则按任务阈值报告三套结果，敏感性/合并的证据身份清楚标注，旧2个标签已打开、此轮修正门后补入学生的顺序如实披露。菜单oracle仅F10/F13/S300/S500，不把调参RHR加入菜单。固定state32×seed2。bootstrap seed2026103032、5000次，先controller再state成簇，两seed整体保留，所有臂/选择配对，重抽中重新选best fixed/schedule/oracle。unknown保留下界0/上界1及配对差识别区间，不重跑、不称已知失败。

TS1 S500-safety CI下界>0；TS2 bestSchedule-bestFixed CI下界>0且>=70%控制器bestSchedule>=bestFixed-.01；TS3a oracle-A3平均遗憾<=.03且A3-safety CI下界>0；TS3b至少3学生F10-F13>=.10且至少3反向，否则IC，满足后A3regret-A0regret CI上界<0；TS6 S500-R28RHR CI下界>0；TS6b S500-tunedRHR CI下界>0。另报alwaysS500-A3配对差及TS4/TS5继承/原教师各管线配对差、单核预热后代价描述。调参RHR用了学生标签，η只在联合教师上确定，比较偏向RHR，如实说明。

每学生nominal outcome、nominal→safety逐条转移；自身停滞=nominal timeout且无碰撞，安全层导致停滞=nominal collision→safety timeout；F13/S500/RHR/tuned各救回数。未知单列；timeout不称deadlock。结论映射严格按任务第五节，有碰撞先披露；不把CI覆盖0当等效性证明。完整树/源码开始结束审计；并行任务新增git条目单列非本轮。全部报告、表、config、代码diff、checkpoint引用/哈希、preflight、网格/选择/门收据均在本目录。
