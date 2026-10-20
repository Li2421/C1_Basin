# 跨type1b开发迭代日志

## 开工与历史核查

上一轮已冻结，62个合格实例、两组机制特征与原锚点标签直接只读复用，不重新训练controller或探测。新一轮按state运行真正的生成器+critic，首先验证16条旧执行记录及冻结生成器/critic的Ring推理。试点先做Flow/扩散行为克隆/强化学习。训练期单seed生成器标签与两seed交叉拟合要求冲突已显式提问；继续独立的复现和原定试点，不擅自增加19840条。当前没有新rollout、新state标签或critic训练。

## 候选准备的历史DB兼容性修正

第一版只读去重在遇到physical_state_json为NULL的历史身份时停止；没有生成候选、训练或rollout。保留原程序和失败日志，v2对所有UID/content继续检查，将无法恢复坐标的旧身份按quality计数为去重证据缺口，不假造零坐标或丢弃该限制。

## critic warm start数值核查（正式训练前）

原GPU默认矩阵乘法下，仿射换归一化后的warm start概率差2.635e-4，未通过自设2e-6核查，原测试及日志保留。相同参数在CPU默认精度差2.980e-7，GPU highest矩阵乘法也为2.980e-7，已定位为默认GPU乘法数值精度。新critic训练与推理将固定float32、matmul highest，仍保持x64=False；不改容差、不改生成器候选、不改controller/物理执行精度。完整原测试将在highest上下文重新执行。

## v1正式critic训练设计与统计核查（标签仍在收集中）

highest精度下全部critic核查通过。源类型集缓存为46个唯一集合，每个full/no_controller/no_state、scratch/warm、seeds17/23/41；2000步、每100步保存，不根据外层标签早停。内层仅源类型平均NLL选初始化/步数；外层NLL和同state排序只用于报告。源标签扰动GPU检查和外层验证指标扰动选择检查均严格不变。统计实现通过小型解析检查，包括相反seed交叉拟合、nominal不入oracle和同UID unknown抵消。这里没有增加候选或额外rollout预算。


2026-10-18 16:43用户已答复：保持原预算，训练完整菜单交叉拟合oracle记为缺口，新评估仍全14动作两seed。USER_TRAINING_ORACLE_DECISION.json及收据记录；不得执行training_seed1扩展。


## 三类型试点完成

26个controller的12480条生成候选训练请求已完成官方合并与postflight；PILOT_REPORT.json/receipt已封存。类型等权按state样本内空间3.2242pp、按controller样本内空间1.0913pp，生成器严格更好3.2242%，不差100%。这只是共同seed0的样本内空间，不是critic已实现提升。试点未改变后续设置。其余36个controller的17280条精确缺失请求已preflight，job25839 array0..15开始。


其余类型采集阶段全局仅本轮16CPU运行；提交前再次squeue确认后，增加2个共享游标worker（remaining_extra，offset16），总18CPU，不增加请求、候选或训练标签预算。


2026-10-18用户最新授权：没有其他任务占用时使用24CPU；GPU仍2shards。全队列squeue只有本轮18CPU，已增加6个共享游标worker（remaining_idle_capacity，offset18，总24），未增加请求预算。RESOURCE_LIMITS.json与USER_CPU_CAPACITY_DECISION收据记录；resource_guard.require每次提交重新检查全队列，无其他任务才允许24，否则回到18上限。后续新eval可用24CPU。


## v1开发结果与v2预先固定的controller分支收缩

v1主方法97.9277%，进度切换96.2081%，配对+1.7196pp，CI[-0.0772,3.7424]；各类型最差-0.6944pp，伤害0。不看controller分支98.2308%，提示需对controller依赖作正则化。v2在已有两分支概率间做严格正权重收缩，lambda=[0.1,0.25,0.5,0.75,1]只按源内留一类型NLL选，lambda=1包含v1；不因外层结果挑lambda或初始化。全部复用276拟合，不增加标签/训练。设计与收据先于v2计算，新评估仍完全未打开。


v2匹配消融补充：双分支完整critic的去state对照应为Q(C,a)+Q(a)，不能继续把旧单分支Q(C,a)当同结构对照。因此补齐92个动作单独分支的三seed拟合，超参数/优化器/源集合与原设置完全相同，零新标签。去controller的两分支都会源内选择同一个Q(h,c,a)，自然退化为原no-controller对照。最终匹配结果记V2B，初步V2的单分支no-state行仅保留为中间诊断。补充设计先于新分支训练。


冻结推理复核首轮逐位失败（最大概率差5.96e-7），原失败日志/代码副本保留。27个外折×模式的均值和保守选臂均无差异。定位为JSON归一化参数丢失dtype：训练h/c均值标准差是float32，JSON列表使算术提升float64后再cast，不看state的两分支（controller加动作、仅动作）逐位0差。修正仅在推理反序列化时恢复h/c参数float32、C参数float64，保持原逐位检查，不改容差或checkpoint，不改物理/历史门。


归一化dtype恢复后，27个外层留出类型×模式的恢复预测全部逐位一致，最大误差0，原自设检查未放宽。V2B开发目标仍未全达成：主提升+1.6755pp，CI[-0.1819,3.8086]；无类型损失>2pp，伤害0。源内选出的正controller权重各外折均0.75，源内NLL均不差于包含的初版lambda1。按prompt冻结当前源内最优版本，不用外层效果在版本间挑选。新eval尚无任何闭环记录。
