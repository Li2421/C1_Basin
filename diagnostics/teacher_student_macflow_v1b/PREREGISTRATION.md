# Round30b frame-consistent students — preregistration

任务规范：docs/prompt_round30b_frame_consistent_students_20261015.md，完整继承其全部要求。所有新增代码和产物仅在本目录；已有源码只读，DB只走官方追加流程。不进行git写操作。资源更新为用户所有任务合计20 CPU、2 GPU shards。

教师、数据与seed顺序严格取round30 TEACHERS.json。仅global indices [0,1,2,3,4,5,6,7,9,11]重训。4个local学生、标签与冻结选择直接复用。旧10个global学生已被取代，原文件保留。TRAINING_CONFIG.json逐字复制round30；唯一语义覆盖：global教师的native动作和trace第0步噪声，都逐机器人用径向/切向R_i^T旋转到local；部署local高斯fold_in(key_for(state,seed,step),i)，输出local_actions_to_world后按原教师方式限速。JAX训练float32，与numpy float64坐标转换的数值差异用单元测试报告；机器人位置取global observation行前2维。所有导入后恢复x64=False。

离线等变诊断先于训练：每教师在train数据用numpy default_rng(2026101501+index)固定均匀选256 state；每state随机角度uniform[-pi,pi]；每state16个独立教师噪声样本，原与旋转state使用相同随机key，比较16样本均值，经相同角度旋转原均值，与旋转state均值的RMS差/原均值RMS；另报均范数比。描述性，不作门。

训练80,000步、batch256、4×512 MLP、Adam3e-4、每500步验证固定8×256样本、最小dev MSE最早选中、seed规则等完全沿用TRAINING_CONFIG。每教师仅一次，不调参、不重训。训练前核教师hash，核旧重建数据hash。能复用的state/观测/归一化不重建。act_scale逐机器人x/y完全相等及全8维是否相等均报告；非各向同性时噪声旋转为近似。

独立蒸馏门固定round30 RNG：2048 state indices default_rng(2026103033)，noise keys split(PRNGKey(2026103034),2048)。目标与噪声转local后比较；非step0且mean cosine>=.90且error norm<=.05m/s占比>=.80，三条全部满足才有效。无效不部署、不重训。全部10个训练、选择、蒸馏门完成并封存后才有闭环。

单元测试：坐标变换每行仅依赖自己的位置；改变其他机器人目标时自身观测与学生动作逐位不变；固定观测噪声确定性；local教师保持round30恒等label/noise与部署一致。round28 pipeline、episode与core的复制版复现16条D-local：前三global+首local，各F10/S500×前2states×seed0，首动作/终止/步数/全部位置逐位一致。复制decentralized_action不改。

能力门round30固定16 pilot states，seed0，safety baseline/S500。两臂均<15%或均>98%排除，少于8个能力门通过学生时整轮终点INCONCLUSIVE，仍报告数值描述。主评估对象为蒸馏有效且能力门合格学生；另外明确报告蒸馏有效完整集合和能力门排除者，避免混称。R5固定16历史probe×η10/η13×200步，先冻结A3/A0/G_E/alwaysS500/alwaysF10，再产生主评估标签。32个round28 eval states×seed0/1×7臂nominal only/safety baseline/F10/F13/S300/S500/B-RHR。教师补充臂全部复用，local学生全部复用，禁止重复rollout。

DB：每入口core先从Basin_C1导入shared_rollout_db与cache_writer并断言包路径/默认journals路径，然后才允许写文件；worker写前再断言。design官方connect/transaction只INSERT OR IGNORE新身份，冲突立即停止；每表原有行hash不变，记录新增行hash。design→manifest/preflight封存→缺失作业→默认append_journal→单进程官方merger→postflight。unknown保留、不重跑、不当已知失败；timeout不称deadlock。

统计沿用round30：controller后state成簇配对bootstrap5000（seed2026103032），每个重抽样内重新选最佳固定/时间表/oracle；unknown下界0上界1。TS1 S500-safety CI下界>0；TS2 bestSchedule-bestFixed CI下界>0且70%学生bestSchedule>=bestFixed-.01；TS3a oracle-A3均遗憾<=.03且A3-safety CI下界>0；TS3b至少3个学生各偏好F10/F13>=.10，否则INCONCLUSIVE；通过前提后A3regret-A0regret CI上界<0；TS6 S500-BRHR CI下界>0。TS4/TS5描述性教师继承和代价。CONFIRMATORY仅本轮新有效学生，DEVELOPMENT追加4个旧local学生（其标签已打开）。能力门限制单列；不把development提升为confirmatory。

每学生报告nominal outcome构成、nominal→safety逐条转移；自身停滞=nominal timeout且无碰撞，安全层导致停滞=nominal collision→safety timeout，unknown单列；F13/S500/BRHR各救回数。结论映射严格按任务第五节，多个蒸馏失败结合离线等变误差解释，不能把相关性当因果。视频不要求。开始/结束完整树审计与import源hash交付。
