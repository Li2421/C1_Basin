# 新实例与新状态的修法选择前瞻确认（CONFIRMATORY）

本预注册在任何本轮训练、仿真和新探测之前封存。主规则只用round30h的46个旧controller、全部五族旧标签与旧诊断，以原A/B网格、原三组右手规则、各族等权下界平均成功率最大化；按A、B、甲/乙/丙循环顺序保留最早最优（浮点tie tolerance1e-14）。拟合结果如下，数值收据先于所有新探测。

{
  "primary": {
    "A": 0.01,
    "B": 20,
    "right_hand_arm": "RHR-A",
    "old_five_family_mean_success": 0.915327380952381
  },
  "secondary": {
    "A": 0.01,
    "B": 20,
    "right_hand_arm": "RHR-A",
    "old_five_family_mean_success": 0.9236607142857143,
    "C": 0.05
  },
  "secondary_A_B_fixed_to_primary": true,
  "source_label_sha256": "687155275d01c9b2b56ef8232ee45f36f34eebacdbbc8b76332989500ea23333",
  "source_diagnostic_sha256": "cbbd48a75f9851e5e4bcff72e26a671d3568eaf42c8207a706a965fcbb3fab99",
  "unknown_policy": "success lower bound",
  "new_data_read": false
}

探索性两级规则保持主规则A/B不动，仅穷举C与原A相同网格[0,.01,.025,.05,.1,.2,.35,.5,.75,1]，freeze>C选右手规则甲，否则右手规则乙；目标、平局与旧源族相同。不得再拟合。

58实例：14个round14A独立seed教师，28个round30f最终round3/step20000学生（闭环重标注14与共享随机种子14），4个全新配方高斯行为克隆，6新模型预测控制，6新A*。全部清单、checkpoint哈希与原父身份写入CONTROLLERS_PENDING_GAUSSIAN/FROZEN_EXISTING_CHECKPOINTS。两学生变体仍属于同一个分布式学生族（28实例）；各族并非独立模型谱系，结论只限规定样本。既有学生在旧状态上曾被评估，本轮前瞻性来自冻结规则、相对round30h的新实例与全新状态，不声称这些checkpoint从无任何历史闭环结果。

[
  {
    "name": "G0",
    "recipe": "base_u_v2_macflow",
    "seed": 2026101701,
    "steps": 12000,
    "batch_size": 256,
    "hidden_dims": [
      256,
      256,
      256
    ],
    "activation": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "validation_batches": 8,
    "validation_batch_size": 256,
    "validation_interval": 250,
    "validation_seed": 2026102701,
    "log_std_bounds": [
      -5,
      1.5
    ],
    "training_sampling": "uniform transitions on exact Flow train split",
    "checkpoint_selection": "minimum fixed dev Gaussian NLL, earliest tie",
    "frame": "global",
    "manifest": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v2_dataset/manifest.json",
    "manifest_sha256": "b49dcafe393b4f5f4103cad8e633c589d0dd8ee08200d52ce512a62941a57d1b",
    "flow_checkpoint": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v2_macflow/best.pkl",
    "flow_checkpoint_sha256": "43e474204cb4294842c7fd7cf3d613ff1359b789ea62bcf2de07a2df4031f1a7",
    "normalization": "exact obs_mean/obs_scale/act_mean/act_scale from corresponding Flow checkpoint config",
    "action_decode": "normalized Gaussian sample; decode; clip raw[-1,1]; per-agent speed bound; original to_world; final float64 speed bound",
    "test_split_never_opened": true
  },
  {
    "name": "G1",
    "recipe": "base_u_v4_macflow",
    "seed": 2026101702,
    "steps": 80000,
    "batch_size": 256,
    "hidden_dims": [
      256,
      256,
      256
    ],
    "activation": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "validation_batches": 8,
    "validation_batch_size": 256,
    "validation_interval": 250,
    "validation_seed": 2026102702,
    "log_std_bounds": [
      -5,
      1.5
    ],
    "training_sampling": "uniform transitions on exact Flow train split",
    "checkpoint_selection": "minimum fixed dev Gaussian NLL, earliest tie",
    "frame": "global",
    "manifest": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v4_dataset/manifest.json",
    "manifest_sha256": "115b20011046267a7be9dd34f5742755c53e7b4c06a0d00aad6b7450e7c17515",
    "flow_checkpoint": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v4_macflow/best.pkl",
    "flow_checkpoint_sha256": "b1a2609ccd42d12982c5a3a45f59c864582b946830c2097d47d5394a80055500",
    "normalization": "exact obs_mean/obs_scale/act_mean/act_scale from corresponding Flow checkpoint config",
    "action_decode": "normalized Gaussian sample; decode; clip raw[-1,1]; per-agent speed bound; original to_world; final float64 speed bound",
    "test_split_never_opened": true
  },
  {
    "name": "G2",
    "recipe": "base_u_plus_w_v4_macflow",
    "seed": 2026101703,
    "steps": 80000,
    "batch_size": 256,
    "hidden_dims": [
      256,
      256,
      256
    ],
    "activation": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "validation_batches": 8,
    "validation_batch_size": 256,
    "validation_interval": 250,
    "validation_seed": 2026102703,
    "log_std_bounds": [
      -5,
      1.5
    ],
    "training_sampling": "uniform transitions on exact Flow train split",
    "checkpoint_selection": "minimum fixed dev Gaussian NLL, earliest tie",
    "frame": "global",
    "manifest": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_plus_w_v4_dataset/manifest.json",
    "manifest_sha256": "bce651f6aeb32710e26001c641c3554e00a11b0d7866c1a977b6f241d07400cf",
    "flow_checkpoint": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_plus_w_v4_macflow/best.pkl",
    "flow_checkpoint_sha256": "e7b6de0db84ee4c69c04d6f1d7d0014a1fba40766d7a25a4cdce4587c14824ad",
    "normalization": "exact obs_mean/obs_scale/act_mean/act_scale from corresponding Flow checkpoint config",
    "action_decode": "normalized Gaussian sample; decode; clip raw[-1,1]; per-agent speed bound; original to_world; final float64 speed bound",
    "test_split_never_opened": true
  },
  {
    "name": "G3",
    "recipe": "base_u_v9_local_macflow",
    "seed": 2026101704,
    "steps": 100000,
    "batch_size": 256,
    "hidden_dims": [
      256,
      256,
      256
    ],
    "activation": "relu",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "validation_batches": 8,
    "validation_batch_size": 256,
    "validation_interval": 250,
    "validation_seed": 2026102704,
    "log_std_bounds": [
      -5,
      1.5
    ],
    "training_sampling": "uniform transitions on exact Flow train split",
    "checkpoint_selection": "minimum fixed dev Gaussian NLL, earliest tie",
    "frame": "local",
    "manifest": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json",
    "manifest_sha256": "dd72095a957acfb9fe47ff9b6478329bb06424c91bd9dd401d6e1b3c74431732",
    "flow_checkpoint": "/home/zhihan/research/Basin_C1/diagnostics/ring_exchange_stage1/base_u_v9_local_macflow/best.pkl",
    "flow_checkpoint_sha256": "3ecacb648a2ac54c0c34ce62dfffe8857f505c950d63ccf526edf4ba24bb1ed6",
    "normalization": "exact obs_mean/obs_scale/act_mean/act_scale from corresponding Flow checkpoint config",
    "action_decode": "normalized Gaussian sample; decode; clip raw[-1,1]; per-agent speed bound; original to_world; final float64 speed bound",
    "test_split_never_opened": true
  }
]

Gaussian逐字复用round33训练函数，MLP256×3 ReLU，8维联合mean与logstd[-5,1.5]，Adam3e-4，batch256；对应原Flow步数、同一train数据与normalization；固定dev8×256，每250步加step0，最低NLL且最早tie选checkpoint，禁止任务标签选权重。只训练上述4个，GPU至多2 shard，不重训既有模型。

新模型预测控制参数：[[30, 64, 100], [30, 256, 20], [20, 128, 50], [20, 128, 150], [10, 128, 50], [30, 128, 150]]

新A*参数：[[0.075, 20, false], [0.075, 20, true], [0.075, 30, false], [0.075, 30, true], [0.1, 10, false], [0.1, 50, false]]

32个新评估状态：原round28 generate/LocalFrameConfig生成器，候选区间[2026101720000,2026101720999]，按顺序接受前32个，排除DB所有uid/content与可解析物理签名（包括signed zero归一），保留不可解析历史记录列表；每状态seed0、1，RNG保持原key_for(uid,seed,physical_step)。STATE_DEDUP_AUDIT先于提交封存。菜单与round30h相同：只加安全层不修正、全程η13、前500步η13后η10、右手规则甲progress20/.35、乙neighbor_right/.8/.10、丙progress40/.25。原物理、事件、安全、修正和右手规则核心逐字复制，不改变任何成功或终止标准。

三道复现门：round33与round30e各16条，round28分布式安全层16条；保存位置/速度/动作dtype、shape、bytes、步数、终点完全一致。复现是用户明确要求的例外；不增补重复任务标签或journal。高斯训练冻结之后，58×16旧R5 probe状态、seed0、安全层300步、关闭终止。诊断函数逐字复用round30h，7维均值定义完全相同：unfinished速度<.01机器人步/1200；当前speed>=.01且40步目标距离减少<.05的机器人秒（主定义不加unfinished，另保留有该条件敏感性）；初始team距离减100/200/300距离；每机器人步执行与nominal之差L2均值；累计角位移unwrap±.15rad的mixed时刻比例。16探测均值作controller输入，不读任务summary/termination。任何数值缺口保留不重跑，不填主特征，受影响部分记缺口。

旧冻结通用critic只使用其原菜单F10/F13/S300/S500。复用14新教师的round14A既有R5（原联合流协议，明确与分布式协议区别）以及28学生round30f既有R5；新Gaussian/MPC/A*共16实例缺少对应R5，按原16×{F10,F13}×200协议仅补512条无标签probe，零任务DB行。原critic权重、source transform不变，不调参；用32新state的已知几何特征作原模型输入，绝不读任务标签。全部诊断与R5先整体封存，再封存所有主、次要、旧critic选择。若旧critic选择菜单外原动作F10或S300，只为该选择补对应64条任务，以评估该对照，不加入主六臂菜单或oracle。这是原菜单对照的必要标签缺口；不重复其他原菜单全臂评估。

全局时序：本预注册与两规则收据→GPU训练4Gaussian并冻结全部58实例→全部诊断与R5冻结→所有新controller主/次要/旧critic选择冻结→design与官方preflight封存→只缺失任务→默认官方journal→ps/squeue确认无其他merger→单进程官方merger→postflight→打开新标签分析。复现门可与GPU训练并行，但全部门完成后才开始新实例诊断。所有入口先加载官方shared_rollout_db再PoC路径、全部导入后断言包和默认journals路径、恢复x64=False。DB登记只INSERT OR IGNORE、逐旧行hash与rollout计数前后相同；未知不重跑，不作失败。timeout不是deadlock。

确认一：主规则减6臂中最佳单一固定臂，5000次配对分层controller→state bootstrap，每次重新取最佳固定臂，95%百分位CI下界>0才PASS。确认二：同样主规则减全体500步时间表。seed2026101710；每族内重抽controller保持28/14/4/6/6，所抽controller内重抽32状态，2个seed成簇；所有策略共享抽样。主要合并58controller等权，另报五族等权。unknown下界0上界1，点值按下界，另给保守差值包络，不以unknown判失败。每族/每controller事后最优都只含六臂，平局按菜单顺序，类型为不修正、时间表（含全程η13）、右手规则；类型准确率与允许并列最优类型的敏感性均报。灾难为某controller相对本族最佳固定臂族平均低严格>10pp，另报unknown可能范围。

确认三描述每族成功率与最佳固定臂差、类型准确率、灾难数。两级规则仅探索性：总体两级−主、MPC族两级−主与配对CI；“明显更好”预先定义其MPC差CI下界>0，不升级为本轮主终点。每controller遗憾相对其六臂事后最优，配对CI条件于冻结选择，不重新拟合。原通用critic仅描述性对照，协议差异单列，不能据其差异重选主规则。

结论映射严格照任务：确认一和二均通过才主方法前瞻确认；只二通过则仅优于默认；一不通过且某族错则分析冻结特征阈值关系；次要MPC通过建议下一轮将两级作为主规则确认，不将探索性改称确认。既有右手菜单曾用族结果调参，历史菜单族信息不是本轮全新发现。

完整性BEGIN覆盖PoC全树、所有列出的历史及依赖全树、全部非并行Basin已有Python源码与两个仓库只读git快照。并行目录按开始squeue/最近24h/untracked规则识别、封存；本轮不导入或写它们，变化独立归因，不通过改外部文件来消除审计差异。所有新代码/缓存/日志/图只写本目录；官方DB流程为唯一例外。最多18 CPU、2 GPU shard，提交前计入running和pending预留；视频不要求。
