完整性审计未完全通过（精确差异清单附后）。AC1 GAP：source N=None，无符合门槛的迁移目标。AC2 GAP：同一P2跨N迁移的平均遗憾=N/App，目标N=[]。五层证据首次观测到的失败或缺口：N=2、第4层，Prespecified three-arm controller capability gate not passed; not universal eta impossibility；互惠基线另在N2起存在第4层资产/复现接口缺口，不能当作性能失败。第5层正式deadlock解决在所有N均未建立。

本轮在round35完成及收据hash核验后才开始，严格顺序执行。N=2/10/20/50使用同一Gap1固定地图、0.62m门、0.16m机器人、0.5m/s、dt=.05与原H_N=2000/8000/16000/40000。4档各16pilot+32评估新状态，共192；按UID和所有可恢复物理三元组去重，无碰撞、无重抽样。4607条旧DB物理记录无法恢复，在STATE_DEDUP_AUDIT中披露。相同seed的任务前缀逐位嵌套；所有新增机器人都跨同一隔墙，共享对向资源竞争，不是独立二机器人案例平铺。几何合法、单机器人路径、联合规划证书、控制器成功、正式死锁解决分列。

历史审计更新了prompt里较早的资产状态：N10、N20、N50已有后来通过导航能力审计的冻结checkpoint，N50为94/96非相向成功、N20为95/96；这些是带原安全层的导航证据，不等于原始nominal或相向成功。四个N用各自checkpoint，不能声称同一Flow跨N。没有重新训练任何模型。旧v1-v4实际逐seed数量、部分B15/fullQ16、数值unknown与每个result哈希均在ASSET_AUDIT。旧池与本轮η10/13/15的精确float64 UID交集为空；近似坐标不算精确复用，旧label也没有被删除或判无效。旧logical停止未观测的未来seed不是失败；旧四次同条件数值尝试不在本轮重跑。

历史执行复现门：Flow→安全层→native的16条完整轨迹覆盖四档N，各4条，首动作u_safe、终止、步数、全位置逐位一致。专家16条完整float64位置轨迹、终止与步数也一致。首次checker错误地把旧float32动作提升到64后与未量化新动作比较，造成16条专家首动作标错；DatasetWriter原源码第120行明确把actions存为float32。已保留原checker/失败报告，按同一原始存储编码比较后首动作字节16/16一致，物理运行仍为原float64，JAX仍False，没有容差放宽或重跑。原生64位首动作本来未被旧dataset保存，这个证据范围在ACTION_RECORDING_SCHEMA_REPAIR中明确标记。旧GapTT pair汇总缺完整轨迹，不冒充历史TT逐位复现；本轮新TT组合函数沿用已复现的policy/projection/native原语，单独单元测试两次同snapshot投影及η进入basis的顺序。

修正为Gap已有TT位置：Flow或P2 nominal→速度界与原goal-stop→原CertifiedHardSafetyFilter→OrthoFlow3→同一投影→native。不是Ring的FF。通用P2为新函数，固定R27P2增益(1,.015,.01)，原距离floor/作用半径，读取真实墙段与几何裕度、目标吸引/机器人及墙排斥，N2/10/20/50/100纯函数单测有界、有限、排列与平移一致，无仿真步。N100单测不代表N100实验。旧互惠基线是带共同安全缩放与额外诊断早停的组合，不是ORCA；旧trace没有首动作数组，不能独立完成字节门。未把它偷偷替换成去掉安全层的路径跟随器，也未改native终止，故该项保留缺口。

各(N,controller)的16pilot只测safety/F10/S13t2；三个臂最高成功率known下界≥20%才进主评估，上界仍<20%记能力门失败，未知跨阈值则INCONCLUSIVE，不换状态、控制器或阈值。门失败限于指定三臂，不能证明连续η族或其他动作都无用，也不能把这些timeout叫deadlock。主评估32新state×2seed×9臂只对通过者；失败行仅显示明确标注的pilot数据，不能混成完整主评估。专家每N32个评估状态各一次，不依据控制器成败筛掉。专家无顺序路线或失败只是未建立联合证书，不是不可解定理。

源N按预注册规则为最小通过者，source主评估先于target。对七个修正臂，用source状态奇偶对侧训练/评估；逐controller选择和所有source controller均值选择都在任何target main标签前冻结。Flow类在其他N换了checkpoint，transfer只作为这组模型间的描述；干净AC2只检验同实现、同固定增益P2。源/目标缺少同一合格controller时不编造动作。旧Ring A3/A0的physical23硬编码4个角度与23维归一化，N2得13维、N10/20/50角度相减维度不匹配；G_E的旧Ring表示也无不变接口，均N/A，未padding、重新训练或用target标签调参。

分析实现/阈值在pilot前冻结；ANALYSIS_MAPPING是前瞻定义澄清：source只选7个修正臂，但目标完整菜单oracle严格用全部9臂，包括nominal和safety。固定oracle为F10/F13/F15，时间表oracle四臂。5000次controller→state成簇配对bootstrap；两seed/各臂配对，保留原state奇偶、bootstrap内重选source与oracle；source/target按同一嵌套物理索引抽样。跨N P2宏平均使用同一state抽样保留相关性；单一P2没有controller总体方差证据。unknown给保守下/上包络，不作为观测失败；同一未知事件的包络可能非最紧。oracle事件率来自固定首选臂。负的crossfit遗憾只是估计下界，不是打败总体oracle。AC1按每个合格目标N的source均值迁移−safety下CI>0；AC2按合格更大N上的P2平均遗憾界≤5pp。各N细节、样本内oracle上界、灾难/伤害与nominal→safety转移均在results/METHOD_METRICS。

P2不使用随机key，seed1只在同一seed0 trace存在并核验hash后做精确确定性派生，没有再模拟相同物理。source数值unknown也原样派生，不重试。两seed仍按同一state簇统计，绝不冒充64个独立物理初态。PERFORMANCE_ACCOUNTING分新物理步与派生步，PER_STEP_PERFORMANCE排除派生副本，MEMORY_PEAK_SUMMARY只描述含多N缓存的worker RSS，不能当每N模型内存。native环境的stalled_groups未用作终止，正式deadlock解决/正常排队时长不是本轮已证明的指标。

官方DB：先用connect/transaction INSERT OR IGNORE新身份、既有登记行逐表前后hash一致；首个准备事务因结构化conditioning需JSON编码而完全回滚，修正发生在任何身份成功登记/提交/仿真之前。所有逻辑seed_key严格为{future_index:i}，匹配官方merger。每阶段preflight封存早于Slurm提交，仅跑direct UID genuinely_missing，已存在unknown不重跑；默认官方journal路径断言、单进程官方merger、postflight原raw hash/标签/UID核验，无直接rollout SQL，无UPDATE/DELETE旧行。旧六分片governor只管其注册旧作业，本轮没有修改它；用户最新总计18核优先。

资源为所有用户任务实际合计18CPU；其他任务进入时只暂停本轮进程组并保留现场，在资源释放后续算。Slurm分配数可能短时20，本轮暂停2个后实际18，暂停前的监控响应窗口不隐瞒；RESOURCE_STATUS保留事件。本轮没有GPU请求、包安装、git写命令、旧源码/环境/控制器/物理/事件改动或视频。编排重载只修正未来blinding元数据并从已完成阶段续接，未中断或重跑rollout。完整性起止清单与数值机械核验分开。

能力门max成功率识别界：

{
  "2": {
    "Flow": [
      0.1875,
      0.1875
    ],
    "P2": [
      0.0625,
      0.0625
    ]
  },
  "10": {
    "Flow": [
      0.0,
      0.0
    ],
    "P2": [
      0.0,
      0.0
    ]
  },
  "20": {
    "Flow": [
      0.0,
      0.0
    ],
    "P2": [
      0.0,
      0.3125
    ]
  },
  "50": {
    "Flow": [
      0.0,
      0.6875
    ],
    "P2": [
      0.0,
      0.1875
    ]
  }
}

旧资产逐seed统计（不把unrun当失败）：

{
  "gap1_eta_scaling_v1": {},
  "gap1_eta_scaling_v2": {
    "10": {
      "pairs": 763,
      "records": 1357,
      "timeout": 1297,
      "unknown": 14,
      "success": 46
    },
    "2": {
      "pairs": 768,
      "records": 9626,
      "success": 1495,
      "timeout": 8122,
      "unknown": 9
    },
    "50": {
      "pairs": 1
    }
  },
  "gap1_eta_scaling_v3_n20": {
    "10": {
      "pairs": 768,
      "records": 1615,
      "timeout": 1528,
      "unknown": 19,
      "success": 68
    },
    "2": {
      "pairs": 768,
      "records": 12288,
      "success": 1521,
      "timeout": 10757,
      "unknown": 10
    },
    "20": {
      "pairs": 768,
      "records": 1610,
      "timeout": 1536,
      "unknown": 65,
      "success": 9
    }
  },
  "gap1_eta_scaling_v4_global128": {
    "10": {
      "pairs": 1280,
      "records": 2856,
      "success": 292,
      "timeout": 2531,
      "unknown": 33
    },
    "20": {
      "pairs": 1280,
      "records": 2689,
      "timeout": 2560,
      "unknown": 124,
      "success": 5
    }
  }
}

完整性精确差异：

- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1d/analysis.py: {"before": null, "after": "d5107dd09c70172321b2b4a9793f153767edb80e41b0c87d58a48c1ae79948c6"}
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1d/analysis_checks.py: {"before": null, "after": "5f868fca265e0186183edd3137c4456e6704a8b7fd1972e68124e59db5289d4b"}
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1d/finish.py: {"before": null, "after": "d9fd08941b3a2b62212fff7d87ed66c96be546014aff7e72d163060afaba0ec9"}
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1d/resume.py: {"before": null, "after": "e874496605be301b6e34153a4ac0682d2bcc5fd53f580136e67ad41a891bb1e0"}
