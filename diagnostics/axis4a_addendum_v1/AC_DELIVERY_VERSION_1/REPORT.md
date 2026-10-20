完整性审计通过；round30路径归属差异0项，其他/未归属差异0项（逐项附后）。GA1=GENERATOR_VALID，中位数max成功率识别界=[93.75, 93.75]%，95%CI=[93.75, 100.0]%；按GA1规则，round35几何能力门结论保留。B未执行（UNDECIDED_NOT_EXECUTED），HZ3缺口/HZ4按对应完整或可用菜单限定报告。C：门的执行正确，N=2 Flow 在相向任务上能力不足。

仅本轮目录新写，R35/R36及源码、checkpoint、环境、原事件/成功标准全部只读。无新训练、无原实验环境的包安装/升级、无git写；本轮plot_env仅用于绘图；没有读取round30中间结果。先审计历史、登记/预检，后提交gates，再提交A。r.9配置14/14逐字段与对应原manifest及默认global/local配置一致，R10与R35geometry两组各16/16首动作、位置全轨迹、步数和终止逐位一致，见CONFIG_UNIT_TEST/CONSISTENCY_GATE。无radius注入行为差异。

A1用R35同一sample_instance development与config构造，种子202610354000起16个，冲突按预注册逐一顺延且记录；UID及可恢复物理三元组去重，无法恢复的历史覆盖见A1_STATE_DEDUP。A2用原32个state_uid排序前16个，逐半径reset检查，无效不替换。r.9同state F10/S500seed0的448条直接引用R10，未重跑或再发journal。A1/A2实际2240新请求合并在A队列，只减少调度空隙、不混合cohort。GA2各r配对差和每controller逐r曲线见A_CAPABILITY_TABLE/CONTROLLER_RADIUS_CURVES；主差是中位数之差，另报controller内配对差的中位数。5000次controller→state成簇抽样，所有r和臂配对；max只作指定描述，unknown给界且不算已知失败。

GA3只有GA1生成器无效且A2下界达20%的半径才触发；每r完整32×2×5，旧A2first16/seed0/F10/S500逐UID复用，其余只补genuinely_missing，合计不以两个阶段重算独立样本。A3/A0/G_E沿用R35冻结选择，在任何A2标签之前已经封存；physical23看不到r/H，不能宣传几何感知。原R35统计实现、菜单、GM1门槛和state交叉拟合不变，GA3全部DEVELOPMENT。遗憾、灾难/伤害和nominal→safety逐条转移在GEOMETRY_MAIN_RESULTS。

B的用户决定必须独立封存，空白不等于放弃或默认候选2；当前状态见PSTAR_UNDECIDED或PSTAR_DECISION.receipt。若获明确授权，候选点值用已冻结精确数值，不用表格舍入数；新Sprog从相同state/key/seed/F13的最长兼容前缀恢复完整原生增广状态，仅执行改臂后的后缀，首事件标志/时钟保留。unknown在分歧前原样保留，不能重跑；切换前终止直接派生。进度检查在动作之前，末端没有虚构动作，fallback=round6H/7；所选pstar在线/离线测试需先封存。HZ3使用三H共享controller/state抽样，HZ4完整修正菜单包含Sprog，旧输入仍保留来源等级；按prompt标CONFIRMATORY（定义修订），不冒充每条都是新独立物理轨迹。未决定/放弃的结果边界分别保留。

C的全部48条N2Flow旧pilot、成功state索引/UID、终止类型与raw/trace哈希在N2_GATE_AUDIT及CSV。原三臂分别1/16、2/16、3/16成功，其余均timeout，没有碰撞或unknown。checkpoint冻结验收依据是非相向导航/分时释放，不能当作同时相向能力保证；后续相向证据另列。改动只在专家checker的float32存储编码比较，Flow历史primitive始终16/16，执行文件冻结hash未变。原生64位专家首动作没有被历史保存，不编造其字节一致性；完整64位位置在修复前已一致。历史门只覆盖实际已有的单投影primitive全轨迹，不扩写成不存在的全TT历史证据。

DB按官方connect/transaction只INSERT OR IGNORE新身份，既有登记行哈希前后一致；同语义旧profile/物理身份保留，不因换目录/编码作废旧标签。默认append_journal、单进程官方merger前ps/squeue确认无其他merger；重复阶段引用不重复写journal。全部unknown不重跑，timeout/排队不称deadlock。EXACT_PREFIX_REUSE_AUDIT.json的prompt文件名不存在，实际R35前缀证据为HORIZON_PREFIX_DEPENDENCY_AUDIT、prefix/test_prefix和原完成review，本轮明确记录而未伪造旧文件。资源按所有任务合计18CPU动态调度，尾部可用工作不足时少于18，不新增实验凑核数。

GA2摘要：

{
  "0.9": {
    "median_max_success_pct": [
      93.75,
      93.75
    ],
    "paired_difference_of_medians_pp": [
      0.0,
      0.0
    ],
    "paired_difference_95CI_pp": [
      0.0,
      0.0
    ]
  },
  "0.8": {
    "median_max_success_pct": [
      0.0,
      0.0
    ],
    "paired_difference_of_medians_pp": [
      -93.75,
      -93.75
    ],
    "paired_difference_95CI_pp": [
      -100.0,
      -87.42187500000114
    ]
  },
  "0.85": {
    "median_max_success_pct": [
      0.0,
      3.125
    ],
    "paired_difference_of_medians_pp": [
      -93.75,
      -90.625
    ],
    "paired_difference_95CI_pp": [
      -100.0,
      -84.375
    ]
  },
  "0.95": {
    "median_max_success_pct": [
      0.0,
      0.0
    ],
    "paired_difference_of_medians_pp": [
      -93.75,
      -93.75
    ],
    "paired_difference_95CI_pp": [
      -100.0,
      -84.375
    ]
  },
  "1.0": {
    "median_max_success_pct": [
      0.0,
      3.125
    ],
    "paired_difference_of_medians_pp": [
      -93.75,
      -90.625
    ],
    "paired_difference_95CI_pp": [
      -100.0,
      -84.375
    ]
  }
}

GA3判定：

{}

完整性逐项差异（归属只按路径，不推断未证实来源）：

[]


# 几何结论的controller范围

GA1阳性对照与GA2半径中位数支持群体层面的几何脆弱，不代表所有controller在所有新半径都不能工作。A2的个体例外如下；这些是同16state/seed0的描述性结果，不改GA1/GA3门。

| controller | r | F10成功率界% | S500成功率界% | max界% |
|---|---|---|---|---|
| base_u_v8_local_macflow | 0.85 | [50.0, 50.0] | [43.75, 43.75] | [50.0, 50.0] |
| base_u_v8_local_macflow | 0.95 | [62.5, 62.5] | [87.5, 87.5] | [87.5, 87.5] |

因此，不能把“中位数接近0”扩写成“每个controller均失败”。完整逐controller曲线与unknown识别界见CONTROLLER_RADIUS_CURVES。原R35专家在各新半径的原32state均32/32成功，仍为可复用的特权联合证据；controller脆弱也不能写成几何不可解。


# 绘图环境补充

首次最终作业22136在导入matplotlib时退出，没有执行任何新仿真，也没有改变统计结果。原错误记录与日志保留。原指定 .venv-c1 未安装/升级任何包；按共同协议缺依赖时在本轮另建环境的规则，在本轮plot_env中安装了matplotlib及绘图依赖，pip缓存/tmp均在本轮目录。

独立环境仅运行冻结plot_curves.py，读取CONTROLLER_RADIUS_CURVES.json生成14个controller的逐半径曲线PNG/PDF，不读取或更改控制器/物理。全部仿真、统计、DB、审计和交付复核仍用原指定Python -B。精确环境版本见PLOT_ENV_PACKAGES.json。本补充不改变GA1/GA2/GA3、B的未决定状态或C的结论。

N2历史选择/验收/相向测试的分cohort表另见N2_HISTORICAL_EVIDENCE_TABLE.md，避免混同dev选择与fresh acceptance。
