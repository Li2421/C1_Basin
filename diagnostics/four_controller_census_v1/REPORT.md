# Round14B Four controller census

Four 的跨 controller η 泛化问题在本轮兼容队列中**未建立**：阶段0两项 headroom 门均不通过，按预注册停止。仅1个 best 与1个参数不同的 final 可用，无法推断全部 Four controller 都不存在泛化问题；阶段压缩与 E 可学习性均 **INCONCLUSIVE / 未执行**。

实际磁盘找到22套 best/final，共44文件，非提示所述23套。严格比较 E8 v13 broad 的网络参数形状、输入框架、环境 fingerprint 及四项归一化数组后，仅该 run 的 best/final 一致，其余21个best不可用（详情 COMPATIBILITY_METADATA_AUDIT.json）；未替换或重拟合归一化器。两者参数不同，final单独为次要组。前向同key重复计算字节一致，导入后显式恢复FP32；历史8例终止原因、步数、第一动作字节完全复现。

32个全新真实t0 state，F10/F13/F15，各seed0/1；384条完整rollout，unknown 2。旧DB及标签按相同语义保留，标准preflight384missing，单进程merger后全部请求有记录；标准planner把unknown列为missing，本轮保留且不重跑；门8例另计。State去重审计记录不可恢复的历史证据，不声称这些证据绝对无重叠。

| group | F10 | F13 | F15 |
|---|---|---|---|
| best | 64/64 (100.0000%) | 1/64 (1.5625%) | 64/64 (100.0000%) |
| final | 61/64 (95.3125%) | 1/64 (1.5625%) | 64/64 (100.0000%) |

阶段0(a)需要至少3个非多数regime且差>=10pp的controller，(b)需要至少3个最优固定臂<=85%的controller。主组只有1个，合计两组也只有2个，不可能达到任何人数门。阶段0判定 **REJECTED**（进入阶段1的假设）；广义Four跨controller问题 **INCONCLUSIVE**（兼容性导致证据不足）。未放宽门槛，未新生成探测或时间表标签。

交付3个完整仿真视频，其中1个正式safety_success与2个诊断成功案例。Four没有原生deadlock判据，deadlock_show / deadlock_resolution 缺失；统一入口已执行，退出码2，视频交付**未完成**，无COMPLETE标记。timeout不称为deadlock。

所有新产物限本目录，共享DB仅标准生命周期写入。历史文件未改动；无新训练。资源与Round14A合计不超过24CPU/3GPU。
