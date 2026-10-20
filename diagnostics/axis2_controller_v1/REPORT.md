完整性审计未完全通过（详见 INTEGRITY_DETAILS.json）。CT1 SUPPORTED：A3 平均交叉拟合遗憾 [-2.1484375, 0.9114583333333334]pp，A3−safety 95%CI [39.7119140625, 73.76464843749999]pp，伤害确定/可能 0/0。CT2 INCONCLUSIVE：固定偏好双向各 2/21 个，遗憾差 CI [-25.846354166666668, -7.552083333333333]pp。CT3 REJECTED：混合−A3 遗憾差 CI [-1.953125, 1.953125]pp。2a 新seed、2b兄弟checkpoint、2c新架构均补齐配对基线，2a 原冻结方法−safety CI=[71.42578125, 93.30357142857143]pp；2b 原冻结方法−safety CI=[53.125, 90.0390625]pp；2cN2 原冻结方法−safety CI=[81.1298076923077, 94.83173076923077]pp；2cTEST 原冻结方法−safety CI=[83.94097222222221, 94.88064236111107]pp；2d 四类各6实例全部通过能力门，是否支持迁移以各CT门槛为准；2e仅整理历史三管线和R30学生，其流程与蒸馏限制保留。

本轮新增 baseline6784、pilot768、main12288（全部24实例保留）；历史复现80，R5无标签probe768。训练6个Gaussian BC与混合类型critic，后者选step500；全部在新类型pilot之前冻结。probe后阶段曾因分析实现尚未封存而按保护检查退出；封存完成后只提交main一次，没有重跑探测或冻结选择，原错误记录和PIPELINE_RESUME.json保留。方法选择与分析实现冻结均早于main提交。本轮提交18 CPU分片，最多2 GPU训练分片，无本任务跨轮并行。其他r30b任务中途增加占用，监控响应前曾短时合计19；初版只暂停1分片的局限随后修正为按总占用动态暂停相应数量的本轮分片。后处理也按空位暂停/恢复整个进程组；所有在途轨迹和统计状态保留，见RESOURCE_REBALANCE.json与STAGE_RESOURCE_REBALANCE.json。不得把这些记录说成全程无资源偏差。

成功、碰撞、timeout、unknown按原仿真语义；unknown上下界保留且不重跑，timeout不称deadlock。所有CI5000次controller→state簇配对bootstrap，双seed保持配对；oracle每次重抽按原state索引奇偶交叉拟合，同时给出样本内上界。CT2/3未通过显著性门槛并不证明效应为零。oracle事件率按已确认成功下界训练后的确定性首选臂计算；unknown引起的可选臂不确定性另体现在成功率和遗憾包络。

CT4使用历史原冻结选择，按原state_uid排序前32、seed0/1与新基线配对。补充A3/A0/GE只读R26冻结预测，沿用原预测state全集聚合并限制到有标签菜单，标DEVELOPMENT；不得把额外事后整理升级为预注册终点。旧六臂的F4等无critic head时不做映射臆造。AXIS_TABLE增加original frozen method列展示CT4的原规则；额外A3/A0/GE列不代替CT4原规则。

DB使用官方登记→提交前preflight→journal→单进程merger→postflight；原含timestep的state行保留，规范三元组使用新的transport身份。原uid、物理、key_for随机流保持不变，80条全轨迹复现证明兼容；不能把这种表示规范化当作旧标签无效的理由。源DB无法恢复的历史物理状态去重范围见 NEW_STATES_DEDUP.json。R5只进入SUMMARY_ONLY来源，不生成success/terminal标签。

混合训练数据只有旧R15 Flow56与R27 M/P12，不含任何新类型pilot或eval标签。MIXED_LOTO_RESULTS.json为描述性留一类型遗憾，预处理/PCA和step1000有旧Flow先验，不是独立前瞻确认。混合模型相对A3同时改变训练类型覆盖与FF/TT输入标志，CT3只支持这一整套预注册配置的效果，不单独识别标志或数据多样性的因果贡献。

历史M/P继续保留R27状态区间修订、R29私有DB及single-projection baseline差异。部署行保留R28 preflight提交时序偏差，R30的448PoC journal越界、chart信息差与6个step0学生。R28特权专家32/32只对应其原32states，不能转作本轮新states的专家成功率。

主交付：AXIS_TABLE.md/csv、METHOD_METRICS.json/csv、results.json、HISTORICAL_RESULTS.json、MIXED_LOTO_RESULTS.json、GAPS.json。完整性差异逐项见 INTEGRITY_DETAILS.json。历史库存、复现、登记/预检/后检、原始轨迹、训练、能力门和冻结收据均在本目录；独立机械核验见 INDEPENDENT_DELIVERY_REVIEW.json。未更改控制器参数、物理、事件和成功标准；无视频要求、无git写操作。

完整性差异清单：

- Basin status added: ?? diagnostics/teacher_student_macflow_v1b/
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/actions.py: None → b5436cd80f6fd5bfd4866696031914a4abead564b6da845e948d2c4c4af5f422
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/analysis.py: None → 42b5b84cd92a3cf0e0b07879b4c6cf4810999d3eac0ec2847b341232a05386ab
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/audit_extra.py: None → e46a7ec4638519293ec4561d42b86d318c0fbd2a735286a91852b5a593e45398
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/audit_utils.py: None → f7e2b8e9e3230d1292a25bef16b1a7df074f2ceb8e831a35976577261d9a8174
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/automate.py: None → 5fe7290141c47e41cdc5616ef8199e1e84f1994edda9a214b311b0e59d914cc8
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/benchmark.py: None → ab27fe674f50584901b04c3947072a103203490429848cbc656af95e25760830
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/components.py: None → 07d4641c0cdf12c2d2bedba17dca9e8418bdc4795c5edb7ce7d7d5a17a984415
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/coordinates.py: None → e82cec36eaa4a934bea4cf8b7fa76cbcaaff99777175529632def227d07e5096
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/core.py: None → ec80cf360051c51931e6af5306f261ca49be3c1ad1dd1a70f5939d9ce686b86a
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/data.py: None → b0cf318da50fc2337c6d8e2b98c84af4bf7f0347dc84ebb5707146a29d048d29
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/deliver.py: None → 1aa084c9795a0e2fd67726b89d38f70dab289e79493fa8d4c071381106dba7ee
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/design.py: None → 9dece3ade3c695235d0363792efe3f86a4a102b4307c6b2f647919deb6e47f66
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/episode_extended.py: None → d7be65a2a7ee630d4d87c5e1e70c9cb3bc3c651758f14371135682aa4a02797a
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/episode_frozen.py: None → befc40a0c912b1a0ee2e172c0fc410c9208abcb3c0bbc6aeb5afb586d8377526
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/equivariance.py: None → d04957a45c65396799a26604b9a483ac6e2be0f02ddc2b30bc373ba464477b39
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/freeze_choices.py: None → e4763a408f9a43da6ce1839dba94ecdbebf593d048310f5951193ff03eec2cc1
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/integrity.py: None → a9edbb236ac8b8c64416af5384e9109a82a0467589d40956400572d07aa1d20a
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/offline.py: None → 5ed9125d747c20b35fdbc928dfa2f4c0900f720996734d4eb608aa59c6da82b3
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/phase_design.py: None → 116301013993e2f471eb7ba7feb9bc816b7c051103c9b5d033a6ca42415adf8a
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/pipeline.py: None → 41603e9841ad5f7d0390577cc5e0722e1d5bcce470423c9388bcce1ca75e3b13
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/plots.py: None → abe3de58beec41c8859c0ee1133cf174c5bcf3e7b32d00ede0af60c17cc0af04
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/postflight.py: None → 2ec523a4b5960883e9144953a2332a95e2e380d96244a6cade3407863785b8b6
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/prepare.py: None → 6a9eb5c8de40aa8356be2e3be1f2bf9d8a9318ac0228837e27a829d172a892bb
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/probes.py: None → af404faf5d8bd7463df44dd01a54dc3a64079ecf067ce91e6c526609bb35dffb
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/registration.py: None → 071a2720c554a1b6f2578389d2025660d3f6292c568c8e823472e3eeaf70ea2f
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/replay_gate.py: None → dfb630a9e636a520411322ffca58abb774c59066d2ac82a5a3333c263048e726
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/run.py: None → 587c801b414fbf1f1f995c9565c5920718af9d972776e2a44cdd98ab88723d42
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/stage.py: None → 7404e78da7692123b20d349ed3212e333ca7cb4e98147e6f3dfcdfd45a103afd
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/student.py: None → e80fe50d0959837a9cdf0e2ad901770ab48d66f1ef0e8ca52c5ccdcb2a07f978
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/train.py: None → fcabd516ecb071bc8bf5d0cdba121b1f38325f477f39e79f6a2305f9380634a1
- existing_source_sha256: /home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/validate.py: None → 77d9577e7a468dec0beab30c8c0abdc4107a4d9889a4d9a78530678fe4d78505

源码清单差异全部是 diagnostics/teacher_student_macflow_v1b/ 内新增的31个Python文件及Git状态；本轮开始时已有的Python内容hash和Git diff hash均无变化。严格的起止清单一致条件仍未通过，不忽略新增文件后改判通过。
