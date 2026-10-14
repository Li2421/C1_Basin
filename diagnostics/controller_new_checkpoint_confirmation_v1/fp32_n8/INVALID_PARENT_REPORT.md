C1、C3均 **INCONCLUSIVE（尚未执行）**；C2的headroom尚无法判断，因为没有产生任何final成功标签。原文件的“14个新真实controller”前提不成立：只有8个final策略与兄弟best不同，另6个策略参数逐字节相同、归一化器及结构一致，已作为best被评估。主实验已锁住，等待cohort澄清，不能据此宣称前瞻确认完成。

## 事实：新颖性复核

14个final文件SHA均不同于best，DB初始checkpoint哈希/rollout身份计数均0，3139份非raw元数据无这些final哈希。然而网络参数字节指纹表明，6个final对应已评估best的完全相同策略：base_u、plus_w_v4、plus_w_v4_wide、plus_w_v5、v2、v3。best与final训练step也相同，文件metadata不同；checkpoint文件新颖不等于controller策略新颖。详见CHECKPOINT_NOVELTY_AUDIT.json、COMPATIBILITY_AUDIT.json、SEMANTIC_CONTROLLER_NOVELTY_AUDIT.json。

本轮确有8个不同的final策略：v4、plus_w_v6、plus_w_v3、plus_w_v4_train_targeted、v10_local、v7_local、v8_local、v9_local。参数距离与结构/normalizer检查已记录，checkpoint和历史代码未修改。

14个run目录仅有best.pkl与final.pkl，共28个文件，无其他snapshot。补足14个真正未评估策略需要其他checkpoint来源，不能把同一策略换文件名充数。

## 事实：冻结与执行状态

预注册/UTC/SHA收据已保存。64个新state（种子2026210000–2026210063）通过DB UID/content与全部可恢复物理初态检查；无法恢复坐标的legacy条目在去重审计中明确列出。448个无标签短探测曾完成并冻结，原source-only、排除同run兄弟的模型与选择也曾冻结。没有任何final成功标签，main/raw为空。

随后发现探测精度违规：poc/field_controller.py的模块导入会设置jax_enable_x64=True；新入口在导入前设置False，导入后被覆盖。该探测并非原R5的FP32语义，原R5_FINAL_FEATURES及由其得到的选择不可用，原始证据与收据全部保留，不假称符合协议。R5差异不能被解释为相同策略突然改变行为。

32例best一致性任务尚未形成任何rollout记录：先启动的两个worker在精度断言处、env.make_env/模拟步之前失败，另两个仍held。主实验held，无final成功数据。已用零模拟步、FP32恢复后的首动作查询验证与历史R5首动作逐字节一致；PRECISION_IMPORT_AUDIT.json给出根因与证据。修复候选代码单独保存，未覆盖已冻结原协议，也未改变模型定义、菜单或阈值。

## 下一步所需澄清

已向用户提供三种cohort方案：仅8个真正新final；寻找其他未评估checkpoint补足14；或8个确认组加6个已知策略回归对照。需要明确方案后，在任何final成功标签产生前写补充预注册并留收据，再以正确FP32探测重新冻结目标描述/选择与协议。不得将这6个同策略checkpoint计作新controller确认。

## 资源与交付限制

用户要求按时长申请，主实验17CPU worker（另留1CPU编排）已提交3小时运行时限，前置依赖保证冻结/门通过前不启动。当前main20252、gate20250剩余任务、gatecheck20251均held。另一用户较早数组申请218任务、每任务1CPU/6GB、并发上限22，不计入用户18CPU额度，但影响实际调度。无GPU worker。

本轮明确不是纯Q视频例外；若继续，仍必须运行Ring统一视频入口。native Ring无deadlock事件，不能用timeout充数。当前没有本轮成功轨迹，未生成视频，也没有COMPLETE标记。

所有新文件在本目录，未修改/删除历史资产或checkpoint，未git commit。DB只新增本轮无标签source_file与预注册身份/初态元数据，probe无success行，final无rollout成功标签。当前results.json仅保存未执行状态，绝不是数值实验结果。
