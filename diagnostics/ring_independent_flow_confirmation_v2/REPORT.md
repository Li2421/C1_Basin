# Round14A independent Ring Flow confirmation

训练代码版本漂移**不足以解释** Round13B 复现失败：历史包与 PoC 全部18文件一致，但CPU重训仍逐数值复现v1的偏差，GPU主门也未通过。预注册的两配方多seed备用门通过，随后用同一历史包与GPU训练14个独立seed Flow。C1 **SUPPORTED**，C3 **SUPPORTED**；C2 headroom 为 **5/14** 个真实 E<5pp，门通过，C2 **SUPPORTED**。

## 复现证据
历史original run文件日期、git43b5264和HEAD训练文件长度、每文件SHA见HISTORICAL_CODE_AUDIT.json；没有仅凭日期认定因果。CPU/GPU原seed3104均使用相同历史包、数据manifest、12000steps/batch256。

| device | best relative error | final relative error | beststep |
|---|---|---|---|
| cpu | 6.693438% | 3.974216% | 11750 |
| gpu | 6.698181% | 3.949186% | 11750 |

主判定阈值均为1%，两设备未通过；参数hash和JAX/device/import路径见PRIMARY_TRAINING_GATE.json及START收据。原beststep12000，新beststep11750。历史代码CPU的best/final与v1一样，因此漂移不是所观察偏差的充分解释；数据内容/原依赖版本/随机数状态仍可能有变化，未确定唯一根因。

仅在双设备失败之后，执行已注册备用门，v2与v9_local各新seed+20001/+20002/+20003；两配方**best和final都要求通过**：

| recipe | endpoint | original-seed historical difference | newseed range | gate |
|---|---|---|---|---|
| base_u_v2_macflow | best_fixed_dev_loss | 0.00869331229 | 0.0228590779 | PASS |
| base_u_v2_macflow | final_fixed_dev_loss | 0.00512549654 | 0.0239474447 | PASS |
| base_u_v9_local_macflow | best_fixed_dev_loss | 2.77669169e-05 | 0.00192743167 | PASS |
| base_u_v9_local_macflow | final_fixed_dev_loss | 3.87169421e-05 | 0.00442877784 | PASS |

备用门允许继续，不等于恢复逐位相同的原训练，也不消除未知历史环境差异。未放宽门槛。14独立训练seed=original+10000，checkpoint best由固定dev loss选取。兼容性门拦截wide原训练日志遗漏512×4层宽度的问题：错误默认256×3训练产物保留并标为无效，补充预注册和收据先于修复训练；以原checkpoint架构、同seed13114重训，探测和成功标签均尚未产生。全部确认checkpoint参数与可读取历史checkpoint及互相均不同，具体数量与未能解析的非同schema资产见CHECKPOINT_NOVELTY_AUDIT。normalizer、网络结构与本配方原best一致，L2/relativeL2与原seed/新seed逐项记录。训练与数据文件hash全部留存。

## 冻结和前瞻证据
先完成448次无标签FP32 R5探测（unknown 0），冻结R5文件和逐trace索引；再冻结source-only13模型（排除同配方原controller）与14源敏感性，PCA2/ridgealpha1与旧源模型逐项一致。G_E Ehat<5pp选F10，否则S500。冻结文件保留旧schema中的sibling_best/excluded_sibling字段，当前均指同配方原source best，确认组为独立训练run而不是同run兄弟。探测与选择的收据均先于任何新controller成功标签。32历史best一致性记录与当前数值loop/adapter/IO绑定/FP32/seed逐项兼容，官方preflight0missing，已精确复用并逐字节比对旧Round9/Round4记录，未重复生成这32条rollout，然后21504条全rollout，64个全新Ring state、6臂、4future seeds。Unknown 183保留且未重跑；timeout不称deadlock。源标签不因输入表示变化而失效。

主动作集F10/F13/S300/S500，F2/F4只供oracle；共同state簇5000 bootstrap（seed202610131201），四个future seeds保留，共享draw，不重采样controller，unknown边界另列。

| endpoint | status |
|---|---|
| C1 | SUPPORTED |
| C1b | INCONCLUSIVE |
| C2 | SUPPORTED |
| C3 | SUPPORTED |
| C4 | DESCRIPTIVE / INCONCLUSIVE（无预注册二元门槛） |

主G_E平均遗憾 0.613839pp，最大 3.906250pp；相对F10平均收益 29.575893pp，95%CI及unknown包络见results.json。C3 Spearman=0.7422925495678401，RMSE=0.0925884063，源均值基线RMSE=0.27010368，比值=0.34278839219819673。C2保留至少2个E<5pp的headroom门，不满足时INCONCLUSIVE。C1b 的最大遗憾保守点虽然低于5pp，但unknown区间为[2.734375, 5.468750]pp，跨过5pp门，因此为INCONCLUSIVE。

每controller点结果如下；unknown上下界保留在results与CSV中，表中regret/gain采用已注册保守点值。所选臂的timeout是horizon终止，unknown是数值异常记录，不称为deadlock。

| controller | G_E arm | oracle point arm | regret(pp) | gainF10(pp) | Ehat/E(pp) | selected-arm terminations |
|---|---|---|---|---|---|---|
| plus_w_v4_wide | S500 | S500 | 0.0000 | 42.1875 | 46.132/42.188 | unknown:1 medianstep=95; timeout:22 medianstep=700 |
| macflow__independent | S500 | S500 | 0.0000 | 49.2188 | 50.180/49.219 | unknown:1 medianstep=196; timeout:41 medianstep=700 |
| v4 | S500 | S300 | 0.3906 | 57.4219 | 36.417/57.422 | unknown:6 medianstep=147.5; timeout:6 medianstep=700 |
| plus_w_v6 | S500 | S500 | 0.0000 | 51.9531 | 44.029/51.953 | unknown:1 medianstep=173; timeout:25 medianstep=700 |
| plus_w_v3 | S500 | S300 | 0.7812 | 1.5625 | 17.098/1.562 | unknown:1 medianstep=177; timeout:40 medianstep=700 |
| plus_w_v4_train_targeted | S500 | S500 | 0.0000 | 50.3906 | 39.783/50.391 | unknown:3 medianstep=129; timeout:11 medianstep=700 |
| v3 | S500 | S300 | 2.7344 | 17.9688 | 23.449/17.969 | unknown:7 medianstep=84; timeout:41 medianstep=700 |
| v2 | S500 | S500 | 0.0000 | 57.4219 | 55.695/57.422 | unknown:1 medianstep=390; timeout:35 medianstep=700 |
| v10_local | F10 | F4 | 3.9062 | 0.0000 | -2.150/-3.906 | unknown:3 medianstep=297; timeout:32 medianstep=700 |
| plus_w_v5 | S500 | S500 | 0.0000 | 42.1875 | 44.950/42.188 | unknown:5 medianstep=240; timeout:21 medianstep=700 |
| v7_local | F10 | F10 | 0.0000 | 0.0000 | -8.865/-9.766 | timeout:11 medianstep=700 |
| plus_w_v4 | S500 | S500 | 0.0000 | 43.7500 | 53.095/43.750 | unknown:6 medianstep=267; timeout:27 medianstep=700 |
| v8_local | F10 | S300 | 0.7812 | 0.0000 | -11.110/0.781 | timeout:2 medianstep=700 |
| v9_local | F10 | F10 | 0.0000 | 0.0000 | -10.763/-3.516 | timeout:4 medianstep=700 |

S300点成功率高于S500的有6/14个，S500更高7/14个，平局1/14个。更早切换获益名单：base_u_v4, base_u_plus_w_v3, base_u_v3, base_u_v10_local, base_u_v7_local, base_u_v9_local。这些是注册的描述性比较，不据此改选G_E。F2/F4成为点oracle的有1/14个，本轮为v10_local的F4（无其它臂平局）；G_E仍按冻结规则选F10。全部六臂成功率/CI、冻结选择、E/L/N、S300vsS500遗憾及F2/F4oracle见CSV、DESCRIPTIVE_SWITCH_ORACLE及图表。失败类型与发生step见FAILURE_RECORDS.json，不把timeout归为deadlock。主结论只覆盖Ring、这14个已有配方的新独立训练seed，不是跨场景或任意配方证明。

## 交付与限制
数值执行完成，视频完整交付为 **INCOMPLETE**。已导出3个真实全episode MP4，连续运动swept安全余量来自实际simulator。Ring原生runtime没有deadlock判据，死锁角色如实缺失；统一入口已运行并以非零码拦截不完整batch，无COMPLETE标记。未知标签与无死锁判据均不改标准。所有新产物限v2，共享DB仅预注册生命周期写入；无历史修改/删除/git commit。资源合计上限24CPU/3GPU。
