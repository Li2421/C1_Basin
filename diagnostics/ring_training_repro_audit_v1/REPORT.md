# Round24 training reproducibility audit

**ROOT_CAUSE_FOUND**：原 `base_u_v2_macflow` 使用 `validation_batches=4`，该参数未写入 config；round13B/14A 重训从 config 恢复时使用了历史训练函数的默认值 8。显式设回 4 后，原 seed、历史代码与原数据的第 0 步和 12,000 步 dev loss 均复现：最终相对误差 0.000005741%，best step 同为 12000，满足0.1%门。

1. 第0步事实：默认8时 train/dev为2.0542898476/2.0659007728，而原值为2.0488712788/2.0563832521；四组归一化数组逐字节相同。初始参数hash已保存，原初始checkpoint不存在，不能直接比较原初始参数。
2. 数据事实：963个archive的states/observations/actions digest均与manifest及元数据一致；manifest SHA与原run的config一致；文件mtime没有晚于原训练日。manifest没有逐archive原始文件SHA，因而同时登记当前SHA与受保护的数组digest，不伪称有不存在的原SHA对照。
3. 依赖事实：jax/jaxlib0.6.2、flax0.10.7、optax0.2.8，当前numpy2.4.6，历史requirements为2.2.6。在本目录隔离venv装2.2.6后，默认8的第0步偏差原样存在。没有修改现有环境。FP64初始化也没有复现原损失。
4. 随机源事实：初始化、train/dev采样及训练key均按seed控制；fixed_train在第一次update之前从同一train采样器抽取，4改8多消耗4×256个样本索引，因而改变后续训练batch顺序。同时dev loss从4批平均变为8批平均，两个门比较的指标本身也不同。4批prefix的train/dev初始均值同时匹配原值，随后完整12000步受控复现验证了原因。
5. 第二配方事实：v9_local正确数据集是base_u_v8_local_dataset。它的原设置为8：第0步两个损失完全一致，第1000步dev loss为0.337218496948，与原值0.337218496948一致；强行用4则不一致。这说明遗漏的非默认参数是配方特定的，并非所有Flow训练都无法复现。

对原代码、数据与原checkpoint没有修改。比较表为COMPARISON_TABLE.csv，逐步事实见STEP1/STEP2/STEP3/STEP4与受控结果；全部对照checkpoint均在本目录，无rollout、DB写入、视频或commit。

**影响评估：** round14A/15的真实rollout标签、冻结选择与其测得的成功率仍对应实际训练出的checkpoint；它们并不会因本次训练审计失效。但v2原4/新8的“同配方独立seed”并非完全相同的训练协议，原dev loss与备用门中新dev loss的比较还混入了验证批数量差异，所以备用门不能据此当成精确配方复现。涉及相同配方覆盖、seed波动的解释应注明该缺口；不能把实际人口上的前瞻选择性能升格为完全复现原训练配方的证明。v9对照则表明这一问题需要逐配方核对。历史报告/标签未被回写，未重复训练任何完整人口。
