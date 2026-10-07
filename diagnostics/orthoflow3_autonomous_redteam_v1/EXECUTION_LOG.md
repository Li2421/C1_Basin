# 执行记录

1. 只读恢复当前已冻结模型、v2及K16报告；确认前项Phase-A采集正在运行，Phase-B/contract审计没有完成报告。
2. 创建独立worktree `audit/orthoflow3-redteam-v1`，记录23个直接科学文件hash；写已审计ledger及四项risk预注册。
3. 读取既有DB，不执行continuation；H2完整16320seed核对及区间重聚合。H1读取当时已完成的train/dev proposal证据，不发起新的proposal批次。
4. 单CPU推理任务1489首次在argmax一致性检查失败；定位到Phase-A按logit、旧K16按float32概率的差别。注册H5后第二个单CPU推理任务完成，76/76冻结eta/score重建通过。这是探针协议修正，不是训练/控制器修复迭代。
5. H5复用候选已有Q，8个Four差异两侧均B15；Ring无选择差异。没有额外eta，没有增加K。
6. 解析停止树区分可用计数似然与有偏stopped ratio，再记录outcome-class取样的校准限定。
7. 隔离生成新报告/typed evidence helpers，不改canonical或并行Phase-A文件。13项审计回归+3项既有basis回归通过。
8. 标准缓存preflight、16320已引用行存在性检查、23+8hash核对完成。发布本目录，不产生新DB记录，不启动后续研究。

## 共享资源

另一个用户任务1483（6CPU）启动后，为维持两个任务合计给实验室留>=6CPU的约定，将先前Phase-A initial array并发从9降至6，重排本方6/7/8子任务，完成结果已在DB；为本审计单CPU短探针再暂时降至5/重排本方5，探针结束恢复6。未取消其他任务，未改科学参数或模型。后续confirmation并发上限12。此举只控制并发，不使任何已完成科学tuple丢失。

本审计GPU使用0。红队推理任务均已退出；前项任务的既有作业不属于此次红队实验，按“不打断先前任务”的要求保留。
