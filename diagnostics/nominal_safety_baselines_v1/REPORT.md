MPC：相对 nominal only +90.10pp（95% CI [83.59, 95.83]），相对 safety baseline +22.92pp（95% CI [2.34, 46.35]）；势场：相对 nominal only +84.38pp（95% CI [70.31, 94.79]），相对 safety baseline +27.08pp（95% CI [7.29, 48.44]）；Flow：相对 nominal only +66.41pp（95% CI [33.98, 96.88]），相对 safety baseline +66.41pp（95% CI [33.98, 96.88]）。以上是与 round27 一致的已确认成功占比差；unknown 单列，完整成功率的识别范围及 CI 包络见 results.json。


| 类别 | nominal only | safety baseline | F10 | F13 | best fixed oracle | A3_full | S500 | full menu oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 0.00 [0.00, 0.00] | 67.19 [39.58, 91.67] | 26.04 [18.75, 33.34] | 91.93 [85.16, 97.66] | 91.93 [85.16, 97.66] | 90.10 [83.59, 95.83] | 90.10 [83.59, 95.83] | 91.93 [85.16, 97.66] |
| P | 15.62 [5.21, 29.69] | 72.92 [51.56, 92.71] | 7.81 [3.12, 13.02] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| Flow | 26.56 [0.00, 58.98] | 26.56 [0.00, 58.98] | 14.84 [4.69, 25.78] | 59.38 [35.16, 78.91] | 59.38 [37.11, 78.91] | 92.97 [86.33, 98.05] | 96.88 [93.75, 99.22] | 96.88 [94.14, 99.22] |

表内为百分比 [95% CI]。5000 次 controller→state 配对分层 bootstrap，两 seed 留在同一 state 簇内；oracle 在每次重抽后重新取 controller 聚合成功率最大臂。最佳固定为 F10/F13/F2，完整菜单为全部六臂；oracle 是描述性事后上界，不是可部署策略。

碰撞率、超时率、数值失败（unknown）率、描述性停滞及逐 controller 八方法结果见 METHOD_METRICS.csv。unknown 未重跑、未归为已知失败；给出 confirmed-success 下界、完整成功率上界、known-only 比率与 unknown bootstrap 包络。timeout 不称 deadlock。两 seed 的确定性势场重复不视为两份独立未来随机性证据。

一致性门 8/8：覆盖 MPC、势场、全部 Flow；终止、步数与首动作 dtype/shape/bytes 完全相同，附加全位置轨迹核对见 CONSISTENCY_GATE.json。冻结 controller 直接加载并验证原 V2 收据和参数/checkpoint hash；未重新实现或调参。导入后及每次构造后恢复 x64=False，逐步断言。

历史审计、预注册 UTC/sha256 收据、2048 manifest 与 preflight/postflight 均保留。第3–8方法复用 round27 原标签，类别六臂与 A3/S500 点值已逐项断言一致；表中 CI 使用本轮统一 bootstrap 重新计算，未改写 round27 原报告。连续运动 swept clearance、完整位置/动作与真实终止保存在 main/traces；collision-free success 由原 simulator 判定。纯 Q 评估按预注册免视频。

**DB 写入范围限制：** 用户只授权新目录写文件，因此 authoritative shared DB 只读备份到本目录 rollout.sqlite；官方 preflight、append-only journal、单进程官方 merger、postflight 在副本运行。全局共享索引未更新；不得声称已完成全局 DB ingestion。没有修改历史资产，没有 git commit。

停滞只用于描述：未成功且最后100个实际步的总目标距离减少<0.05；正常等待或后退也可能满足，不提供死锁认证。A3 只沿用 round27 冻结选择，未使用本轮 baseline 标签训练、选择或调整。

碰撞率 / 超时率（%）：

| 类别 | nominal only | safety baseline | F10 | F13 | best fixed oracle | A3_full | S500 | full menu oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 100.00 / 0.00 | 0.00 / 32.81 | 0.00 / 73.96 | 0.00 / 8.07 | 0.00 / 8.07 | 0.00 / 9.90 | 0.00 / 9.90 | 0.00 / 8.07 |
| P | 84.38 / 0.00 | 0.00 / 27.08 | 0.00 / 91.67 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Flow | 49.22 / 24.22 | 0.00 / 73.44 | 0.00 / 84.77 | 0.00 / 40.62 | 0.00 / 40.62 | 0.00 / 7.03 | 0.00 / 3.12 | 0.00 / 3.12 |
