# P0 决策补充（P2 设计冻结前）

P0 已按 `PREREGISTRATION_P0.md` 在原生 Ring 的旧标签上完成 9 个数据子集、27 个三随机种子网络拟合，并比较两种 checkpoint 选择规则。全部测试 state 为已打开的 256 个，故只属于 DEVELOPMENT。

等 2048 格标签预算下，验证 NLL 规则的成功率：128×16 为 90.82%，256×8 为 91.04%，512×4 为 90.34%。相对每次 bootstrap 内重估的全部 K16 最佳固定 η 的同时 95% CI 分别为 +[0.62,4.49]pp、+[1.05,4.55]pp、+[0.00,4.16]pp。按事前选择规则冻结 **256 state×8 η**。

在该分配上，验证 selected-success 选 checkpoint 与 NLL 的配对开发差为 −0.14pp，95% CI [−1.02,0.78]pp；按事前规则冻结 **验证 NLL**。这只是设计校准，不证明 P2 的效果。全部逐子集、逐 fit、未知区间和模型检查点见 `P0_RESULTS.json`、`P0_TRAINING.json`、`p0_fits/`。
