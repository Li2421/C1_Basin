# Round18 handoff

η2 缺口 B1 INCONCLUSIVE_NO_HEADROOM（TEST η2 型 2/18，门槛≥3；C2−C0 CI [-2.3509636969430514, 0.031696127589206334]）；完整标准 B2 REJECTED（C2 平均/最大遗憾 1.87/10.12pp）；受控描述性对照显示：两种因素的独立收益均未由描述性CI确认（η2表示 C2−C1 0.35pp，TRAIN-new覆盖 C2−C3 -0.11pp）。

Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。先读 `PREREGISTRATION.md`、预标签澄清、`FEATURES_FROZEN.receipt.json`、`TEST_FREEZE.receipt.json`、两阶段DB postflight审计。
训练计划与36 checkpoint见 `TRAINING_PLAN.json`、`checkpoints/`；无标签探测见 `probe.py`、`FEATURE_MATRIX.npz`；Q标签见 `train_LABELS.npz`、`test_LABELS.npz`，-1=unknown。模型与CV见 `critic/`。
模型选择、阈值与TEST动作在TEST标签前冻结。`results.json`是数值主记录；重新训练或改规则需另开实验。
v3 TEST与TRAIN-new共享配方/架构，不能称独立新配方。本轮纯Q实验按视频例外执行。
