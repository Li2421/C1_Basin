# Round18 Ring η2 gap

η2 缺口 B1 INCONCLUSIVE_NO_HEADROOM（TEST η2 型 2/18，门槛≥3；C2−C0 CI [-2.3509636969430514, 0.031696127589206334]）；完整标准 B2 REJECTED（C2 平均/最大遗憾 1.87/10.12pp）；受控描述性对照显示：两种因素的独立收益均未由描述性CI确认（η2表示 C2−C1 0.35pp，TRAIN-new覆盖 C2−C3 -0.11pp）。

## 预注册门槛

B1 INCONCLUSIVE_NO_HEADROOM；B2 REJECTED；B3 SUPPORTED；B4 REJECTED。配对 bootstrap 每项10000次、按48个TEST state成簇；详见 `results.json`。
B2 灾难数 0。TEST unknown 427 条，保留为unknown且未重跑；timeout未称deadlock。

## 析因与范围

C1−C0 -0.52pp；C3−C0 -0.06pp；C2−C1 0.35pp；C2−C3 -0.11pp。各95%CI见 `results.json`。
C1−C0 同时包含Round17与TRAIN-new覆盖；C3−C0 同时包含Round17覆盖与表示变化。C2−C1隔离新增η2表示，C2−C3隔离TRAIN-new覆盖。这是预标签澄清，详见 `PREREGISTRATION_AMENDMENT.md`。
v3家族TRAIN-new与TEST共享配方及架构，仅seed不同；TEST state全新。TRAIN-new发生率是开发数据，TEST结果是前瞻。
每个controller选择、F2优势、η2探测统计与相关、按架构/seed的v3发生率见 `results.json`。图见 `MODEL_REGRET.svg`、`CONTROLLER_CHOICES.svg`、`ETA2_PROBE_SCATTER.svg`。

## 证据

36 checkpoint 新颖性与FP32兼容性、109 controller短探测、两次冻结、32例一致性门、TRAIN-new与TEST完整rollout、DB preflight/merge/postflight均在本目录留有审计。按AGENTS.md纯Q例外不运行视频入口。
