# Round17 handoff

报告的编辑性补充说明 K2 仅因 A3 最大遗憾 6.36pp 超出 5pp 而未通过；A3 平均遗憾 1.70pp 优于 G_E 的 5.45pp，故不能从 K2 失败推出 G_E 整体更好。该补充未改变冻结模型、选择、标签或 `results.json`。

冻结的通用 critic A3 在新训练配方 controller 上：K1 SUPPORTED，相对 A0 固定规则的平均遗憾差 -4.23pp，95% CI [-5.4466342630931655, -2.957332494765258]；K2 REJECTED，A3/G_E 平均遗憾 1.70/5.45pp、A3 最大遗憾 6.36pp；K3 SUPPORTED，F10 低至少10pp 的新 controller 数 0。

## Reproduce

Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。先阅读 `PREREGISTRATION.md`、预标签澄清、`FINAL_FREEZE.receipt.json` 与 `POSTFLIGHT_AUDIT.json`。`results.json` 为主数值记录；`LABELS.npz` 保留所有 seed 标签，其中 -1=unknown。
训练入口 `train.sbatch`；无标签探测 `probe_r5.py`/`probe.sbatch`；A3-all `fit_a3_all.py`；选择冻结 `freeze_choices.py`；标签设计/执行 `design.py`/`run.py`；DB 合并 `merge.py`；最终分析 `analyze.py`。任何重新训练或标签重算必须另开实验，不覆盖本批冻结证据。

## Scope

N2 是共享旧数据集的新架构配方。N1 是新数据集，但只有4个controller，均值比较仅描述。K3 对所有可用新 controller 执行。纯 Q 视频例外已预注册。
