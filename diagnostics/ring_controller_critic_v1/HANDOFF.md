# Round16 handoff

通用 critic 在独立训练的新 controller 上：Q1 REJECTED，A1 相对 A0 的遗憾差 -0.28pp（95% CI -0.61 至 0.00）；Q2 REJECTED，A1 遗憾 1.51pp、G_E 0.89pp、A1 最大遗憾 7.81pp；逐 state 价值 Q3 REJECTED，遗憾差 -0.28pp（95% CI -0.61 至 0.00）。

## Reproduction

Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。从仓库根目录运行 `python -B diagnostics/ring_controller_critic_v1/evaluate.py`；它首先校验模型冻结收据和历史输入哈希，再读取外部标签。训练代码是 `fit.py`，Slurm 启动配置为 `train.sbatch`（4 CPU / 1 GPU shard，先 7 折再全源重训）。T2 A2 补充预测由 `extend_t2_a2.py` 用冻结 checkpoint 和 Round11 R6 生成；仅用于 T2 描述，不参与 T1 门槛。`REPORT.md` 的解释段是根据固定 `results.json` 补写的，重新运行 `evaluate.py` 会重写该解释段。

## Artifacts

`PREREGISTRATION.md` 和收据固定设计；`DATA_AUDIT.json` 只含预冻结身份和形状；`POSTFREEZE_LABEL_AUDIT.json` 记录 unknown 和可用动作；`fits/` 含每模型每 fit checkpoint、变换、CV NLL 和选步；`predictions/` 含每模型每 fit 的 CV/训练/可用外部预测和冻结 G_E 选择；`MODEL_FREEZE.receipt.json` 封存原始模型与分析代码；`MODEL_FREEZE_AMENDMENT_V2.receipt.json` 记录冻结后非模型修正；`results.json` 是完整数值结果。

## Limits

全部结论为 DEVELOPMENT；T1 标签已在历史 round14A 打开，不能声称全新前瞻确认。T2 A2 使用原有 Round11 R6，仅作为事后可用性纠正的描述性补充；T1/T3 无兼容 R6。未补跑探测。纯 Q 实验依据 AGENTS.md 无视频交付。
