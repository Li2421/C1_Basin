# 异质训练集 / 验证集 + critic 自检（助手自跑，2026-10-26）

非正式证据：rollout 用隔离执行副本（`/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness`，执行版本 cross_scene_si_v1，3 个快控制器，种子 0），未写共享数据库；启用的四项工程加速逐位核对一致。

- 场景：`scenes_design.json`、`scenes_extra.json`（候选），`scenes_set.json`（入选 36 个）；`SPLIT.json` 为预登记规则（设计状态平均成功率 10–90%、最好 η ≥ 30%；每类固定种子抽 2 个进验证集）。
- 标签：`labels/*.jsonl`（design / extra = 状态 0–7；fresh = 验证集状态 8–23、训练集状态 8–15；全部 49 候选）。
- 异质核对：`VERIFY.json`（选择只用状态 0–7，计分只用新状态）。
- critic 验证：`CRITIC_VALIDATION.json`（训练集状态 0–15 训练，验证集新状态 8–23 计分）。
- 代码：`code/`（job 临时目录原样复制，路径仍指向原位置）。
- generator（2026-10-26）：`GEN_FROZEN.json`（16 个提案与 critic 选择，rollout 前冻结），`labels/generator_proposals.jsonl`（验证集新状态 8–23 × 3 控制器 × 16 提案），`GEN_VALIDATION.json`。
- 纠错（2026-10-26）：generator 目标由"每状态并列最好"改为"每场景最好 1 个"（训练集内留一检验 `gen_loso.jsonl`）；critic 加训练场景上留一 generator 提案的标签（`gen_top1_train.jsonl`）；结果 `FIX_CRITIC_TOP1.json`。
- 最终（2026-10-27）：3 种子 generator 组合 + 重训 critic；12 个新测试场景（`scenes_test.json`，新种子 2026102791，同一筛选规则），全部选择冻结于 `FINAL_FROZEN.json` 后才跑测试新状态；结果 `FINAL_RESULTS.json`。
