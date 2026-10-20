# Round20 handoff

状态：D1 REJECTED；D2 REJECTED；D3 REJECTED；数值 postflight PASS；视频 gate FAIL。

权威要求：`docs/prompt_round20_cross_scene_probe_selection_20261014.md` 与仓库 `AGENTS.md`。

本目录关键文件：`PREREGISTRATION.md` 及 `.receipt.json`；`CONSISTENCY_GATE.json`；`GEOMETRIES.json`、`FRESH_STATES.json`、`STATE_DEDUP_AUDIT.json`；`pilot_*`、`probe_*`、`main_*` preflight/postflight；`VALIDITY_GATE.json`；`PROBE_SELECTION_FREEZE.json` 及收据；`results.json`；两张 PNG；`videos/manifest.json`；`REPORT.md`。

runtime 继承 E8 FF 与 round13 η schedule。pilot 两个 Ring 半径变体均排除；主评估为 Toy/DB 6 变体 × 3 算子 × 40 state × 7 臂 × 2 seed。所有目标 episode 完整 raw/trace 已进入共享 DB；unknown 不重跑。probe 为无标签部分轨迹，DB 模式没有成功字段可供合并，保留独立 preflight、12 个本地无标签 journal（`probe_JOURNAL_INDEX.json`）及 postflight。pilot/probe 并发造成额外 64 条无标签 Ring probe，已排除，不参与冻结选择。

视频缺口由 `VIDEO_GATE.json` 逐角色记录，统一入口非零，不能宣布完整 batch。继续研究时优先检验更能区分目标到达静止与堵塞的预注册探测，但本轮禁止事后替换规则。
