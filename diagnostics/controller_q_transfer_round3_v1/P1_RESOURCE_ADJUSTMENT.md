# P1 调度续跑记录

2026-10-11 13:07 UTC 左右，P1 24 worker 在 Slurm 2 小时限制的约 35 分钟处产出约 8,000/19,968 个 raw。不同 checkpoint 的 episode 用时不均，部分 worker 约 160–200 个/34 分钟，按此速度可能超过最初 2 小时 walltime。尝试以 `scontrol update JobId=19266 TimeLimit=04:00:00` 延长 walltime，Slurm 返回 `Access/permission denied`；不修改 P1 样本、标签、执行代码、gate 或 worker CPU 上限。

为避免 walltime 截断导致已完成 raw 无法合并，预先提交一个依赖原数组 `afterany` 的同一 worker 续跑数组，每任务仍为 1 CPU、总并发不超过 24。`p1_resume.slurm` 若该 worker 的 `P1_WORKER_i.json` 已标为完成即退出，否则按原始 `P1_MISSING_CASES.json` 和同一 `p1_run.py` 运行；每个既有 raw 以精确 case 校验并 journal 追加，不重复 rollout。原始 pending merge/analysis 改为只依赖续跑数组成功的单进程 merge/postflight/analyze。此调整仅改变调度窗口，不改变科学设计或 outcome 定义。续跑产生的 raw 数和 postflight 必须与冻结的 genuinely missing 19,968 一致。
