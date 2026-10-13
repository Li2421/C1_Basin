# P2′ merger 外键错误修复记录

2026-10-11 UTC，P2′ 的 67,584 个冻结 UID 均已有 raw 后，自动合并任务 19342 在首个 `source_file` 插入时因 `experiment_uid` 外键失败而停止；依赖训练 19344 因而未运行。只读检查确认共享 DB 的 `experiment` 表没有 P2′ 的行，`source_file` 与 `rollout` 中属于该实验的记录均为 0。state、controller_config、η 实体已由设计/preflight 登记。原因是 `p2prime_design.py` 漏登独立实验行，P1 的实验行使用根目录路径且 DB 对 `experiment.path` 有唯一约束。

修复仅将冻结的 P2′ 实验身份登记到唯一子目录 `p2prime/`，记录 protocol SHA、原始设计代码 SHA、预算与分片 metadata；不改 raw、journal、UID、标签或统计规则。登记后重新运行原 `p2prime_merge.slurm`，再检查 postflight。旧失败日志保留，新的合并日志独立命名。若 merger 再失败，保留已入库来源并使用相同精确 UID 的幂等合并，不重跑 rollout。
