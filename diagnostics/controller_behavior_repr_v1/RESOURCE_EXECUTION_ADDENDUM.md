# Resource execution addendum

UTC 2026-10-13T05:28:06.010569+00:00

用户明确要求本轮优先，和其他任务合计18CPU/3GPU shard，覆盖prompt原4CPU上限。原预注册不改。实时其他Slurm作业0；GPU真实LOCO拟合继续原3job×1CPU，新融合LOPO主/敏感性拟合转为Slurm CPU后端，每job1核、最大14并发，另留1CPU编排，总计18。JAX x64=false、seed17/23/41、optimizer/validation/训练预算及全部臂不变。CPU/GPU浮点backend差异记录，不能声称bitwise一致；不按终点选择backend。此时本轮新终点评价尚未执行。原GPU队列中尚未开始的blend indices35..54取消调度后转CPU，属于同一预定fit，不删臂、不重复优化。
