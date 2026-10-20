"""Completion index only after all five final receipts verify."""
from common import *

def main():
 rounds=[('32','axis1_state_v1'),('33','axis2_controller_v1'),('34','axis3_cross_scene_v1'),('35','axis4a_geometry_horizon_v1'),('36','axis4b_agent_count_v1')];checks=[];links=[]
 for n,name in rounds:
  folder=D/name;receipt=folder/'NUMERICAL_COMPLETE.receipt.json';q=read(receipt)
  for f,h in q['files'].items():assert Path(f).exists() and sha(f)==h,(n,f)
  checks.append(dict(round=n,receipt=str(receipt),receipt_sha256=sha(receipt),verified_files=len(q['files']),report_sha256=sha(folder/'REPORT.md')))
  links.append(f"| {n} | [报告](../{name}/REPORT.md) | [结果表](../{name}/AXIS_TABLE.md) | [完成收据](../{name}/NUMERICAL_COMPLETE.receipt.json) |")
 write(HERE/'FIVE_ROUND_FINAL_HASH_REVIEW.json',dict(utc=utc(),passed=True,checks=checks,note='Only final active receipts; PRE_REVIEW / PRE_DOCUMENTATION archived receipts intentionally retain historical document hashes. Numeric completion is distinct from source-integrity pass.'))
 text='''# Round32–36 顺序执行完成索引

五轮依次完成；本索引只有在五份最终完成收据及其全部文件 SHA256 校验通过后生成。数值交付完成不表示所有实验假设通过，也不表示源码起止完整性审计通过。每轮首段、缺口表和审计差异是结论的一部分。

| 轮次 | 报告 | 方法结果 | 收据 |
|---|---|---|---|
'''+ '\n'.join(links)+'''

- Round32：指定 cohort 的 ST1/ST2 数值标准通过，ST3 未检测出提升；14 个测试 checkpoint 与 A3 训练 checkpoint 无精确重合，纯 State 轴前提缺口保留。
- Round33：CT1 支持；CT2 样本方向覆盖不足，无法判定；CT3 混合 critic 的优势标准未通过。
- Round34：四场景 XS1 无法完整判定，可用三场景子集未通过；XS2 未通过。Toy 的旧 FP64 与协议 FP32 复现门冲突，保留前瞻缺口。
- Round35：HZ1 两个新预算均支持相对时间表；HZ2 结果按各预算报告。p* 字面定义无法覆盖所有 episode，未擅自采用补充定义，HZ3 缺口、HZ4 协议完整菜单无法判定。四个半径均未通过 controller 能力门，保留缺口。
- Round36：见本轮 REPORT 首段、CAPABILITY_GATE、FIVE_LAYER_EVIDENCE 与 TRANSFER_TABLE。未通过能力门不能解释为修正无效，规划器失败不能证明任务不可解，timeout/排队不能解释为 deadlock。

所有轮次均保留完整性起止差异、接口/能力/定义缺口与 unknown。历史收据中的文档旧 hash 由明确的最终复核或文档修订收据接续，不修改旧证据。最终文件核验见 FIVE_ROUND_FINAL_HASH_REVIEW.json。本轮额外的轨迹派生审计见 ADDITIONAL_VERIFICATION.receipt.json 和 REUSE_QUALIFICATION.md；后期资源调度与实际CPU/Slurm分配差别见 RESOURCE_OPERATIONAL_NOTES.md、EXPERT_CPU_HANDOFF.receipt.json、TAIL_SCHEDULING.md；18核目标与正常尾部不足18核的实际占用见TAIL_CPU_UTILIZATION.md。
'''
 (HERE/'COMPLETION_INDEX.md').write_text(text)
 seal('FIVE_ROUND_COMPLETION.receipt.json',[HERE/'FIVE_ROUND_FINAL_HASH_REVIEW.json',HERE/'COMPLETION_INDEX.md',HERE/'EXPERT_CPU_HANDOFF.receipt.json',HERE/'TAIL_SCHEDULING.receipt.json',HERE/'TAIL_CPU_UTILIZATION.md'],sequential_rounds=[32,33,34,35,36],all_final_hashes_verified=True)
if __name__=='__main__':main()
