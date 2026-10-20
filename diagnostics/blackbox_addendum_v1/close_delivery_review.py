from core import *
import re, subprocess
v=read(HERE/'FINAL_VERIFICATION.json')
for k in ['receipt_hashes_passed','imported_sources_passed','source_protection_passed','checkpoint_and_manifest_assets_passed']:assert v[k],k
m=read(HERE/'RESOURCE_MONITOR.json')
events=[e for e in m['events'] if e.get('allocated_total',0)>18]
r=dict(utc=utc(),sample_max_allocated=max(s['allocated_total'] for s in m['samples']),sample_max_known_active=max(s['known_actual_total'] for s in m['samples']),over_cap_allocation_events=events,current_experiment_GPU_shards=0,interpretation='Periodic samples do not bound intersample peaks. One logged total allocation of 19 triggered pausing one own worker. Strict never-over-18 allocation cannot be claimed; other task scheduling was independent.')
write(HERE/'RESOURCE_EXCEPTION.json',r)
p='资源核对：定期采样中的总分配和已知活动进程数最高均为18；日志另记录2026年10月16日04:43:51 UTC总分配瞬时达到19，并暂停本轮一个进程，约2秒后其他任务退出才恢复。定期采样不能排除采样间峰值，因此不能声称总分配全程严格未超过18。该偏差及原始事件保存在RESOURCE_EXCEPTION.json；本轮未使用图形处理器。'
for name in ['REPORT.md','HANDOFF.md']:
 path=HERE/name
 assert p not in path.read_text()
 path.write_text(path.read_text().rstrip()+'\n\n'+p+'\n')
write(HERE/'REPAIRS_RESOLVED.json',dict(utc=utc(),unresolved_technical_execution_errors=[],resolved=[dict(issue='Initial E replay import failed before physical execution',evidence='replay_E_IMPORT_FAILURE_RESOLVED.receipt.json',no_unknown_retry=True),dict(issue='Timing timezone metadata unavailable',resolution='Standard-library UTC+08 metadata analysis only; timing traces not repeated',evidence='E_COMPLETE.receipt.json'),dict(issue='Default environment matplotlib unavailable',resolution='Existing read-only plotting vendor; no package install',evidence='A_CURVE_PLOTS.receipt.json')]))
checks=[]
for name in ['REPORT.md','HANDOFF.md']:
 text=(HERE/name).read_text();bad=[]
 for alias in ['F10','F13','F15','S500','Srel','Sprog','RHR','DM1','DM3','A3','A0','G_E','TT','FF']:
  if re.search(r'(?<![A-Za-z0-9_])'+alias+r'(?![A-Za-z0-9_])',text):bad.append(alias)
 assert not bad,(name,bad)
 checks.append(dict(file=name,sha256=sha(HERE/name),no_unexpanded_controller_aliases=True))
required=['REPORT.md','HANDOFF.md','results.json','GAPS.md','AXIS_TABLE.md','METHOD_METRICS.json','A_RESULTS.json','B_RESULTS.json','D_RESULTS.json','E_RESULTS.json','BLACKBOX_SUBSET_RESULTS.json','N2_GATE_AUDIT.json','MPC_TRANSFER_PER_INSTANCE_TABLE.md','A_CONTROLLER_RADIUS_CURVES.pdf','PSTAR_DECISION.receipt.json']
assert all((HERE/n).exists() for n in required)
write(HERE/'DOCUMENT_REVIEW.json',dict(utc=utc(),checks=checks,first_paragraph=(HERE/'REPORT.md').read_text().split('\n\n')[0],required_files=required,parts=dict(A='Complete; conditional main not triggered',B='Complete via compatible historical evidence after independent calibration',C='Complete read-only',D='Complete',E='Replication, timing and all capability gates complete; main and transfer blocked by gates',F='Complete DEVELOPMENT posthoc subset'),gaps_acknowledged=True,resource_exception_disclosed=True,all_authorized_branches_resolved=True))
s=read(HERE/'OPERATOR_STATE.json');s.update(utc=utc(),completed=['F','C','A','B','D','E'],active=None,status='COMPLETE',pending_review_items=[],final_sequence='Completed; no coordinator restart. E main/transfer not eligible. See COMPLETION.receipt.json.',resource_exception='One transient 19 CPU allocation event disclosed in RESOURCE_EXCEPTION.json');write(HERE/'OPERATOR_STATE.json',s)
write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='COMPLETE',completed_parts=['A','B','C','D','E','F'],E_scope='All capability gates complete; main and transfer blocked by gates',gaps_file=str(HERE/'GAPS.md'),report=str(HERE/'REPORT.md'),resource_exception=str(HERE/'RESOURCE_EXCEPTION.json')))
print('Reviewed all six sections; COMPLETE set; waiting for governor exit before sealing.')
