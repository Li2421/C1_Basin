"""Final B delivery, preserving the original A/C version and all frozen code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from pipeline import call
import collections

def main():
 assert (HERE/'B_COMPLETE.receipt.json').exists()
 d=read(HERE/'PSTAR_DECISION.receipt.json');assert d['selection']=='CANDIDATE_1' and d['pstar']==.9681620884830543
 counts=collections.Counter();cost=collections.Counter();kinds=collections.Counter();switches={};preserved_unknown=[]
 for c in read(HERE/'B_CASES.json'):
  q=read(HERE/'B/raw'/f"{c['rollout_uid']}.json");ref=q['prefix_provenance'];assert ref is not None
  counts[str(c['H'])]+=1;kinds[ref['compatibility']['kind']]+=1
  for k in ['new_physical_steps','prefix_reused_steps']:cost[k]+=q[k]
  counts['unknown']+=q['numerical_failure'];switches[q['rollout_uid']]=q['progress_switch_step']
  if ref['compatibility']['kind']=='UNKNOWN_REUSE':
   old=read(ref['raw_path']);assert old['error']==q['error'] and q['new_physical_steps']==0;preserved_unknown.append(q['rollout_uid'])
  first=ref['compatibility']['first_action_difference']
  if q['new_physical_steps']>0:assert q['progress_switch_step']==first==q['prefix_reused_steps']
 write(HERE/'B_PREFIX_REUSE_AUDIT.json',dict(utc=utc(),passed=True,counts=counts,cost=cost,kinds=kinds,preserved_unknown=preserved_unknown,unknown_retries=0,selected_progress_threshold=d['pstar'],switch_steps=switches,scope='Per-request longest same-horizon F13 source; full prefix bytes, event/native integration, DB hashes independently checked by review_B'))
 call('immutability.py');call('audit.py','END');call('delivery_B.py')
 b=read(HERE/'B_RESULTS.json')
 note='''\n\nB本次修订：用户明确选择候选1，仅用实际观测到p(500)的699条旧S500轨迹计算中位数，p*=0.9681620884830543。195条提前成功及2条提前unknown不进入阈值计算，不补读、不作终点保持、不重跑。该阈值是到达第500步子集的条件中位数，并非全部896条的中位数。决定原文/确认/UTC/SHA见PSTAR_DECISION.receipt.json，早于选定阈值测试和B提交。HZ3/HZ4采用CONFIRMATORY（定义修订），旧标签和前缀保持来源等级；共享前缀不代表独立样本。HZ3未检出差异时不表示等效。\n\n本次交付为A/C完成后的B增补版本。旧A/C全部顶层文件的原字节保存在AC_DELIVERY_VERSION_1/，路径映射与哈希见AC_DELIVERY_VERSION_1.manifest.json；其完成收据的旧绝对路径引用按此映射核验。当前完整结果以NUMERICAL_COMPLETE_B.receipt.json和B_FINAL_COMPLETE.receipt.json为准；原NUMERICAL_COMPLETE/SUPPLEMENTARY为A/C历史版本。B的逐轨迹前缀复用、实际新增步数和unknown保留见B_PREFIX_REUSE_AUDIT.json。\n'''
 p=HERE/'REPORT.md';p.write_text(p.read_text()+note+'\n'+(HERE/'GEOMETRY_SCOPE.md').read_text()+'\n'+(HERE/'PLOT_ENV_QUALIFICATION.md').read_text()+'\nN2不同历史cohort见N2_HISTORICAL_EVIDENCE_TABLE.md。\n')
 p=HERE/'HANDOFF.md';p.write_text(p.read_text()+note+'\nB已完成，不再等待决定或重跑。启动编号及封存依次见PSTAR_DECISION、SELECTED_PROGRESS_TEST、B_PREFLIGHT、B_SUBMISSION、B_COMPLETE。最终统计用冻结analyze_B.py，交付适配仅改说明及完成收据名称，见B_DELIVERY_ADAPTER_FROZEN.receipt。\n')
 call('review_B.py')
 # Verify every original A/C top-level snapshot independent of updated current delivery.
 saved=read(HERE/'AC_DELIVERY_VERSION_1.manifest.json')['files'];assert all(sha(v['snapshot'])==v['sha256'] for v in saved.values())
 frozen=['B_ANALYSIS_FROZEN.receipt.json','B_DELIVERY_ADAPTER_FROZEN.receipt.json','B_RESUMPTION_FROZEN.receipt.json','PSTAR_DECISION.receipt.json','SELECTED_PROGRESS_TEST.receipt.json']
 for name in frozen:
  for f,h in read(HERE/name)['files'].items():assert sha(f)==h,(name,f)
 r=read(HERE/'INDEPENDENT_DELIVERY_REVIEW.json')
 write(HERE/'FINAL_STATUS.json',dict(utc=utc(),A_C_complete=True,B_executed=True,B_status='COMPLETE_CANDIDATE_1',pstar=d['pstar'],GA1='GENERATOR_VALID',GA3='NOT_TRIGGERED',C=read(HERE/'N2_GATE_AUDIT.json')['conclusion'],HZ3=b['HZ3'],HZ4={H:q['HZ4'] for H,q in b['parts'].items()},source_integrity_passed=r['source_integrity_passed'],DB_records=r['authoritative_DB_records'],unknowns=r['unknown_query_counts'],unknown_retries=0,completion_receipt='NUMERICAL_COMPLETE_B.receipt.json',previous_delivery_snapshot='AC_DELIVERY_VERSION_1.manifest.json'))
 write(HERE/'ACTIVE_WORK_STATE.json',dict(utc=utc(),round=37,status='A/B/C COMPLETE',B_selection='CANDIDATE_1',pstar=d['pstar'],remaining=[],review='INDEPENDENT_DELIVERY_REVIEW.json',completion='B_FINAL_COMPLETE.receipt.json'))
 seal('SUPPLEMENTARY_DELIVERY_B.receipt.json',[HERE/'NUMERICAL_COMPLETE_B.receipt.json',HERE/'PSTAR_DECISION.receipt.json',HERE/'B_PREFIX_REUSE_AUDIT.json',HERE/'B_COMPLETE.receipt.json',HERE/'B_DELIVERY_ADAPTER_FROZEN.receipt.json',HERE/'AC_DELIVERY_VERSION_1.receipt.json',HERE/'FINAL_STATUS.json',HERE/'ACTIVE_WORK_STATE.json'],A_C_snapshot_verified=True,unknown_retries=0)
 seal('B_FINAL_COMPLETE.receipt.json',[HERE/'NUMERICAL_COMPLETE_B.receipt.json',HERE/'SUPPLEMENTARY_DELIVERY_B.receipt.json',HERE/'FINAL_STATUS.json'],round37_complete=True)
 print('A/B/C final delivery complete',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'B_FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
