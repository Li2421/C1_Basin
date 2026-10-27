"""Close user-stopped old circle labels after all current episodes have landed."""
from experiment import *
from atomic_write import write
from resources_corrections_20261021 import snapshot
from registry import verify_receipt
import time,subprocess,collections

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 stopped=read(HERE/'V2_CORRECTION_OLD_CIRCLE_DRAIN.json');jobs=set(stopped['old_circle_jobs'])
 while jobs&{r['job_id'] for r in snapshot()['jobs']}:time.sleep(10)
 verify_receipt('CBS_FLAT_CACHE_MIGRATION_COMPLETE.receipt.json')
 for path,expected in read(HERE/'CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json').items():assert sha(path)==expected,path
 folder=HERE/'learning_labels_lit_circle_five';rows=[];outcomes=collections.Counter();trace_checked=set()
 for path in sorted((folder/'raw').glob('*.json')):
  raw=read(path);marker=folder/'journal_receipts'/path.name;receipt=read(marker);assert receipt['raw_sha256']==sha(path)
  if raw['trace_sha256'] not in trace_checked:assert sha(raw['trace_path'])==raw['trace_sha256'];trace_checked.add(raw['trace_sha256'])
  rows.append(dict(rollout_uid=raw['rollout_uid'],raw=str(path),raw_sha256=sha(path),trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],journal_receipt=str(marker),journal_receipt_sha256=sha(marker)))
  outcomes[raw['outcome']]+=1
 write(HERE/'OLD_CIRCLE_LEARNING_LANDED_20261021.json',dict(utc=utc(),rows=rows,unique_traces_checked=len(trace_checked)))
 # The same serialized official merger also safely merges any newly landed
 # revised fixed evidence; no copied DB or worker SQLite writes are used.
 with (HERE/'OLD_CIRCLE_FINAL_MERGE_20261021.log').open('a') as out:
  result=subprocess.run([sys.executable,'-B',str(HERE/'db_merge.py')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
 assert result.returncode==0
 write(HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.json',dict(utc=utc(),reason='用户决定停止：修正语义修订',planned_labels=6144,landed_preserved=len(rows),unwritten_not_to_resume=6144-len(rows),landed_outcomes=dict(outcomes),new_unknowns_from_user_stop=0,old_evaluation_not_run=1536,all_old_worker_jobs_ended=True,hard_cancelled_jobs=0,all_landed_raw_and_trace_hashes_verified=True,all_landed_journals_merged_by_official_merger=True,old_pre_cache_raw_hashes_unchanged=True,new_revised_stages_independent=True))
 seal('OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json',[HERE/n for n in ('close_old_circle_learning_20261021.py','OLD_CIRCLE_LEARNING_LANDED_20261021.json','OLD_CIRCLE_LEARNING_STOPPED_20261021.json','V2_CORRECTION_OLD_CIRCLE_DRAIN.json','OLD_CIRCLE_FINAL_MERGE_20261021.log')],user_decision=True,no_unknown_relabeling=True,no_resume=True,old_evaluation_not_run=True)
 with (HERE/'ITERATION_LOG.md').open('a') as out:out.write('\n\n'+utc()+': Old circle label flow closed under user correction instruction: '+str(len(rows))+' landed results preserved with verified raw/journal/trace hashes and official merged journals; '+str(6144-len(rows))+' unwritten labels will not be resumed, old1536 evaluation not run,0hard cancellations/0stop-induced unknowns. See OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json. Revised flow continues independently.\n')
 print(dict(landed=len(rows),unwritten_stopped=6144-len(rows),outcomes=dict(outcomes)),flush=True)
if __name__=='__main__':main()
