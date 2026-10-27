"""Live corrected-scope request counts; cancellation is distinct from unknown."""
from experiment import *
from atomic_write import write
from resources_corrections_release_20261023_quota18_now_20261022 import snapshot
import time,fcntl

SCENES=list(NAMES);FIXED=[16896,29568,4320,17280];EVAL=[3072,3072,1536,3072]
REUSE={}
def count(stage,planned):
 path=HERE/f'{stage}_REUSED_ROWS.json'
 if path.exists() and str(path) not in REUSE:REUSE[str(path)]=len(read(path))
 reused=REUSE.get(str(path),0);landed=sum(1 for _ in (HERE/stage/'raw').glob('*.json'))
 return dict(stage=stage,completed=reused+landed,planned=planned,remaining=planned-reused-landed,exact_reuse=reused,landed_new_uid_records=landed)

def tick():
 stages=[count(f'{layer}_revised_fixed_{s}',n) for layer in ('distributed','central') for s,n in zip(SCENES,FIXED)]
 labels=[count('learning_labels_revised_20261021_'+s,7680) for s in SCENES];evaluation=[count('learning_eval_revised_20261021_'+s,n) for s,n in zip(SCENES,EVAL)]
 done=sum(r['completed'] for r in stages+labels+evaluation);assert all(0<=r['completed']<=r['planned'] for r in stages+labels+evaluation)
 old_circle=sum(1 for _ in (HERE/'learning_labels_lit_circle_five/raw').glob('*.json'));old_done=219584+old_circle
 optional=[]
 for stage,scope in read(HERE/'OPTIONAL_SCOPE.json')['stages'].items():optional.append(count(stage,scope['planned_requests']))
 optional_done=sum(r['completed'] for r in optional);ability=119808
 all_done=old_done+ability+done+optional_done;all_plan=old_done+ability+177600+127168
 state=read(HERE/'STATUS.json');resources=snapshot()
 result=dict(utc=utc(),phase=state['phase'],scope='User correction revision. Logical requests include exact reuse; percentages are not elapsed-time completion or physical trajectory counts.',main_registered_or_preflight_verified_completed=done,main_planned_logical_requests=177600,main_remaining=177600-done,main_request_percent=100*done/177600,revised_fixed_stages=stages,revised_learning_label_stages=labels,revised_learning_eval_stages=evaluation,original_semantics=dict(completed=old_done,circle_labels_landed=old_circle,circle_labels_planned=6144,circle_unwritten_stopped=6144-old_circle,circle_old_evaluation_stopped=1536,stop_reason='用户决定停止：修正语义修订',unknowns_from_stopping=0,old_circle_workers_may_still_drain=not (HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json').exists()),ability_gate_completed=ability,second_stage_and_other_optional=optional,all_v2_completed=all_done,all_v2_planned_logical_requests=all_plan,all_v2_remaining=all_plan-all_done,all_v2_request_percent=100*all_done/all_plan,resources=resources)
 write(HERE/'LIVE_PROGRESS_PERCENT.json',result);write(HERE/'CORRECTION_PROGRESS_20261021.json',result)
 text=f"{result['utc']}。阶段：{result['phase']}。修订主线{done}/177600（{100*done/177600:.4f}%），还需{177600-done}个逻辑请求；含精确复用，不等于时间进度或实际新轨迹。原语义已有{old_done}个主线请求完成，旧圆形标签保留{old_circle}条，其余不再派发、不算unknown。资源上限{resources['cpu_limit']}CPU含1本地、{resources['gpu_shard_limit']}GPU份额；10月23日09:00新加坡时间降至18CPU；当前Slurm{resources['scheduled_cpus']}CPU/{resources['gpu_shards']}GPU份额。\n"
 (HERE/'STATUS.md').write_text(text)
 return result

def main():
 with (HERE/'CORRECTION_PROGRESS.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(HERE/'CORRECTION_PROGRESS_PID_20261021.json',dict(utc=utc(),pid=os.getpid()))
  while not (HERE/'V2_COMPLETE.json').exists():tick();time.sleep(60)
  tick()
if __name__=='__main__':main()
