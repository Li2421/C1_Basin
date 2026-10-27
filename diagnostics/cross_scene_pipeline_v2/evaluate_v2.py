"""All frozen decisions -> timed pilot -> exact missing-only main -> audit."""
from common import *
from requests_v2 import prepare,design
from rollout_schedule_v2 import run
from results import result
import subprocess,numpy as np

def checkpoint_compatibility():
    old=ROOT/'diagnostics/cross_scene_correction_v1';sources={}
    for name in('runtime_new.py','runtime_checkpointed.py','checkpoint_store.py'):
        assert sha(HERE/name)==sha(old/name);sources[str(HERE/name)]=sha(HERE/name)
    historical={}
    for name in('CHECKPOINT_REPLAY_CHECK.json','CHECKPOINT_PLANNER_SCOPE_CHECK.json','CHECKPOINT_STRICT_BITWISE_AUDIT.json'):
        p=old/name;v=read_evidence(p);assert v['passed'];historical[str(p)]=sha(p)
    assert read(HERE/'REPLAY_CHECK.json')['passed']and read(HERE/'REQUEST_RECONSTRUCTION_CHECK.json')['passed']
    write(HERE/'CHECKPOINT_COMPATIBILITY.json',dict(passed=True,implementation_sources=sources,historical_proofs=historical,
        scope='Unchanged new-scene PA/CBS numerical loops and snapshot implementation; six full local replays passed; cache root only relocated.',checkpoints_are_not_rollout_labels=True))

def pilot_cost():
    measured={}
    for c in read(HERE/'pilot_CASES.json'):
        r=result('pilot',c['rollout_uid']);key=c['state']['scene']+'|'+c['controller_payload']['nominal']['type_code'];measured.setdefault(key,[]).append(r['elapsed_seconds'])
    historical=read_evidence(ROOT/'diagnostics/cross_scene_pipeline_v1/ACTUAL_COST.json')['per_request'];old={}
    for r in historical.values():old.setdefault(r['scene']+'|'+r['type_code'],[]).append(r['seconds'])
    total=0;counts={}
    with dbm.connect(True)as db:
        for c in design('main'):
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():continue
            key=c['state']['scene']+'|'+c['controller_payload']['nominal']['type_code'];cost=float(np.mean(measured.get(key)or old.get(key)or[60.]));total+=cost
            row=counts.setdefault(key,dict(requests=0,cpu_seconds=0));row['requests']+=1;row['cpu_seconds']+=cost
    write(HERE/'PILOT_TIMINGS.json',dict(per_scene_controller={k:dict(all_seconds=v,mean_seconds=float(np.mean(v)),max_seconds=max(v))for k,v in measured.items()},expected_missing_main_cpu_hours=total/3600,
        queue_prediction=counts,compatible_v1_timings_used_for_unmeasured_types=True,limits='New eta can produce planning tails; timing is not an outcome label.'))
    lines=['# 计时试跑\n\n','对每个不超出训练范围的留出场景，使用一个既定合格控制器、两个状态与场景级提案0/1；优先冲突搜索多机规划（conflict_based_search），否则既定顺序首个类型。相同请求按UID去重，精确缓存直接复用。\n\n',
        f'正式缺失请求预测合计{total/3600:.2f}单CPU小时；并行墙钟不能由均值可靠推出，优先级时空A*规划的尾部可能主导。\n\n','| 场景与控制器身份 | 单局秒数 | 均值 | 最大值 |\n|---|---|---:|---:|\n']
    from names_v2 import SCENE_NAMES
    for key,v in measured.items():
        n,t=key.split('|');lines.append(f"| {SCENE_NAMES[n]}（{n}），{t} | {', '.join(f'{x:.3f}'for x in v)} | {np.mean(v):.3f} | {max(v):.3f} |\n")
    (HERE/'PILOT_COST.md').write_text(''.join(lines))

def main():
    assert read(HERE/'ALL_FOLDS_FROZEN.receipt.json')['folds']==31
    checkpoint_compatibility()
    if not(HERE/'EXECUTION_FROZEN.receipt.json').exists():
        seal('EXECUTION_FROZEN.receipt.json',[HERE/n for n in('requests_corrected.py','requests_v2.py','rollout_worker_resume.py','rollout_pool_v2.py','rollout_pool_v2.sbatch','runtime_new.py','runtime_checkpointed.py','checkpoint_store.py','CHECKPOINT_COMPATIBILITY.json','REQUEST_RECONSTRUCTION_CHECK.json')],
            all_choices_already_frozen=True,hard_new_rollout_limit=1300,seed=0,new_rollouts_before_freeze=0)
    for stage in('pilot','main'):
        if not(HERE/f'{stage}_PREFLIGHT.receipt.json').exists():prepare(stage)
        run(stage)
        if stage=='pilot':pilot_cost()
    subprocess.run([sys.executable,'-B',str(HERE/'statistics_v2.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver_v2.py')],cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(HERE/'EVALUATION_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
