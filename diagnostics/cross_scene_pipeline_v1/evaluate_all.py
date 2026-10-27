"""Frozen proposals -> official preflight -> timed pilot -> missing main requests."""
from common import *
from requests_new import prepare,design
from schedule import run,jobs,cancel
from results import result
import numpy as np,subprocess

def checkpoint_compatibility():
    old=ROOT/'diagnostics/cross_scene_correction_v1';sources={}
    for name in ('runtime_new.py','runtime_checkpointed.py','checkpoint_store.py'):
        assert sha(HERE/name)==sha(old/name);sources[str(HERE/name)]=sha(HERE/name)
    historical={}
    for name in ('CHECKPOINT_REPLAY_CHECK.json','CHECKPOINT_PLANNER_SCOPE_CHECK.json','CHECKPOINT_STRICT_BITWISE_AUDIT.json'):
        p=old/name;v=read_evidence(p);assert v['passed'];historical[str(p)]=sha(p)
    write(HERE/'CHECKPOINT_COMPATIBILITY.json',dict(passed=True,implementation_sources=sources,historical_proofs=historical,
        scope='Only frozen new-scene PA/CBS execution; numerical loops and snapshot code byte-identical to historical proof. Native kernel changes only cache root, six full local replays passed.',
        checkpoints_are_not_rollout_labels=True))

def pilot_cost():
    table={}
    for c in read(HERE/'pilot_CASES.json'):
        r=result('pilot',c['rollout_uid']);key=c['pipeline_scene']+'|'+c['controller_payload']['nominal']['type_code']
        table.setdefault(key,[]).append(float(r['elapsed_seconds']))
    measured={k:dict(n=len(v),mean_seconds=float(np.mean(v)),max_seconds=max(v),all_seconds=v)for k,v in table.items()}
    cases=design('main');historical=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/PILOT_TIMINGS.json')['per_scene_controller']
    # Historical timings for literature are read from the compatible eta13
    # request sources. They are cost predictors only, not new eta labels.
    from requests_new import literature_templates
    for c in literature_templates().values():
        key=c['state']['scene']+'|'+c['controller_payload']['nominal']['type_code']
        if key in historical:continue
        p=Path(c.get('reuse_source',''))
        if p.is_file():
            r=read_evidence(p);historical[key]=dict(mean_seconds=float(r['elapsed_seconds']),max_seconds=float(r['elapsed_seconds']),n=1,cost_only_source=str(p))
    totals={};cpu=0.
    with dbm.connect(True)as db:
        for c in cases:
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():continue
            key=c['pipeline_scene']+'|'+c['controller_payload']['nominal']['type_code'];same_scene=[v for k,v in measured.items()if k.startswith(c['pipeline_scene']+'|')]
            ref=measured.get(key)or historical.get(key)or same_scene[0];seconds=ref['mean_seconds'];cpu+=seconds
            row=totals.setdefault(key,dict(requests=0,seconds=0.,basis='pilot'if key in measured else 'historical compatible controller timing'if key in historical else 'same-scene pilot controller proxy'))
            row['requests']+=1;row['seconds']+=seconds
    from schedule import quota
    workers=max(1,quota(jobs()));payload=dict(per_scene_controller=measured,main_cost_by_scene_controller=totals,
        expected_missing_main_cpu_hours=cpu/3600,worker_reference=workers,expected_main_wall_hours=cpu/3600/workers,
        limitations='Pilot2 states x2 generated proposals per scene; other controller costs use compatible historical timing, not new outcome labels. Planning tails can dominate.')
    write(HERE/'PILOT_TIMINGS.json',payload)
    lines=['# 计时试跑\n\n',f'11个场景各2状态×2生成器提案；使用该场景已有合格冲突搜索控制器，若无则取既定顺序首个实例。正式缺失请求预计 {cpu/3600:.2f} CPU小时，按当前{workers}个worker约 {cpu/3600/workers:.2f} 小时。\n\n',
        '未测类型的成本用已有相同控制器执行语义的旧计时作预测，不替代新η结果；规划尾部可能使均值估计失准。\n\n',
        '| 场景/控制器类型 | 单局秒数（4局） | 平均秒 | 最长秒 |\n|---|---|---:|---:|\n']
    for k,r in measured.items():lines.append(f"| {k} | {', '.join(f'{v:.3f}'for v in r['all_seconds'])} | {r['mean_seconds']:.3f} | {r['max_seconds']:.3f} |\n")
    lines.append('\n| 正式场景/类型 | 预计缺失局数 | CPU小时 | 成本依据 |\n|---|---:|---:|---|\n')
    for k,r in totals.items():lines.append(f"| {k} | {r['requests']} | {r['seconds']/3600:.3f} | {r['basis']} |\n")
    (HERE/'PILOT_COST.md').write_text(''.join(lines))

def main():
    assert read(HERE/'ALL_FOLDS_FROZEN.receipt.json')['folds']==31
    checkpoint_compatibility()
    if not(HERE/'STATISTICS_FROZEN.receipt.json').exists():
        seal('STATISTICS_FROZEN.receipt.json',[HERE/n for n in ('statistics_new.py','requests_new.py','results.py','rollout_worker.py','evaluate_all.py')],
            seed=2026102517,replicates=5000,primary='pipeline minus same-source-pool global fixed,11 deduplicated LOSO scenes,95% lower>0',new_rollouts_before_freeze=0)
    for stage in ('pilot','main'):
        if not(HERE/f'{stage}_COMPLETE.json').exists():
            if not(HERE/f'{stage}_PREFLIGHT.receipt.json').exists():prepare(stage)
            run(stage)
            if stage=='pilot':pilot_cost()
            write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),stage=stage));print('COMPLETED',stage,flush=True)
    subprocess.run([sys.executable,'-B',str(HERE/'statistics_new.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver.py')],cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(HERE/'EVALUATION_FATAL.json',dict(utc=utc(),type=type(e).__name__,message=str(e),traceback=__import__('traceback').format_exc()))
        for r in jobs():
            if r['name'].startswith('xpl_'):cancel(r['id'],True)
        raise
