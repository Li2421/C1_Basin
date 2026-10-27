"""Inspect all saved central-circle numerical unknowns without solver calls."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from projection_adapter import constraints
from shared_control.hard_projection import HardProjectionConfig
from states import walls
import numpy as np
import re


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix='CENTRAL_CIRCLE_UNKNOWN_STATIC_REVIEW_20261023'
    assert not (HERE/(prefix+'.receipt.json')).exists()
    verify_receipt('REVISED_CENTRAL_FIXED_COMPLETE_20261021.receipt.json')
    stage=HERE/'central_revised_fixed_lit_circle_five'
    hp=HardProjectionConfig();rows=[]
    for path in sorted((stage/'raw').glob('*.json')):
        raw=read(path)
        if not raw['numerical_failure']:continue
        assert sha(raw['trace_path'])==raw['trace_sha256']
        c=raw['case'];cfg=c['scenario_payload']['config'];error=raw['error']
        assert error['type']=='CBFSolverError',str(path)
        with np.load(raw['trace_path']) as z:
            p=z['positions'][-1];actions=z['actions'];A,lower,kinds=constraints(dict(positions=p,walls=walls('lit_circle_five'),config=cfg),hp)
            distance=np.linalg.norm(p[:,None]-p[None,:],axis=-1);distance[np.eye(len(p),dtype=bool)]=np.inf
            message=error['message'].replace("\\'", "'")
            excess=[float(x) for x in re.findall(r"'max_speed_excess': ([+\-0-9.eE]+)",message)]
            rows.append(dict(rollout_uid=raw['rollout_uid'],raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],
                controller=c['controller_payload']['nominal']['controller'],method=c['method'],state=c['state']['index'],seed=c['seed'],
                deterministic_alias=bool(raw.get('deterministic_source')),error_step=error['step'],failure_type=error['type'],
                message_prefix=message[:150],reported_candidate_speed_excesses=excess,
                positions_finite=bool(np.isfinite(p).all()),executed_actions_finite=bool(np.isfinite(actions).all()),
                minimum_pair_center_distance=float(distance.min()),d_safe=2*cfg['agent_radius']+cfg['agent_collision_margin']+hp.separation_buffer,
                minimum_joint_zero_input_linear_margin=float((-lower).min()),joint_zero_input_strictly_feasible=bool(np.all(lower<=0)),
                last_executed_max_speed=float(np.linalg.norm(actions,axis=-1).max(initial=0)),configured_max_speed=cfg['max_speed'],
                physical_collision=int(raw['collision']),minimum_swept_clearance=raw['minimum_swept_clearance'],
                saved_actions=len(actions),unknown_retained_unchanged=True))
    assert len(rows)==24,len(rows)
    result=dict(utc=utc(),scope='All24 logical central revised fixed circle unknowns, including deterministic aliases',
        unique_saved_traces=len({r['trace_sha256'] for r in rows}),strictly_zero_feasible=sum(r['joint_zero_input_strictly_feasible'] for r in rows),
        all_saved_positions_and_executed_actions_finite=all(r['positions_finite'] and r['executed_actions_finite'] for r in rows),
        physical_collisions=sum(r['physical_collision'] for r in rows),reviews=rows,
        interpretation='Inspect only saved physical evidence and original linear constraints. Rejected solver candidates were not executed. Preserve unknown; do not modify solver or tolerances, replay, or relabel.',
        new_rollouts=0,relabelled=0,solver_calls=0)
    write(HERE/(prefix+'.json'),result)
    text=f"集中式圆形固定修正的全部24个逻辑数值unknown（{result['unique_saved_traces']}份不同保存轨迹）静态核查。只读取保存位置、已执行动作与原约束；不调用solver、不重放、不改unknown或容差。严格零输入可行的逻辑请求数为{result['strictly_zero_feasible']}/24；全部保存位置和已执行动作有限={result['all_saved_positions_and_executed_actions_finite']}，物理碰撞数={result['physical_collisions']}。拒绝的求解器候选没有被执行，不能把其数值当作实际速度。\n\n|controller|方法|state/seed|错误step|零输入最小联合线性余量|日志候选速度超限|\n|---|---|---|---:|---:|---|\n"
    for r in rows:text+=f"|{r['controller']}|{r['method']}|{r['state']}/{r['seed']}|{r['error_step']}|{r['minimum_joint_zero_input_linear_margin']:.12g}|{r['reported_candidate_speed_excesses']}|\n"
    (HERE/(prefix+'.md')).write_text(text)
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'.json'),HERE/(prefix+'.md'),HERE/'REVISED_CENTRAL_FIXED_COMPLETE_20261021.receipt.json',HERE/'CORRECTION_REVISION.receipt.json'],
         saved_trace_only=True,new_rollouts=0,solver_calls=0,relabelled=0)
    print(json.dumps({k:v for k,v in result.items() if k not in ('reviews','interpretation')}),flush=True)


if __name__=='__main__':main()
