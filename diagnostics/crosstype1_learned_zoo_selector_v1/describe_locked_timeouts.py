"""Descriptive trajectory audit of safety timeouts, without another endpoint test."""
from common import *


def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists()
    target=HERE/'LOCKED_TIMEOUT_TRAJECTORY_DIAGNOSTICS.json';assert not target.exists()
    design=HERE/'LOCKED_TIMEOUT_DESCRIPTION_DESIGN.json'
    write(design,dict(utc=utc(),evidence_level='DESCRIPTIVE_POST_CONFIRMATION',
        rationale='Requested description of actual safety timeouts; unconditional probe averages do not describe the episodes that timed out.',
        definitions='Reuse the frozen 0.05 m/s speed and 0.05 m per 1 second progress thresholds. Report all steps from 20 onward and final 100 steps. Show all-robot-step and unfinished-robot-step denominators.',
        classification_threshold_added=False,deadlock_verified=False,extra_hypothesis_test=False,
        frozen_models_or_endpoints_changed=False,new_rollouts=0))
    p=read(HERE/'locked_evaluation_protocol.json');cfg=p['native_manifest']['scenario_config']
    cases=read(HERE/'locked_evaluation_CASES.json');rows=[];inputs=[]
    for c in cases:
        if c['arm']!='safety_baseline':continue
        path=HERE/'locked_evaluation/raw'/f"{c['rollout_uid']}.json";record=read(path)
        if not record.get('timeout') or record.get('numerical_failure'):continue
        assert sha(record['trace_path'])==record['trace_sha256']
        with np.load(record['trace_path'],allow_pickle=False) as z:
            positions=z['positions'];velocities=z['velocities'];goals=z['goals']
        d=np.linalg.norm(goals[None]-positions,axis=-1);speed=np.linalg.norm(velocities,axis=-1)
        remaining=d>cfg['goal_tolerance'];window=round(1/cfg['dt']);n=len(positions)-1
        assert n==700 and np.isfinite(positions).all() and np.isfinite(velocities).all()
        diagnostics={}
        for label,start in (('from_step20',max(20,window)),('last100_steps',max(n-99,window))):
            idx=np.arange(start,n+1);active=remaining[idx]
            frozen=active&(speed[idx]<.05)
            moving=active&(speed[idx]>=.05)&(d[idx-window]-d[idx]<.05)
            diagnostics[label]=dict(robot_steps=int(active.size),unfinished_robot_steps=int(active.sum()),
                strict_freeze_robot_steps=int(frozen.sum()),moving_no_progress_robot_steps=int(moving.sum()),
                strict_freeze_fraction_all=float(frozen.mean()),moving_no_progress_fraction_all=float(moving.mean()),
                strict_freeze_fraction_unfinished=float(frozen.sum()/active.sum()) if active.any() else None,
                moving_no_progress_fraction_unfinished=float(moving.sum()/active.sum()) if active.any() else None)
        rows.append(dict(controller=c['base_controller_name'],state_index=c['state_index'],seed=c['seed'],
            raw_path=str(path),raw_sha256=sha(path),trace_path=record['trace_path'],trace_sha256=record['trace_sha256'],
            final_unfinished_robots=int(remaining[-1].sum()),diagnostics=diagnostics))
        inputs.append(path)
    aggregate={}
    for key in ('from_step20','last100_steps'):
        total={field:sum(r['diagnostics'][key][field] for r in rows) for field in
               ('robot_steps','unfinished_robot_steps','strict_freeze_robot_steps','moving_no_progress_robot_steps')}
        for field in ('strict_freeze','moving_no_progress'):
            for suffix,denom in (('all','robot_steps'),('unfinished','unfinished_robot_steps')):
                total[f'{field}_fraction_{suffix}']=total[f'{field}_robot_steps']/total[denom] if total[denom] else None
        aggregate[key]=total
    write(target,dict(utc=utc(),evidence_level='DESCRIPTIVE_POST_CONFIRMATION',episodes=len(rows),
        aggregate=aggregate,episodes_detail=rows,new_rollouts=0,deadlock_verified=False,
        inference_limit='Kinematic stopping or slow progress does not establish a jointly solvable deadlock. These describe the actual safety timeout traces, without changing the locked endpoints.'))
    seal('LOCKED_TIMEOUT_TRAJECTORY_DIAGNOSTICS.receipt.json',[target,design,HERE/'describe_locked_timeouts.py',*inputs])
    print(json.dumps(dict(episodes=len(rows),aggregate=aggregate),indent=2))


if __name__=='__main__':main()
