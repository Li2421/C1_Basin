"""Retrospective full-trace math audit; not an independent efficacy test."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.terminal_progress_guard import trajectory
from single_integrator.environment import Config,GiveWayEnv


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU scoring only')
    jax.config.update('jax_enable_x64',True)
    folder=ROOT/'results/c1_early_gradient_v2'
    out=ROOT/'results/c1_guard_trace_audit_v1'
    if out.exists():raise FileExistsError('new output required')
    protocol=json.loads((folder/'protocol.json').read_text())
    records=json.loads((folder/'records.json').read_text())
    config=Config(**protocol['environment']);goals=jnp.asarray(GiveWayEnv(config).goals)
    fn=jax.jit(lambda b,a,u,m,to:trajectory(b,a,u,goals,m,terminal_timeout=to,
        dt=config.dt,max_speed=config.max_speed,goal_tolerance=config.goal_tolerance,
        hold_seconds=config.deadlock_hold_seconds,progress_window_seconds=config.progress_window_seconds,
        progress_epsilon=config.progress_epsilon,speed_epsilon_fraction=config.speed_epsilon_fraction))
    values=[];padding_checks=0
    for row in records:
        path=folder/f'{row["arm"]}_{row["rid"]}_{row["seed"]}.npz'
        with np.load(path) as tr:
            n=len(tr['applied']);pad=config.max_steps-n
            tail=np.repeat(tr['positions_after'][-1:],pad,axis=0)
            before=np.concatenate([tr['positions_before'],tail]);after=np.concatenate([tr['positions_after'],tail])
            u=np.pad(tr['applied'],((0,pad),(0,0)));alive=np.arange(config.max_steps)<n
            result={k:float(v) for k,v in fn(before,after,u,alive,row['timeout']).items()}
            assert all(np.isfinite(v) for v in result.values())
            base,risk=result['base_risk'],result['J_live']
            assert risk>=base-1e-12
            assert (risk>1) if row['either_deadlock'] else (risk<=1)
            if row['success']:assert result['terminal_progress_guard']==0
            if pad and padding_checks<12:
                before[n:]=0;after[n:]=0
                other={k:float(v) for k,v in fn(before,after,u,alive,row['timeout']).items()}
                np.testing.assert_allclose(list(result.values()),list(other.values()),atol=1e-12,rtol=1e-12)
                padding_checks+=1
        values.append(dict(rid=row['rid'],seed=row['seed'],arm=row['arm'],outcome=row['outcome'],
                           deadlock=row['either_deadlock'],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**result))
    dead=[x['J_live'] for x in values if x['deadlock']];nondead=[x['J_live'] for x in values if not x['deadlock']]
    summaries={label:dict(n=sum(x['outcome']==label for x in values),
        old_mean=float(np.mean([x['base_risk'] for x in values if x['outcome']==label])),
        new_mean=float(np.mean([x['J_live'] for x in values if x['outcome']==label])))
        for label in sorted({x['outcome'] for x in values})}
    sources=['single_integrator/c1/risk/terminal_progress_guard.py','single_integrator/c1/risk/ordered_guidance.py',
             'single_integrator/c1/risk/exact_margin.py','scripts/audit_c1_progress_guard_traces.py']
    report=dict(scope='already-seen development trace audit only',n=len(values),
        strict_ordering=min(dead)>max(nondead),min_deadlock=min(dead),max_non_deadlock=max(nondead),
        padding_invariance_checks=padding_checks,success_guard_zero=True,original_risk_upper_envelope=True,
        outcome_means=summaries,sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    out.mkdir();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'scores.json').write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
