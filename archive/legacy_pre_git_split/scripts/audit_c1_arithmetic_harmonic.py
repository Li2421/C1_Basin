"""50 cached traces: new-candidate event adapter audit, not future prediction."""
from pathlib import Path
import json
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.arithmetic_harmonic import trajectory
from single_integrator.environment import Config,GiveWayEnv


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU only')
    jax.config.update('jax_enable_x64',True)
    source=ROOT/'results/c1_risk_two_tests_v1';out=ROOT/'results/c1_arithmetic_harmonic_adapter_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    protocol=json.loads((source/'protocol.json').read_text());records=json.loads((source/'records.json').read_text())
    chosen=np.random.default_rng(2026091861).choice(len(records),50,replace=False)
    config=Config(**protocol['environment']);goals=jnp.asarray(GiveWayEnv(config).goals)
    fn=jax.jit(lambda b,a,u,m,to:trajectory(b,a,u,goals,m,terminal_timeout=to,
        dt=config.dt,max_speed=config.max_speed,goal_tolerance=config.goal_tolerance,
        hold_seconds=config.deadlock_hold_seconds,progress_window_seconds=config.progress_window_seconds,
        progress_epsilon=config.progress_epsilon,speed_epsilon_fraction=config.speed_epsilon_fraction))
    paths=['single_integrator/c1/risk/arithmetic_harmonic.py','scripts/audit_c1_arithmetic_harmonic.py']
    for p in paths:
        dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    values=[];padded=0
    for index in chosen:
        row=records[index];path=source/f'{row["arm"]}_{row["rid"]}_{row["seed"]}.npz'
        with np.load(path) as tr:
            n=len(tr['applied']);padding=config.max_steps-n
            tail=np.repeat(tr['positions_after'][-1:],padding,axis=0)
            b=np.concatenate([tr['positions_before'],tail]);a=np.concatenate([tr['positions_after'],tail])
            u=np.pad(tr['applied'],((0,padding),(0,0)));mask=np.arange(config.max_steps)<n
            result={k:float(v) for k,v in fn(b,a,u,mask,row['timeout']).items()}
            assert all(np.isfinite(v) for v in result.values())
            score=result['J_live'];ok=(score>1) if row['either_deadlock'] else (score<=1)
            if padding:
                b[n:]=123.;a[n:]=-321.;u[n:]=7.
                alternative=fn(b,a,u,mask,row['timeout'])
                np.testing.assert_allclose(score,float(alternative['J_live']),rtol=1e-12,atol=1e-12)
                padded+=1
        values.append(dict(rid=row['rid'],seed=row['seed'],arm=row['arm'],outcome=row['outcome'],
            deadlock=row['either_deadlock'],sign_agreement=bool(ok),
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**result))
    dead=[x['J_live'] for x in values if x['deadlock']];nondead=[x['J_live'] for x in values if not x['deadlock']]
    result=dict(scope='new risk on seen traces: code/event agreement only, not early prediction or efficacy',
        n=50,selection_seed=2026091861,selection='uniform without outcome filtering',
        sign_agreement=all(x['sign_agreement'] for x in values),padding_checks=padded,
        deadlocks=len(dead),nondeadlocks=len(nondead),minimum_deadlock=min(dead) if dead else None,
        maximum_nondeadlock=max(nondead) if nondead else None,
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (out/'scores.json').write_text(json.dumps(values,indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['sign_agreement']:raise RuntimeError('event adapter mismatch')


if __name__=='__main__':main()
