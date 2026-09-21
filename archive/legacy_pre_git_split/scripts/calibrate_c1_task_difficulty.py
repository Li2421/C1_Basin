"""50 Safety-only CPU replays to calibrate a distribution, never select cases."""
from pathlib import Path
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,source_hashes,digest
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU only')
    out=ROOT/'results/c1_difficulty_calibration_v1'
    if out.exists():raise FileExistsError(out)
    params,field,plant,cbf,baseline=setup();goals=np.asarray(GiveWayEnv(plant).goals)
    rng=np.random.default_rng(2026091870)
    q=rng.uniform(size=(25,2));y=rng.uniform(-.025,.025,(25,2))
    ranges=dict(original=(.55,1.05),closer_interaction=(.30,.55))
    starts={}
    for name,(lo,hi) in ranges.items():
        x=lo+(hi-lo)*q
        starts[name]=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
        for initial in starts[name]:GiveWayEnv(plant).reset(initial)
    out.mkdir()
    sources=source_hashes()
    for p in ['scripts/calibrate_c1_task_difficulty.py','single_integrator/c1/evaluate_deadlock_union.py']:
        sources[p]=digest(ROOT/p)
    atomic_save(out/'protocol.json',dict(scope='Safety-only task calibration, no candidate efficacy',
        initial_seed=2026091870,noise_seed=82900,rid_offset=64000,replays=50,
        starts={k:v.tolist() for k,v in starts.items()},ranges=ranges,
        pairing='same uniform quantiles, y coordinates and execution noise across the two distributions',
        eligibility='at least 5 deadlocks of 25 cases (20%), all original safety checks zero',
        selection='prefer closer_interaction if eligible; otherwise original if eligible; otherwise no distribution selected',
        prohibition='no per-case outcome filtering, no C1 or archived risk scoring, no automatic candidate rerun',
        baseline_sha256=digest(baseline),environment=plant.to_dict(),cbf=cbf.to_dict(),sources=sources))
    for p in sources:
        dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    rows=[];begun=time.monotonic()
    for rid in range(25):
        draws=noise(82900,64000+rid)
        for name in ranges:
            row,tr=execute(params,field,starts[name][rid],draws,plant,cbf)
            event='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
            label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                goal_errors=np.linalg.norm(tr['positions_after']-goals,axis=-1)),event,plant.dt)
            row.update(rid=rid,distribution=name,outcome=label,either_deadlock=label in ('safe_deadlock','stalled_deadlock'))
            rows.append(row);np.savez_compressed(out/f'{name}_{rid}.npz',**tr)
            atomic_save(out/'records.json',rows)
        print(json.dumps(dict(cases_per_distribution=rid+1,replays=len(rows),elapsed=time.monotonic()-begun)),flush=True)
    summaries={}
    for name in ranges:
        subset=[r for r in rows if r['distribution']==name]
        dead=sum(r['either_deadlock'] for r in subset);n=len(subset);p=dead/n;z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        summaries[name]=dict(n=n,deadlocks=dead,successes=sum(r['success'] for r in subset),
            ordinary_timeouts=sum(r['outcome']=='other_timeout' for r in subset),
            deadlock_rate=p,wilson95=[center-half,center+half],eligible=dead>=5)
    selected=next((name for name in ['closer_interaction','original'] if summaries[name]['eligible']),None)
    assert all(digest(ROOT/p)==sha for p,sha in sources.items())
    atomic_save(out/'complete.json',dict(replays=len(rows),elapsed=time.monotonic()-begun,
        distributions=summaries,selected_distribution=selected,
        use='freeze distribution only; new candidate evaluations require fresh starts/noises and their own difficulty gate',
        independent_test_opened=False,training_launched=False))


if __name__=='__main__':main()
