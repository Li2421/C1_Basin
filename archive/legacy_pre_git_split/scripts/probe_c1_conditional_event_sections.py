"""50 CPU event observations for conditional-integration feasibility only.

No risk surrogate, controller change, gradient step, or candidate-path choice.
Finite grid observations cannot certify all intervening event boundaries.
"""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise,digest,source_hashes
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU only')
    out=ROOT/'results/c1_conditional_event_sections_v1'
    if out.exists():raise FileExistsError(out)
    pilot=ROOT/'results/c1_difficulty_calibration_v1/complete.json'
    assert json.loads(pilot.read_text())['selected_distribution']=='closer_interaction'
    params,field,plant,cbf,baseline=setup();goals=np.asarray(GiveWayEnv(plant).goals)
    rng=np.random.default_rng(2026091871)
    x=rng.uniform(.30,.55,(5,2));y=rng.uniform(-.025,.025,(5,2))
    starts=np.stack([np.c_[-x[:,0],y[:,0]],np.c_[x[:,1],y[:,1]]],1)
    direction=np.zeros((plant.max_steps,4))
    direction[:100]=np.random.default_rng(2026091872).normal(size=(100,4))
    direction/=np.linalg.norm(direction)
    z_grid=np.arange(-4.5,5.,1.)
    out.mkdir()
    sources=source_hashes()
    for p in ['scripts/probe_c1_conditional_event_sections.py','single_integrator/c1/evaluate_deadlock_union.py']:
        sources[p]=digest(ROOT/p)
    atomic_save(out/'protocol.json',dict(scope='new conditional integration feasibility, not efficacy or certified probability',
        starts=starts.tolist(),direction_seed=2026091872,initial_seed=2026091871,
        noise_seed=82910,rid_offset=65000,z_grid=z_grid.tolist(),replays=50,
        direction='fixed unit vector on existing first100-step Gaussian latent coordinates, independent of states/parameters/noise/outcomes',
        conditional_noise='eta=xi-w*dot(w,xi); replay eta+z*w with unchanged current policy',
        interpretation='multiple observed label transitions refute a single-threshold assumption; no observed transition does not certify absence of roots',
        pilot_sha256=digest(pilot),environment=plant.to_dict(),cbf=cbf.to_dict(),baseline_sha256=digest(baseline),sources=sources))
    for p in sources:
        dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    np.save(out/'direction.npy',direction)
    rows=[];begun=time.monotonic()
    for rid,initial in enumerate(starts):
        xi=np.asarray(noise(82910,65000+rid),dtype=np.float64)
        eta=xi-direction*np.sum(direction*xi)
        assert abs(np.sum(eta*direction))<1e-12
        np.save(out/f'conditioned_noise_{rid}.npy',eta)
        for index,z in enumerate(z_grid):
            row,tr=execute(params,field,initial,jnp.asarray(eta+z*direction),plant,cbf)
            first='safe_deadlock' if row['any_deadlock'] else 'success' if row['success'] else 'other_timeout'
            label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(tr['applied'].reshape(-1,2,2),axis=-1).max(axis=-1),
                goal_errors=np.linalg.norm(tr['positions_after']-goals,axis=-1)),first,plant.dt)
            row.update(rid=rid,grid_index=index,z=float(z),outcome=label,either_deadlock=label in ('safe_deadlock','stalled_deadlock'))
            rows.append(row);np.savez_compressed(out/f'trace_{rid}_{index}.npz',**tr)
            atomic_save(out/'records.json',rows)
        print(json.dumps(dict(sections_completed=rid+1,replays=len(rows),elapsed=time.monotonic()-begun)),flush=True)
    sections=[]
    for rid in range(5):
        subset=[r for r in rows if r['rid']==rid];labels=np.array([r['either_deadlock'] for r in subset])
        sections.append(dict(rid=rid,event_sequence=labels.astype(int).tolist(),
            outcomes=[r['outcome'] for r in subset],observed_transitions=int(np.count_nonzero(labels[1:]!=labels[:-1]))))
    assert all(digest(ROOT/p)==sha for p,sha in sources.items())
    atomic_save(out/'complete.json',dict(replays=len(rows),elapsed=time.monotonic()-begun,sections=sections,
        event_interval_completeness_certified=False,probability_upper_bound_certified=False,
        risk_gradient_validated=False,independent_test_opened=False,training_launched=False))


if __name__=='__main__':main()
