"""Matched-controller probe: committed Flow0 action at t0, frozen Flow1 thereafter.

Workers write explicit-identity envelopes; they never write SQLite. The legacy
heuristic ingestor must not be used because it assumes Toy always uses Flow0.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TOY = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0] = [str(TOY), str(ROOT)]
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import canonical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, required=True)
    args = ap.parse_args()
    frozen = json.loads((OUT/'protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == frozen['runtime_sha256']
    pf = json.loads((OUT/'cache_preflight.json').read_text())
    missing = {(r['state_uid'], r['eta_uid'], r['controller_uid'], s)
               for r in pf['details'] for s in r['missing_seeds']}
    tasks = []
    for r in json.loads((OUT/'pair_manifest.json').read_text()):
        for i in range(16):
            if (r['state_uid'], r['eta_uid'], frozen['alternate_controller_uid'], canonical({'future_index':i})) in missing:
                tasks.append((r,i))
    tasks = [t for j,t in enumerate(tasks) if j % args.shards == args.shard]
    import jax
    import jax.numpy as jnp
    from single_integrator.evaluate import load_policy
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from shared_control.basis_families import get_basis_family
    jax.config.update('jax_enable_x64', True)
    policies = []
    for k in (0,1):
        path = Path(frozen['flow_paths'][str(k)])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen['flow_sha256'][str(k)]
        policy, prov = load_policy(path)
        assert prov['evaluation_environment'] == frozen['environment']
        policies.append(policy)
    samples = [jax.jit(lambda obs,key,p=p: p.sample_actions(obs[None],seed=key)[0]) for p in policies]
    cfg, cbf = Config(**frozen['environment']), CBFConfig(**frozen['cbf'])
    basis = get_basis_family('orthoflow3')
    expected = np.load(frozen['features_path'])['h_raw']
    exp = frozen['experiment_uid']
    journal_dir = ROOT/'shared_rollout_db/journals'/exp
    done = set()
    if journal_dir.exists():
        for path in journal_dir.glob('*.jsonl'):
            for line in path.read_text().splitlines():
                r=json.loads(line)['record'];done.add((r['state_uid'],r['eta_uid'],r['future_index']))
    pending=[]; completed=0; started=time.monotonic()
    for pair,seed in tasks:
        if (pair['state_uid'],pair['eta_uid'],seed) in done: continue
        env=GiveWayEnv(cfg);env.reset(np.asarray(pair['initial_positions'],np.float64))
        eta=np.asarray(pair['eta'],np.float64)
        current=jax.random.fold_in(jax.random.PRNGKey(42),pair['rollout_id'])
        future=jax.random.fold_in(jax.random.PRNGKey(42+seed),pair['rollout_id'])
        error=None;jdef=0.;feature_error=None;response={};min_wall=float('inf');min_agent=float('inf')
        for step in range(cfg.max_steps):
            try:
                obs=jnp.asarray(np.asarray(env.observation(),np.float32))
                key=jax.random.fold_in(current if step==0 else future,step)
                raw=np.asarray(samples[0 if step==0 else 1](obs,key),np.float64)
                flow=bounded_nominal(raw,cfg.max_speed)
                A, lower, _=barrier_constraints(env.snapshot(),cbf)
                safe,*_=project_velocity_with_retry(flow,A,lower,cfg.max_speed,cbf)
                correction=basis.compute(env.positions,env.goals,safe,cfg.max_speed).correction(eta)
                executed,*_=project_velocity_with_retry(safe+correction,A,lower,cfg.max_speed,cbf)
                assert np.min(A@executed.reshape(4)-lower)>=-cbf.feasibility_tol
                if step==0:
                    h,_=StartupAwareFeatureBuilder().build(env,{'u_flow':flow,'u_safe':safe},cfg,cbf)
                    feature_error=float(np.max(np.abs(np.asarray(h,np.float32)-expected[pair['episode_index']])))
                    np.testing.assert_allclose(np.asarray(h,np.float32),expected[pair['episode_index']],rtol=1e-6,atol=1e-7)
                    response['committed_t0_raw_flow']=raw.tolist()
                    response['committed_t0_executed']=executed.tolist()
                if step==1:
                    # Same physical state and same continuation noise. Diagnostic
                    # Flow0 response is never executed; Flow1 drives the rollout.
                    response['flow1_raw_t1']=raw.tolist()
                    oldraw=np.asarray(samples[0](obs,key),np.float64)
                    response['flow0_raw_t1']=oldraw.tolist()
                    response['raw_response_distance']=float(np.linalg.norm(raw-oldraw))
                    try:
                        oldflow=bounded_nominal(oldraw,cfg.max_speed)
                        oldsafe,*_=project_velocity_with_retry(oldflow,A,lower,cfg.max_speed,cbf)
                        oldcorr=basis.compute(env.positions,env.goals,oldsafe,cfg.max_speed).correction(eta)
                        oldexecuted,*_=project_velocity_with_retry(oldsafe+oldcorr,A,lower,cfg.max_speed,cbf)
                        response['safe_response_distance']=float(np.linalg.norm(safe-oldsafe))
                        response['executed_response_distance']=float(np.linalg.norm(executed-oldexecuted))
                    except Exception as exc:
                        response['counterfactual_projection_error']=str(exc)
                jdef+=cfg.dt*float(np.sum((executed-safe)**2))
                _,_,_,info=env.step(executed)
                min_wall=min(min_wall,float(info['min_swept_wall_distance']))
                min_agent=min(min_agent,float(info['min_swept_agent_distance']))
                if env.done:break
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc),'step':step};break
        summary=env.summary()
        outcome=('numerical_failure' if error else 'collision' if summary['wall_collision'] or summary['agent_collision']
                 else 'success' if summary['success'] else 'deadlock' if summary['deadlock'] else 'timeout')
        record={**pair,'controller_uid':frozen['alternate_controller_uid'],'future_index':seed,
                'success':outcome=='success','deadlock':outcome=='deadlock','timeout':outcome=='timeout',
                'collision':outcome=='collision','numerical_failure':outcome=='numerical_failure','outcome':outcome,
                'episode_length':env.step_count,'J_def':jdef,'feature_max_abs_error':feature_error,
                'minimum_wall_clearance':min_wall if np.isfinite(min_wall) else None,
                'minimum_agent_clearance':min_agent if np.isfinite(min_agent) else None,
                'execution_error':error,'response':response,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        pending.append({'schema':'explicit_controller_rollout_v1','record':record})
        completed+=1
        if len(pending)>=32:
            append_journal(pending,exp,f'shard{args.shard}');pending=[]
            print(json.dumps({'shard':args.shard,'completed':completed}),flush=True)
    if pending:append_journal(pending,exp,f'shard{args.shard}')
    print(json.dumps({'shard':args.shard,'new':completed,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
