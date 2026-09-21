"""Paired SI evaluation with frozen nominal policy and post-hoc filter seam."""
import argparse
from datetime import datetime
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import flax.serialization
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'flowbc'))
sys.path.insert(0,str(ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official'))
sys.path.insert(0,str(ROOT/'scripts'))
from giveway_flowbc_agent import GiveWayFlowBCAgent, get_config
from giveway_initial_state import sample_initial_positions
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.filters import FilterResult
from single_integrator.outcomes import annotate, aggregate_outcomes, PROTOCOL


def load_policy(path, allow_legacy=False):
    with Path(path).open('rb') as f:checkpoint=pickle.load(f)
    provenance=checkpoint.get('si_metadata')
    if provenance is None and not allow_legacy:
        raise ValueError('Checkpoint has no SI training provenance. PID-trained models are diagnostic-only; explicitly use --allow_legacy_checkpoint to probe them.')
    config=get_config()
    if 'config' in checkpoint:config.update(checkpoint['config'])
    policy=GiveWayFlowBCAgent.create(0,jnp.zeros((1,2,10)),jnp.zeros((1,2,2)),config)
    policy=flax.serialization.from_state_dict(policy,checkpoint['agent'])
    return policy,provenance


def filter_factory(spec):
    if spec=='none':return None,None
    module,name=spec.split(':',1)
    factory=getattr(importlib.import_module(module),name)
    path=Path(inspect.getfile(factory))
    return factory,dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def rollout(policy,initial,config,seed,rollout_id,action_filter=None,cbf_config=None,
            c1_geometry_config=None):
    if c1_geometry_config is not None and cbf_config is None:
        raise ValueError('C1 geometry requires the authoritative CBF configuration and projection')
    if cbf_config is not None:
        from single_integrator.cbf import barrier_geometry
        from single_integrator.outcomes import first_event
        if not all((config.terminate_on_collision, config.terminate_on_success, config.terminate_on_deadlock)):
            raise ValueError('First-event evaluation requires all terminal events enabled')
    env=GiveWayEnv(config);env.reset(initial)
    # Common random numbers depend only on experiment seed, rollout index and
    # time index, never on filter outcomes or a previous episode's length.
    episode_key=jax.random.fold_in(jax.random.PRNGKey(seed),rollout_id)
    buffers={k:[] for k in ['trajectory_id','observations','positions_before','raw_policy_velocity','nominal_velocity','executed_velocity','filter_delta','nominal_projection_delta','filter_status','filter_diagnostics','filter_seconds','positions','velocities','goal_errors','wall_distances','agent_surface_distance','min_swept_wall_distance','min_swept_agent_distance','wall_collision','agent_collision','endpoint_wall_collision','endpoint_agent_collision','outside_workspace','window_ready','window_progress','candidate_deadlock','stuck_timer','max_speed','speeds','deadlock_trigger_timestep','deadlock','task_success','tracking_error','integration_residual']}
    if cbf_config is not None:
        buffers.update({k: [] for k in ['u_nom','u_safe','intervention_norm','pairwise_h','min_wall_h','qp_status']})
        initial_geometry=barrier_geometry(env.snapshot(),cbf_config)
    for step in range(config.max_steps):
        obs=env.observation();before=env.positions.copy()
        raw=np.asarray(policy.sample_actions(jnp.asarray(obs[None]),seed=jax.random.fold_in(episode_key,step)))[0]
        nominal=(policy.nominal_control(raw,config.max_speed) if hasattr(policy,'nominal_control')
                 else bounded_nominal(raw,config.max_speed))
        snapshot=env.snapshot()
        start=time.perf_counter()
        result=FilterResult(nominal.copy(),status='unfiltered') if action_filter is None else action_filter(snapshot,nominal.copy())
        if not isinstance(result,FilterResult):raise TypeError('Action filters must return FilterResult, with explicit diagnostics/fallback status')
        elapsed=time.perf_counter()-start
        executed=np.array(result.velocity,dtype=np.float64,copy=True)
        geometry=barrier_geometry(snapshot,cbf_config) if cbf_config is not None else None
        c1_geometry=None
        if c1_geometry_config is not None:
            # C1 Stage 1: confirmed CBF geometry only.  No cone distance, r_t,
            # trajectory risk, correction model, or learning signal.
            from single_integrator.c1.risk.deadlock_geometry import extract_deadlock_geometry
            c1_geometry=extract_deadlock_geometry(snapshot, nominal, executed, cbf_config, c1_geometry_config)
        _,_,done,info=env.step(executed)  # Rejects invalid/overspeed output; never clips it.
        values=dict(trajectory_id=rollout_id,observations=obs,positions_before=before,raw_policy_velocity=raw,nominal_velocity=nominal,executed_velocity=executed,filter_delta=executed-nominal,nominal_projection_delta=nominal-raw,filter_status=result.status,filter_diagnostics=json.dumps(result.diagnostics,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),filter_seconds=elapsed)
        if geometry is not None:
            values.update(u_nom=nominal,u_safe=executed,intervention_norm=float(np.linalg.norm(executed-nominal)),pairwise_h=geometry['pairwise_h'],min_wall_h=geometry['min_wall_h'],qp_status=result.status)
            if c1_geometry is not None:
                values.update(c1_task_force=c1_geometry['task_force'], c1_h=c1_geometry['h'],
                              c1_cbf_residual=c1_geometry['cbf_residual'], c1_rho=c1_geometry['rho'],
                              c1_safety_force_blocks=c1_geometry['safety_force_blocks'],
                              c1_active_mask=c1_geometry['active_mask'],
                              c1_active_count=len(c1_geometry['active_indices']),
                              c1_completely_deadlock_free=c1_geometry['completely_deadlock_free'],
                              c1_cone_distance_defined=c1_geometry['cone_distance_defined'])
                if step == 0:
                    buffers.update({key: [] for key in ['c1_task_force','c1_h','c1_cbf_residual','c1_rho','c1_safety_force_blocks','c1_active_mask','c1_active_count','c1_completely_deadlock_free','c1_cone_distance_defined']})
        for key in buffers:
            buffers[key].append(values[key] if key in values else info[key])
        if done:break
    summary=annotate(env.summary())
    summary.update(rollout_id=rollout_id,termination=info['termination'],initial_positions=np.asarray(initial).tolist(),completion_time=env.first_success_step*config.dt if env.first_success_step is not None else None,max_tracking_error=max(buffers['tracking_error']),max_integration_residual=max(buffers['integration_residual']),intervened_steps=int(sum(np.linalg.norm(d)>1e-10 for d in buffers['filter_delta'])),nominal_projection_steps=int(sum(np.linalg.norm(d)>1e-10 for d in buffers['nominal_projection_delta'])))
    if cbf_config is not None:
        outcome=first_event(info)
        interventions=np.asarray(buffers['intervention_norm'])
        # Swept minima include the terminal action interval, not only sampled endpoints.
        center_min=min(buffers['min_swept_agent_distance'])+2*config.agent_radius
        summary.update(outcome=outcome,success=outcome=='success',collision_free_success=outcome=='success',
                       completion_time=summary['episode_steps']*config.dt if outcome=='success' else None,
                       min_pairwise_h=min(initial_geometry['pairwise_h'],center_min**2-initial_geometry['d_safe']**2),
                       min_wall_h=min(initial_geometry['min_wall_h'],min(buffers['min_swept_wall_distance'])-config.wall_collision_margin-cbf_config.separation_buffer),
                       mean_intervention=float(interventions.mean()),
                       intervened_steps=int(np.count_nonzero(interventions>cbf_config.intervention_tol)),
                       intervention_rate=float(np.mean(interventions>cbf_config.intervention_tol)),
                       simultaneous_collision=bool(info['wall_collision'] and info['agent_collision']))
    return summary,{k:np.asarray(v) for k,v in buffers.items()}


def plot_rollout(path,arrays,initial,config,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env=GiveWayEnv(config)
    fig,ax=plt.subplots(figsize=(9,3.5))
    for a,b in env.walls:ax.plot([a[0],b[0]],[a[1],b[1]],'k-',lw=2)
    for i,color in enumerate(['royalblue','tomato']):
        positions=np.concatenate([np.asarray(initial)[None],arrays['positions']])
        ax.plot(positions[:,i,0],positions[:,i,1],color=color,label=f'Agent {i}')
        ax.scatter(env.goals[i,0],env.goals[i,1],marker='*',s=100,color=color)
    ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend();fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/give_way_si_short_v1')
    p.add_argument('--split',choices=['val','test','random'],default='test')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n_rollouts',type=int,default=25)
    p.add_argument('--out_dir',type=Path)
    p.add_argument('--action_filter',default='none',help='none, or module:factory(config_dict) returning a FilterResult callable')
    p.add_argument('--allow_legacy_checkpoint',action='store_true')
    p.add_argument('--fixed_horizon',action='store_true',help='Explicit secondary protocol: detect but do not terminate on success/deadlock')
    p.add_argument('--wall_layout_scale',type=float,default=1.,help='Scale wall segment coordinates about origin; goals, starts, radii and policy unchanged')
    p.add_argument('--progress_window_seconds',type=float,default=2.)
    p.add_argument('--deadlock_hold_seconds',type=float,default=5.)
    p.add_argument('--progress_epsilon',type=float,default=.01)
    p.add_argument('--speed_epsilon_fraction',type=float,default=.05)
    p.add_argument('--c1_deadlock_geometry',action='store_true',help='Opt-in C1 Stage-1 CBF geometry only; requires the authoritative CBF filter.')
    p.add_argument('--c1_rho',type=float,default=.05,help='Shared CBF look-ahead threshold rho for opt-in C1 geometry.')
    p.add_argument('--c1_active_tol',type=float,default=1e-7,help='CBF residual equality tolerance for opt-in C1 geometry.')
    args=p.parse_args()
    if args.n_rollouts<=0:raise ValueError('n_rollouts must be positive')
    if args.c1_deadlock_geometry:
        if args.action_filter!='single_integrator.cbf:cbf_factory':
            raise ValueError('--c1_deadlock_geometry requires --action_filter single_integrator.cbf:cbf_factory')
        from single_integrator.cbf import CBFConfig
        from single_integrator.c1.risk.deadlock_geometry import DeadlockGeometryConfig
        c1_cbf_config=CBFConfig()
        c1_geometry_config=DeadlockGeometryConfig(rho=args.c1_rho,active_tol=args.c1_active_tol)
    else:
        c1_cbf_config=None
        c1_geometry_config=None
    policy,provenance=load_policy(args.checkpoint,args.allow_legacy_checkpoint)
    config=Config(**provenance['evaluation_environment']) if provenance else Config()
    dataset_meta=json.loads((args.dataset/'environment.json').read_text())
    if provenance and dataset_meta['evaluation_environment']!=provenance['evaluation_environment']:raise ValueError('Dataset and checkpoint evaluation protocol mismatch')
    if provenance and dataset_meta.get('schema')!='giveway_si_dataset_v1':raise ValueError('SI initial states required')
    from dataclasses import replace
    config=replace(config,wall_layout_scale=args.wall_layout_scale,progress_window_seconds=args.progress_window_seconds,
                   deadlock_hold_seconds=args.deadlock_hold_seconds,progress_epsilon=args.progress_epsilon,
                   speed_epsilon_fraction=args.speed_epsilon_fraction,terminate_on_collision=True)
    if args.fixed_horizon:
        config=replace(config,terminate_on_success=False,terminate_on_deadlock=False)
    factory,filter_source=filter_factory(args.action_filter)
    out=args.out_dir or ROOT/'single_integrator'/('eval_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Evaluation output must be empty')
    runtime_sources={'policy':Path(inspect.getfile(GiveWayFlowBCAgent)), 'networks':Path(inspect.getfile(importlib.import_module('utils.networks'))), 'flax_utils':Path(inspect.getfile(importlib.import_module('utils.flax_utils'))), 'evaluator':Path(__file__), 'outcomes':ROOT/'single_integrator/outcomes.py'}
    runtime_code_sha256={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in runtime_sources.items()}
    metadata=dict(outcome_protocol=PROTOCOL,runtime_code_sha256=runtime_code_sha256,method='stage_i_flow_bc' if provenance else 'legacy_checkpoint_diagnostic',checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),environment=config.to_dict(),environment_fingerprint=config.fingerprint,seed=args.seed,split=args.split,n_rollouts=args.n_rollouts,action_filter=args.action_filter,filter_source=filter_source,policy_provenance=provenance,rng_protocol='fold_in(fold_in(PRNGKey(seed), rollout_id), step)',collision_protocol='swept geometry, no collision response; endpoints also logged',plant_source_sha256=hashlib.sha256((ROOT/'single_integrator/environment.py').read_bytes()).hexdigest(),c1_deadlock_geometry=(dict(config=c1_geometry_config.to_dict(),cbf=c1_cbf_config.to_dict(),status='geometry_only_no_cone_distance_or_risk') if args.c1_deadlock_geometry else None))
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    rng=np.random.default_rng(args.seed);summaries=[]
    for rid in range(args.n_rollouts):
        pair=-1
        if args.split=='random':initial=sample_initial_positions(rng)
        else:
            pair=(200 if args.split=='val' else 225)+rid%25
            with np.load(args.dataset/f'raw/episode_{pair*2:04d}.npz') as z:initial=z['initial_positions'].copy()
        action_filter=factory(config.to_dict()) if factory else None
        summary,arrays=rollout(policy,initial,config,args.seed,rid,action_filter,cbf_config=c1_cbf_config,c1_geometry_config=c1_geometry_config)
        summary['pair_id']=pair;summaries.append(summary)
        np.savez_compressed(out/f'rollout_{rid:04d}.npz',**arrays,initial_positions=initial,pair_id=pair,environment_fingerprint=config.fingerprint)
        plot_rollout(out/f'rollout_{rid:04d}.png',arrays,initial,config,f"SI rollout {rid}: {summary['termination']}")
        print(f"rollout={rid} steps={summary['episode_steps']} success={summary['success']} walls={summary['wall_collision']} agents={summary['agent_collision']} deadlock={summary['deadlock']}",flush=True)
    aggregate={k+'_rate':float(np.mean([s[k] for s in summaries])) for k in ['success','collision_free_success','wall_collision','agent_collision','deadlock']}
    aggregate.update(n_deadlock=sum(s['deadlock'] for s in summaries),collision_termination_rate=float(np.mean([s['termination']=='collision' for s in summaries])),n_rollouts=len(summaries),timeout_rate=float(np.mean([s['termination']=='timeout' for s in summaries])),max_tracking_error=max(s['max_tracking_error'] for s in summaries),max_integration_residual=max(s['max_integration_residual'] for s in summaries))
    aggregate.update(aggregate_outcomes(summaries))
    (out/'summary.json').write_text(json.dumps(dict(aggregate=aggregate,rollouts=summaries),indent=2))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
