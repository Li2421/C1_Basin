"""Development-only local-action audit for a trained Ring Stage-I checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
import numpy as np
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import RingExchangeEnv
from .protocol import RingExchangeScenario


def _bounded(agent, obs, key, maximum):
    raw = np.asarray(agent.sample_actions(obs[None], key)[0], dtype=np.float64)
    bounded = np.asarray(sample_bounded_actions(agent, obs[None], key)[0], dtype=np.float64)
    norm = np.linalg.norm(bounded, axis=-1, keepdims=True)
    bounded *= np.minimum(1., (maximum - 1e-8) / np.maximum(norm, 1e-12))
    return raw, bounded


def _phase(position, goal, expert_action, tolerance):
    if np.linalg.norm(goal - position) <= tolerance:
        return "goal_wait"
    radial = position / max(np.linalg.norm(position), 1e-12)
    tangent = np.array((-radial[1], radial[0]))
    ur, ut = float(expert_action @ radial), float(expert_action @ tangent)
    if abs(ut) >= abs(ur): return "arc"
    return "entry_radial" if ur < 0 else "exit_radial"


def audit(dataset, checkpoint, output, *, seed=451, stride=3):
    dataset, checkpoint, output = Path(dataset), Path(checkpoint), Path(output)
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((dataset/'manifest.json').read_text())
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=manifest['environment_fingerprint'])
    scenario=RingExchangeScenario()
    rows=[r for r in manifest['files'] if r['split']=='dev' and r['source']=='nominal']
    data=[]; outcomes=[]; z_values=[]; raw_norm=[]; component_clip=[]
    for case,row in enumerate(rows):
        with np.load(dataset/row['file'],allow_pickle=False) as d: initial=json.loads(str(d['initial_state_json'].item()))
        env=RingExchangeEnv(); env.reset(initial['positions'],velocities=initial['velocities'],goals=initial['goals'])
        terminal='timeout'
        for step in range(env.config.max_steps):
            obs=env.observation(); key=jax.random.fold_in(jax.random.PRNGKey(seed+case),step)
            raw,action=_bounded(agent,obs,key,env.config.max_speed)
            raw_norm.extend(np.linalg.norm(raw,axis=-1)); component_clip.extend(np.abs(raw).reshape(-1)>=.999)
            # Requery only valid pre-collision development policy states.
            if step % stride == 0:
                state={'positions':env.positions.copy(),'velocities':env.velocities.copy(),'goals':env.goals.copy(),
                       'split':'dev','recovery':step>0}
                expert=scenario.expert(state,np.random.default_rng(seed+case*10000+step))
                if expert.success:
                    target=np.asarray(expert.actions[0],dtype=np.float64)
                    zero=RingExchangeEnv(); zero.reset(env.positions,velocities=np.zeros((4,2)),goals=env.goals)
                    _,without_v=_bounded(agent,zero.observation(),key,env.config.max_speed)
                    flat=obs.reshape(4,23); indices=(2,3,8,9,12,13,16,17)
                    z=(flat[:,indices].reshape(-1)-np.asarray(agent.config['obs_mean']).reshape(4,23)[:,indices].reshape(-1))/np.asarray(agent.config['obs_scale']).reshape(4,23)[:,indices].reshape(-1)
                    z_values.extend(np.abs(z))
                    for i in range(4):
                        radial=env.positions[i]/max(np.linalg.norm(env.positions[i]),1e-12); tangent=np.array((-radial[1],radial[0]))
                        delta=action[i]-target[i]
                        data.append({'phase':_phase(env.positions[i],env.goals[i],target[i],env.config.goal_tolerance),
                                     'squared_error':float(np.mean(delta*delta)), 'radial_error':float(delta@radial),
                                     'tangent_error':float(delta@tangent), 'expert_radial':float(target[i]@radial),
                                     'expert_tangent':float(target[i]@tangent), 'policy_radial':float(action[i]@radial),
                                     'policy_tangent':float(action[i]@tangent),
                                     'velocity_ablation_delta':float(np.linalg.norm(action[i]-without_v[i]))})
            _,_,done,info=env.step(action); terminal=str(info['termination'])
            if done: break
        outcomes.append(terminal)
    summary={}
    for phase in ('entry_radial','arc','exit_radial','goal_wait'):
        x=[r for r in data if r['phase']==phase]
        summary[phase]={'agent_samples':len(x),'rmse_to_local_expert':float(np.sqrt(np.mean([r['squared_error'] for r in x]))) if x else None,
                        'mean_radial_error':float(np.mean([r['radial_error'] for r in x])) if x else None,
                        'mean_tangent_error':float(np.mean([r['tangent_error'] for r in x])) if x else None,
                        'mean_policy_radial':float(np.mean([r['policy_radial'] for r in x])) if x else None,
                        'mean_expert_radial':float(np.mean([r['expert_radial'] for r in x])) if x else None,
                        'mean_policy_tangent':float(np.mean([r['policy_tangent'] for r in x])) if x else None,
                        'mean_expert_tangent':float(np.mean([r['expert_tangent'] for r in x])) if x else None,
                        'velocity_zero_action_delta':float(np.mean([r['velocity_ablation_delta'] for r in x])) if x else None}
    result={'scope':'development nominal policy states only; no test loaded or evaluated','checkpoint':str(checkpoint),'cases':len(rows),'stride':stride,
            'terminations':{k:outcomes.count(k) for k in sorted(set(outcomes))},'phase_comparison':summary,
            'sampling_and_normalization':{'raw_speed_gt_environment_limit':float(np.mean(np.asarray(raw_norm)>0.52)),
                                          'raw_component_at_internal_clip':float(np.mean(component_clip)),
                                          'velocity_feature_abs_z_p99':float(np.quantile(z_values,.99)),
                                          'velocity_feature_abs_z_max':float(np.max(z_values)),
                                          'velocity_feature_abs_z_gt3':float(np.mean(np.asarray(z_values)>3))}}
    output.mkdir(parents=True); (output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--stride',type=int,default=3)
    a=p.parse_args();print(json.dumps(audit(a.dataset,a.checkpoint,a.output,stride=a.stride),indent=2))
