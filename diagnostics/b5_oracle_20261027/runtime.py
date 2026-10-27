"""Read-only adapter to the frozen hetero-set production runtime."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAP = ROOT / 'diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027'
H = SNAP / 'scene_search/harness'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(H))
import bootstrap
import runtime_new as RN
from shared_control.correction_b5 import FrozenYield, YieldConfig, tangent, target


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def dump(path, x):
    Path(path).write_text(json.dumps(clean(x), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


# The sole inserted statement is after the original B3 construction and before
# the original second safety projection. Every other line is the production source.
SOURCE = inspect.getsource(RN.run)
ANCHOR = '            second=[f.project(target[i:i+1])'
assert SOURCE.count(ANCHOR) == 1
ADAPTED = SOURCE.replace('def run(c):', 'def run(c, hook):').replace(
    ANCHOR, '            target=hook(env,v,filters,t,eta,target)\n'+ANCHOR)
namespace = dict(vars(RN))
exec(compile(ADAPTED, str(OUT/'production_hook_generated.py'), 'exec'), namespace)
run_hook = namespace['run']


def make_hook(spec):
    role = None
    history = {}
    def hook(env, safe, filters, t, eta3, original):
        nonlocal role
        mode = spec['mode']
        if mode == 'b3': return original
        if mode.startswith('rhr_'):
            d = np.linalg.norm(env.goals-env.positions, axis=1)
            history[t] = d.copy(); history.pop(t-41, None)
            active = (history[t-40]-d < .05)&(d>env.config.goal_tolerance) if t>=40 else np.zeros(len(d),bool)
            if not active.any(): return original
            # Match historical RHR: safety-only second projection precedes perturbation.
            out = np.concatenate([f.project(original[i:i+1])[0] for i,f in enumerate(filters)])
            q = env.goals-env.positions if mode=='rhr_goal' else env.positions
            for i in np.flatnonzero(active):
                out[i] += .35*np.array([q[i,1],-q[i,0]])/max(np.linalg.norm(q[i]),1e-30)
            return out
        cfg = env.config
        eta = np.asarray(spec['eta'], float).copy()
        if spec.get('schedule') == 'release_third' and t >= cfg.max_steps//3:
            eta = np.array([*spec['release_eta'], 0., 0.])
        if role is None and spec.get('role'):
            role = FrozenYield(YieldConfig(**spec['role']), env.positions, env.goals,
                               RN.contact_radius(cfg)/1.25)
        fields = np.stack([RN.distributed_fields(env.positions, env.goals, safe, i,
                          cfg.max_speed, cfg.goal_tolerance, RN.contact_radius(cfg)) for i in range(env.num_agents)])
        tang = tangent(spec.get('recipe','goal'),env.positions,fields[:,0],cfg.max_speed,
                       RN.contact_radius(cfg),filters)
        return target(safe,fields,tang,role,env.positions,cfg.max_speed,eta,mode,
                      tuple(spec.get('keep',[0,1,2,3,4])))
    return hook


def make_case(scene, controller, state_index, spec, seed=0):
    from shared_rollout_db.src.rollout_db import uid, eta_identity, canonical
    state = scene['states'][state_index]
    scenario = dict(version='cross_scene_si_v1', config=scene['config'], walls=scene['walls'],
        bottleneck_config=scene.get('bottleneck_config'),factor=scene['factor'],
        plant='frozen_swap_step',scene_definition_sha256=scene['definition_sha256'])
    scid = uid('scenario', scenario)
    physical = {k:state[k] for k in ('positions','velocities','goals')}
    sid = uid('state',dict(scenario=scid,physical=physical))
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT/'shared_control/correction_b5.py', OUT/'runtime.py',
                         ROOT/'shared_control/hard_projection.py']}
    # All frozen backend Python files included, including plant and nominal dependencies.
    sources['frozen_backend_tree'] = digest({str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(H.rglob('*.py'))})
    payload = dict(nominal=controller, b5_spec=spec, execution_sources=sources,
        safety_layer='distributed',precision='jax_x64_false',rng='2026102004/state_uid/seed/t',
        correction_semantics='unchanged_contact_gate_plus_opt_in_b5_v1')
    cid = uid('ctl',dict(scenario=scid,payload=payload))
    eta3 = list(spec['eta'][:3]); eid=eta_identity(eta3)[0]; sk=canonical(dict(future_index=seed))
    return dict(state=state,scenario_payload=scenario,controller_payload=payload,
        scenario_uid=scid,state_uid=sid,controller_uid=cid,eta_uid=eid,seed_key=sk,
        rollout_uid=uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk)),
        seed=seed,eta_a=eta3,eta_b=spec.get('eta_b',eta3),theta=spec.get('theta'),
        scene=scene['name'],state_index=state_index,spec=spec)


def run(case):
    result, arrays = run_hook(case,make_hook(case['spec']))
    cfg = case['scenario_payload']['config']; p=arrays['positions']; goal=arrays['goals']
    distances=np.linalg.norm(goal[None]-p,axis=-1).sum(-1)
    # Explicit diagnostic proxy, not an additional success/deadlock predicate.
    reached=np.flatnonzero(distances <= distances[0]-.1)
    result.update(local_progress_time=float(reached[0]*cfg['dt']) if len(reached) else None,
                  completion_time=result['episode_length']*cfg['dt'] if result['success'] else None,
                  truncated_time=result['episode_length']*cfg['dt'] if result['success'] else cfg['max_steps']*cfg['dt'],
                  horizon_time=cfg['max_steps']*cfg['dt'])
    return clean(result), arrays
