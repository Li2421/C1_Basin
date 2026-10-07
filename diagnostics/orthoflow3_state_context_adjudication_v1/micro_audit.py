"""Source-only truncated physical probes. Never label task success or update DB."""
import argparse
import hashlib
import time
import numpy as np
import jax
from .audit import OUT, SRC, read, write
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20


def main(controller):
    jax.config.update('jax_enable_x64',False)
    protocol=read(OUT/'micro_protocol.json')
    states=read(SRC/'states.json')
    physical={r['state_uid']:r['physical'] for r in read(SRC/'physical.json')}
    chosen=sorted([s for s in states if s['split']=='train'],key=lambda s:hashlib.sha256(s['source_group'].encode()).hexdigest())[:4]
    pairs=read(SRC/'pairs.json')
    profile=next(r for r in read(SRC/'protocol.json')['profiles'] if r['name']==controller)
    rt=h20.rc.RichRuntime('ring_exchange',profile['path'])
    core=rt.core;cfg=core.cfg
    radius=cfg.agent_radius
    results=[];started=time.time()
    for state in chosen:
        candidates=[r for r in pairs if r['state_uid']==state['uid']]
        for ei in protocol['eta_indices']:
            candidate=candidates[ei];eta=np.array(candidate['eta'])
            env=core.reset(physical[state['uid']]);trajectory=[]
            for t in range(protocol['max_steps']):
                key=jax.random.fold_in(jax.random.PRNGKey(protocol['probe_root']),t)
                flow=(rt.base_flow if t==0 or rt.alt_flow is None else rt.alt_flow)(env,key)
                safe=core.project(env,flow)
                corrected=safe+core.basis.compute(env.positions,env.goals,safe,cfg.max_speed).correction(eta)
                executed=safe if np.array_equal(eta,np.zeros(3)) else core.project(env,corrected)
                positions=env.positions
                pair=min(np.linalg.norm(positions[i]-positions[j])-2*radius for i in range(len(positions)) for j in range(i))
                rad=np.linalg.norm(positions,axis=1)
                obstacle_gap=min(float((rad-cfg.obstacle_radius-radius).min()),float((cfg.outer_radius-rad-radius).min()))
                row={'step':t,'seconds':t*cfg.dt,'min_pair_surface_gap':float(pair),
                     'min_obstacle_surface_gap':obstacle_gap,
                     'nominal_safety_L2_over_speed':float(np.linalg.norm(safe-flow)/cfg.max_speed),
                     'corrected_projection_L2_over_speed':float(np.linalg.norm(executed-corrected)/cfg.max_speed),
                     'goal_distance_mean':float(np.linalg.norm(env.goals-env.positions,axis=1).mean())}
                trajectory.append(row)
                env.step(executed)
                if env.done: break
            def first(predicate):
                hits=[r['step'] for r in trajectory if predicate(r)]
                return hits[0] if hits else None
            results.append({'state_uid':state['uid'],'eta_uid':candidate['eta_uid'],'eta_index':ei,
                'controller':controller,'checkpoint_sha256':profile['sha256'],
                'first_material_safety_step':first(lambda r:max(r['nominal_safety_L2_over_speed'],r['corrected_projection_L2_over_speed'])>=.01),
                'first_pair_gap_under_half_meter_step':first(lambda r:r['min_pair_surface_gap']<=.5),
                'trajectory':trajectory})
            print(controller,len(results),'steps',len(trajectory),flush=True)
    write(f'micro_{controller}.json',{'protocol':protocol,'new_task_continuations':0,
         'physical_steps':sum(len(r['trajectory']) for r in results),'elapsed_seconds':time.time()-started,'rows':results})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controller',required=True)
    main(parser.parse_args().controller)
