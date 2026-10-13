"""Read-only audit before extending controller mechanisms across environments."""
import json
from pathlib import Path
import numpy as np
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,FIELD
from diagnostics.controller_q_diversity_v1.experiment import write,sha
from diagnostics.physical_state_critic_formal_v1.physical_entities import templates

HERE=Path(__file__).resolve().parent

def main():
    rows=[];t=templates()
    for scene in ['double_bottleneck','four_way_intersection']:
        rt=native_runtime(scene);source=HERE.parent/'physical_state_critic_formal_v1'/f'states_{scene}.json'
        d=json.loads(source.read_text());rng=np.random.default_rng(2026101106)
        chosen=[]
        for split,n in [('train',32),('validation',8)]:
            pool=[s for s in d['states'] if s['split']==split]
            for idx in sorted(rng.choice(len(pool),n,replace=False)):
                s=pool[idx];env=rt.make_env(s)
                np.testing.assert_allclose(env.positions,s['physical']['positions'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(env.velocities,s['physical']['velocities'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(env.goals,s['physical']['goals'],rtol=0,atol=1e-6)
                chosen.append(s)
        cfg=rt.config.to_dict() if hasattr(rt.config,'to_dict') else vars(rt.config)
        from importlib import import_module
        layout=import_module(scene+'.scenario').build_layout(rt.config)
        actual=[dict(a=np.asarray(w[0]).tolist(),b=np.asarray(w[1]).tolist(),radius=float(cfg.get('wall_radius',0.)),curvature=0.,interior_sign=0.) for w in layout.walls]
        map_equal=(actual==t[scene]['obstacles'])
        radii=all(x==rt.config.agent_radius for x in t[scene]['radius'])
        tolerances=all(x==rt.config.goal_tolerance for x in t[scene]['goal_tolerance'])
        write(HERE/f'candidate_states_{scene}.json',dict(states=chosen,source_file=str(source),source_sha256=sha(source),outcome_blind_uniform_subset=True))
        rows.append(dict(scene=scene,config=cfg,cbf=rt.cbf.to_dict(),checkpoint=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint),
            FF_steps=rt.steps,act_mean=rt.mean.tolist(),act_scale=rt.scale.tolist(),map_adapter_matches=map_equal,radius_matches=radii,goal_tolerance_matches=tolerances,
            selected_states=len(chosen),physical_steps=0))
    write(HERE/'four_scene_runtime_audit.json',dict(rows=rows,new_rollouts=0,
        legacy_LOSO_runtime='run_frozen_test.FrozenNewRuntime.rollout computes project(project(flow)+g_eta). It is a terminal intervention, despite FF in later launcher names.',
        current_runtime='poc.field_controller.Runtime.integrate injects g_eta into each of10pseudo-time vector-field steps with projection.',
        consequence='Legacy terminal-intervention outcome labels are incompatible with actual field-intervention Q labels. Physical state reuse is still valid after exact native reset/config audit. This semantic distinction does not by itself explain failure between two current correctly-matched FF scenes.'))
    print(json.dumps([{k:v for k,v in r.items() if k in ['scene','map_adapter_matches','radius_matches','goal_tolerance_matches','selected_states']} for r in rows]),flush=True)

if __name__=='__main__':main()
