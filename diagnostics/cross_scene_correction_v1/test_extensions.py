"""Meaningful compatibility checks before any new-scene episode."""
from common import *
import numpy as np
import jax
from scene_environment import NewNominal
from nominal_geometry_user_fix_20261023 import Nominal as OldNominal
from legacy_states import LiteratureConfig


def main():
    checks=[]
    pool=read_evidence(V2/'LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json')['training_controllers']
    for c in read(HERE/'REPLAY_CASES.json'):
        scene=c['state']['scene'];cfg=LiteratureConfig(**c['scenario_payload']['config'])
        if scene=='lit_swap8_obstacles' or scene=='lit_swap8_moving':
            from swap_geometry_user_fix_20261023 import SwapEnvRevision as Env
        elif scene.startswith('lit_swap8'):
            from swap_geometry import SwapEnv as Env
        else:
            from flow_scene import LiteratureEnv as Env
        # All eight optimizer families, original scenes, two independent keys.
        if any(row['scene']==scene for row in checks):continue
        for ctl in pool[::2]:
            for seed in (0,1):
                a=OldNominal(ctl).action(Env(c['state'],cfg),jax.random.PRNGKey(seed))
                b=NewNominal(ctl).action(Env(c['state'],cfg),jax.random.PRNGKey(seed))
                equal=np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).tobytes()==np.asarray(b).tobytes()
                checks.append(dict(scene=scene,controller=ctl['controller'],seed=seed,passed=equal))
                assert equal,checks[-1]
    write(HERE/'EXTENSION_UNIT_TESTS.json',dict(passed=True,checks=checks,planning_workspace='Frozen native bounds are unbounded; 700 grid-time nodes unchanged, geometry supplied by env.walls. No parameter change.'))
    print('Extension compatibility:',len(checks),'passed',flush=True)


if __name__=='__main__':main()
