"""Static new-name adapters; numerical loop and old-scene branches remain exact."""
from common import *
import difflib,ast

def main():
    assert read(HERE/'MODEL_FROZEN.receipt.json')['test_rollouts_before_freeze']==0
    rows=[]
    def derive(src,name,edits):
        before=src.read_text();after=before
        for a,b in edits:
            assert after.count(a)==1,(name,a);after=after.replace(a,b)
        inverse=after
        for a,b in reversed(edits):
            assert inverse.count(b)==1;inverse=inverse.replace(b,a)
        assert inverse==before;ast.parse(after)
        target=HERE/name;target.write_text(after)
        target.with_suffix('.py.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(src),tofile=str(target))))
        rows.append(dict(source=str(src),source_sha256=sha(src),target=str(target),target_sha256=sha(target),
            inverse_transform_exact=True,edits=edits))
    derive(HERE/'controllers_new.py','controllers_prospective.py',[("scene.startswith('xsc_')","scene.startswith(('xsc_','xsp_'))")])
    derive(HERE/'scene_environment.py','prospective_environment.py',[
        ('from controllers_new import Nominal as GeometryNominal','from controllers_prospective import Nominal as GeometryNominal'),
        ("env.config.scene.startswith('xsc_')","env.config.scene.startswith(('xsc_','xsp_'))")])
    for source,target in [('runtime_new.py','runtime_prospective.py'),('runtime_checkpointed.py','runtime_prospective_checkpointed.py')]:
        derive(HERE/source,target,[('from scene_environment import NewEnv as SwapEnv','from prospective_environment import NewEnv as SwapEnv'),
            ('from scene_environment import NewNominal as Nominal','from prospective_environment import NewNominal as Nominal')])
    derive(HERE/'checkpoint_store.py','prospective_checkpoint_store.py',[
        ("('runtime_new.py','runtime_checkpointed.py','checkpoint_store.py')",
         "('runtime_prospective.py','runtime_prospective_checkpointed.py','prospective_checkpoint_store.py')")])
    write(HERE/'PROSPECTIVE_EXTENSION_BUILD.json',dict(passed=True,files=rows,
        scope='Only recognize xsp_ names using existing generic fixed-wall algorithms; environment/loop/physics/RNG/parameters unchanged.',
        checkpoint_numerics='Exact copied historical snapshot implementation and loop; cache-root relocation only in native code.'))
    # A structural restoration check uses only initial definitions, no physical steps.
    import pickle,numpy as np
    from prospective_environment import NewEnv,NewNominal
    from legacy_states import LiteratureConfig
    pool=read(V2/'LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json')['training_controllers'];checks=[]
    for scene in read(HERE/'SCENES.json'):
        if scene['family']!='bottleneck' or not scene['eligible']:continue
        for ctl in pool:
            if ctl['type_code']not in ('prioritized_astar','conflict_based_search'):continue
            env=NewEnv(scene['states'][0],LiteratureConfig(**scene['config']));actor=NewNominal(ctl)
            e,a=pickle.loads(pickle.dumps((env,actor),protocol=pickle.HIGHEST_PROTOCOL))
            for k in ('positions','velocities','goals','walls'):assert getattr(env,k).tobytes()==getattr(e,k).tobytes()
            assert canonical(actor.__dict__)==canonical(a.__dict__)
            checks.append(dict(scene=scene['name'],controller=ctl['controller'],initial_state_pickle_exact=True,physical_steps=0))
    write(HERE/'EXECUTION_CHECK.json',dict(passed=True,static_inverse_checks=rows,initial_serialization_checks=checks,
        historical_full_replay=sha(HERE/'REPLAY_CHECK.json'),new_test_physical_steps=0))

if __name__=='__main__':main()
