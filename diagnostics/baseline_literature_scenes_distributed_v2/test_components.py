"""Required algebra, locality, basis, N1 and unchanged-plant checks."""
from experiment import *
from dataclasses import replace
from flow_scene import LiteratureEnv
from legacy_flow_scene import LiteratureEnv as OriginalEnv
from states import config
from distributed_safety import LocalSafety,local_basis,project_independent
from projection_memo import MemoSafetyFilter
from projection_adapter import constraints
from nominal_adapter import EmptyWallControllerView
from controllers import wall_penalties,bound
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig
import shared_rollout_db
from shared_rollout_db.src import cache_writer
import jax
import numpy as np

def main():
    prereg();rng=np.random.default_rng(2026102101);rows=[]
    assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
    assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
    jax.config.update('jax_enable_x64',False)
    assert not jax.config.x64_enabled
    for scene in NAMES:
        state=read(HERE/f'FLOW_STATES_{scene}.json')[0]
        solo=dict(state,N=1,positions=state['positions'][:1],goals=state['goals'][:1],velocities=[[0.,0.]])
        cfg=replace(config(scene,state['N']),num_agents=1)
        env=LiteratureEnv(solo,cfg);snap=env.snapshot();central=MemoSafetyFilter();maximum=0.
        for _ in range(24):
            target=rng.normal(size=(1,2))*cfg.max_speed*3
            left=np.asarray(central(snap,target).velocity)
            right=LocalSafety(snap,0).project(target)[0]
            assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes(),scene
            maximum=max(maximum,float(np.max(abs(left-right))))
        rows.append(dict(test='N1_exact_projection',scene=scene,targets=24,max_absolute_error=maximum,byte_equal=True))
        full=LiteratureEnv(state);snap=full.snapshot();A,lower,kinds=constraints(snap,HardProjectionConfig())
        for coefficient in (.5,1.):
            filters=[LocalSafety(snap,i,coefficient=coefficient,problem=(A,lower,kinds)) for i in range(state['N'])]
            for k,kind in enumerate(kinds):
                if kind[0]!='pair':continue
                i,j=kind[1:];parts=[]
                for robot in (i,j):
                    ix=filters[robot].kinds.index(kind)
                    np.testing.assert_array_equal(filters[robot].A[ix],A[k,2*robot:2*robot+2])
                    parts.append(filters[robot].lower[ix])
                assert parts[0]+parts[1]==2*coefficient*lower[k]
        target=rng.normal(size=(state['N'],2))*full.config.max_speed
        original,_,_=project_independent(snap,target)
        permutation=np.arange(state['N'])[::-1]
        permuted=dict(snap,positions=np.asarray(snap['positions'])[permutation])
        other,_,_=project_independent(permuted,target[permutation])
        error=float(np.max(abs(original[permutation]-other)))
        assert error<=2e-6,('permutation',scene,error)
        changed=target.copy();changed[1:]*=-7
        independent,_,_=project_independent(snap,changed)
        np.testing.assert_array_equal(original[0],independent[0])
        rows.append(dict(test='pair_algebra_permutation_locality',scene=scene,permutation_max_error=error,coefficient_checks=[.5,1.]))
    native=get_basis_family('orthoflow3');max_basis=0.
    for n in (2,3,5,14,100):
        p=rng.normal(size=(n,2));g=rng.normal(size=(n,2));v=rng.normal(size=(n,2))*.3
        reference=native.compute(p,g,v,.7).values
        for i in range(n):
            own=local_basis(p,g,v,i,np.inf,.7)
            for k in range(3):
                error=float(np.max(abs(own[k]-reference[k][i])));max_basis=max(max_basis,error)
                assert error<=2e-14,(n,i,k,error)
            altered=v.copy();altered[np.arange(n)!=i]*=100
            np.testing.assert_array_equal(own,local_basis(p,g,altered,i,np.inf,.7))
    rows.append(dict(test='local_basis_native_components_and_own_safe_only',N=[2,3,5,14,100],max_absolute_error=max_basis,tolerance=2e-14))
    assert LiteratureEnv.step is OriginalEnv.step and LiteratureEnv.reset is OriginalEnv.reset
    collision_checks=0
    for scene in ('lit_narrow_passage','lit_doorway','lit_circle_five'):
        state=read(HERE/f'FLOW_STATES_{scene}.json')[0];cfg=config(scene,state['N'])
        old=OriginalEnv(state,cfg);new=LiteratureEnv(state,cfg)
        assert old.walls.tobytes()==new.walls.tobytes()
        for _ in range(100):
            target=bound(old.positions.mean(0)-old.positions,cfg.max_speed)
            _,done,a=old.step(target,diagnose_stalls=False);_,done2,b=new.step(target,diagnose_stalls=False)
            assert done==done2 and a['termination']==b['termination'] and a['collision']==b['collision']
            assert a['swept_clearance']==b['swept_clearance']
            assert old.positions.tobytes()==new.positions.tobytes()
            if done:collision_checks+=bool(a['collision']);break
    assert collision_checks>=1
    dense=LiteratureEnv(read(HERE/'FLOW_STATES_lit_crowded_random.json')[0]);view=EmptyWallControllerView(dense)
    points=rng.normal(size=(4,9,3,2))*1e3
    inner,outer=wall_penalties(points,view)
    assert not len(dense.walls) and not np.any(inner) and not np.any(outer)
    rows.append(dict(test='native_reset_step_and_swept_events_unchanged',collisions_exercised=collision_checks,empty_wall_cost_exact_zero=True))
    write(HERE/'UNIT_TESTS.json',dict(utc=utc(),passed=True,tests=rows,jax_enable_x64=bool(jax.config.x64_enabled),
        db_module=str(Path(shared_rollout_db.__file__).resolve()),journal_root=str(cache_writer.ROOT/'journals')))
    seal('UNIT_TESTS.receipt.json',[HERE/'UNIT_TESTS.json',HERE/'test_components.py',HERE/'distributed_safety.py',HERE/'flow_scene.py',HERE/'nominal_adapter.py'],all_required_component_checks_passed=True)
    print('All component tests passed',len(rows),'groups',flush=True)

if __name__=='__main__':main()
