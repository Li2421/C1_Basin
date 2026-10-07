import numpy as np
from four_way_intersection.environment import Config, FourWayIntersectionEnv
from four_way_intersection.expert import CentralizedExpert, CrossingOrder, all_crossing_orders
from four_way_intersection.scenario import sample_initial_state
from four_way_intersection.rollout import crossing_order_signature

def test_open_geometry_and_observation_schema():
    env=FourWayIntersectionEnv(); assert env.observation().shape==(4,18)
    # The central conflict area is free: an agent at origin has large boundary clearance.
    w,_=env.distances(np.zeros((4,2))+np.array(((0,0),(1,1),(-1,-1),(1,-1))))
    assert w.min()>3

def test_all_order_hypotheses_are_available_and_two_modes_succeed():
    env=FourWayIntersectionEnv(); expert=CentralizedExpert(); assert len(all_crossing_orders())==24
    a=expert.plan_hypothesis(env,CrossingOrder((0,2,1,3)))
    b=expert.plan_hypothesis(env,CrossingOrder((1,3,0,2)))
    assert a.success and b.success and not a.collision and not b.collision
    assert crossing_order_signature(a.positions) != crossing_order_signature(b.positions)

def test_broad_split_samples_and_expert_pilot():
    cfg=Config();expert=CentralizedExpert(); starts=[]
    for k in range(30):
        p,v,_=sample_initial_state(cfg,"development",k);starts.append(p);env=FourWayIntersectionEnv(cfg);env.reset(p,v);plan=expert.plan(env)
        assert plan.success and not plan.collision and plan.episode_steps<cfg.max_steps
        assert plan.actions.shape==(plan.episode_steps,4,2)
        np.testing.assert_allclose(np.diff(plan.positions,axis=0),cfg.dt*plan.actions,atol=3e-14)
    assert np.std(np.asarray(starts)[:,:,0])>.15

def test_hard_safety_smoke_has_all_six_pairwise_rows():
    from four_way_intersection.stage1_runner import hard_safety_smoke
    result=hard_safety_smoke()
    assert result["num_pair_constraints"] == 6
    assert len(result["pair_indices"]) == 6

def test_recovery_goal_agent_is_held_and_removed_from_crossing_schedule():
    env=FourWayIntersectionEnv()
    positions=env.positions.copy(); positions[0]=env.goals[0]
    env.reset(positions)
    plan=CentralizedExpert().plan(env)
    assert plan.success
    np.testing.assert_allclose(plan.actions[:,0],0.,atol=1e-14)
    np.testing.assert_allclose(plan.positions[:,0],np.repeat(env.goals[0][None],len(plan.positions),axis=0),atol=1e-14)
