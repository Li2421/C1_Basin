import numpy as np
from four_way_intersection.environment import Config
from four_way_intersection.lane_local import FourWayLaneLocalEnv, local_to_world, representation_fingerprint, world_to_local
from four_way_intersection.lane_local_protocol import FourWayLaneLocalScenario

def test_action_rotation_is_orthonormal_and_round_trips():
    world=np.asarray(((.2,-.3),(.4,.1),(-.1,.5),(.3,-.2)))
    local=world_to_local(world)
    np.testing.assert_allclose(local_to_world(local),world,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(local,axis=1),np.linalg.norm(world,axis=1),atol=1e-14)

def test_goal_is_forward_positive_and_local_env_keeps_world_dynamics():
    env=FourWayLaneLocalEnv();obs=env.observation()
    assert obs.shape==(4,18)
    assert np.all(obs[:,4]>0)  # goal displacement forward coordinate
    start=env.positions.copy();local=np.asarray(((.3,.1),(.3,.1),(.3,.1),(.3,.1)))
    env.step(local_to_world(local));np.testing.assert_allclose(env.positions-start,env.config.dt*local_to_world(local),atol=1e-14)

def test_expert_data_use_local_actions_and_separate_fingerprint():
    scenario=FourWayLaneLocalScenario(); state=scenario.sample_initial_state("train",np.random.default_rng(3)); continuation=scenario.expert(state,np.random.default_rng(4))
    assert continuation.success and continuation.observations.shape[1:]==(4,18)
    np.testing.assert_allclose(np.diff(continuation.states,axis=0),scenario.config.dt*local_to_world(continuation.actions),atol=3e-7)
    assert scenario.environment_fingerprint==representation_fingerprint(Config())
    assert scenario.environment_fingerprint != Config().fingerprint
