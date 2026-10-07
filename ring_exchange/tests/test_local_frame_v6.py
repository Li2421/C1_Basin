import math
from types import SimpleNamespace
import numpy as np
from ring_exchange.environment import LocalFrameConfig, RingExchangeEnv, sample_initial_state
from ring_exchange.expert import CentralizedExpert, CirculationHypothesis
from ring_exchange.expert_v6 import LocalFrameCentralizedExpert
from ring_exchange.local_frame import local_actions_to_world, local_observation, world_actions_to_local
from ring_exchange.protocol_v6 import RingExchangeLocalFrameScenario


def _rotate(values, angle):
    matrix=np.array(((math.cos(angle),-math.sin(angle)),(math.sin(angle),math.cos(angle))))
    return np.asarray(values)@matrix.T


def test_local_v6_has_new_fingerprint_and_rotation_invariant_physical_observation():
    cfg=LocalFrameConfig()
    try:
        LocalFrameConfig(schema="ring_exchange_single_integrator_v1")
    except ValueError as error:
        assert "requires the v2 local-frame schema" in str(error)
    else:
        raise AssertionError("local representation must not share the v1 fingerprint")
    instance=sample_initial_state("development",7,cfg); angle=.91
    original=local_observation(instance.positions,instance.velocities,instance.goals,cfg)
    rotated=local_observation(_rotate(instance.positions,angle),_rotate(instance.velocities,angle),_rotate(instance.goals,angle),cfg)
    assert original.shape==(4,23)
    assert np.allclose(original,rotated,atol=2e-6)


def test_local_action_round_trip_and_expert_encoding_are_physical():
    cfg=LocalFrameConfig(); instance=sample_initial_state("train",5,cfg)
    env=RingExchangeEnv(cfg,instance); world=CentralizedExpert().plan_hypothesis(env,CirculationHypothesis("cw"))
    local=LocalFrameCentralizedExpert().plan_hypothesis(RingExchangeEnv(cfg,instance),CirculationHypothesis("cw"))
    assert local.success and local.actions.shape==world.actions.shape
    decoded=np.stack([local_actions_to_world(local.actions[t],local.positions[t]) for t in range(len(local.actions))])
    assert np.allclose(decoded,world.actions,atol=2e-8)
    assert np.allclose(world_actions_to_local(world.actions[3],world.positions[3]),local.actions[3],atol=2e-8)


def test_v6_recovery_restores_world_velocity_from_local_label():
    scenario=RingExchangeLocalFrameScenario(); initial=scenario.sample_initial_state("train",np.random.default_rng(3))
    continuation=scenario.expert(initial,np.random.default_rng(4)); assert continuation.success
    trajectory=SimpleNamespace(states=continuation.states,actions=continuation.actions,initial_state=initial,split="train")
    state=scenario.recovery_state_at(trajectory,4)
    expected=local_actions_to_world(continuation.actions[3],continuation.states[3])
    assert np.allclose(state["velocities"],expected)
    assert scenario.valid_state(state)
