import numpy as np
from ring_exchange import RingExchangeEnv, sample_initial_state
from ring_exchange.expert import (CentralizedExpert, CirculationHypothesis, circulation_signature,
                                  infer_circulation_hypothesis)
from ring_exchange.protocol import RingExchangeScenario
from ring_exchange.safety import all_pairwise_and_obstacle_constraints, polygonal_obstacle_snapshot
from types import SimpleNamespace


def test_broad_independent_split_draws_are_safe():
    first = sample_initial_state("train", 9)
    second = sample_initial_state("development", 9)
    assert not np.allclose(first.positions, second.positions)
    for split in ("train", "development", "test"):
        for seed in range(10):
            env = RingExchangeEnv(instance=sample_initial_state(split, seed))
            wall, pair = env.distances()
            assert wall.min() > env.config.collision_margin
            assert pair.min() > 0
            assert env.observation().shape == (4, 23)


def test_both_circulation_hypotheses_are_valid_on_comparable_state():
    instance = sample_initial_state("development", 4)
    planner = CentralizedExpert()
    plans = []
    for direction in ("cw", "ccw"):
        plan = planner.plan_hypothesis(RingExchangeEnv(instance=instance), CirculationHypothesis(direction))
        assert plan.success and not plan.collision
        assert plan.actions.shape[1:] == (4, 2)
        assert plan.observations.shape[1:] == (4, 23)
        assert circulation_signature(plan.positions) == direction
        plans.append(plan)
    # Directional paths may have different durations; their first few motion
    # samples already establish topologically distinct realizations.
    assert not np.allclose(plans[0].positions[30], plans[1].positions[30])


def test_swept_obstacle_collision_is_detected():
    env = RingExchangeEnv(instance=sample_initial_state("train", 2))
    # A deliberately invalid inward command only serves to test swept event
    # semantics; normal expert starts/routes remain in the wide annulus.
    action = -env.positions / np.linalg.norm(env.positions, axis=-1, keepdims=True) * env.config.max_speed
    for _ in range(100):
        _, _, done, info = env.step(action)
        if done:
            break
    assert done and info["obstacle_collision"]


def test_hard_safety_snapshot_contains_central_and_outer_boundaries():
    env = RingExchangeEnv(instance=sample_initial_state("development", 7))
    snapshot = polygonal_obstacle_snapshot(env)
    assert len(snapshot["walls"]) == 96
    assert snapshot["wall_names"].count("central_obstacle") == 48
    assert snapshot["wall_names"].count("outer_boundary") == 48
    matrix, lower, geometry = all_pairwise_and_obstacle_constraints(env)
    assert matrix.shape == (6 + 4 * 96, 8)
    assert lower.shape == (6 + 4 * 96,)
    assert geometry["min_wall_h"] > 0


def test_protocol_preserves_goals_without_deploying_mode_label():
    scenario = RingExchangeScenario()
    initial = scenario.sample_initial_state("train", np.random.default_rng(37))
    recovery = scenario.perturb_state(initial, np.random.default_rng(38))
    assert np.array_equal(initial["goals"], recovery["goals"])
    assert "circulation" not in initial and "circulation" not in recovery
    continuation = scenario.expert(initial, np.random.default_rng(39))
    assert continuation.success
    assert continuation.observations.shape[1:] == (4, 23)
    assert continuation.actions.shape[1:] == (4, 2)
    trajectory = SimpleNamespace(states=continuation.states, actions=continuation.actions,
                                 initial_state=initial, split="train")
    restored = scenario.recovery_state_at(trajectory, 3)
    assert np.array_equal(restored["goals"], initial["goals"])
    assert np.allclose(restored["velocities"], continuation.actions[2])


def test_recovery_requery_continues_physically_visible_circulation():
    scenario = RingExchangeScenario()
    initial = scenario.sample_initial_state("train", np.random.default_rng(18))
    env = RingExchangeEnv()
    env.reset(initial["positions"], velocities=initial["velocities"], goals=initial["goals"])
    plan = CentralizedExpert().plan_hypothesis(env, CirculationHypothesis("cw"))
    trajectory = SimpleNamespace(states=plan.positions, actions=plan.actions,
                                 initial_state=initial, split="train")
    # This source is past radial entry and has a coherent CW tangent velocity.
    source = scenario.recovery_state_at(trajectory, 30)
    assert infer_circulation_hypothesis(source["positions"], source["velocities"]).direction == "cw"
    continuation = scenario.expert(source, np.random.default_rng(19))
    assert continuation.success
    assert continuation.metadata["circulation_mode"] == "cw"
    assert continuation.metadata["expert_mode_selection"] == "velocity_inferred_continuation"


def test_near_goal_safe_chord_overrides_inferred_circulation_direction():
    """An angular overshoot must not be labelled as another full ring lap."""
    instance = sample_initial_state("development", 22)
    positions = instance.positions.copy()
    goals = positions.copy()
    theta = np.arctan2(positions[0, 1], positions[0, 0])
    radius = np.linalg.norm(positions[0])
    # This is a safe, sub-1.10m chord in the wide annulus, but its positive
    # tangent direction intentionally disagrees with a forced CW hypothesis.
    goals[0] = radius * np.array((np.cos(theta + 0.20), np.sin(theta + 0.20)))
    radial = positions[0] / radius
    velocities = np.zeros((4, 2))
    velocities[0] = 0.2 * np.array((-radial[1], radial[0]))
    env = RingExchangeEnv()
    env.reset(positions, velocities=velocities, goals=goals)
    plan = CentralizedExpert().plan_hypothesis(env, CirculationHypothesis("cw"))
    assert plan.success
    first = plan.actions[0, 0]
    assert np.dot(first, np.array((-radial[1], radial[0]))) > 0.0
