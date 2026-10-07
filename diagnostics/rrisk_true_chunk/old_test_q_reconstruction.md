# Frozen old TEST-Q reconstruction

This reconstruction was written before any new persistent-correction outcome.
It does not reinterpret the authoritative old result.

- Perturbed variable: `action_offsets[action_step]`, a joint 4-D residual
  added to `u_safe + r` before the second projection. Frozen `G_phi` had zero
  output, so the nonzero diagnostic term was exactly this offset.
- Physical execution: `u_exec=Pi_U(x)(u_safe+r+offset)`. The perturbation was
  therefore pre-Pi_2, while outcomes used the post-projection executed action.
- Duration: exactly action index 100, one physical step = 0.05 s. Every later
  offset was zero.
- Magnitude: four frozen random unit vectors and their negatives, each scaled
  to exact pre-projection L2 norm 0.2, plus baseline. This gave 9 arms per
  state.
- State set: old frozen deadlocking trajectory IDs 0, 2, 3, all immediately
  before action 100, alive and with the historical deadlock latch false.
- Randomness: action 100's Flow latent was frozen by the common prefix seed;
  independent continuation randomness began at action 101.
- Continuation: unchanged frozen Flow-BC, Pi_1, Pi_2, environment, first-event
  termination, historical latch, and `D_H` definition.
- Counts: 3 states x 9 arms x 8 continuation seeds = 216 continuation
  rollouts; all 27 arms were 8/8 `D_H`, giving 108/108 within-state unordered
  score-different pairs with no empirical true-risk difference.
- Projection result: every one of the 108 executed-action pairs remained
  distinguishable; the failure was not mass aliasing.

Why 27 arms ended 8/8 deadlock is not identified by that experiment. The
states were not already latched. In the pre-existing reference traces, action
100 preceded the deadlock event action by 197, 130, and 261 action indices for
rids 0, 2, and 3 respectively (9.85, 6.50, and 13.05 s by index difference).
Thus they were not uniformly adjacent to `D_H`. A norm-0.2 one-step offset can
alter position by at most roughly 0.01 m before being removed, so insufficient
temporal commitment is a plausible hypothesis, not an established explanation.
The new experiment tests that hypothesis using true continuation outcomes.
