# Pair-1 ranking-failure attribution

## Result

**MULTI_GROUP_SHORTCUT**

The two Pair-1 members necessarily use different valid OOF folds. Their reported OOF logit difference is therefore decomposed exactly relative to each fold's train-mean normalized reference; the explicit fold-model offset is retained rather than incorrectly assigning it to an input dimension. In addition, straight-path integrated gradients were computed under each fold model separately and symmetrically averaged. Flow seeds do not match across these states, so the 64 paths use deterministic seed-rank pairing and are labeled as comparable, not matched.

Mean decomposition across 3 seeds x 64 variant-rank pairs:

- Actual OOF nonzero-minus-zero logit: **-1.302027** (wrong ordering).
- Symmetric within-model feature/path effect: **+0.611240**.
- Cross-fold model effect: **-1.913268**.
- Maximum absolute IG completeness residual: `1.814e-06`.

Crucially, this reversal does **not** survive same-model scoring:

- Under the zero state's `anchor_D2_pair228` OOF model, logits are `2.380337` (zero) and `3.489386` (nonzero), a correctly ordered margin of `+1.109050`.
- Under the nonzero state's `anchor_D4_pair227` OOF model, logits are `0.964878` and `1.078309`, also correctly ordered by `+0.113431`.

Thus the cross-fold Pair-1 reversal is dominated by fold-specific model/score offset, not by a single common gate ranking the two clouds backwards. The `MULTI_GROUP_SHORTCUT` label describes why the held-out zero itself sits at an intervention-like absolute score: compared under its own fold gate with its fixed same-label training neighbor, geometry/observation, history/monitor, and control/projection all materially elevate the zero state's logit. It must not be read as claiming that these feature effects alone account for the cross-fold numerical reversal.

### Actual OOF train-mean-reference feature contributions

Positive values support the oracle-correct ordering; negative values support the observed wrong ordering.

| Feature group | Mean contribution | SD | source eta² | label eta² |
|---|---:|---:|---:|---:|
| geometry_observation | -0.733323 | 0.280106 | 0.715 | 0.110 |
| episode_time | +0.485480 | 0.112741 | 0.681 | 0.024 |
| control_projection | -0.295081 | 0.206798 | 0.651 | 0.120 |
| history_monitor | -0.210197 | 0.045727 | 0.686 | 0.152 |
| u_safe | -0.184390 | 0.165456 | 0.665 | 0.064 |
| inter_agent_relative | -0.153746 | 0.110353 | 0.699 | 0.139 |
| goal_relative | -0.153734 | 0.080123 | 0.728 | 0.123 |
| u_flow | +0.018882 | 0.210475 | 0.687 | 0.040 |

Dominant wrong-order groups: geometry_observation, control_projection, history_monitor, u_safe, inter_agent_relative, goal_relative. Dominant correct-order groups: episode_time, u_flow.

The fixed nearest opposite-label training state is `R_D4_s95101006_p60`; the fixed nearest same-label training state is `N_r045_s131`. Their per-seed, per-variant score and IG comparisons are in `pair1_neighbor_comparison.csv`.

## Interpretation boundary

Integrated gradients explain the frozen networks locally; they do not establish causality. Source-group eta-squared is reported only as association and is expected to be upward-biased when many groups are small. No checkpoint, feature, label, threshold, state, or rollout was changed.
