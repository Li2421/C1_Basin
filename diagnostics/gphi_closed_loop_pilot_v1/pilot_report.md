# Learned G_phi full-episode closed-loop pilot

## Result

**GPHI_CLOSED_LOOP_FAIL**

The frozen seed-23/epoch-1171 checkpoint was evaluated on 128 fresh matched
complete episodes. Deployment used only FlowBC, the first frozen projection,
the startup-aware 214-D feature builder, deterministic G_phi, and the second
frozen projection. There were no online eta, basin, oracle, or gate calls.

| Controller | Success | Deadlock | Timeout | Collision | Other | Q (Wilson 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Safety | 128 | 0 | 0 | 0 | 0 | 1.0000 [0.9709, 1.0000] |
| Learned G_phi | 79 | 1 | 48 | 0 | 0 | 0.6172 [0.5307, 0.6968] |

The paired difference is `-0.3828125`, with conservative paired 95% CI
`[-0.481804, -0.242598]`. There were 0 rescues, 49 broken Safety successes,
79 joint successes, and 0 joint failures. Exact two-sided McNemar
`p=3.55e-15`. The result is conclusive at 128, so the frozen extension to 256
was not run.

## Safety and deformation

Hard safety remained intact: zero agent/wall collisions, invalid actions,
NaN/Inf execution errors, projection failures, or solver retries.

Learned `J_def` mean/median/P95/max was
`0.160329 / 0.106164 / 0.355471 / 0.522123`. Failed learned episodes had mean
`J_def=0.268926`, versus `0.092971` for successful episodes.

Steps 0--40 contributed mean `J_def=0.025466`, or 20.54% of total episode
deformation on average. Mean raw/executed correction norm during startup was
`0.095382 / 0.081324`, versus `0.063441 / 0.045435` after startup.

Mean/median/P95 second-projection rewrite was
`0.029840 / 0.004550 / 0.127181`. Using the frozen CBF `intervention_tol=1e-6`
only as a reporting threshold, 61.90% of timesteps had a substantial rewrite.

## Failure association

G_phi applied a nontrivial executed correction on 100% of timesteps and on
100% of late-quartile timesteps. Failures were strongly associated with
persistent correction: mean executed-correction norm was 1.92x the successful
episode value (`r=0.888` with failure), and post-startup deformation was 3.35x
higher (`r=0.815`).

OOD distance was elevated overall, but only weakly associated with failure:
episode-mean OOD was `0.3366` for failures versus `0.3121` for successes, with
`r=0.141`. Projection rewrite was associated with failure but had no solver or
safety violation. The dominant supported issue is therefore
**INTERVENTION_TIMING_WINDOW / persistent over-correction**, rather than OOD
alone.

The smallest next experiment is one frozen-checkpoint, oracle-free cadence
ablation on the same matched seeds: apply G_phi only at the already
Stage-1-supported `H=8` decision cadence (0.4 s), with zero added correction
between decision instants. This changes one timing variable and directly tests
whether continuous application caused the failures; it does not retrain G_phi
or introduce a learned gate.
