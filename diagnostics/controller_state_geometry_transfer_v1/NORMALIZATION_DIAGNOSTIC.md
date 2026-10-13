# Bounded, label-free normalization failure diagnostic

The source-input audit (feature_shift_audit.json), before reading factor final
performance, found source-constant coordinates becoming100 standard deviations
away in FAMILY,200 in COUNT rawN, and2250 in TIME rawH; the mechanism time margin
reaches29.8 source standard deviations in TIME. Thus a negative transfer result
could be a normalization artifact rather than evidence that physical quantities
are useless. This diagnostic is exploratory and does not replace primary arms.

Run exactly one alternative ENT+mechanism encoding on FAMILY,COUNT,TIME only,
seeds17/23/41, same labels/loss/6000steps. Subtract SOURCE feature mean, divide by
fixed physical units [1,60,1,1,1,4,10,1] for [clearance_ratio,time_margin_seconds,
detour_ratio,holes,passage_count,demand,approach_seconds,wall_clearance_metres].
No variance scaling, test-fitted statistics, outcome-based feature selection or
hyperparameter sweep. Raw/ENT/primary-mechanism results remain immutable. Nine
additional fits, zero rollouts, at most2GPU shards and12GB. Compare paired
success and Q MSE to ENT and the original mechanism arm; report all results.
