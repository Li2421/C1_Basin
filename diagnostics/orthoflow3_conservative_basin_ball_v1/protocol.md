# OrthoFlow3 conservative basin-ball geometry audit v1

This is a model-free fixed-eta OrthoFlow3 audit. The frozen state population
is a deterministic outcome-blind hash permutation of the exact 120 Q-v2
eligible states; the first 24 states are selected. Geometry uses the
authoritative active search box `[0.5,1.25] x [-0.5,0.5] x [0,0.75]`, affine
center `[0.875,0,0.375]`, and scales `[0.75,1,0.75]`. Explicit eta zero is a
special valid controller value retained outside the active Sobol box.

The planned center is selected from the frozen zero-plus-23-Sobol common cloud
by maximum normalized distance to the nearest `<7/8` screening failure, with
cloud-index tie break. It is promoted to B63 without using Q or J. Eighteen
fixed normalized directions comprise six signed axes plus twelve deterministic
Fibonacci-sphere directions. Ray radii are `0.05,0.10,0.20,0.35,0.50`, with up
to three deterministic bisections. Six limiting directions receive B63
confirmation and the final radius is `0.85 * min(confirmed radii)`.

Before new rollouts, a hard cost audit is required. A scientifically useful
nontrivial 24-state audit necessarily requires at least 20,544 new
continuations after centers are available: minimum ray screens plus six B63
boundary extensions, mandatory independent inside-ball screening/promotions,
and the mandatory outside shell. This already exceeds the 15,000 automatic
cap before common-cloud completion, center promotion, additional coarse radii,
or bisection. Consequently no new rollout is authorized in this run; frozen
manifests and exact cache/cost accounting are produced and the audit is marked
underresolved rather than weakening B63 or reducing the prescribed population.
