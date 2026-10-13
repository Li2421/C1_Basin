# Prospective amendment: one bounded initial-state-support pilot

2026-10-11; see addendum_state_support_receipt.json for UTC and SHA256.
This amendment is AFTER the initial pilot and its single preregistered bay-depth
revision both failed. Those failures remain final preregistered results. The
original geometry-revision search is closed; this is a separately labelled,
exploratory developmental attempt, not a retroactive change of its gate.

Evidence motivating it: pilot mean Q shows eta12 for isotropic and eta4 for
anisotropic are almost universally successful. Increasing bay depth left that
saturation intact. The archived starts concentrate in opposite staging areas
(x around +/- .7 to 1), while relevant encounter order and bay occupancy have
little coverage. Expanding initial state support directly tests this explanation
without changing radii, tasks, success, Flow, FF, controller or eta.

Keep bay_top=.68 and the original Toy width family. Generate true-t0 safe starts
with RNG SeedSequence([2026101109,state_index]), goals unchanged (+/-1.09,0),
velocities zero, native empty monitor. Four equally weighted strata by index%4:
0: independent opposing x distances uniform [.20,1.05];
1: agent0 distance [.18,.45], agent1 [.65,1.05];
2: reverse distances;
3: one agent initially inside the bay (alternate agent identity by index//4),
x uniform[-.02,.02], y uniform[.38,.50], other agent in its original opposing
staging side at distance [.65,1.05]. Corridor y uniform[-.025,.025].
Safety validated with the unchanged simulator for ALL widths before outcomes.
No repositioning in response to outcome. These starts represent ordinary safe
encounter/recovery configurations; no intermediate monitor history is fabricated.
They are a new initial-state distribution and MUST NOT be called the old t0
training distribution or a demonstrated baseline deadlock distribution.

Pilot states0..15, widths .41 and .59, C2,K8,future seeds0/1:1024 new rollouts.
Apply original quantitative gate without change, bootstrap16 physical states.
If point thresholds pass but uncertainty fails, allow one Q4 extension (1024
more labels). No further state/geometry/candidate/controller changes. If this
fails, the strong-state premise is INCONCLUSIVE and further main-panel claims
are confined to the original low-headroom source; do not infer a method defect.

If this passes, freeze main data with the same distribution: states16..31 train,
32..39 final test,40..43 target calibration. Widths/eta/C/learner seeds and MANY,
FEW,FEW_x2,ALL,EXTRAP definitions from PREREGISTRATION remain, offsets relative
to state16. Test new-state labels Q4, train Q2, FEW_x2 Q4, target calibration Q2.
All final-test outcomes are generated only after this gate. No final labels are
used for calibration or to select feature variants.

Resource amendment: raise total NEW-rollout ceiling once from8192 to16384 for
this finite support attempt plus a minimal independent-factor panel. Pilot
runtime evidence ~1.5s/rollout gives6.83 CPU hours at ceiling; pessimistic5s gives
22.8 CPU hours. Maximum24 CPU slots, at most4 GPU shards; raw storage<100MB,
models/traces<2GB. No automatic budget growth beyond this amendment.

Incidental visibility disclosure: while checking source-job completion, the
last lines of the independently running historical analysis log were visible,
including boolean MLP evaluability/defect flags. No numerical final-test Q or
selection score was used to formulate this amendment. New final states32..39
above have never been generated or evaluated. The original final states remain
an already-existing test panel, not a newly sealed prospective one.
