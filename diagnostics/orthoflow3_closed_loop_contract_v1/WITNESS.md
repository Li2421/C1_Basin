# Confirmed remaining-horizon alias

Two outcome-blind TRAIN parents per primary scenario were selected before
rollouts. Zero-action waiting produced legal live states at step 1 and H−1.
No physical variable was edited counterfactually. Same reference-policy probe
and future seed streams were used within pairs; physical DB identities differ.

All four pairs have **exactly identical Phase-B normalized inputs**. The late
state cannot succeed: a far agent's remaining goal distance exceeds vmax×dt
plus goal tolerance. This is an independent reachability bound, not a B15 flip.

For the frozen selected eta, early/late successes are Four 16/16 vs 0/16;
Four 11/11 certified plus 5 numerical unknown vs 0/16; Ring 12/16 vs 0/16;
Ring 16/16 vs 0/16. Eta-zero controls give Four 10/16 and 8/16, Ring 16/16
and 16/16, versus 0/16 in every late state. No valid collision occurred.
All numerical attempts remain separate; no unknown is counted as failure.

Classification: REPRESENTATION_ALIASING. The Q labels remain valid for their
original physical snapshots; this is not a physical replay/label corruption.

Minimal repair: append the physically normalized remaining horizon, without
altering simulator horizon, eta schedule, loss, model family or labels.
Refit only the now dimension-incompatible learner weights from scratch using
the same three Phase-B training seeds/budgets. No witness label is added to
training. The counterexample is removed at the representation level; whether
the learner uses the extra field well remains a separate validation question.

Evidence: waiting_probe/preregistration.json, states.json, proposals.json,
results.json and DB-referenced seed outcomes. Source: Phase-B transformed()
and inputs() omit time; TrainingRuntime.reset() restores step_count; native
environment step() terminates at config.max_steps. The isolated patch is
repair.py; original Phase-B artifacts remain unchanged.
