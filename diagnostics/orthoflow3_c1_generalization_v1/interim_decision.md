# Interim decision after 32/48 independent Toy states

The full 48-state cohort, generator checkpoint, all 768 stochastic proposals,
critic/eta-only scores and selections were frozen before any outcomes. Batches
were predetermined as states 0–15, 16–31 and 32–47. No model or threshold has
been changed after viewing the first two batches.

The first 32 states have full matched Q16 outcomes on identical K=16 pools:
oracle 32/32 B15, frozen Toy W1 critic 32/32, continuous eta-only kernel
28/32 and eta-only MLP 22/32. Both completed batches passed global DB
postflight with 4,096/4,096 exact reuse and zero missing/conflicts. The kernel
comparison has four paired rescues and zero breaks but remains small-sample;
therefore batch 3 proceeds as originally frozen. This decision is about
replicating the fixed comparison, not selecting a new architecture.

Interpretation constraints: this is a same-distribution new-IC/source-family
test with unseen exact eta proposals, not an arbitrary-OOD or cross-mechanism
test. The independent Four-Way/Ring/DB confirmation task is still running and
its uncompleted outputs are not counted as evidence.
