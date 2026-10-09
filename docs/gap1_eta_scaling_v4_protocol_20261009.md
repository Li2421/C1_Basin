# Predeclared Gap1 eta training-data continuation

The v3 audit found no robust eta in N=20's fixed 16-candidate pool and only
four robust N=10 candidates, all in TRAIN states. This continuation tests
whether a **larger global, state-independent candidate pool** gives enough
coverage for state-conditioned learning. It does not change Flow checkpoints,
OrthoFlow3 semantics or support, the hard safety projection, the simulator,
the B15 criterion, or the future-seed convention. It does not sample toward
observed successful eta or deadlock states.

Before evaluating any new eta outcome, freeze a single scrambled Sobol(3)
sequence (seed 20261001) of 128 points in the unchanged support
`[0.5,1.25] × [-0.5,0.5] × [0,0.75]`. Its first 16 points match the v3
global pool, but **all physical states are newly generated** with master
seed 20261010, disjoint from v3's TRAIN/VAL/TEST states. Each N=2/10/20 has
32 TRAIN, 8 VAL, and 8 TEST states. Those new TEST states are identified in
the immutable design but their rollout outcomes remain unopened until the
candidate count is fixed using TRAIN/VAL only.

Use nested global prefixes K=32, 64, 128. First evaluate B15 logical labels
for the first 32 eta on all N=10/N=20 TRAIN and VAL states. N=2 is deferred
because its v3 pool already has usable coverage; the final selected global
prefix will nevertheless be used for N=2 as well. Expand to the next global
prefix if either N=10 or N=20 has fewer than 8/32 TRAIN states or 2/8 VAL
states with at least one robust eta. The threshold is checked separately for
each N. Stop at 128 even if coverage is still inadequate, and report the
negative result. The expansion uses the same fixed Sobol order for every
state and N; no state or eta receives outcome-based extra sampling. Open the
untouched new TEST states once only if TRAIN/VAL coverage passes at a fixed
prefix. If it never passes, retain TEST unopened for a later independent
protocol. Only if TRAIN/VAL and TEST provide meaningful basin coverage
proceed to complete Q16 labels for generator/Q training. Early B15 decisions
remain distinct from Q16.

Rollouts use the exact accepted `Flow → bound → goal-stop → hard projection
→ OrthoFlow3 → hard projection → simulator` chain and the frozen N-specific
Flow checkpoints. The shared rollout database must be registered and
preflighted before Slurm submission. All workers commit exact per-seed
records and the CPU governor limits Gap1 jobs to six shards. The old v3 TEST
states are never reused or used to choose a new eta region.

The immutable design was created as `datasets/gap1_eta_scaling_v4_global128/`.
It contains 144 states and 18,432 planned state-eta pairs; the full Q16
envelope is 294,912 future rollouts. The shared-cache preflight found zero
exact/partial reuse and zero ambiguous identities. The v4 states have zero
physical-state overlap with v3; the first 16 eta coordinates and exact IDs
match v3. Only the first-stage 2,560 N=10/N=20 TRAIN/VAL pairs were submitted,
as Slurm arrays 14986, 14989, and 14990. Array indices are mapped uniquely
to physical-state index 0–39 and eta index 0–31; no TEST pair is in these
arrays. `bottleneck_family/audit_gap1_eta_v4_prefix.py` reads only TRAIN/VAL
pair files and applies the predeclared coverage gate.
