# Semantic unified controller audit — final report

## Final classification

**RECOVERY_ENTRY_NOT_PREDICTABLE**

The staged audit stopped at the required Stage-2 gate.  State-driven local
correction is a useful clock-free partial replacement for H8, and persistent
eta remains a real recovery primitive, but the generic matched-outcome data did
not support a learnable conservative decision boundary for entering recovery.
No three-way mode head, exit head, shared multi-head model, or pristine final
test was run after this failure.

## Completed stages

1. **Stage 0 integrity — PASS.** All frozen assets matched; the semantic state
   machine and no-H/L invariants passed their tests.
2. **Stage 1 state-driven local — `STATE_DRIVEN_LOCAL_PARTIAL`.** On the
   inspected 200-episode development cohort: Safety 135/200, semantic local
   178/200, frozen H8 reference 188/200.  Semantic local rescued 43/59 Safety
   timeouts, 0/6 strict deadlocks, and broke 0/135 Safety successes.  Mean
   `J_def=0.023719`; 3,855 local events; mean/median spacing 2.862/1 steps.
3. **Stage 2 recovery entry — `RECOVERY_ENTRY_NOT_PREDICTABLE`.** 160 new
   generic WIDE sources yielded 126 available predeclared anchors and Safety
   outcomes 102 success/53 timeout/5 strict deadlock.  Sixty generic anchors
   were evaluated under 32 matched futures and S/L/R, producing 5,760 complete
   branches and 1,769,844 physical transitions with no safety/numerical event.

Stages 3--5 were not authorized after the Stage-2 stop.  The final-test
directory remains empty and no final manifest was generated.

## Stage-2 evidence

At the continuation level, S/L/R success was 1251/1920, 1278/1920, and
1434/1920.  Recovery versus Safety produced 271 rescues and 88 breaks
(`Delta Q=+0.0953`); versus Local it produced 243 rescues and 87 breaks
(`Delta Q=+0.0813`).  These are matched continuations clustered within 60
source states, not independent WIDE episodes.

The frozen conservative confidence rule resolved only 9/60 decision states:
4 train and 5 validation.  Every resolved target was `ENTER_RECOVERY_NOW`;
51/60 remained unresolved.  There were zero supported Safety or Local targets.
Training `214 -> 64 -> 64 -> 3` on this one-class set would make recovery entry
nearly unconditional, reproducing the explicitly prohibited degeneracy.

## Answers to the falsification questions

1. **Can state-driven LOCAL replace H8?** Partially.  It retains 72.9% of H8's
   timeout rescues with zero breaks, but Q is 0.890 rather than 0.940.
2. **Natural timing:** mean/median event spacing is 2.862/1 physical steps;
   spacing was observed, never targeted.  It does not recreate H=8.
3. **Can NORMAL identify persistent recovery entry?** Not established.  The
   primitive has benefit, but conservative generic labels are one-class.
4. **Does recovery add strict-deadlock capacity?** Prior takeover evidence says
   yes; this audit did not establish when to invoke it online.
5. **Can recovery exit to Safety?** Not tested because entry failed its gate.
6. **Can one shared model represent everything?** Not tested; modular semantics
   were incomplete.
7. **Does the usable state-driven local controller contain fixed H/L?** No.
   A complete S/L/R/C/X semantic controller was not established.
8. **Does it preserve the H8 high-water mark?** No: 0.890 vs 0.940 on the
   inspected development cohort.

## Integrity and safety

All 5,760 Stage-2 branches completed.  Each R branch predicted eta once and
latched it; paired current/future Flow streams matched.  There were no agent or
wall collisions, invalid actions, NaN/Inf events, or projection failures.
Independent review found and fixed fail-closed provenance gaps in the unused
full-loop evaluator; 8 main tests plus 4 source-freeze tests pass.

## Remaining limitation and next experiment

The single blocking limitation is **recovery-entry supervision**: existing
evidence identifies some states where R is best but does not conservatively
identify a deployment-learnable non-entry region without forcing unresolved
S-vs-L choices.

The smallest justified next experiment is a predeclared partial-preference
mode-head pilot using the already-frozen evidence: label R only when strictly
supported and label the action set `{S,L}` only where R is strictly disfavored,
without choosing between S and L.  Validate it on new development roots before
any exit training.  This experiment was not launched here.

## Budget and resources

- New branch continuations: 8,384 / 20,000.
- All new rollouts including Stage-2 source collection: 8,544.
- New physical transitions: 2,737,424 / 10,000,000.
- Maximum concurrent allocation: 2 GPU shards, 4 CPU threads, 30 GB requested
  memory; hidden BLAS threads were constrained.
- Recorded simulation wall-clock envelope: 2,941 s (49.0 min).
