# Stage A recovery-primitive readiness audit

## Audit scope and integrity

This report audits the completed, development-only evidence in
`diagnostics/recovery_takeover_primitive_v1/`.  It does not use the new
single-segment sources, launch a rollout, train a head, or claim that an entry
or exit policy is feasible.

The reused primitive experiment is complete and internally consistent:

- its final manifest SHA-256 map matches every listed output byte-for-byte;
- its frozen development-manifest SHA-256 is
  `3e57e9baba575897b62bdbc3b57cd5b8e7c76676dbcfd6980c11104341944656`;
- all 248 failure-state files and all 248 nominal-match records exist, and all
  recorded state hashes match;
- exact augmented-state restoration has maximum reported error `0.0`;
- all 248 Continue-Safety failure branches reproduce the original Safety
  outcome and terminal step;
- all 248 nominal Continue-Safety branches reproduce success;
- dense Direct-g was re-queried every recovery step;
- structured eta was predicted exactly once at takeover and remained fixed;
- matched Flow semantics passed, and neither experiment used an exit rule,
  H8, a burst, a hybrid controller, online oracle, or eta search;
- no controller produced an agent collision, wall collision, invalid action,
  NaN/Inf, or projection-solver failure.

The source cohort contains 200 independent root episodes with Safety outcomes
`138 success / 54 timeout / 8 strict deadlock`.  It produced 248 failure
takeover states (62 failing sources at each of four offsets), 248 matched
nominal controls, and 1,488 branch continuations.  Repeated offsets from one
root are correlated and are not treated as independent episodes.  The nominal
controls cover 42 unique successful root episodes and 56 unique augmented
states; their pooled break counts therefore are evidence, not certification.

## Reused takeover evidence

| Candidate | Offset | Timeout rescue | Strict-deadlock rescue | False-trigger break |
|---|---:|---:|---:|---:|
| Dense Direct-g | -8 s | 3/54 | 3/8 | 1/62 |
| Dense Direct-g | -4 s | 0/54 | 0/8 | 0/62 |
| Dense Direct-g | -2 s | 1/54 | 2/8 | 0/62 |
| Dense Direct-g | -1 s | 1/54 | 0/8 | 0/62 |
| Persistent structured eta | -8 s | 1/54 | 6/8 | 2/62 |
| Persistent structured eta | -4 s | 1/54 | 7/8 | 0/62 |
| Persistent structured eta | -2 s | 0/54 | 6/8 | 0/62 |
| Persistent structured eta | -1 s | 1/54 | 5/8 | 0/62 |

Across correlated offsets, Direct-g rescued `5/216` timeout branches and
`5/32` strict-deadlock branches.  Its strict-deadlock recovery disappears at
-4 s and -1 s, and the prior audit classified it
`NOT_A_USEFUL_RECOVERY_PRIMITIVE`.  This is not credible enough to spend the
bounded labeling budget on entry/exit heads merely to manufacture usefulness.

Persistent eta rescued `3/216` timeout branches and `24/32`
strict-deadlock branches.  Crucially, it retained strict-deadlock rescue at
-4 s (`7/8`), -2 s (`6/8`), and -1 s (`5/8`).  The result supports genuine
mid/late takeover capacity for the strict-deadlock slice, not timeout recovery.
Its pooled false-trigger break count was `2/248`; because nominal sources are
reused, this does not establish the eventual 5% calibration budget by itself.

## Stage A disposition

| System | Frozen recovery controller | Stage A disposition |
|---|---|---|
| System G | dense Direct-g, SHA-256 `340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700` | **NOT READY FOR HEAD TRAINING**: observed rescue is weak and inconsistent |
| System ETA | one-shot persistent eta, SHA-256 `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095` | **READY FOR STAGE B**, specifically as a strict-deadlock recovery candidate |

Thus `RECOVERY_PRIMITIVE_NOT_ESTABLISHED` does not apply globally: System ETA
passes the primitive gate for strict-deadlock recovery.  This says nothing yet
about whether a learned exit returns successfully to Safety, whether a learned
entry respects break limits, or whether the final single-segment hierarchy
improves full-episode success.  System G remains documented as the predeclared
comparison but should not proceed to expensive head labeling unless the parent
explicitly revises the Stage A gate on new evidence.

## Frozen generic source plan

`prepare_sources.py` froze four independent authoritative WIDE source groups
before any new outcome was evaluated:

| Split | Sources | IC root | Flow root | Predeclared label anchors |
|---|---:|---:|---:|---:|
| Train | 40 | `2026100101` | `2026100102` | 80 |
| Validation | 20 | `2026100111` | `2026100112` | 40 |
| Calibration | 20 | `2026100121` | `2026100122` | 0 |
| Final test | 200 | `2026100131` | `2026100132` | 0 |

For every train/validation root, exactly two distinct absolute global steps are
drawn uniformly without replacement from `[80, 769]` using a deterministic
source-ID-derived RNG.  The requested steps are fixed without reading the
trajectory.  A later collector may materialize a request only if the source is
nonterminal immediately before that decision; otherwise it records the request
as unavailable and does not replace it.  The rule never uses terminal/failure
time, outcome class, geometry, bottleneck location, model error, or a
counterfactual result.  All states, Flow variants, branches, and later derived
states remain grouped with their root source.

Calibration and final-test groups intentionally have no supervised decision
anchors.  They are reserved for full closed-loop evaluation after validation
selection; the 200 final sources are additionally frozen in
`final_test_manifest.json` and must remain untouched until that stage.

Freshness checks passed across all 280 new sources:

- exact overlap between new splits: `0`;
- exact overlap with 1,806 prior initial-state records (1,308 unique): `0`;
- minimum L2 distance to any prior initial state: `0.0023966944`;
- seed collisions and source-ID collisions: `0`;
- replacements, rejection sampling, outcome filtering, and geometry filtering:
  `0`.

Frozen artifact hashes:

- `source_split_manifest.json`:
  `23a90d125caa2697ad74e14243eb7b5971fa95eb75bb9f1bc13cb462632cf21d`
- `decision_anchor_plan.csv`:
  `88e65d773ab392658f09f9123bb6146370ed3d00faf07b0135c084f3ad7cb161`
- `final_test_manifest.json`:
  `0d449b25a15cdc0e5688dacff268454ecbd89725ec07ac06bbd76ee79806208e`

The preparation stage used zero new branch continuations and zero physical
simulation steps.  The global limits remain 12,000 new branch continuations and
6,000,000 physical steps.  The number of materialized decision states is
deliberately unresolved until outcome-blind Safety collection determines which
predeclared absolute steps are nonterminal.

## Readiness conclusion

Stage A is complete.  The scientifically supported dependency is:

`System ETA strict-deadlock primitive -> bootstrap exit labels -> validate an
actual finite recovery segment followed by Safety -> temporally aligned entry
labels -> closed-loop validation/calibration`.

System ETA may proceed to Stage B under the frozen budget.  No feasibility is
claimed yet for the learned single-segment system, and the final-test cohort
must not be touched until entry/exit training, validation, and calibration are
fully frozen.
