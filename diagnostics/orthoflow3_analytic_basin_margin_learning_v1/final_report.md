# OrthoFlow3 analytic Basin search and margin learning v1

## Decision

**ANALYTIC_BASIN_REPRESENTATION_FAILS**. The representation gate failed, so no network, held-out TEST controller evaluation, or Fresh-WIDE cohort was run.

## Evidence and acquisition

The audit reused 798 TRAIN/VAL exact-Q64 state–eta records (703 B63, 95 non-B63). All 271 historical TEST records were quarantined and omitted from fitting and selection. Because seven VAL states had inadequate independent multimodal evidence, 43 frozen neighbor-target eta probes were evaluated directly at Q64: 17 were B63 and 26 non-B63. This cost 2,752 continuations and 1,373,981 physical steps. The final selection inventory contains 841 exact TRAIN/VAL records.

## Predeclared candidates

- ncb_affine: false=1, full recall=0.550, retained recall=0.408, neighbor coverage=0.500, universal=0.773, complexity=11, FAIL
- racs: false=0, full recall=0.167, retained recall=0.167, neighbor coverage=0.333, universal=0.727, complexity=20, FAIL
- rfse: false=0, full recall=0.417, retained recall=0.000, neighbor coverage=0.500, universal=0.864, complexity=12, FAIL
- two_lobe: false=0, full recall=0.000, retained recall=0.000, neighbor coverage=0.333, universal=0.600, complexity=22, FAIL

RFSE reports the best of its frozen exponent variants, `(4,4)`; `(2,2)` and `(4,2)` also failed. No candidate reached all simultaneous requirements: zero retained false inclusion, full median recall >=0.60, 6/8 states >=0.40, retained median recall >=0.35, neighbor coverage >=0.60, and TRAIN universal coverage <=0.75. Therefore fresh retained-interior validation was not scientifically warranted.

## Property diagnosis and synthesis

The main failure was a precision–coverage conflict around sharp, state-dependent exclusions. Three low-complexity forms were synthesized under the unchanged gates:

1. **Clipped NCB (19 scalars):** NCB-AFFINE intersected with at most two learned halfspaces. Full recall 0.526, retained recall 0.224, neighbor coverage 0.500; FAIL.
2. **RAP (12 scalars):** rotated asymmetric parallelotope. Full recall 0.618, retained recall 0.333, but only 5/8 states met recall 0.40, neighbor coverage 0.250, and universal retained coverage 0.955; FAIL.
3. **TWO_RAP (24 scalars):** union of two independently eroded RAP components. Full recall 0.333, retained recall 0, neighbor coverage 0.583, universal coverage 0.800; FAIL.

The synthesis limit was exhausted. Adding more flexibility without new held-out boundary evidence would not be scientifically justified.

No representation was accepted, so retained set-label counts are TRAIN 0 and VAL 0; no checkpoint or held-out controller metrics exist.

## Scientific interpretation

- Basin existence remains established: robust persistent eta values exist for every labeled true-t0 state.
- Low-complexity analytic representability was **not established under the frozen gates**.
- Precision could often be made perfect, but only by losing too much independent/multimodal recall.
- Enlarging sets improved recall but recreated a near-universal retained eta, which would again permit constant-output margin-loss collapse.
- Since no label representation passed, margin-loss learnability and closed-loop generalization were not tested. This is not evidence that Basin supervision itself fails.

## Explicit answers

- Did an analytic low-complexity Basin representation exist? **Not among the four predeclared and three justified synthesized families under the required precision/recall/degeneracy gates.**
- Did Basin margin supervision solve point-target ambiguity? **Unresolved; training was correctly blocked before this question could be tested.**

## Next step

Collect a small, state-held-out candidate-aware exact-Q64 set focused on sharp exclusion boundaries and retained-interior coverage, then reassess whether a low-complexity conditional inequality family is identifiable. Do not train until the unchanged representation gate passes.
