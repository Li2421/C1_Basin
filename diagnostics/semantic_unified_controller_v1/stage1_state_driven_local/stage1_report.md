# Stage 1 — State-driven local correction

## Decision

**STATE_DRIVEN_LOCAL_PARTIAL — PROCEED_STAGE2.**

The learned state-driven S/L policy is useful and safe enough to retain as the
NORMAL policy for the Stage-2 recovery-entry experiment, but it has not fully
replaced the frozen H=8 high-water mark.  On the 200-episode external
development cohort it rescued 43/59 Safety timeouts (72.9%), versus 59/59 for
the authoritative frozen H=8 reference, and achieved 178/200 rather than
188/200 successes.  It introduced no breaks on the 135 Safety-success episodes.

The 200-episode cohort is an already-inspected **development** cohort, not the
pristine final test.  The H=8 numbers are an external frozen reference from
`gphi_h8_fresh_unseen_generalization_v1`; H=8 was not used inside the semantic
controller.

## Frozen semantic policy

- At every NORMAL timestep, a 214-D head chooses Safety or one local correction.
- A local event queries frozen Direct-g `c29800f6...` once for that physical
  step; the next timestep receives a new state-driven decision.
- Selected head: `214 -> 64 -> 64 -> 1`, SiLU, seed 17, epoch 1,
  `lambda_break=0`, checkpoint SHA256
  `96b82740e4581d9e7634aa917508d536b76323faf5c25a0c3a36fad4cb14a79a`.
- Frozen deployment threshold: probability 0.75.
- There is no fixed period, cadence phase, cooldown, or burst length.  In
  particular, there is no fixed H or fixed-duration L in this policy.

## Full-episode evidence

| Cohort | Controller | Success | Q | Timeout rescue | Strict-deadlock rescue | Break | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation (20) | Safety | 16 | 0.80 | — | — | — | 0 |
| validation (20) | frozen H8 reference | 19 | 0.95 | 3/4 | n/a (0 cases) | 0 | 0.020856 |
| validation (20) | semantic local | 20 | 1.00 | 4/4 | n/a (0 cases) | 0 | 0.024533 |
| calibration (20) | Safety | 14 | 0.70 | — | — | — | 0 |
| calibration (20) | frozen H8 reference | 17 | 0.85 | 5/5 | 0/1 | 2 | 0.026935 |
| calibration (20) | semantic local | 19 | 0.95 | 5/5 | 0/1 | 0 | 0.022998 |
| external development (200) | Safety | 135 | 0.675 | — | — | — | 0 |
| external development (200) | frozen H8 reference | 188 | 0.940 | 59/59 | 0/6 | 6 | 0.024892 |
| external development (200) | semantic local | 178 | 0.890 | 43/59 | 0/6 | 0 | 0.023719 |

The semantic policy's external-development paired outcome against Safety was
43 rescues, 0 breaks, net rescue +43, and Delta Q +0.215.  All 22 remaining
failures were timeouts; no collision or strict-deadlock terminal outcome was
introduced by the semantic policy.

## Emergent intervention timing

On the 200-episode development cohort the policy made 3,855 local events:

- events/episode: mean 19.275, median 14;
- inter-event spacing: mean 2.8616 steps (0.1431 s), median 1 step (0.05 s),
  P95 8 steps;
- mean J_def: 0.023719.

These spacings are observations, not training targets or timing rules.  The
median spacing of one step is direct evidence that the learned policy did not
merely recreate the periodic H=8 schedule.

## Label uncertainty and limitation

Both paired-label rounds used 32 matched future streams per decision state and
preserved downstream-policy lineage.  All 36 queried inputs remained unresolved
under the predeclared equivalence test; no-discordance at n=32 was not promoted
to proven equivalence.  Full-episode validation/calibration nevertheless showed
useful behavior, but the external-development gap of 10 successes versus H=8
precludes `STATE_DRIVEN_LOCAL_SUPPORTED`.

Stage 2 may therefore test whether a state-driven persistent-eta recovery entry
adds strict-deadlock capability while keeping this partial local policy frozen.
No final-test episode was consumed in Stage 1.

## Integrity and safety

- Frozen Direct-g SHA256 matched `c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e`.
- Paired branch results: 2,304/2,304 complete; zero execution errors.
- Full-loop results reported zero agent collisions, wall collisions, or
  execution errors.
- Source grouping remained disjoint; final-test use was forbidden.

