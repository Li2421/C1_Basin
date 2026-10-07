# Semantic unified controller audit protocol

Status: **frozen before outcome experiments** (Stage 0 only).

## Scientific objective

Test, in stages, whether the frozen Give-Way primitives can be controlled by
state-dependent semantic decisions rather than periodic timing.  The target
controller has a `NORMAL` mode and a one-use `RECOVERY` mode.  No deployment
decision may depend on a periodic phase, fixed cadence, fixed burst length, or
future outcome.

## Frozen low-level primitives

- `S` — execute authoritative Safety.
- `L` — query frozen Direct-g `c29800f6...` once, execute its correction after
  the second hard projection, then decide afresh in `NORMAL` on the next step.
- `R` — query frozen structured-eta `2481028b...` once, latch the clipped and
  de-normalized eta, enter `RECOVERY`, and execute its first projected
  structured correction on the same physical step.
- `C` — continue the latched structured-eta recovery; recompute the frozen
  state-dependent bases and apply the second projection each step.
- `X` — exit permanently to Safety and execute Safety on that same step.

Physical terminal-event priority is applied before any neural decision.
Absolute episode time, Flow state, physical state, monitor memory, histories,
timers, and latches are never reset at a mode transition.  Official episodes
use 850 transitions at `dt=0.05` seconds.

## Stage gates

1. **Integrity:** freeze/hash implementations, checkpoints, and exact state
   machine; run CPU-only transition tests.
2. **State-driven local:** learn `S` versus `L` from matched full-continuation
   outcomes.  Round 0 is Safety-only bootstrap; Round 1 uses the learned
   downstream S/L policy.  Stop if most of the H=8 timeout advantage cannot be
   reproduced without a clock.
3. **Recovery entry:** learn `S/L/R` only if Stage 1 passes.  Persistent eta is
   predicted once on `R`; no oracle or future failure label is available at
   deployment.
4. **Recovery exit:** learn `C/X` from recovery-visited states, with
   `(h_t, eta_latched)` input and at most one relabel round.  No fixed duration.
5. **Shared model:** only if the modular semantic policy works, distill the
   already-defined targets into the fixed 218-input multi-head architecture.
6. **Development and final evaluation:** model/threshold choices use only
   validation/calibration.  The 200-source final WIDE manifest is frozen before
   rollout and is never used for tuning.

## Data and labels

Root sources are split before branch outcomes; all derived states and Flow
variants stay with their root.  Training states are sampled generically from
the authoritative WIDE distribution, at location-independent nonterminal
indices, with no failure-conditioned or bottleneck sampling.  Paired branches
share future exogenous random streams.

Decision supervision is based on complete-continuation rescue and break
evidence, not action imitation.  Unresolved comparisons remain unresolved.
Task success dominates deformation; deformation is only a bounded tie-break
after predeclared practical-equivalence analysis.  All uncertainty is reported
at root-source level.

## Frozen budget and resources

- At most 20,000 new branch continuations.
- At most 10,000,000 new physical simulation steps.
- Ordinarily 4–8 CPU threads total with BLAS/OpenMP oversubscription disabled.
- Shared-user resource policy limits GPU shards to the observed-server tier;
  Stage 0 uses zero GPU shards.

No stage may silently exceed these limits.  Existing exact cached branches may
be reused only when state, controller, code, and random-stream hashes match.

## Non-goals

No H/L schedule, repeated recovery segment, controller retraining, online eta
search, Basis-as-set, variable-D model, recurrence, scenario change, or test-set
tuning is authorized.  A failed stage is reported rather than hidden by adding
machinery.
