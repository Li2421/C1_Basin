# OrthoFlow3 representation migration protocol

Status: frozen before new migration rollouts.

This audit migrates only the eta basis/oracle/data representation.  It does
not train a controller and does not change the environment, FlowBC, either
safety projection, monitor, horizon, or fixed-eta continuation semantics.

## Authoritative representation

The unique validated implementation is `P1-OrthoFlow3` in
`diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py`, with the
globally fixed scale artifact `P1_SCALE.json`.  The exact archived code is the
reference; prose is not used to reconstruct it.  Its frozen common 256-point
Sobol design and bounds come from
`diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json`.

Historical P0 source files are immutable.  Future callers use the new
explicit selector in `shared_control/basis_families.py`; environment config and
checkpoint fingerprints are not changed.

## Stage A

1. Compare the new P0 selector with both historical P0 implementations.
2. Compare the new OrthoFlow3 selector with every field produced by the
   archived implementation on deterministic numerical trials and archived
   basis states.
3. Reproduce archived selected-center result records from the frozen archive.
4. On startup, warm, two-agent and four-agent states, verify eta=0 produces an
   exact zero correction and the second-projected action matches Safety.

No Stage B rollout may start unless these checks pass.

## Frozen 32-state subset

Apply `numpy.random.default_rng(2026092801).permutation(424)` to the immutable
424-state manifest and take the first 32 states.  Selection is performed
without inspecting source category, P0 label, or outcome.

## OrthoFlow3 search and robust continuation bridge

The candidate domain, common 256 points, exact-flow screen, normalized
eight-neighbor density ranking, distance-to-zero tie break, and maximum of
eight candidates are copied from the validated experiment.  For migration to
the existing B63 dataset pipeline, the matched 64 Flow streams already frozen
for each source state are retained; this keeps P0/OrthoFlow3 capacity paired.

1. Evaluate all 256 frozen candidates on the first matched Flow stream.
2. Select all exact-flow successes if at most eight, otherwise the archived
   deterministic top-eight rule.
3. Evaluate selected candidates on the first 16 matched streams.
4. Promote candidates in descending 16-stream success, then lower successful
   mean J_def, then eta index, to the full 64-stream B63 test while budget
   permits.  A candidate is B63 only at >=63/64.
5. Canonical eta is chosen only among confirmed B63 candidates by minimum
   successful mean J_def, then eta index.  Failed and incomplete candidates
   remain in the candidate dataset.
6. Eta=0 uses existing exact state/seed Safety records after Stage-A numerical
   invariance proves representation independence.

Interpolation uses confirmed robust endpoints only.  Test alpha 0.25, 0.5,
0.75 on eight matched streams; two failures reject B63.  Promote screen-perfect
points only if the global 15,000-rollout ceiling permits.  Unpromoted points
remain explicitly unresolved.

## Resource and stop rules

- Maximum 15,000 new continuation rollouts.
- Maximum 8,000,000 new physical steps.
- Full-424 rebuild may start automatically only if its complete projection is
  within both limits and approximately 60 minutes under then-permitted
  resources.
- Maximum two GPU shards under clearly light use; at most six CPU threads for
  normal work.  BLAS/OpenMP oversubscription is disabled.
- All raw candidates, including failures, are retained.

