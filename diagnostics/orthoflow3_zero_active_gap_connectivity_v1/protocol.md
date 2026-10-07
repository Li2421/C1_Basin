# OrthoFlow3 zero/ACTIVE gap connectivity audit v1

## Frozen design

- Population: the first 12 states, in exact frozen order, from the 32-state
  OrthoFlow3 representation-migration cohort. No outcome-based selection or
  ZERO/ACTIVE rebalancing was performed.
- Controller: frozen OrthoFlow3 basis implementation SHA256
  `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`,
  fixed eta for the continuation, bases recomputed every physical step, and
  the unchanged two-projection safety stack.
- Robust success: B63 means at least 63 successes among the exact 64 matched
  migration Flow seeds. An 8/8 result is screening evidence only.
- Endpoint A: first cached nonzero B63 candidate in frozen numeric eta-index
  order. Endpoint B, when available: cached B63 candidate farthest from A in
  normalized eta space, with eta index as tie-break.
- Radial alpha grid: `0, .05, .10, .20, .30, .40, .50, .60, .75, .90, 1`.
- Promotion: unresolved endpoints; adjacent 8/8 versus <=6/8 transitions and
  their neighbors; 7/8 ambiguities and neighbors; and surprising interior
  misses between B63 endpoints. B63 semantics were not weakened.
- Zero-neighborhood: 12 deterministic frozen Sobol ACTIVE anchors at alpha
  `.05, .10, .20`; every observed non-8/8 representative was promoted to 64.
- Conditional bridge search: the common 64-point bridge cloud and k=6 graph
  would run only for a ZERO-sufficient state with a confirmed interior radial
  B63 failure gap. No state met that trigger, so bridge outputs are empty with
  explicit schemas rather than post-outcome replacement samples.

## Bridge domain

`E_bridge = conv({0} union ACTIVE_BOX) = {alpha*x : alpha in [0,1],
x in ACTIVE_BOX}`. The exact definition and normalization are stored in
`bridge_domain_definition.json`.

All state manifests, endpoints, radial coordinates, and zero-neighborhood
directions were frozen before new rollout outcomes. Learned Q and J_def were
not used for candidate selection, promotion, topology classification, or the
final decision.
