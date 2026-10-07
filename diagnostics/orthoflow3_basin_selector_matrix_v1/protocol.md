# OrthoFlow3 basin learning and selector matrix v1 — cost preflight

## Frozen scientific design

- Authoritative OrthoFlow3 SHA256: `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.
- Eta domain: the archived continuous `E_bridge = conv({0} union ACTIVE_BOX)`; the historical disconnected domain is forbidden.
- Only independently validated spherical inner-ball labels are eligible. Rejected axis-aligned ellipsoids are forbidden.
- Minimum source-separated label split: TRAIN 24 / VAL 8 / TEST 8.
- Preferred split: 40 / 10 / 10.
- Robustness remains B63 (`>=63/64`), with the accepted center, limiting-ray, containment, and independent inside-ball validation semantics unchanged.
- Downstream matrix, if authorized after label construction: frozen `G_LOWJ`; three seeds each for `G_CENTER_DIRECT`, `H_REG`, and `H_MARGIN`; selectors CENTER, DEF, and VAL-selected TRADEOFF; no Q/J training.

## Mandatory preflight decision

Only six eligible continuous-domain balls exist: TRAIN 3 / VAL 1 / TEST 2. Reaching the minimum split therefore requires 34 additional verified balls: TRAIN +21 / VAL +7 / TEST +6.

The accepted six-state oracle required 12,688 new continuations and 3,498,278 physical steps. Outcome-blind extrapolation of that exact accepted protocol to 34 additional states requires approximately 71,899 new continuations and 19,823,575 physical steps. The original component-wise preflight scales to 74,936 continuations and 20,719,804 physical steps.

Both estimates exceed the automatic caps of 30,000 continuations and 12,000,000 physical steps. Six shards reduce projected rollout wall time to approximately 74–77 minutes, but do not reduce these scientific costs.

Per the frozen resource policy, execution stops before any rollout or model training with:

`BASIN_SELECTOR_MATRIX_DATASET_REQUIRES_APPROVAL`

No robustness criterion, ray certification, or inside-ball validation was weakened.
