# OrthoFlow3 large-margin ball transfer pilot v1

## Decision

**LARGE_MARGIN_BALL_TRANSFER_STRONGLY_SUPPORTED**

All 19 exact available same-trajectory neighbors of the five margin-eligible anchors accepted the **unshrunk** source ball. Every transferred center was 64/64, every one of the 228 independent interior screening points was 8/8, and all 76 predeclared robust interior promotions were 64/64. No internal false inclusion, collision, projection failure, or numerical failure occurred.

## Source balls and transfer yield

| Anchor state | Split/source group | Radius | Eligibility | Offsets evaluated | Usable transfers |
|---|---|---:|---|---|---:|
| `N_r104_s125` | VAL / `baseline_r104` | 0.015938 | margin too small | none | 0 |
| `ZR_P_r052_m080_s95400004_p253` | TRAIN / `baseline_r052` | 0.353621 | eligible | -4, -1, +1, +4 | 4/4 |
| `S_r043_p02` | TEST / `baseline_r043` | 0.233750 | eligible | -1, +1, +4; t-4 unavailable | 3/3 |
| `N_r076_s119` | TRAIN / `baseline_r076` | 0.249688 | eligible | -4, -1, +1, +4 | 4/4 |
| `R_D2_s95101009_p112` | TEST / `anchor_D2_pair228` | 0.265625 | eligible | -4, -1, +1, +4 | 4/4 |
| `N_r004_m120` | TRAIN / `baseline_r004` | 0.353621 | eligible | -4, -1, +1, +4 | 4/4 |

The exact neighbor identifiers are `T__<anchor_state_id>__d<offset>` and are frozen with complete state/history/Flow provenance in `neighbor_manifest.json`.

## Robust transfer results

- Center B63 transfer: **19/19**, all exactly 64/64.
- Accepted shrink factors: s=1.00: **19**; s=0.85: 0; s=0.75: 0; failures: 0.
- Training-usable balls satisfying both r>=0.20 and r/r_anchor>=0.75: **19/19**.
- Interior screening: **228/228 points were 8/8**.
- Mandatory inside B63: **76/76 points were 64/64**.
- Confirmed internal false inclusions: **0**.
- Transferred radius mean/median/min/max: **0.294288 / 0.265625 / 0.233750 / 0.353621**.
- Retained-radius mean/median/min: **1.0 / 1.0 / 1.0**.
- Retained-volume fraction `(r_transfer/r_anchor)^3`: **1.0 for every transfer**.
- Outside shell at 1.10r: **114/114 points were 8/8**. The accepted transferred balls remain conservative.
- Normalized h distance ranged from 0.156 to 13.285 and physical-state displacement from 0.00152 to 0.07579; every case retained s=1.00, so a distance/shrink correlation is not identifiable in this pilot.

## Dataset interpretation

The observed yield is **3.8 usable new labels per eligible expensive anchor**, or **4.8 total labeled states per anchor family**. These neighbors improve local state coverage, not source diversity, and all states in one anchor family must remain in the same split.

At this observed yield, a source-separated 24/8/8 dataset needs about 5 TRAIN, 2 VAL, and 2 TEST expensive anchor groups (9 total), with an estimated combined cost of 33,806 continuations and 7.84M steps. Given the five eligible groups already available (3 TRAIN, 0 VAL, 2 TEST), the projected incremental need is four new full balls—two TRAIN and two VAL—plus their transfers: approximately **15,025 continuations and 3.48M steps**. This is materially below the prior brute-force estimate of 75k continuations and 20.7M steps for 34 independently reconstructed balls.

## Integrity and runtime

- Exact compatible cache reuse: 0 continuations; the existing caches used incompatible current-Flow/future-stream identities.
- New execution: 8,208 continuations and 1,440,300 physical steps.
- Rollout critical-path wall time: 999.95 s (16.67 min); maximum concurrent GPU shards: 2; 2 CPU threads and 8 GiB RAM allocated per shard.
- Authoritative OrthoFlow3 SHA256 remains `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.
- No controller or safety-projection code was modified.

No G, H, Q, J, center regression, or margin learner was trained.
