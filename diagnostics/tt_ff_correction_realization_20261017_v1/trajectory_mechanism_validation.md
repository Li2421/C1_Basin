# Saved trajectory mechanism validation

Eight immutable historical JSON/NPZ pairs were reused from `diagnostics/placement_boundary_mechanism_audit_v1/{results_full,results_h80}`. Exact paths and file hashes appear in `trajectory_metrics.json`. New physical rollouts and new physical steps: **0**.

The four full episodes use historical candidate 4,
`eta=(0.7037163688801229,-0.24619798455387354,0.11380863189697266)`.
The other four are 80-step zero-correction baselines. These do not substitute for trajectories at eta*=(0.5,0,0.5).

## Compatibility and measurement

The frozen difficult-state manifests, checkpoint, four registered source hashes, and historical noise protocol were audited. All four registered source hashes still match. At 13 selected recorded action states, reconstructed actions agree with the saved action exactly at reported floating precision. Every saved noise key in these eight trajectories matches the hard-field schedule: SHA256 state-UID token, future_root=2026100601, seed index 0, physical step. Position increments agree with `dt*action` to floating precision (per-file residuals in JSON). Stored outcomes remain the historical evaluator outcomes; they were not inferred from endpoint motion.

Positions are sampled at step boundaries including the initial state. We compute `chiY=y2-D` and `chiP=x1-x2-D`, D=0.3201. A strict diagnostic Yield–Hold–Pass sequence requires a first chiY>0, a later first chiP>0, and chiY>0 throughout the interval between them. Linear held-action position interpolation preserves the endpoint lower bound between samples. We separately report minimum chiY during **all** horizontal overlap; approach states may overlap in x before yielding is complete, so this stricter quantity is not used to erase an observed later sequence.

## Results

| Initial state suffix | Pipeline | Actual outcome | First chiY>0 | First chiP>0 | Clearance maintained after yielding until passing? |
|---|---|---|---|---|---|
| 820001 | TT | success, 286 steps | never | 152 | No sequence under the specified chiY |
| 820001 | FF | deadlock, 356 steps | never | never | No |
| 820013 | TT | deadlock, 236 steps | never | never | No |
| 820013 | FF | success, 455 steps | 259 | 292 | **Yes**, min chiY=0.00228197465 m, duration 1.65 s |

For FF 820013, the interpolated x-order crossing occurs in interval 265→266. At that crossing y2=0.3464130 m, chiY=0.0263130 m, and lateral separation y2-y1=0.3632945 m>D. Thus this is evidence of complete local yielding, maintaining clearance, and passing, followed by actual task success. It is **PARTIAL** evidence for the mechanism, restricted to this historical state/candidate/seed/pipeline. It is not evidence that eta* has the same behavior.

TT 820001 is an instructive difference between a sufficient indicator and necessity. Its maximum y2 is only 0.3001317 m, so chiY never becomes positive. Nevertheless at its x-order crossing y2=0.3000181 m and y2-y1=0.3280106 m>D: robot 1 is slightly below the nominal centerline. This successful passage is geometrically compatible and contradicts the assertion that every successful pass must have y2>D. It does not contradict a theorem explicitly conditioned on y1=0 and sufficient lateral clearance.

The zero-correction 80-step records are all **truncated**, not complete episodes. None reaches either positive threshold within the saved window. Their full task outcome, deadlock status, and eventual passage are UNKNOWN. Their `success=null` is retained; no timeout or deadlock was invented.

## Outcomes kept separate

Among the four reused full episodes: 2 success, 2 actual detector deadlock, 0 collision, 0 timeout, 0 numerical failure. Among the other four records: 4 diagnostic truncations. The actual deadlock monitor uses a 2 s progress window, progress epsilon 0.01 m, speed threshold 0.025 m/s, and 5 s hold. These detector outcomes are not inferred merely from low progress, and truncation is not timeout.

For the **exact witness eta***, full local Yield–Hold–Pass and full-episode task success remain **UNKNOWN**. The global cache contains 102 exact-eta rows only under an AMBIGUOUS controller profile with unknown checkpoint/safety/RNG/scenario identity (`witness_ambiguous_cache_detail.json`). Those rows were not promoted to compatible Toy evidence. The hard-state K16 bank lacks eta*. The new probes close the action/substep gap only; no full-episode claim is made from first-step signs.
