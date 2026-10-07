# Direct eta basin geometry and failure audit

## Decision

**ZERO_ACTIVE_FACTORING_JUSTIFIED**

The smallest justified next model is a discrete ZERO/ACTIVE decision followed by deterministic active-eta regression. A conditional normalizing flow is **not justified** by the observed basin geometry.

No production controller was trained or modified. Only diagnostic CPU probes were fit. The frozen `G_eta` checkpoint remained SHA256 `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`.

## 424-state target geometry

- ZERO: **242/424**; ACTIVE: **182/424**.
- Distinct canonical eta vectors: **14**. Most frequent: (0.0, 0.0, 0.0) × 242, (0.375, -0.4375, -0.0625) × 103, (0.40625, -0.5, 0.0) × 22, (0.40625, -0.4375, -0.0625) × 22, (0.4375, -0.53125, 0.0) × 8.
- Multiple stored B63 candidates: **176/424**.
- Multiple near-equivalent candidates: **73/413** states with metadata; 11 strict additions lack `E_near` metadata.
- Active targets on a union-envelope boundary: **27/182**.

This is a large exact zero spike plus a small repeated active lattice, not one continuous Gaussian-like target cloud.

## Feature identifiability

- Exact cross-state h collisions: **0**; <=1e-12 normalized collisions: **0**.
- State-centroid rank-1 ZERO/ACTIVE mismatch: **9.20%**; large normalized eta jump >=0.5: **10.61%**.
- Source-grouped linear ZERO/ACTIVE probe: AUROC **0.9340**, AUPRC **0.9165**, BAcc **0.8570**.
- Source-grouped MLP: AUROC **0.9405**, AUPRC **0.9185**, BAcc **0.9202**.
- Random-state shortcut: linear AUROC/BAcc **0.9870/0.9312**; MLP **1.0000/1.0000**.

Random splitting is optimistic, but grouped performance remains strong. The current 214-D representation contains substantial OFF/ACTIVE information; this is not a representation-aliasing result.

## Basin geometry and MSE

- Uniform 32-state subset composition: 19 canonical ZERO, 13 ACTIVE.
- Fixed learned eta is B63 for **30/32** states (2027/2048 successes).
- All **39/39** farthest-endpoint interpolation points passed the 16-seed screen.
- **29/39** were promoted to matched64 and all promoted points were B63; all **13/13** midpoints were confirmed B63.
- Fully confirmed alpha=.25/.5/.75 farthest-endpoint edges: **3/13**. Remaining states retain an unconfirmed 16/16 alpha point, so exhaustive topology is unresolved.
- Empirically resolved single-component / multi-component / unresolved states: **0 / 0 / 32**. Raw confirmed-edge graphs contain isolated sampled points, but absence of an edge is not evidence of a failure barrier and is therefore not counted as multimodality.
- Normalized canonical eta-L2 versus actual learned-eta Q Spearman: **-0.3350**; nearest confirmed-basin distance versus Q: **-0.4188**.

The mandatory 128-Sobol × 32 screen was not launched because its 32,768 new continuations alone would violate the 15,000 ceiling. Accordingly, no exhaustive connectedness claim is made. Still, the absence of any failing midpoint between confirmed robust endpoints—and 29 matched64 successful interpolants—provides no affirmative evidence for a multimodal generator.

## Fresh-WIDE step-0 root-cause decomposition

The 32 episodes were selected by a seeded uniform permutation before new outcomes.

- ZERO_SUFFICIENT: **14/32**.
- ACTIVE_ORACLE_NEEDED: **15/32**.
- NO_ROBUST_ETA_FOUND in the original frozen finite search: **3/32**.
- Original fixed-eta oracle capacity: **29/32 = 0.9062**.
- An additional **1** state had a robust learned eta outside the finite oracle candidate set, so demonstrated fixed-eta-family capacity is at least **30/32**.
- Current fixed learned eta B63: **10/32 = 0.3125**.
- Capacity-learning gap: **19/32 = 0.5938**.

Current `G_eta` taxonomy:

- ZERO_FALSE_ACTIVATION: **11**.
- ZERO_BUT_HARMLESS_ACTIVE: **3**.
- ACTIVE_MISS_TO_NEAR_ZERO: **0**.
- ACTIVE_WRONG_BASIN_OR_OUTSIDE_BASIN: **9**.
- ACTIVE_NONCANONICAL_BUT_SUCCESSFUL: **6**.
- CLIPPING_ASSOCIATED_FAILURE: **0**.
- OTHER_UNRESOLVED: **3**.

The learned model emitted materially active eta on every fresh32 state (minimum physical eta norm >0.18, no clipping). It broke most states for which zero was already robust and also missed some active basins. The oracle family itself retained high capacity, so global fixed eta is not the primary bottleneck on this sample.

## Model-class decision

1. **Representation redesign:** not supported—no exact aliases and strong grouped ZERO/ACTIVE prediction.
2. **Global fixed-eta family limited:** not supported as primary explanation—offline oracle capacity is 29/32.
3. **Conditional normalizing flow / mixture:** **NO**—no confirmed successful-endpoint/failing-midpoint case, and every promoted interpolant succeeded.
4. **Basin-aware set-valued loss:** scientifically plausible later because canonical eta is not locally smooth and many states have several valid candidates, but it is not the smallest fix for the dominant fresh error.
5. **ZERO/ACTIVE factoring:** directly supported by the zero spike, strong grouped identifiability, and the fresh ZERO false-activation failures.

## Limitation and smallest next experiment

The active basin cloud is sparse and inherited from two historical oracle domains; only the original general-WIDE finite eight-candidate domain was used for fresh G3. `NO_ROBUST_ETA_FOUND` therefore means not found in that frozen finite search, not a proof against every continuous eta.

The single smallest next experiment is a controlled diagnostic **zero/active-factored eta model** on the identical 424-state dataset and grouped splits: predict `p_active(h)`; output exact eta=0 when inactive; otherwise use a deterministic active-eta regressor. Select on validation only and evaluate once on a new development WIDE cohort. Do not introduce a flow unless a later, adequately sampled same-state topology audit demonstrates robust endpoints separated by confirmed failure regions.
