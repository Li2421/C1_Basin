# Analytic Basin candidate protocol

All eta geometry is evaluated in the frozen normalized `E_bridge` coordinates and intersected with the 14 authoritative domain halfspaces. TEST states are quarantined. State-specific fits use exact-Q64 fitting positives and all exact non-B63 points as hard negative evidence; provenance-frozen B63 points are held out for recall. Sparse VAL acquisition used the nine robust target modes frozen by the prior 40-state dataset: even mode indices were fitting probes and odd mode indices were held out, plus one deterministic point-search holdout/state.

All fits use deterministic constrained grids. The optimization priority is zero retained false inclusion, retained fitting-positive coverage, full fitting-positive coverage, then volume/complexity. Erosion is frozen at `gamma_tan=0.85`, `gamma_norm=0.75`. A candidate that fails any cached recall, precision, nondegeneracy, multimodal-coverage, or universal-constant gate is not subjected to fresh retained-interior validation.

Candidate order is NCB-AFFINE, RACS, RFSE `(2,2)/(4,2)/(4,4)`, then TWO_LOBE. RFSE `(4,4)` is the reported family representative because it had the highest VAL full recall among its three frozen exponent variants; none passed.
