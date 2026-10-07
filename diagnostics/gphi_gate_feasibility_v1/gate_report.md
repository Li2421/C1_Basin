# G_phi Gate Feasibility V1

## Decision

**GATE_PARTIALLY_LEARNABLE**

This was a strictly offline binary feasibility audit. It trained no correction head, changed no oracle/data/controller semantics, and ran no closed-loop control.

## Oracle label and data

`y_gate=0` iff `eta=(0,0,0)` satisfies `B_63`; otherwise `y_gate=1` when a nonzero success-constrained correction is required. Labels came only from Dataset V4 oracle metadata, never from a correction-norm threshold.

- States: 324 = 194 gate-0 / 130 gate-1.
- Split counts: {'train': {'states': 221, 'gate_0': 137, 'gate_1': 84, 'categories': {'NORMAL': 100, 'PRE_DEADLOCK': 45, 'RECOVERY': 76}, 'close_boundary_gate_0': 9, 'close_boundary_gate_1': 9}, 'validation': {'states': 45, 'gate_0': 26, 'gate_1': 19, 'categories': {'NORMAL': 10, 'PRE_DEADLOCK': 5, 'RECOVERY': 30}, 'close_boundary_gate_0': 2, 'close_boundary_gate_1': 2}, 'test': {'states': 58, 'gate_0': 31, 'gate_1': 27, 'categories': {'NORMAL': 10, 'PRE_DEADLOCK': 10, 'RECOVERY': 38}, 'close_boundary_gate_0': 4, 'close_boundary_gate_1': 4}}.
- Source-group leakage: none. All 64 Flow variants of every state remain colocated.

## Selected classifier

- Architecture: `[214, 64, 64, 1]`, seed 41, standard unweighted BCE.
- Validation-selected threshold: **0.603043**; test was evaluated once after freezing it.
- Test balanced accuracy/AUROC/AUPRC: **0.9008/0.9821/0.9825**.
- Test FPR/FNR: **0.1613/0.0370**.
- Recovery-only balanced accuracy/AUROC/FPR/FNR: **0.8515/0.9608/0.2381/0.0588**.
- Held-out close-boundary balanced accuracy/AUROC: **0.5000/0.5000**.
- Matched pairs ranking: **0.8667** overall and **0.6667** on validation/test pairs.

Nearest-neighbor boundary disagreement was 50.00%. Linear/nonlinear and source-group diagnostics are in the accompanying files. The 214-D input contains useful global gate signal: **True**; it is sufficient for reliable held-out recovery-boundary generalization: **False**.
