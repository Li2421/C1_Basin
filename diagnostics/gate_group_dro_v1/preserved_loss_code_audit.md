# Preserved BCE-loss audit

`group_dro_gate_runner.py` preserves the former active source-balanced BCE
line verbatim under the marker `BASELINE_BCE_PRESERVED` in both JIT step
implementations.  The original line was commented rather than deleted.

`loss_mode=source_balanced_bce` executes that same ordinary per-example BCE
after the already frozen source-group → state → Flow-variant sampler.

`loss_mode=group_dro_bce` leaves the per-example BCE unchanged and changes
only aggregation: it computes a mean loss for every **training** source
group, then uses `sum_g q_g L_g`.  `q` starts uniform and is updated in
log-space by the documented exponentiated-gradient equivalent of
`q_g <- q_g exp(eta_q stop_gradient(L_g))`, normalized over training groups.

No loss, q, normalization, validation threshold, or early-stop operation
reads validation or outer-test groups except the pre-existing validation
evaluation used strictly for early stopping and threshold selection.
