# Phase decomposition execution preregistration

Evidence level: DEVELOPMENT. Authoritative definitions and predictions: docs/prompt_round10_addendum_phase_decomposition_20261013.md and its designer receipt (SHA256 2ffa9e43d2dd94a1726fa5f5e0bb0b52d59b93af8bb88cc2214926770baacd9e; UTC 2026-10-13T09:18:52+00:00).

## Start gate and disclosure

The authoritative addendum requires round10 marked COMPLETE. At preparation time the existing marker is NUMERICAL_COMPLETE_VIDEO_INCOMPLETE, with no COMPLETE.json. Execution is blocked pending explicit user clarification whether numerical completion suffices for this pure analysis. Do not modify round10 markers. Preparing this document is not executing the analysis.

The implementing agent previously worked on round10 and saw its numerical results before receiving this addendum. This receipt therefore freezes implementation choices; it does not claim the implementing agent is outcome blind. The independent designer receipt predates round10 endpoint opening (2026-10-13T09:59:37.263707+00:00). No new round10 label arrays or outcomes have been opened for this task before this registration; status metadata were checked solely for the start gate.

## Historical evidence and compatibility

Reuse round10 LABELS.npz, protocol.json, main_CASES.json, raw outcome records and existing postflight/endpoint receipts. They contain all 14 real Flow, 64 prospective states, seeds 0–3, arms F10/F13/Freg/S300/S400/S500. No simulator/controller change, training, inference, rollout, or DB write is necessary. Freg exists only for three controllers; unavailable cells must be masked, never treated as zero-success arms.

Reuse round7 controller_behavior_repr_v1/FEATURES.npz and frozen data.py/stage1.py representation definitions. For R3 use common96-state R3_96 means, consistent with round7 stage1's 14-controller LOCO; do not substitute R3_real's mixed128/96-state means. For R5 drop columns105:110 from the160-dimensional existing features for every controller, giving155 dimensions; no imputation. Only14 real controllers participate, no blended controllers. Existing round7 models predicted historical eta preferences, not these phase effects: their preprocessing can be reused, but fresh source-only ridge fits and new analysis of frozen labels are the evidence gap.

## Fixed statistical implementation

Use success lower bound (unknown success=0) as the primary point rate; upper bound adds unknown success=1. Main E=S500-F10, L=S500-F13, N=F13-F10. Sensitivity uses S300/S400 in E and L. Bounds for differences are A_lower-B_upper and A_upper-B_lower. The lower-rate point differences obey E-L=N exactly; do not present this identity as a hypothesis test.

State-cluster bootstrap: NumPy default_rng seed202610131001, 5000 resamples of64 states with replacement, retaining all four seeds together. Reuse each state draw across all controllers and arms; percentile95% intervals. Report point-estimate sampling intervals and unknown-aware envelopes formed from the2.5th percentile of lower-bound replicates and97.5th percentile of upper-bound replicates. No controller-resampling is substituted for the required state clustering.

P1: among N<0 controllers at least half have E>=0.10 AND among all14 at least5 have E>=0.10 and L>=0.10. P2: E>0 count>=11 AND L>0 count>=11, with strict negative/zero/positive N counts. Apply these exact point-estimate rules; separately describe unknown sensitivity, without changing thresholds. Use the addendum's H_compress conclusion mapping; all interpretations DEVELOPMENT.

P3 LOCO: for each held controller source13 means determine the selected action, with no target labels used in selection. Fixed set F10,F13; expanded set F10,F13,S300,S400,S500. Tie break by that listed order. Oracle is target's best available arm among all6, including Freg. Primary regret=max(target lower rates)-selected target lower rate; also report unknown-compatible regret bounds with shared selected-arm cancellation (lower>=0; upper=max(other arm upper-selected lower,0)). P3 passes iff expanded mean regret<=fixed mean regret/3 and expanded maximum<=0.05. Target labels are used only for evaluation. Report all fold choices, source means, and target regrets.

P4: each fold fits source13 equal-weight feature z-score (population sd<1e-6 ->scale1), source-only SVD PCA2, orient each component's largest absolute loading positive. Target only projects. Ridge alpha1 with unpenalized intercept and no additional PCA-score scaling, exactly round7 stage1 centered ridge. Separate E,L,N targets and LOCO predictions. Report Spearman and exact three-way sign equality (zero is separate), plus source-mean predictor reference to make uniform-sign accuracy interpretable. No tuning or invented formal predictability threshold: P4 has no directional prediction or supplied inferential cutoff; state observed metric evidence and limitations explicitly rather than manufacture a binary significance gate.

1NN: Euclidean in the same source-fitted PCA2, ties by frozen controller order; select nearest source's highest lower-rate arm in the expanded menu, ties by menu order. Evaluate against the same six-arm target oracle; compare mean and maximum regrets to P3. No target outcome enters descriptor fitting, neighbor selection, or action choice.

For exceptions inspect only existing raw outcome/failure/event-step records, distinguish timeout, collision and numerical unknown from actual deadlock. Do not infer event steps from rendered frames.

## Scope and delivery

All writes only diagnostics/controller_phase_decomposition_v1/. No shared DB writes, no changes to old assets, no git commit, no videos or rollout jobs. Deliver code, controller E/L/N and sensitivity tables with CI, P1–P4 judgments, E–L scatter colored by N sign, results.json, REPORT.md and HANDOFF.md. Report's first paragraph directly answers H_compress and what P4 evidence shows about learning the decomposed relation; facts and interpretations are separated. Keep unresolved start gate visible; no completion claim before actual execution and verification.
