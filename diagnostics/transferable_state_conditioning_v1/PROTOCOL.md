# Transferable physical-state η selection: prospective protocol

This study tests whether correct deployment-time physical state changes the
selected three-dimensional η beneficially within an unseen scene. The critic
may read only physical state and η. Controller, Flow, hard-safety execution,
candidate proposals and success evaluation remain frozen. This is a pure-Q
study; `AGENTS.md` exempts it from video delivery.

The historical stochastic proposal generators were already trained with
scene-specific data (the DB/Four/Ring generator jointly on those three
scenes, and Toy's generator on Toy). Thus four-scene LOSO here means **critic
label/checkpoint transfer conditional on a frozen pretrained proposal
generator**. It is not end-to-end zero-shot transfer of the whole pipeline.
The same candidates are presented to every critic and the within-held-out-
scene state derangement keeps that generator and scene fixed, so the causal
critic-state test remains valid under this narrower interpretation.
The common Sobol anchor bank was frozen in the historical matched protocol,
independently of these held-out labels and before this study's target-support
audit. The label-blind η-support audit diagnoses distribution shift; it is
not used to choose a target-scene-specific bank, normalization or checkpoint.

## Reuse and current evidence

`formal_exact_db_audit.json` is a read-only full-database census of every
frozen formal request by exact rollout UID. Training data may reuse labels
only with matching state, η, controller and seed semantics. In particular,
the 512-state matched Ring FF cache uses another execution profile and is
not mixed with four-scene LOSO. Re-encoding an exactly compatible cached
physical state does not invalidate its rollout label. Original raw results,
journals, caches and checkpoints are retained.
The historical strict LOSO source rows also have controller/RNG identity
hashes different from the frozen new true-t0 LOSO profile in every scene.
They remain useful DEVELOPMENT evidence, but their labels are not silently
reassigned to the new source states or controllers. This semantic mismatch,
not the physical-only feature encoding, is why many new exact rollout slots
are required.

The historical four-scene targets and the first 48 Toy formal TEST states
were opened in earlier studies; they are DEVELOPMENT evidence. Old strict
LOSO data contain no TRAIN state with η success-rate range ≥0.25 in either
Four-Way or Ring, so those outcomes cannot support a strong cross-scene
interaction claim. The completed strict-source critic and its eta-only
baseline are diagnostic controls, not final models.

## Frozen state partitions

- TRAIN: formal LOSO global-distribution 512 independent true-t0 states per
  scene, K16 frozen stochastic proposals, four rollout seeds per pair.
- VALIDATION: separate 32 states per scene, K16, 16 seeds per pair. For each
  outer LOSO fold, only the three source-scene TRAIN/VALIDATION labels enter
  checkpoint and method selection.
- DEVELOPMENT: historical LOSO targets and Toy formal TEST indices 0–47.
  Their labels may diagnose mechanisms but do not certify the final claim.
- FINAL CONFIRMATION: formal LOSO TEST states, except Toy indices 0–47.
  Eligible Toy indices 48–127 and all 128 DB/Four/Ring indices were frozen
  before this protocol. No TEST label is used to select architecture,
  checkpoint, hyperparameter, normalization, threshold or anchor bank.

The final primary denominator includes **every** eligible state in each
scene, regardless of outcome. If any final label is opened while modifying
the method, that state is reclassified DEVELOPMENT; a replacement final
state must be sampled independently from the same global generator before
claiming confirmation.

State manifests record the unchanged samplers: Toy WIDE-IC uniform initial
positions, DB accepted global initial states with three prespecified
regimes, Four-Way native global train/development/test draws, and Ring native
development draws. All physical states are true-t0 and unique across the
frozen splits. DB's geometric/expert acceptance happens before outcomes and
does not target critic failure regions. Proposal pools are generated after
physical states are frozen and before any rollout label is opened.

## Modeling and source-only decisions

The first fair comparison uses the existing masked physical-entity critic
and an eta-only control fitted to the **same** source labels. The eta-only
control must have a head wide enough to eliminate the known narrow-head
capacity confound. Source-only ablations may compare raw/relative physical
encoding, concat and eta-prior-plus-residual objectives, and BCE/ranking
losses. Select a single procedure using only source-scene validation,
including leave-one-source-scene-out transfer validation if needed. Primary
checkpoint criterion is validation selection regret; pairwise η ranking
resolves near ties and NLL is calibration only. Freeze model and prediction
hashes before reading FINAL CONFIRMATION outcomes.

The prespecified baseline repair comparison fits two seeds each of the
ordinary concat critic, the head-matched eta-only critic, and a frozen
eta-prior-plus-physical-residual critic. The residual has an explicit
state×η interaction and optimizes BCE plus a within-state pairwise ranking
term. Its eta prior is the source-validation-selected eta-only checkpoint.
Within each fold, choose concat or residual by the lowest source-validation
selection-regret bucket (width 1/512); use source-validation pairwise
ranking, then same-scene mismatch mean-success gap, then NLL as tie-breaks.
This rule is implemented in `choose_final_models.py` and must be run before
any FINAL CONFIRMATION label. If the source-only broad-η probe warrants a
different training support, it remains a DEVELOPMENT experiment and any
method change must be frozen before final outcomes.

## Final falsification

On the identical frozen K16 proposal set and physical states, evaluate
Q(η), matched Q(s,η), and many independently seeded same-scene state
derangements (minimum 100) with paired 16-seed rollout outcomes. Add Safety
and per-state oracle. Report B15 success (≥15/16), mean success, collision,
numerical failure, deadlock/timeout, regret and state-bootstrap 95% CIs.
The primary matched-minus-mismatched analysis uses each physical state as
the unit; no rollout-level pseudoreplication.

The stochastic K16 candidates differ by state. Consequently, candidate
**index** is not a fixed η policy, and exact η-pair reversals across states
are not identifiable from that panel. For the fixed-policy/reversal audit,
use the 16 η vectors already frozen in
`../physical_state_critic_formal_v1/matched_protocol.json` as a separate
common anchor bank. Apply the identical bank to a prespecified natural
subset of FINAL CONFIRMATION states in every scene, with discovery seeds
0–7 and independent evaluation seeds 8–15. The anchor bank and subset must
be frozen before any anchor outcomes. On this bank compare all 16 constant
η policies, source-selected best fixed, evaluation-posthoc best fixed and
bootstrap-reselected best fixed. Strong reversal cases require an absolute
discovery success difference ≥3/8, and the evaluation half checks both
preference persistence and critic switch direction. Pool **all** eligible
pair/state cases; no representative pair determines the aggregate result.
The secondary adaptive-demand subset is defined only from discovery seeds:
`max_eta q_disc(s,eta) - q_disc(s,eta_fixed_disc) >= 3/8`, with the discovery
aggregate best fixed eta and lowest-index deterministic tie-break. Report
its size and evaluation-half paired effects without replacing the natural
all-state primary result.
All other argmax-based selector/oracle ties also use the lowest η index.

State-effect correlation uses per-η centering across states on the common
anchor bank. For state mismatch, the η bank and scene are unchanged and only
the physical-state input is permuted. Report both stochastic-proposal
deployment and common-bank mechanism results separately.

No final verdict is issued from a selected subset or from a LOSO aggregate
gain that survives state permutation. The three allowed verdicts are the
user-specified SUCCESS, PARTIAL and FAILURE categories; a resource-blocked
run remains incomplete and receives no scientific verdict.

The prespecified cross-scene decision rule uses the **all-state natural**
proposal panel as primary. A held-out scene counts as causal decision value
only when matched-minus-within-scene-mismatched mean success has a positive
state-bootstrap 95% CI lower bound, and the common-bank audit also shows
positive matched-minus-mismatched evaluation-half success and positive
matched-minus-bootstrap-reselected-best-fixed evaluation-half success.
Reversal accuracy on the independent half must be directionally better than
eta-only on the pooled strong cases; lack of strong cases is reported as
non-identifiability, not an automatic pass. The global SUCCESS verdict
requires at least two held-out scenes with causal decision value, a positive
equal-scene-weighted LOSO matched-minus-mismatched estimate, no material
collision regression, and fresh-state within-scene evidence. One supported
held-out scene without the global pattern is PARTIAL. If no held-out scene
passes after source coverage and justified representation/objective repairs,
the verdict is FAILURE. These rules are frozen before FINAL CONFIRMATION
outcomes; effect estimates and uncertainty remain more informative than the
category label.

Safety is evaluated on the same physical states and 16 seeds. The global
SUCCESS verdict additionally requires a positive equal-scene-weighted
Q(s,η)−Safety mean-success state-bootstrap CI lower bound; a critic that
adapts but cannot beat frozen Safety does not establish a useful final
pipeline gain.
