# Frozen-pipeline audit (before new labels)

The historical experiments are two distinct protocols. The formally matched
within-scene TT/FF study is
`diagnostics/orthoflow3_tt_ff_matched_learnability_v1/plan.json` and
`protocol.json`: Toy Give-Way and Ring only, 32/8/16 independent
TRAIN/VAL/TEST states per scene, one public Sobol K16 eta bank shared by TT
and FF, Q4 TRAIN and Q16 VAL/TEST, identical physical states and future seed
indices across placements. TT is the native old action path in
`execution.py`; FF calls the archived field runner. Original checkpoint
selection used validation count NLL, which this task explicitly supersedes.
The Ring FF physical-only expansion is
`diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json`:
512/32/64 states, unchanged K16 bank and FF controller, Q4/Q16/Q16.

The prior four-scene LOSO study is
`diagnostics/orthoflow3_loso_root_cause_v1` plus the first held-out Ring
fold in `diagnostics/orthoflow3_cross_scene_zero_shot_v1`. It evaluated FF
on Toy 48, DB 24, Four-Way 24 and Ring 60 target states, with the frozen
scenario-specific stochastic K16 generator proposals. Its historical source
training pairs were irregular and often early-stopped; its eta-only baseline
was source-validation selected from MLP/kernel using NLL. The present task
requires new matched Q4, 512-independent-state source training and a new
selection-based checkpoint rule, so those old checkpoints are not comparable
primary baselines. The old target panels have been opened and cannot serve as
the requested wholly fresh 128-state TEST sets.

The archived strict LOSO reports (`orthoflow3_loso_root_cause_v1/final_report.txt`
and `orthoflow3_loso_partial_count_v1/final_report.txt`) explicitly state that
no fold had a stable state-aware advantage over its matched source-only
eta baseline; the earlier Ring fold even failed badly. Any prior narrative
that “some LOSO scenes clearly benefited” conflicts with these executed
frozen reports. The new 128-state results must be compared to those archived
figures as historical context, not represented as preserving a proven LOSO
gain.

The `physical23` representation in the Ring expansion is an invertible
four-agent ring-local encoding. It is undefined for Toy's two agents and for
DB/Four geometry. The existing cross-scene masked physical-entity adapter is
`diagnostics/orthoflow3_unified_representation_v1/representation.py`. Its
default `entities()` includes nominal/bounded Flow and controller-derived
flags, so it cannot be used unmodified as the new critic input. The user
confirmed use of its masked entity mechanism with those channels removed.
`physical_entities.py` constructs only positions, velocities, goals, fixed
map geometry and physical robot dimensions; it zeros Flow, policy-frame and
global controller/monitor channels. It adds no scene ID, response or new
physical feature. Ring within-scene FF retains the validated `physical23`
encoder; LOSO uses the approved common masked entity encoder.

The shared rollout DB policy in `shared_rollout_db/README.md` requires
complete request manifests, exact semantic preflight, missing-only execution,
append-only worker journals and a single DB merge. B15 remains at least 15
successes among the fixed 16 evaluation seeds. Numerical uncertainty must
retain lower/upper bounds. This pure Q experiment is exempt from videos under
the current `AGENTS.md`.

Working interpretation for the new protocol: preserve TT and FF for the two
historically matched scenes; use FF for DB and Four-Way within-scene and for
all four LOSO folds. The two historical K16 mechanisms (fixed Sobol bank for
matched TT/FF, frozen stochastic generator for LOSO) must remain separate
panels, with their outputs reported separately. No TEST label will select a
checkpoint, threshold, representation, normalization, or candidate pool.

## Toy cache identity correction before LOSO model training

The historical Toy LOSO Flow folds `rollout_id` into its current and future
PRNG keys. The original cache seed key contained only `future_index`; a new
state with the same initial positions but a different `rollout_id` could
therefore falsely reuse a historical outcome. The initial preflight found 64
such Toy validation slots. Before any LOSO checkpoint or TEST outcome, the
Toy request seed key was corrected to include both `future_index` and the
frozen per-state `rollout_id`; all 64 became genuinely missing. State samples,
K16 η proposals, Flow weights, controller actions and evaluator were not
changed. The affected Toy source array was cancelled and relaunched. Ten
old pilot/partial shard receipts and their raw seed records remain retained
as superseded evidence but their old rollout UIDs are absent from the
corrected cases. Exact pre/post hashes are in
`loso/toy_seed_identity_correction_audit.json`.
