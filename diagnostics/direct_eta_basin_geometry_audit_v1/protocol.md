# Direct eta basin geometry and failure audit — frozen protocol

## Scientific scope

This is a root-cause/data-geometry audit of the frozen one-shot persistent fixed-D eta predictor. It keeps separate the underlying robust success set, the canonical minimum-deformation oracle target, and the learned eta prediction. It trains no production controller and changes no controller, basis, feature, environment, or scenario.

## Frozen primary assets

- Structured eta checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz`, expected SHA256 `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`.
- Dataset: `/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_strict_deadlock_v1/samples.npz`, expected 424 unique augmented states and 27,136 214-D Flow feature variants.
- Canonical eta labels: authoritative `eta_best_metadata_only`; never inferred from g.
- Eta domain: `[0,-0.53125,-0.125]` to `[1.25,0.5,0.75]`, including exact eta=0.
- Environment: authoritative Give-Way, horizon 850 physical transitions, dt 0.05 s, both hard projections and frozen event priority.

## State weighting and splits

- Stage A and state-level Stage B–D statistics count each of the 424 unique augmented states once.
- All 64 feature variants remain grouped by state/source for diagnostic probes.
- The expensive basin subset is the first 32 states in a seeded uniform permutation of all 424 states, frozen before new basin outcomes.
- The fresh capacity subset is the first 32 episodes in an independent seeded uniform permutation of the existing unconditioned `gphi_structured_eta_fresh_wide_v1` cohort, without outcome conditioning.

## Expensive-work budget

- Hard pre-authorized ceiling: 15,000 **new** fixed-eta continuation rollouts.
- Cached state/candidate/seed results are reused only when state, eta, code, horizon, and continuation semantics match.
- Existing training oracle caches already cover eta=0 and the original fixed candidate cloud for 413/424 states. The 11 strict-deadlock states use their separate frozen B63 capacity artifacts.
- The pre-existing active-state cloud contains nine tested candidates per state (zero plus eight frozen candidates); 168/171 non-strict active states already have multiple B63 points. Therefore Stage E begins from this exact cache rather than mechanically adding 128 Sobol points. Where this cloud cannot support a component conclusion, the result is `UNRESOLVED_SPARSE`; no mathematical connectedness claim is allowed.
- Any projected plan exceeding 15,000 new continuations stops before launch rather than weakening B63.

## Allowed learning

Only CPU diagnostic probes are allowed: regularized linear/logistic models and a 214→64→64→1 SiLU diagnostic MLP. No probe is a deployment artifact.

## Interpretation priority

1. Fresh-WIDE global fixed-eta oracle capacity.
2. Zero/active semantic split.
3. Feature identifiability.
4. Empirical active-basin interpolation/connectivity.
5. Canonical-label smoothness and MSE validity.

No recommended model is trained in this task.
