# Exact validation reuse for unchanged state instances

An independently validated state instance need not be re-probed merely because another state's parameters were updated under the SAME frozen family, fitting recipe, hyperparameters and erosion. Reuse requires exact equality of stored mathematical parameters (family, planes, quadratic coefficients), exact equality of that state's fitting-positive and hard-negative eta-key sets, and unchanged primary heldout membership. Its validation points must remain excluded from fitting. State/h/xi0, seed, basis, horizon, projections and success semantics must already pass the authoritative exact-Q64 inventory audit.

This does NOT permit ignoring failed states or substituting a new model under an old validation result. All12 states remain in the final recall/precision gate; zero confirmed non-B63 is still mandatory. If the family, recipe, erosion, state parameters or fitting evidence changes, prior validation cannot validate that changed instance. Record reused independent points separately from genuinely new validation.

Current allowed comparison: DB instances in minimal_polyhedral_r5_corrected versus same-family refinement_r6. DB24-point v5 batch passed and is NOT listed in development_validation_batches.json. Toy v5 failed and cannot validate any refit; its replacement validation must be genuinely new.
