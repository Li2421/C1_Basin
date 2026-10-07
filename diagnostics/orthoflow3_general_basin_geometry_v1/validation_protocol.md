# Prospective retained-validation design (frozen before selection)

Only a candidate passing all applicable cached geometry gates may enter final retained validation. A diagnostic falsification batch for a failed family is not mislabeled as final validation.

For each adequately sampled state, generate a deterministic scrambled Sobol geometric candidate pool of 2^16 points in normalized E_bridge bounding box, reject points outside E_bridge and outside the frozen retained analytic set. This is ONLY numerical set sampling; no broad Sobol rollout sweep is performed. Scramble seed is SHA256(state_id+'retained-validation-v1') first8hex modulo2^32. Deduplicate exact previously queried state-eta tuples and include only genuinely fresh points.

Select6 retained points by farthest-point coverage: begin at the feasible pool point closest to its own pool mean; repeatedly choose the point with largest distance to already chosen points, breaking ties by frozen pool index. This covers central and extremal retained geometry without choosing based on Q outcomes. Selection and fitted parameter hashes are saved before execution.

All6 per state are evaluated at exact64 matched future seeds. Zero non-B63 is required across the entire final validation batch. A single failure rejects that frozen instance/family validation attempt. Do not shrink/repair it in place and pretend the same batch validates the repair. Any later justified synthesis/refit round needs fresh validation, and prior failed-validation observations become disclosed development evidence.

A passing finite validation batch remains EMPIRICAL conservative support, not a proof of continuum safety or a population-Q certificate. Fresh-validation uncertainty and sample size are reported.
