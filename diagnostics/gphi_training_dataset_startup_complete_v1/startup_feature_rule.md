# Startup-aware 214-D feature rule

This experiment preserves the authoritative V3 feature mapping and changes
only how an incomplete goal-error history is presented to it.

For the causal history available at physical timestep `t`,

```text
[e_0, e_1, ..., e_t]
```

the wrapper left-pads with `e_0` until there are exactly 41 entries. At step 0
the history is 41 copies of `e_0`. Histories with at least 41 real entries use
their ordinary last-41 tail. No future entry is read or synthesized.

The implementation is composition-based: it gives a read-through environment
view with only `distance_history` overridden to the frozen V3 `FeatureBuilder`.
The authoritative builder continues to compute every one of the 214 dimensions.
The wrapper does not edit the original builder or any controller code.

## Frozen sources

- Projection: `/home/zhihan/research/02_C1_Toy_GiveWay/single_integrator/cbf.py`
  (`841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48`)
- Exact identical-problem retry:
  `/home/zhihan/research/Basin_C1/diagnostics/success_basin_multimodality/exact_projector.py`
  (`e29d510dc1752f138bdfcc491f8f5008dbcd215282301396852bdbfd754ac544`)
- V3 feature implementation:
  `/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v2/finalize_dataset.py`
  (`5ec2ba5ab447a707022f693719731595a76d92e6352b3f66c19caedf2baf2341`)

The reusable module asserts all three paths and hashes before importing the
projection or building a feature. A mismatched same-named projection import is
an immediate error.

## Continuity contract

- Steps 0–39: causal earliest-value padding is active.
- Step 40: 41 real history values first exist; padding becomes inactive.
- Step 40 and later: the wrapper must equal the original FeatureBuilder exactly.
- Only history-derived segments may respond to a change in supplied history:
  `recent_progress_2s`, `history_start_step`, and
  `goal_error_history_tail_41`.

Run the isolated audit with:

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  audit_startup_feature_builder.py
```

The audit advances three deterministic 41-step diagnostic traces. It performs
no oracle continuation, label generation, dataset merge, or training.
