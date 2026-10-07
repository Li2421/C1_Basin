# Frozen protocol

- Freeze the 12 anchors, mode IDs, selector checkpoint, temperature 1.1172859072763657,
  threshold 0.95, original 128/32/32 split, and TRAIN-only normalization.
- Build labels only for the frozen selector's chosen mode on TRAIN/VAL states.
- Assign an eta to exactly one nearest anchor only when normalized distance is at most 0.20.
- A positive is 63/64, 31/32, 16/16, or 8/8. A negative is at most
  60/64, 29/32, 14/16, or 6/8. Intermediate outcomes remain ambiguous.
- A usable local set needs two positives separated by at least 0.05 normalized units.
- For underresolved selected pairs, evaluate 16 deterministic Sobol offsets inside
  E_bridge, radius at most 0.15: TRAIN uses 8 seeds and VAL uses 16.
- Train only set-distance plus negative-margin loss (mu=0.05, lambda=1), never point MSE.
- Select among seeds 17/23/41 using VAL set loss only. Test on the already frozen,
  source-isolated 64-state fresh cohort; fixed outcomes are reused and only moving eta
  outcomes are newly evaluated with the same 64 continuation indices.

