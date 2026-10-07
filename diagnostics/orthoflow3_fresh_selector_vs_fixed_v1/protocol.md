# Frozen protocol

- Freeze the 12-mode codebook, `mlp_seed17` checkpoint, temperature
  1.1172859072763657, threshold 0.95, and TRAIN-prior mode 0 before any
  fresh outcome is observed.
- Select up to 64 states by SHA256(source group), then state ID, after
  excluding every source group and state in the prior 128/32/32 split.
- Use matched continuation indices 0..63 for Safety, fixed mode, and selector.
- Only selector-non-B63 states receive the conditional 12-mode oracle audit.
- Primary uncertainty is a deterministic 200,000-replicate paired state
  bootstrap of mean Q64(selector)-mean Q64(fixed), with seed 20260929.
- Classification is frozen as follows: codebook-limited if at least half of
  selector-non-B63 states have no B63 codebook mode; weak if selector has more
  paired breaks than rescues or lower mean Q64/B63 count; replicated if the
  paired mean-Q CI lower bound is positive, B63 count improves, and gains are
  concentrated among switched states; fixed-nearly-sufficient if fixed is
  B63 on at least 90% and selector gains at most two B63 states or less than
  0.02 mean Q64; otherwise underresolved.

