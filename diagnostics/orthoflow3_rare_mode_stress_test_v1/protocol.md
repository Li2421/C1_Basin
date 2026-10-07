# ORTHOFLOW3_RARE_MODE_STRESS_TEST_V1

This audit uses cached compatible Toy true-t0 evidence only. Candidate mining
uses the frozen 128 TRAIN source groups from `orthoflow3_shared_eta_codebook_v1`.
VAL/TEST outcomes are read only after the TRAIN-only availability decision.

A cached eta is considered prevalence-identifiable only when the frozen TRAIN
panel has at least 16 matched trials for all 128 states. Exact Q64 uses B63;
Q32 and Q16 use the declared 30/32 and 15/16 proxies. Partially observed,
adaptively acquired eta are listed but are not treated as prevalence estimates.

Near-duplicate candidate coordinates are clustered at normalized eta distance
0.05. Evidence is never transferred between coordinates inside a cluster; the
representative is the actual eta with the widest TRAIN coverage.

If fewer than three identifiable candidates have 10--30% TRAIN prevalence, the
Toy experiment stops without rollout or training as
`RARE_MODES_NOT_AVAILABLE_FROM_CACHE`.

