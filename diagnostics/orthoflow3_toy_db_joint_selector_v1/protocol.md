# Toy–DB joint selector protocol

- No rollout is generated. Canonical mode IDs, Toy eta, transformed DB eta, and the Toy-to-DB transform are immutable.
- Toy 214-D and DB 80-D inputs have incompatible dimensions and semantics. Each receives a scenario-specific Dense(64)+SiLU adapter. A Dense(64)+SiLU trunk and Dense(12) canonical-mode head are shared.
- Training loss is exactly `0.5 * mean_BCE_Toy + 0.5 * mean_BCE_DB`; each BCE uses the existing binomial success fraction for every state/mode.
- Seeds are 17, 23, 41. Checkpoints are selected on VAL by average scenario NLL, average top-1 empirical Q, worst-scenario NLL, minimum scenario top-1 Q, then lower seed.
- DB TRAIN subsets are nested, outcome-blind SHA256 selections of 16/32/64 states. Toy TRAIN always uses all 128 states.
- TEST outcome files remain unopened until all checkpoint/seed choices are frozen.
