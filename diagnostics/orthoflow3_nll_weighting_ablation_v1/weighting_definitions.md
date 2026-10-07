# Frozen weighting definitions

- W0: pair weight `min(n_trials,16)`.
- W1: every canonical pair has weight 1.
- W2: pair weight `1/M_state`, so every state has equal total weight.
- W3: within each state, pair weight is `sqrt(min(n,16))`, normalized to state total 1.

All variants use the same architecture, seed-specific initialization, minibatch indices, optimizer, learning rate, 4,000 steps, and 50-step evaluation cadence. Checkpoints minimize the variant's VAL probability objective. No ranking loss is used.
