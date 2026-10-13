# Prospective selected-only Toy FF Q(s,η) confirmation

48 physical TEST states were fixed before critic scoring; all five selectors and Safety
used identical state-specific K16 proposals and 16 fixed future seeds.
Only selected η outcomes were simulated, so no oracle is estimated.

| Method | B15 confirmed/48 | B15 possible/48 | Mean success lower–upper | Collisions |
|---|---:|---:|---:|---:|
| within_state | 39/48 | 39/48 | 0.844–0.844 | 0 |
| within_eta_wide | 33/48 | 33/48 | 0.812–0.812 | 0 |
| within_state_shuffled | 30/48 | 30/48 | 0.724–0.724 | 0 |
| loso_state | 28/48 | 28/48 | 0.622–0.622 | 0 |
| loso_eta_wide | 32/48 | 32/48 | 0.750–0.750 | 0 |
| Safety | 17/48 | 17/48 | 0.483–0.483 | 0 |

| Paired comparison | B15 difference [95% state CI] | Mean Q difference [95% state CI] |
|---|---:|---:|
| within_state_minus_within_eta_wide | +12.5 [+0.0, +25.0] pp | +3.1 [-7.4, +14.1] pp |
| within_state_minus_within_state_shuffled | +18.8 [+6.2, +31.2] pp | +12.0 [+0.9, +23.4] pp |
| within_state_minus_Safety | +45.8 [+29.2, +62.5] pp | +36.1 [+21.0, +51.0] pp |
| within_state_minus_loso_state | +22.9 [+8.3, +37.5] pp | +22.1 [+8.9, +36.1] pp |
| within_state_minus_loso_eta_wide | +14.6 [+0.0, +29.2] pp | +9.4 [-2.3, +21.6] pp |
| loso_state_minus_loso_eta_wide | -8.3 [-27.1, +10.4] pp | -12.8 [-28.9, +3.5] pp |
| loso_state_minus_Safety | +22.9 [+4.2, +39.6] pp | +13.9 [-2.7, +30.5] pp |

Toy's original historical target set was not reused as this TEST set.
The 48 states are the first 48 of the previously frozen, globally sampled fresh Toy TEST panel.
The source model checkpoints and choices were frozen before these rollout outcomes.
