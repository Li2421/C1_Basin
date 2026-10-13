# Part 3 Round 3 run log — 2026-10-12

## Historical audit before analysis

- Reused `diagnostics/controller_q_transfer_round3_v1/P1_PROTOCOL.json`, `P1_LABELS.npz`, all 14 `p1_features/*.npz`, P1 report/decomposition, and the Round 2 outcome-free R5 feature files and nearest-source diagnosis. They have matching controller order/state UIDs and the same Ring FF execution semantics. Q4 labels are development/opened; no new outcome or simulator run was required.
- Round 2 nearest-source was a single descriptor-neighbor Q lookup. The missing evidence was whether source-only feature scaling/regularized multi-source reading changes held-checkpoint Q and K16 selection. The new source matching answers this without training a network or generating labels.

## Freeze and execution

1. Wrote `PREREGISTRATION.md` and `ablate.py`, compiled the script, and stored SHA256 for both plus P1 labels/protocol in `frozen_hashes.sha256` **before** running the ablation. The opened P1 data make this a prospective *analysis rule*, not an independent new-label preregistration.
2. Ran on CPU with the project venv:

   ```bash
   /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_transfer_feasibility_round3_20261012/ablate.py > diagnostics/part3_transfer_feasibility_round3_20261012/ablation.log 2>&1
   ```

3. `results.json` and `outer_predictions.npz` contain all 14 outer held-checkpoint predictions and scores. Every model/metric/temperature/match weight was chosen from only the other 13 checkpoints using inner LOCO Q midpoint MSE. Held checkpoint Q entered only scoring. All 272 P1 numerical-unknown seeds remain interval-censored.
4. After inspecting results, added **secondary descriptive** `diagnose.py`; it computes paired checkpoint contrasts, η selection concentration, split-half Q4 noise/headroom, and state×η residual cross-products from the frozen predictions. This file and its `diagnostics.json` do not change the primary gate or models:

   ```bash
   /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_transfer_feasibility_round3_20261012/diagnose.py > diagnostics/part3_transfer_feasibility_round3_20261012/diagnostics.log 2>&1
   ```

5. Recorded SHA256 of all reused old R3/R4 and Round 2 R5 NPZ files in `feature_hashes_postrun.sha256` for future replay. These feature hashes were audited **after** the run; the frozen pre-run receipt covers the plan, analysis script, labels and protocol.
6. Ran `qa.py` after reporting. It verified frozen hashes, finite 6×14×24×16 predictions, every report link, and a one-fold target-label firewall: replacing all Q values of the held checkpoint left R5's source-only chosen configuration and outer prediction bitwise unchanged. Details are in `qa.json` and `qa.log`.

## Accounting

New rollout: **0**. GPU: **0**. New `env.step`: **0**. Changed checkpoint, controller, simulator, evaluator, projection or historical label: **0**. Q MSE comparisons and 14-checkpoint bootstrap intervals are descriptive for this selected, opened inventory. The gate in `PREREGISTRATION.md` failed; no held Flow Q16 work was initiated.
