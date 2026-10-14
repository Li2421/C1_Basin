# Phase decomposition handoff

COMPLETE (pure analysis; all conclusions DEVELOPMENT). Round10 numerical-completion gate explicitly authorized by user; round10 video status remains unchanged.

H_compress SUPPORTED: P1 9/12 eta10-preferring controllers have E>=10pp; 10/14 have both E,L>=10pp. P2 E>0 11/14, L>0 14/14; N negative12, positive2.

P3 FAIL: expanded source-average selection mean regret1.0324pp vs fixed32.8962pp, but max5.8594pp>5. Unknown cannot rescue the whole gate because v9's5.0781pp deficit has no unknown.

P4: R5-PCA2 predicts E with Spearman0.8418, sign13/14, RMSE8.59pp; L Spearman0.2088 and its sign100% only equals uniform-positive source-mean reference. Partial positive evidence for learning E; full decomposed controller–Q learnability not established. R3/R5 1NN mean2.6507pp, max24.2188pp, worse than source-average menu selection. No post-hoc predictability cutoff was invented.

Read REPORT.md and results.json. Exact failure/termination steps in EXCEPTION_FAILURE_RECORDS.csv and EXCEPTION_OUTCOME_STEPS.csv; timeout is not deadlock. No new rollout or DB write. All18688 raw labels/hashes independently matched frozen evidence. Input hashes unchanged.

Reuse commands: specified venv Python -B analyze.py, then -B verify_and_report.py (from this directory or with absolute paths). Models/descriptors only source13 controllers per fold. R3 common96-state means; R5 frozen155D main scheme. No new checkpoint training or controller inference.

Next research candidate: structured E prediction, with independent-controller validation and a better L model; do not claim that descriptor-based action selection improved, or that the universal <=5pp guarantee passed.
