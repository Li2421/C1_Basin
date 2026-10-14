# Round12 supplement: eight genuinely different final policies, correct FP32

This supplement applies the user's explicit instruction after the semantic-novelty and precision audits. Original PREREGISTRATION.md and receipts remain preserved; no final success label has been generated or read. The original14-controller configuration is superseded before its success endpoint.

## Authorized cohort and unchanged rules

Confirmation cohort comprises only these8 final policies whose parameter bytes differ from sibling best:
- base_u_v4_macflow
- base_u_plus_w_v6_macflow
- base_u_plus_w_v3_macflow
- base_u_plus_w_v4_train_targeted_macflow
- base_u_v10_local_macflow
- base_u_v7_local_macflow
- base_u_v8_local_macflow
- base_u_v9_local_macflow

Six identical-policy finals are excluded from all subsequent probes and evaluation:base_u_macflow,base_u_plus_w_v4_macflow,base_u_plus_w_v4_wide_macflow,base_u_plus_w_v5_macflow,base_u_v2_macflow,base_u_v3_macflow. Report their parameter byte hashes and exclusion reasons only. No regression rollout for these six. Their previously produced invalid-precision unlabeled probes are retained as incident evidence, not used as confirmation or recomputed.

Main size8x64x6x4=12288 fullrollouts; corrected R5 size8x16x2=256 unlabeled200-step probes. Sources remain14 historicalbest policies. Primary E model excludes target's same-run brother,13 sources; secondary14source model sensitivity only. Source-only155D R5,zscore/PCA2, ridgealpha1/unpenalizedintercept unchanged. G_E Ehat<5pp ->F10 elseS500. Mainmenu F10,F13,S300,S500, oracle-onlyF2/F4. No model specification, arm, threshold or CI procedure changed.

C1: average gainvsF10 lower95>0, meanregret<=2pp, no controller lossvsF10>10pp. C1b maxregret<=5pp. C2 paired regretimprovement CIvsS500, only if at least2 actualE<5pp; otherwiseINCONCLUSIVE/noheadroom. C3 Spearman>=0.6 ANDRMSE<=0.7 source-mean-baselineRMSE. C4 unchanged descriptivecountsE>0,L>0,both>=10pp, compared with14best using denominators/proportions. Original unknown-conservative statuses and literal point gate reporting retained. Fullcohort condition now n==8; that administrative change does not lower any threshold.

Reducing the controller count to n=8 reduces precision for C3 and averages/CIs; C3 and CI uncertainty are expected to widen, not grounds to relax gates. Actual state-cluster intervals remain data-dependent, so every interval is not mathematically guaranteed wider. Maintain5000 shared64-state clusteredbootstrap draws, seed202610131201, fourfutureseeds perstate, no controllerresampling. Do not tune based on n=8 results.

## Precision incident and correction

poc/field_controller.py:25 changes globaljax_enable_x64=True on import. The first new probe entry setFalse before import, which was overwritten; old probe448raws,R5 features and target choices are invalid for this confirmation, preserved unchanged. Source model definition/thresholds were not tuned. Initial gate workers were stopped by x64 assertion before any physical simulatorstep; no final success label exists. A zero-stepquery with explicitFalse AFTER imports exactly reproduced a historicalR5 firstaction byte-for-byte (PRECISION_IMPORT_AUDIT.json).

Corrected code explicitly setsjax_enable_x64=False after all numerical/controller imports and beforecheckpointload andaction calls. Record x64False in every probe andfullrollout, assert checkpointsparameters/sampleoutputsfloat32 andretained physicalnumpyfloat64 unchanged. Repeat pure forwardcompatibility on8finals with correctprecision, without env.step/task labels. Do not modify controller, physics, projection or success functions. Original32 best cases must reproduce outcome,length,first-actionfloat64bytes exactly before main.

Before generating anyfinal success labels: complete256correct probes, independently auditstats/precision, freezeR5features withUTC/SHA, retrain existing fixed source-onlyEfits andfreeze all8Ehat/G_E choices withUTC/SHA, verifyall32consistencycases. The sourcefit parameters should agree with original fixed sourcefits; only corrected target observations may change projections/choices. This is repair to the alreadyrequiredFP32 semantics, not a post-label feature/model search. No priorFP64-derived feature or choice is reused; compatible old/source evidence continues to be reused. Unknowns undercorrectsemantics neverrerun, neverfillzero. Earlier wrong-precision evidence is scientifically incompatible, preserved not silently overwritten or relabeled as ordinaryunknown.

## States, DB, storage and queue

Reuse the original outcome-blind64states generated beforelabels, seeds2026210000–2026210063, and originalsuccessfuldedup receipt. They were registered as plannedstates but never produced final success evidence. Check identity/physicalhashes stillsame andall8final task-success DB rowsstill0. No adaptive newdraw. New active namespace/directory fp32_n8/ inside the authorizedround12directory. New experimentUID distinguishes correctedcode/precision from invalid preparation; dedup is checked against all compatibleevidence, not just namespace. Original files andreceipts stayunchanged; invalidityannotatedinnewmanifest/report. Sixexcludedcheckpoints never appear in newplannedrollouts.

Pipeline: compatibleforwardaudit ->metadataonlyunlabeledpreflight ->256probes ->stats/precisionaudit+R5freeze ->source-onlymodels+choicefreeze ->DBdesign/preflight ->32bestgate ->singlemerger/gateverification ->12288 genuinelymissingmain ->singlemerger/postflight ->frozenendpointanalysis ->mandatoryRingvideoentry/report. Final starts mustcarry bothfreezehashes andpassedgate. Inspect newfinaloutcomes onlyafter allrequestedraw files present. Standardworker-localcache_writerjournals/singlemerger; no workerSQLitewrites.

All outputs inside diagnostics/controller_new_checkpoint_confirmation_v1/, except explicitDBlifecycle. Cancel supersededheldjobs; reserve correctedpipeline using17CPUworker+1orchestration budget,totaluser<=18CPU,<=3GPU,noGPU needed. Main request2h runtime limit for smallercohort, shortstages15min; queuewaitexcluded. Videos remain mandatory, no pureQexemption. Do not fabricate deadlockcases or rename timeout. If strictvideoentry lacks trueRingdeadlock roles, returnnonzero and retainNUMERICAL_COMPLETE_VIDEO_INCOMPLETE, no COMPLETE batchmarker.

Reportfirstparagraph answers C1,C3,C2headroom for8cohort; disclose6parameter-identicalexclusions, oldprecisionincident, guardinterception, correction/validation, allfrozenreceipts, and lackofindependenttraining-run/cross-sceneconfirmation. Only newFP32 finaloutcomes aftercorrectedfreezes count asprospective. Preserve alloriginalexperimentalhistory, no gitcommit.
