# Conditional C2 low-cost fallback; before C1 final outcomes

Purpose: directly test whether controller-dependent Q patterns can transfer through the five requested representations on the C1 cache, if the selection-transfer gate is not met. This follows docs §3's Q/calibration fallback. It cannot replace the full C2 independent64-state confirmatory selection study, and negative results cannot identify representation failure because N1 demands far more states.

Gate: execute only after C1 final audit. If C1 qualifies, evaluate formal C2 budget first. Otherwise, preserve C2-main as NOT EXECUTED—GATE FAILED, run this DEVELOPMENT Q-only diagnostic without new rollouts.

Data: C1 original32 states, originalK16. Outcome-blind random permutation seed2026101123:16 TRAIN/8 VAL/8 TEST. Train controllers k=-1,0,.5; held k=-.5 interpolation and1 extrapolation. Use source-controller seeds0–3 for train/validation, held-controller seeds4–7 for evaluation. Same-state held-controller results on16TRAIN states separately from8 fully held states. No targetlabels in normalization/checkpoint selection. Tiny support and MCprecision strictly disclosed.

Representations all padded to256 channels, same23physical state andeta3, same64x64SiLU scalarMLP, AdamW historicalhyperparameters12000steps/checkpoints50/100/200/every200,3seeds17/23/41, common step by3fit mean sourceVALNLL. R0zeros; R1source onehot3 with unknown held identity allzeros (noadaptation); R2truek; R3existing peragent compact64 flattened256; R4local projected effectivebasis ateta0 andeta13, centered finite difference eps.001 (48channels, rotated toagent-goal frame and divided byvmax). Staticquerykey2026101121 independent of labels. R4+zeroeta action (56channels) and unprojectedRB (24channels) are prespecified mechanism ablations. No physical steps for descriptor extraction.

Mismatch: for heldk=-.5 usek=.5 descriptor, for heldk=1 usek=-1; retain actualstate andeta. R0must exactlymatch; R1unknown-versus-known contrast not interpreted as deployment gain.

Primary fallback outcomes heldcontroller Q probabilityMSE,NLL andcontroller×eta centeredpatterncorrelation. Selectiongain/regret/matched−mismatched success computed only descriptively from cached closed-loop labels, never advertised as qualified selection transfer. Sourcefixedeta chosen sourceTRAIN; targetfixedeta discovery0–3 versus eval4–7 explicitly oracle calibration, hindsightbestfixed additionalreference. Paired physicalstatebootstrap5000; all newmodel fits DEVELOPMENT. At most21 additional tinyfits and zero rollout.
