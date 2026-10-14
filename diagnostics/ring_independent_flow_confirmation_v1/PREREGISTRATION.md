# Round13b independent Ring Flow confirmation: preregistration

UTC: 2026-10-13T14:37:38.242412+00:00

## Prior evidence and gap

Read AGENTS.md, shared_rollout_db/README.md, docs/batch_videos.md, round12 fp32_n8 preregistration, receipts, report and handoff, all 14 training config/summary files, and new_benchmark_common/training.py. Reuse round10 source14 best-controller E labels, round7/R12 R5 source features and 16 probe states, round12 FP32 repair and frozen model/selector definition, round10/R12 32-case replay gate, shared DB compatible old evidence, and round12 video provenance. Round12's 8 final checkpoints were same-run siblings; independent-seed checkpoint evidence is absent. New training is necessary to answer the independent-run question. New target R5 and full rollout labels cannot be inferred from old checkpoints or predictions. Existing rollout labels remain valid when execution semantics match; changing critic representation alone does not invalidate them.

## Training plan

Use each original config and the exact manifest hash below. Change only seed to original+10000. Preserve steps, batch size, all sampling/model options and fixed-dev best checkpoint selection. First retrain base_u_v2 at original seed 3104 and 12000 steps. Compare parameter hash and full dev loss curve; both best and final fixed dev relative errors must be <=1%, otherwise stop before new-seed training. Check every new parameter hash against all historical checkpoints; report parameter L2 and relative L2 to its original best. All generated files remain within this directory.

- base_u_macflow: 3102 -> 13102; 5000 steps; diagnostics/ring_exchange_stage1/base_u_dataset; manifest SHA256 609f45f7543b922db2f0ca5b317f06334e392948a2f70c22dbc3296502c8da86
- base_u_plus_w_v3_macflow: 3110 -> 13110; 60000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v3_dataset; manifest SHA256 bd611622e9c8a80948e0407479ee33dbb52298f3eb59a203a5f787809645de8b
- base_u_plus_w_v4_macflow: 3113 -> 13113; 80000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v4_dataset; manifest SHA256 bce651f6aeb32710e26001c641c3554e00a11b0d7866c1a977b6f241d07400cf
- base_u_plus_w_v4_train_targeted_macflow: 3115 -> 13115; 80000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v4_train_targeted_dataset; manifest SHA256 1ea5d99b338a00fe467e8188392a938b77e001ccb3897ebd57e978c5ca31cf69
- base_u_plus_w_v4_wide_macflow: 3114 -> 13114; 80000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v4_dataset; manifest SHA256 bce651f6aeb32710e26001c641c3554e00a11b0d7866c1a977b6f241d07400cf
- base_u_plus_w_v5_macflow: 3116 -> 13116; 100000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v5_dataset; manifest SHA256 04d12430ab57a28affb1ded33d9aac8ba9656274e073a2c1057c159cdfd616a5
- base_u_plus_w_v6_macflow: 3122 -> 13122; 100000 steps; diagnostics/ring_exchange_stage1/base_u_plus_w_v6_dataset; manifest SHA256 6b664b19b590baf530e709e692447d30b2cff72387b276ebaa3728ede395532e
- base_u_v10_local_macflow: 3127 -> 13127; 100000 steps; diagnostics/ring_exchange_stage1/base_u_v10_local_dataset; manifest SHA256 dd72095a957acfb9fe47ff9b6478329bb06424c91bd9dd401d6e1b3c74431732
- base_u_v2_macflow: 3104 -> 13104; 12000 steps; diagnostics/ring_exchange_stage1/base_u_v2_dataset; manifest SHA256 b49dcafe393b4f5f4103cad8e633c589d0dd8ee08200d52ce512a62941a57d1b
- base_u_v3_macflow: 3109 -> 13109; 60000 steps; diagnostics/ring_exchange_stage1/base_u_v3_dataset; manifest SHA256 d7987615ee03bdd33f18d873c527856e19ca491be52a50d9d61ed963ec7bd242
- base_u_v4_macflow: 3112 -> 13112; 80000 steps; diagnostics/ring_exchange_stage1/base_u_v4_dataset; manifest SHA256 115b20011046267a7be9dd34f5742755c53e7b4c06a0d00aad6b7450e7c17515
- base_u_v7_local_macflow: 3124 -> 13124; 100000 steps; diagnostics/ring_exchange_stage1/base_u_v7_local_dataset; manifest SHA256 1f64eb949e360fa0a99816ed118760435a22475956e0dbd895fdbdebc88447dd
- base_u_v8_local_macflow: 3125 -> 13125; 100000 steps; diagnostics/ring_exchange_stage1/base_u_v8_local_dataset; manifest SHA256 4dc96b770ddb0e1a7981b50df5b7d64c03b700ba09ce6146d381b0dd36769f0c
- base_u_v9_local_macflow: 3126 -> 13126; 100000 steps; diagnostics/ring_exchange_stage1/base_u_v8_local_dataset; manifest SHA256 4dc96b770ddb0e1a7981b50df5b7d64c03b700ba09ce6146d381b0dd36769f0c

## Frozen prospective protocol

Before any new controller success label, complete and freeze FP32 R5 probes: 16 historical states x eta10/eta13 x 200 steps per controller, same round12 stats and source-only 155D preprocessing. Then fit fixed ridge alpha=1 with R5-PCA2 inputs and round10 best-controller E targets; primary excludes the same-recipe original best (13 sources), sensitivity uses all14. Unpenalized intercept, no clipping. Freeze every Ehat and G_E choice; Ehat<5pp chooses F10, otherwise S500. No target outcome is used in probes or fitting. Full menu F10,F13,S300,S500; F2,F4 oracle only. 64 new true-t0 Ring states by round10 generator, deduplicated by UID and physical content against all DB states, including round12. Four future seeds per state. Exact 14x64x6x4=21504 requests, design/preflight/missing-only/journal/single merger/postflight. Replay 32 historical best records exactly before main outcomes. Unknown remains unknown and is never rerun. Timeout is not deadlock.

## Frozen endpoints

Reuse round12 5000 shared-state-cluster bootstrap (seed 202610131201), four seeds kept together; unknown lower/upper envelopes and conservative status logic. C1: G_E-F10 mean gain 95% lower>0, mean regret<=2pp, every controller gain>=-10pp. C1b secondary max regret<=5pp. C2 requires >=2 actual controller E<5pp, then paired G_E-S500 improvement CI lower>0. C3: Spearman(Ehat,E)>=0.6 and RMSE <=0.7 times source-mean baseline RMSE. C4 descriptive E,L signs only. Report S300 vs S500 regret and F2/F4 oracle prevalence. No threshold, model, arm or state adaptation after seeing new outcomes. Full Ring video gate runs before complete-batch claim; true deadlock roles may remain unavailable under unchanged native runtime criterion, and that gap will be reported.

## Resource and scope

Maximum combined user load 18 CPU shards and 3 GPU shards; inspect live queue before jobs. No core controller/physics/safety/success edits. No historical edits, deletions or git commit. New artifacts only diagnostics/ring_independent_flow_confirmation_v1; mandatory shared SQLite lifecycle writes are the documented DB exception.
