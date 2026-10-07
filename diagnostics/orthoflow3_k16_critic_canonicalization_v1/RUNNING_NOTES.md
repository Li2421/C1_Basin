# Unfinished execution handoff (not a final report)

Current phase: **A, rollout acquisition running**. Phase B and both queued audits have **not** begun. No new critic checkpoint exists yet. Do not claim readiness.

## Commands / environment

- CWD `/home/zhihan/research/Basin_C1`; Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`.
- Use `PYTHONPATH` equal cwd, `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=''`, OMP/OPENBLAS threads1. JAX prints a CUDA plugin warning even in CPU mode; it has not prevented execution.
- Preserve dirty README/scripts/test.sh and untracked canonical directories. No canonical code or old model/dataset has been edited by this task.
- Current jobs: 1442 (24 shards,18 concurrently) v2 K16 Q; 1461 (18 shards,12 concurrent, waits for1442_0..17) extra initial-state Q;1462 collects after both1442 and1461;1463 three canonical critic training seeds after1462. Monitor total resource use, normally leave >=6CPU and>=2GPU shards for lab across user's tasks. No GPU used here. Increase1461 to18 only once1442 is fully finished and no other user task consumes reserved share.
- **Resource update16:22local:** other user task1467 (`c1-generalize`,3x2CPU+3GPU) started. Reduced1442 to12active workers and requeued ONLY own shards12..17 (allcompleted DBrows retained; checked27lastjournalrows,0missing).1461 throttle now6 while<=6tail1442shards may coexist;1465 throttle12. Increase caps only based on actual other-task allocation and reserve rule. `sacct` is disabled; use squeue/scontrol and run_shard JSONs for completion.1467 is NOT ours; do not alter it.
- **Update16:28–16:32local:**1467 left queue; restored1442/1465 throttle18. Removed1461dependency so it can use spare CPU immediately; initially2workers, now3as1442active dropped to15. **Maintain1461cap <=18 - running1442 - other user's CPUs** (and account for other active own workers) until1442finishes, then increase as allowed. Collect1462still depends on BOTH1442 and1461. Original245-state batch >42kexecutions,9shards completed; initial expansion running. No PhaseB work started.
- Additional dependent jobs queued:1464 (`freeze_confirmation_a.sbatch`, after1463) executes pre-registered `review_phase_a.py` guard (including direct canonical hash checks) and freezes accepted critic ONLY if guards pass, then creates fresh cohort/proposals. If no candidate passes it exits2 and no cohort is created.1465 (36shards%18, after1464) fullQ16 fresh confirmation. No PhaseB job exists.

## Implemented, tested so far

- `phase_a.py freeze/preflight/run/collect`: reuses original models and rollout runtimes, DB preflight, immediate insertion, full canonical Q16 for all17proposals and independent eta0 reference. Numerical failures initial+3identical retries; retained uncertified, never imputed. Existing uncertified entries not re-executed ad infinitum.
- All245 v2 states frozen.4165proposal tuples.70560total seed slots incl B0;3199cached;67361missing initially.
- Exact all48 historical validation mean/first4sample matches.
- `initial_expansion.py`: v2 metadata shows Four timestep101..550 and Ring154..350, no t0. Outcome-blind hash selection of32train+8validation source parents per new scenario.80additional conditioning states,23040missing Q slots incl B0. No test sources. Stored under `initial_expansion/`; no change to v2 or the first frozen proposal manifest.
- `train_phase_a.py`: canonical unchanged Critic, empirical-Q BCE,AdamW1e-3/wd1e-4/clip5,4000steps max,3seeds17/23/41,96state draws/scenario/step. Equal scenario/uniform state,80% uniform frozen proposal+20% v2 full>=16valid-seed Q. Partial/uncertified fractions not treated as complete Q16. No superseded ranking/LCB experiments. Logs per-state/source exposure. Uses both v2+initial evidence; validation is64conditions (16DB+24Four+24Ring), not64independent parents.
- Training is gated on complete seed acquisition and no valid collision. `metrics` keeps unknown robust status and Q bounds separate.
- `confirmation_a.py` written but **not run**. Requires `phase_a/critic_frozen.json` first; then creates fresh24DB/24Four/60Ring, original distributions. Generator unchanged,16stochastic+frozen mean candidate,critic choices frozen before outcomes. Q16 allproposals and hard-safety-only B0 with matched future seeds. Do not create this cohort before dev decisions finish.
- `test_phase_a.py`:4tests passed: hashes unchanged, original splits/shapes finite, exact nested proposals/no extras, extra parents inherit split.
- `provenance.py` recorded Stage-I hashes, current Ring safety18419e...,basisf78c1b...; Double EXPECTED source/checkpoint hashes pass. runtime raw inherited `provenance_experiment=basin_dataset_v1` is documented; new DB experiment/source/stage/proposal-hash identify actual acquisition.
- New code syntax compiled. Full relevant regressions still required at final completion.

## Next work

1. Monitor1442/1461 and resources; completed rollouts preserved in globalDB. Do not repeat compatible seeds.
2. Ensure1462collect succeeds and1463trains. Compare each dev critic to old on exactly same proposals. Inspect Ring gap/exploitation and DB/Four <=5pp guard. `acceptance_protocol.json` and `review_phase_a.py` now implement the dev guard, manifest freeze and DEVELOPMENT_FREEZE.md. Need inspect results and finalize full training-data manifest/report; do not silently pick a failing checkpoint. Additional conventional train/dev data only if evidence clearly demands it.
3. Freeze critic choice (`development_frozen`,checkpoint/checkpoint_sha256,generator_sha256) only after dev decisions. Run fresh confirmation once; no post-confirmation adaptation. Finish/freeze/report Phase A before B starts.
4. Then implement Phase B in isolated files/derived dataset. Ring suffix -> radial/CCW local frame. Four physical C4/agent canonicalization, shared equivalent-slot normalization, physical context handling. Preserve original v2 labels and physical state identity. No upstream MACFlow changes. Distinguish passive transformed Flow vs actively rerun non-equivariant Four MACFlow; do not claim h-only canonicalization guarantees full closed-loop symmetry. Use latest architectures/recipe, generator-first retraining, train/dev performance+metamorphic checks, then fresh held-out rotations. Finish required outputs/status.
5. Execute queued contract audit and then complete red-team task per `TASK_QUEUE.md` and original full user instructions, reading actual accepted artifacts. They are separate stages, not authorization to modify current phase concurrently. Astra delegation allowed only in those specified narrow read-only audit roles.

Latest observed progress when this note was written: >10k new rollout attempts, no recorded numerical or collision attempts at that point,18 CPU workers active. This is not a frozen scientific result.
