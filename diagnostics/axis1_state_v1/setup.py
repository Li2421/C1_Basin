import sys,json,hashlib,difflib
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(H/n).write_text(json.dumps(x,indent=2)+'\n')
sources={}
for src,dst in [('diagnostics/controller_eta_switch_time_v1/core.py','core.py'),('diagnostics/controller_eta_switch_time_v1/run.py','historical_runner.py')]:
 p=R/src;s=p.read_text();t=s
 if dst=='core.py':
  t=t.replace("EXP='exp_controller_eta_switch_time_v1'","EXP='exp_axis1_state_v1'")
 (H/dst).write_text(t);(H/(dst+'.diff')).write_text(''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(p),tofile=str(H/dst))));sources[str(p)]=sha(p)
# Copy only requested generator function; source imports bind to immutable classes.
p=R/'ring_exchange/environment.py';s=p.read_text();a=s.index('def sample_instance(');b=s.index('\ndef sample_initial_state',a)
t=s[a:b].replace('np.random.default_rng(_split_seed(split, seed))','np.random.default_rng(np.random.SeedSequence([32015491, int(seed)]))').replace('-0.20, 0.20','-0.40, 0.40').replace('-0.18, 0.18','-0.36, 0.36').replace('2.30, 2.68','1.40, 2.90').replace('0.0, 0.11','0.0, 0.30')
(H/'shift_generator.py').write_text('import math\nimport numpy as np\nfrom ring_exchange.environment import Config,RingInstance,_angle_vector\n\n'+t)
(H/'shift_generator.diff').write_text(''.join(difflib.unified_diff(s[a:b].splitlines(True),t.splitlines(True),fromfile=str(p),tofile=str(H/'shift_generator.py'))));sources[str(p)]=sha(p)
write('COPIED_SOURCES.json',sources)
text='''# Round32 Axis1 State preregistration

UTC: UTC_TOKEN

Read and inherit docs/prompt_round32_axis1_state_20261015.md and the complete common protocol (sha256 in receipt). No existing source edits, no training, no videos, no cross-round parallel execution. Only this directory and official identity-registration/journal/merger DB writes are permitted.

Historical audit: round10 actual protocol/main_CASES/PROSPECTIVE_FRESH_STATES are authoritative, not obsolete FRESH_STATES. R16 frozen A3/A0 and R15/R7 frozen R5 are reused; R26 aligned T2 results are DEVELOPMENT. R2 gain 5.21pp [2.65,7.88] is DEVELOPMENT. New-state endpoints cannot be answered by opened historical labels. Exact identities must be preflighted, cached unknown preserved, no reruns.

1a: round3 generate('round32_axis1_id',64,202610320000,native_protocol), calling original development sampler, same as actual R10. Fixed candidate interval 202610320000..202610320063. Any DB UID/content/physical duplicate stops 1a; no redraw. 1b: copied sampler, only requested constants changed and SeedSequence tag 32015491 (distinct from 11939/27183/41771). Candidate seeds 202610321000..202610321999, take first 32 passing original environment reset and full DB/content/physical dedup; reject without outcome access. Stop 1b if fewer than 32. Unrecoverable legacy physical coordinates are disclosed. All 96 state identities/physical features frozen before evaluation labels.

Fourteen original R10 best checkpoints, Ring FF chain='field', H700, dt=.05, pseudo steps10, x64=False after all imports. Original key_for future_root2026100403, seed0..1 and state rng_namespace unchanged. Arms nominal_only, safety_baseline eta=(0,0,0), F10,F13,S300,S500: 10752+5376=16128. New nominal function calls rt.nominal(env,key) (traced=False), executes returned flow directly. No safety or correction; simulator unchanged. Unit tests for new arms mandatory.

Reproduction: before new-state rollout reproduce 16 actual round10 records. Deterministically first two states, seed0..1, four schedule-dev controllers, F10. Full position arrays, first action float64 bytes, termination and step count must match exactly. Reproduction uses new validation identities with original physical coordinates/rng namespace, old controller identities; new validation IDs keep expected history separate from DB reuse. Gate preflight sealed before submission; no main submission until gate passes. New execution function is separately written, references immutable copied historical make_runtime and key_for; copied historical code numerical logic unchanged. Transport normalization is local, no module monkeypatches.

Menu ['F10','F13','S300','S500'] is intersection with frozen R16 eight-arm menu. Ensemble seeds17,23,41 probability mean; controller choice argmax(mean over states), state choice argmax per state, ties menu order. A3 and A0 each frozen for both levels before main labels. Frozen R5 from controller_behavior_repr_v1/FEATURES.npz, matching names; physical23 from frozen code on actual new coordinates. G_E uses R16 frozen transform and G_E_FROZEN_CHOICES coefficients with threshold .05 (F10 if below, else S500). No refit/probes required.

Identity DB: use official connect+transaction INSERT OR IGNORE only permitted tables, exact collisions checked before insertion; preserve content hashes of all preexisting registration rows; record inserted full rows + sha256. New DB physical_state_json omits redundant timestep0 to satisfy official registered_exact_uid_v1 merger equality; state content identity still includes timestep0. Original rng namespace and full future root in state provenance/controller payload; seed_key canonical {'future_index':seed}, matching official merger semantics. Existing execution-equivalent identities reused; gate uses validation state identities. Journal records explicitly carry cache_identity_schema, scenario_uid, exact UID, eta, physical triples, checkpoint hash and future_index. Official merger single process, no custom ingestor. Unknown excluded by planner may appear missing: suppress rerun explicitly by direct exact-UID presence check, report separately.

Endpoints: ST1 (1a) and ST2 (1b) pass iff controller A3 cross-fit four-arm oracle mean regret<=3pp AND paired A3-safety 95% lower CI>0. ST3 separately per cohort and pooled equal state weighting: state-A3 minus controller-A3, CI lower>0 gain, upper<0 harm, else undetected. Cross-fit controller oracle: choose on even study_index states, evaluate odd and swap. State oracle: choose on seed0, evaluate seed1 and swap. Also report in-sample upper oracle, state-vs-controller potential. Fixed oracle only F10/F13; complete oracle four arms. Cross-fit selection ties menu order. CI 5000 paired nested bootstrap replicates, RNG202610153200: controllers resampled first, within each sampled controller resample state clusters with seeds kept together; recompute cross-fit oracles in each replicate preserving fixed halves. Unknown lower success indicator, upper adds numerical unknown; do not relabel unknown as failure; conservative comparisons report both envelopes, threshold-crossing unknown implies INCONCLUSIVE. All rates denominator includes all planned episodes. Catastrophe/harm counts use >=10pp relative F10/safety per controller. Nominal-to-safety paired transition counts and collision/timeout/unknown required. Shift description stratified by min start/goal radius (<2.3 vs >=2.3), max initial speed (>0.11 vs <=0.11), angular asymmetry (start residual spread>.4 vs <=.4); no effect-based thresholds.

ST1-ST3 CONFIRMATORY only with preregistration, frozen choices before labels, passed reproduction gate and complete data. Blocked items explicitly N/A/gaps, not empirical rejection. History DEVELOPMENT rows T2 R16/R26 and R2 always preserved. Required deliveries integrity begin/end+comparison, history, gate, DB pre/postflight/insert receipts, results.json, AXIS_TABLE.md/csv common method columns, gaps, REPORT first-paragraph integrity+ST1/2/3, HANDOFF. Resource total<=18CPU,<=2GPU; squeue before work/submission; this round CPU only. Finish round32 before reading/executing round33.
'''.replace('UTC_TOKEN',datetime.now(timezone.utc).isoformat())
(H/'PREREGISTRATION.md').write_text(text)
write('PREREGISTRATION.receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(H/'PREREGISTRATION.md'),'prompt_sha256':sha(R/'docs/prompt_round32_axis1_state_20261015.md'),'common_protocol_sha256':sha(R/'docs/generalization_axes_protocol_20261015.md')})
print('Preregistration sealed')
