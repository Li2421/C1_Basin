from core import *
import copy,sqlite3
from shared_rollout_db.src import rollout_db as dbmod
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
def localdb():dbmod.DB_PATH=HERE/'rollout.sqlite'
def main():
 assert not (HERE/'PREREGISTRATION.md').exists()
 p=read(OLD/'main_protocol.json');receipt=read(OLD/'CONTROLLER_IMPLEMENTATION_V2_FROZEN.receipt.json')
 for path,h in receipt['files'].items():assert sha(path)==h,path
 assert sha(OLD/'controllers.py')==p['controller_code_sha256'];assert sha(OLD/'run.py')==p['runner_sha256']
 paths=[OLD/x for x in ['controllers.py','run.py','core.py','base_protocol.json','main_protocol.json','main_CASES.json','LABELS.npz','FROZEN_CHOICES.json','REPORT.md','results.json','EVAL_STATES.json','CONTROLLER_IMPLEMENTATION_V2_FROZEN.receipt.json']]
 for v in p['base_profiles'].values():
  v=v['payload']
  for key,hkey in [('flow_checkpoint_path','flow_checkpoint_sha256'),('adapter_manifest_path','adapter_manifest_sha256')]:
   if v.get(key):
    path=Path(v[key]);paths.append(path)
    if v.get(hkey):assert sha(path)==v[hkey],path
 paths += [FIELD/'poc/field_controller.py',ROOT/'new_benchmark_common/safety_eta3.py',ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py',ROOT/'diagnostics/controller_q_diversity_v1/runtime.py',ROOT/'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/execution.py',ROOT/'ring_exchange/environment.py',ROOT/'shared_control/hard_projection.py']
 write(HERE/'FROZEN_INPUTS.json',dict(utc=utc(),files={str(x):sha(x) for x in paths},controller_parameter_payloads={n:v['payload'] for n,v in p['base_profiles'].items()}))
 # Read-only SQLite backup: all new writes, including the exact cache index,
 # remain in the only user-authorized directory.
 with connect(True) as src,sqlite3.connect(HERE/'rollout.sqlite') as dst:src.backup(dst)
 localdb()
 with connect(True) as db:
  hist=[dict(r) for r in db.execute("SELECT controller_uid,config_json FROM controller_config WHERE config_json LIKE '%no_safety%' OR config_json LIKE '%hard_safety%' OR config_json LIKE '%nominal_only%'")]
  state_rows=[dict(r) for r in db.execute('SELECT state_uid,controller_uid,seed_key,outcome FROM rollout WHERE state_uid IN ('+','.join('?' for _ in p['states'])+')',[s['state_uid'] for s in p['states']])]
 baseline_hist=[]
 for h in hist:
  v=json.loads(h['config_json']);baseline_hist.append(dict(controller_uid=h['controller_uid'],payload=v,matching_requested_states=sum(r['controller_uid']==h['controller_uid'] for r in state_rows)))
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),reused_round27=str(OLD),reused_labels=6144,reused_states=32,reused_controllers=16,reused_choices='A3_full frozen',historical_baseline_candidates=baseline_hist,requested_state_existing_rows=len(state_rows),compatible_baseline_rows_on_requested_states=sum(h['matching_requested_states'] for h in baseline_hist),evidence_gap='No equivalent baseline label for requested state/controller/seed unless the attached candidate audit proves exact compatibility. Orthoflow3 labels cannot answer uncorrected baseline outcomes.',DB_policy='Read-only authoritative DB snapshot; local official preflight/journal/merge/postflight, global DB unchanged to obey new-directory-only writes.'))
 assert not any(h['matching_requested_states'] for h in baseline_hist),'Historical candidate requires semantic reuse audit before rollout'
 text=f'''Round29 preregistration — frozen at {utc()} UTC

Authority: docs/prompt_round29_nominal_safety_baselines_20261015.md.
Read AGENTS.md, shared_rollout_db/README.md, round27 REPORT/HANDOFF and frozen code.
16 frozen controllers (M0–M5, P0–P5, four Flow TT), accepted round27 32 states,
seed 0–1 and identical exact seed keys; 2 chains: no_safety and hard_safety.
2048 complete rollouts, only genuinely missing; retain unknown without rerunning.
Controller source loaded directly from round27; parameters/checkpoints hashes
verified against V2 receipt and main protocol. FROZEN_INPUTS is the input receipt.
No controller tuning, physics/projection/event/success changes or training.
After imports AND each lazy runtime construction restore jax_enable_x64=False,
as round27 protocol requires; assert at each physical step and record status.
no_safety executes native nominal directly; hard_safety executes exactly one
native safety.project(nominal). These follow safety_eta3 chain dispatch with
round27 RNG key_for; neither computes a correction. Orthoflow3 gate calls the
frozen Observed Runtime.action(old), with unchanged schedule and RNG.

Before baseline execution replay 8 known round27 main records: M0/M5, P0/P5,
all four Flow references; deterministic first eligible known record, alternating
F10/S500 when possible. Compare native termination, episode_length, and first
action dtype/shape/bytes against stored traces. Also compare full positions.
Stop on any mismatch. These 8 explicitly required validation repeats are listed
separately from missing baseline execution; never rerun unknown.

Metrics: category and controller × eight methods: nominal only, safety baseline,
F10, F13, best fixed oracle (max of aggregate F10/F13/F2 per controller), frozen
A3_full, S500, full-menu oracle (max aggregate of six arms per controller).
Items 3–8 reuse LABELS and choices; verify point estimates against round27.
Unknown is neither known failure nor success: report identified success interval
[success/total,(success+unknown)/total], known-only rate and count. To reproduce
round27 exactly, table point values and CIs describe confirmed-success fraction
(lower bound), explicitly not a failure classification. Also give upper bounds
and bootstrap envelopes allowing unknown outcomes. Numerical failure rate is
unknown incidence. Collision/timeout/stagnation separately, swept simulator
clearances from every step, success requires all agents collision-free arrival.
Stagnation descriptive: no success, >=100 steps, final 100-step reduction in
sum goal distances <0.05; never call it deadlock.
5000 paired hierarchical bootstrap replicates: resample controllers first within
category, then 32 state clusters within sampled controller, keep both seeds and
all methods paired. Per-controller CIs resample states only. Bootstrap seed
202610152901 + group index. Recompute aggregate oracle maxima each replicate.
Key contrasts A3_full minus each baseline, percentile 95% CI plus unknown envelope.

Pure Q evaluation data: no video entry, per AGENTS.md explicit exception.
No git commit; no historical asset changes. All files (including local cache DB,
journals, Slurm logs/caches) written only here. The shared authoritative DB is
read-only snapshotted, since writing it would violate the file scope; official
planner and single-process merger operate on this local snapshot. Global index
ingestion is deferred; this limitation must appear in REPORT/HANDOFF.
CPU shard cap across ALL user's workstation jobs 18, GPU shard cap 2; use CPU
only, wait for available resources and count existing jobs before execution.
Design → preflight → gate → missing-only → journal → single merge → postflight.
'''
 (HERE/'PREREGISTRATION.md').write_text(text);write(HERE/'PREREGISTRATION.receipt.json',dict(utc=utc(),sha256=sha(HERE/'PREREGISTRATION.md'),inputs_sha256=sha(HERE/'FROZEN_INPUTS.json')))
 oldcases=read(OLD/'main_CASES.json');gate=[]
 for i,n in enumerate(['M0','M5','P0','P5']+[n for n in p['controller_order'] if n.startswith('base_')]):
  arm=['F10','S500'][i%2]
  c=next(c for c in oldcases if c['base_controller_name']==n and c['arm']==arm and not read(OLD/'main/raw'/f"{c['rollout_uid']}.json")['numerical_failure'])
  gate.append(dict(c,chain='orthoflow3'))
 write(HERE/'gate_CASES.json',gate)
 gm=HERE/'gate_PLANNED_ROLLOUTS.json';write(gm,dict(requests=[dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[c['seed_key']]) for c in gate]));write(HERE/'gate_PREFLIGHT.json',preflight(gm))
 profiles={};rows=[]
 for n,old in p['base_profiles'].items():
  for chain in ['no_safety','hard_safety']:
   v=copy.deepcopy(old['payload']);v.update(chain=chain,placement='nominal' if chain=='no_safety' else 'nominal_project',conditioning=chain,execution_semantics='round27_native_nominal_'+chain+'_v1');profile=dict(controller_uid=uid('ctl',v),payload=v);profiles[n+'__'+chain]=profile
   for oc in oldcases:
    if oc['base_controller_name']!=n or oc['arm']!='F10':continue
    c=copy.deepcopy(oc);c.update(chain=chain,arm=chain,controller_name=n+'__'+chain,controller_uid=profile['controller_uid'],eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_step=700)
    c['eta_uid']=eta_identity([0,0,0])[0];c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));rows.append(c)
 p.update(controllers=profiles,experiment_uid=EXP);write(HERE/'protocol.json',p);write(HERE/'main_CASES.json',rows);assert len(rows)==2048
 with connect() as db,transaction(db):
  db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round29 nominal safety baselines',str(HERE),sha(HERE/'PREREGISTRATION.md'),sha(HERE/'runtime.py'),canonical(dict(pure_Q=True,local_snapshot=True))))
  eid,eta,hx=eta_identity([0,0,0]);db.execute('INSERT OR IGNORE INTO eta VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)',(eid,*eta,canonical(eta),hx))
  for ctl in profiles.values():
   v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v.get('flow_checkpoint_sha256'),None,uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])) if v['chain']=='hard_safety' else None,str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
 manifest=HERE/'main_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[c['seed_key']]) for c in rows]));q=preflight(manifest);write(HERE/'main_PREFLIGHT.json',q)
 assert q['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 write(HERE/'main_MISSING_CASES.json',missing);write(HERE/'DESIGN_RECEIPT.json',dict(utc=utc(),planned=2048,missing=len(missing),gate=8,protocol_sha256=sha(HERE/'protocol.json')))
 print('DESIGN',len(missing),flush=True)
if __name__=='__main__':main()
