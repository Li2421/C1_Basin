from common import *
from shared_rollout_db.src.rollout_db import eta_identity
import collections

def main():
 v3=ROOT/'datasets/gap1_eta_scaling_v3_n20';v1=ROOT/'datasets/gap1_eta_scaling_v1';a=read(v3/'design_manifest.json');b=read(v1/'design_manifest.json');assets={}
 manifests={2:ROOT/'diagnostics/gap_flow_competence_v3_recovery/frozen/gap1_n2_macflow_competence_v1.json',10:ROOT/'docs/gap1_n10_freeze_20261008.json',20:ROOT/'diagnostics/gap_flow_scale_20261007/frozen/gap1_n20_flow_navigation_v1.freeze.json',50:ROOT/'diagnostics/gap_flow_scale_20261007/frozen/gap1_n50_flow_navigation_v1_manifest.json'}
 for n in [2,10,20,50]:
  spec=(b if n==50 else a)['scenario_info'][str(n)];cp=ROOT/spec['checkpoint'];mf=manifests[n]
  if not mf.exists() and n==2:mf=next((ROOT/'diagnostics/gap_flow_competence_v3_recovery/frozen').glob('*.json'))
  frozen=read(mf);assert sha(cp)==spec['controller']['flow_checkpoint_sha256'];source=ROOT/spec['source_manifest'];assert sha(source)==spec['source_manifest_sha256'];assets[str(n)]=dict(N=n,H=spec['config']['max_steps'],config=spec['config'],checkpoint=str(cp),checkpoint_sha256=sha(cp),source_manifest=str(source),source_manifest_sha256=sha(source),freeze_manifest=str(mf),freeze_manifest_sha256=sha(mf),freeze=frozen,controller=spec['controller'],controller_uid=spec['controller_uid'],old_scenario_uid=spec['scenario_uid'],expert='bottleneck_family.expert.SequentialExpert',same_checkpoint_cross_N=False)
 # Exact menu values are read back from the authoritative DB, never rounded.
 values=read(D/'controller_universal_eta_schedule_v1/protocol.json')['eta'];etas={}
 with rollout_db.connect(True) as db:
  for i in [10,13,15]:
   eid=eta_identity(values[i])[0];row=db.execute('SELECT * FROM eta WHERE eta_uid=?',(eid,)).fetchone();assert row;etas[str(i)]=dict(eta_uid=eid,values=[row['eta1'],row['eta2'],row['eta3']],canonical_hex=row['canonical_hex']);assert eta_identity(etas[str(i)]['values'])[0]==eid
 old=[]
 for path in [ROOT/'datasets/gap1_eta_scaling_v1',ROOT/'datasets/gap1_eta_scaling_v2',v3,ROOT/'datasets/gap1_eta_scaling_v4_global128']:
  design=read(path/'design_manifest.json');pool=read(path/'eta_pool.json');poolids={eta_identity(x)[0] for x in pool['eta']};counts=collections.defaultdict(collections.Counter);files=[]
  for f in sorted((path/'rollouts').glob('n*/pair_*/result.json')):
   d=read(f);n=str(d['N']);counts[n]['pairs']+=1
   for r in d.get('seeds',{}).values():
    counts[n]['records']+=1;counts[n]['unknown' if r.get('numerical_failure') else 'success' if r.get('success') else 'collision' if r.get('collision') else 'timeout']+=1
   files.append(dict(path=str(f),sha256=sha(f)))
  old.append(dict(path=str(path),design_sha256=sha(path/'design_manifest.json'),pool_sha256=sha(path/'eta_pool.json'),counts={n:dict(v) for n,v in counts.items()},matching_fixed_menu=[i for i,e in etas.items() if e['eta_uid'] in poolids],has_scheduled_arm_labels=False,full_trajectories_in_pair_records=False,result_file_hashes=files))
 write(HERE/'ASSET_AUDIT.json',dict(utc=utc(),assets=assets,eta=etas,shared_map='Gap1, opening0.62, fixed16x8 rooms, radius.16, speed.5,dt.05',scale_independent_controller=dict(potential_P2='New generic geometry implementation with fixedR27P2 gains(1,.015,.01)',reciprocal='Historical safety-first composite, not ORCA. No stored action arrays in archived traces; strict first-action byte gate not independently verifiable. It also terminates on an auxiliary diagnostic, incompatible with native termination. Keep as historical descriptive asset and gap for nominal transfer; do not silently remove internal scale or change terminal standard.'),history_labels=old,necessity='New32 physical states and16 pilots eachN required by prompt; no controller training; original fixed labels retain validity where semantics match, schedules and new genericP2 evidence absent. All new states must dedup before simulation.',historical_prompt_updates='Latest frozenN10/N20/N50 acceptance supersedes older failed candidates. v3 labels substantially completed; partialQ16/earlyB15 not failure or full probabilities.',source_code_sha256={str(f):sha(f) for f in [ROOT/'docs/prompt_round36_axis4b_agent_count_20261015.md',ROOT/'docs/generalization_axes_protocol_20261015.md',ROOT/'AGENTS.md',ROOT/'shared_rollout_db/README.md',ROOT/'docs/scalable_scene_audit_20261007.md',ROOT/'bottleneck_family/README.md',*[ROOT/'bottleneck_family'/x for x in ['environment.py','scenario.py','expert.py','evaluate_flow.py','evaluate_safety_audit.py','evaluate_reciprocal_baseline.py','observation.py','run_gap1_eta.py']],D/'other_controller_classes_v1/controllers.py',ROOT/'new_benchmark_common/macflow.py',ROOT/'shared_control/basis_families.py',ROOT/'shared_control/diagnostic_corrector.py',ROOT/'shared_control/hard_projection.py',D/'double_bottleneck_eta3_basin/tools/exact_projection_retry.py']}));print('Asset audit complete', {n:v['H'] for n,v in assets.items()},flush=True)
if __name__=='__main__':main()
