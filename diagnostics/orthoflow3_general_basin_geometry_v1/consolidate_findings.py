#!/usr/bin/env python3
"""Compact deterministic summaries; no automatic scientific acceptance."""
from audit import *

def median(rows,k):
 a=[float(r[k]) for r in rows if r.get(k) not in ('',None)];return float(np.median(a)) if a else None
def main():
 tag=sys.argv[1];inv=read(HERE/'exact_q64_inventory.csv');metrics=[r for r in read(HERE/'corrected_frozen_model_metrics.csv') if r['tag']==tag] if (HERE/'corrected_frozen_model_metrics.csv').exists() else read(HERE/f'family_metrics_{tag}.csv');adequate={r['state_id'] for r in metrics}
 sig=[r for r in read(HERE/'per_state_geometry_signature.csv') if r['state_id'] in adequate]
 stars=read(HERE/'star_convexity.csv');lines=read(HERE/'conditional_interval_structure.csv');paths=read(HERE/'connectivity_paths.csv');holes=read(HERE/'hole_vs_notch.csv')
 per_scenario=[]
 for scenario in sorted({r['scenario'] for r in inv}):
  ss=[r for r in sig if r['scenario']==scenario];ids={r['state_id'] for r in ss};ii=[r for r in inv if r['scenario']==scenario and inside((eta(r)-AFF)/SCALE)[0]];mm=[r for r in metrics if r['state_id'] in ids];ll=[r for r in lines if r['state_id'] in ids]
  record=dict(scenario=scenario,total_states=len({r['state_id'] for r in ii}),adequate_states=len(ids),exact_eta=len(ii),B63=sum(r['B63']=='True' for r in ii),median_anisotropy=median(ss,'tangent1_normal_ratio'),median_diameter=median(ss,'diameter'),median_strict_sampled_component_fraction=median(ss,'sampled_link_component_fraction'),confirmed_enclosed_holes=0,median_full_recall=median(mm,'full_recall'),median_retained_recall=median(mm,'retained_recall'),fraction_full_recall_ge_half=float(np.mean([float(r['full_recall'])>=.5 for r in mm])))
  for axis in '123':
   l=[r for r in ll if str(r['axis'])==axis];record[f'axis{axis}_lines']=len(l);record[f'axis{axis}_multiple_success_runs']=sum(int(r['sampled_success_runs'])>=2 for r in l);record[f'axis{axis}_single_success_run']=sum(int(r['sampled_success_runs'])==1 for r in l)
  per_scenario.append(record)
 write('cross_scenario_geometry_signature.csv',per_scenario)
 common=json.load(open(HERE/'common_core_summary.json'));cross=read(HERE/'cross_scenario_common_eta_matrix.csv')
 summary=dict(inventory=json.load(open(HERE/'inventory_summary.json')),adequate_states=len(adequate),fit_tag=tag,scenarios=per_scenario,strict_main_component_dominated_states=sum(float(r['sampled_link_component_fraction'])>=.9 for r in sig),strict_sampled_component_range=[min(float(r['sampled_link_component_fraction']) for r in sig),max(float(r['sampled_link_component_fraction']) for r in sig)],
 star_segments=dict(all_sampled_success=sum(r['all_three_B63']=='True' for r in stars),segments=len(stars),interior_B63=sum(int(r['B63']) for r in stars),interior_tests=sum(int(r['tested']) for r in stars)),
 success_detours=[r for r in paths if r['kind']=='success_detour'],failure_corridor_classes=dict(Counter(r['classification'] for r in holes)),confirmed_enclosed_holes=0,holes_absent_proven=False,
 common_core_best=max(common,key=lambda r:r['B63']),common_core_positive_volume_verified=False,cross_scenario_common_modes_B63=sum(r['B63']=='True' for r in cross),cross_scenario_common_modes_n=len(cross),training_runs=0,
 caveats=['sampled links do not certify continuous paths','no sampled-link graph spans90percent; no true disconnectedness inferred','positive heldout is retrospective; repeated model comparison is adaptive','all exact negatives hard constraints; adaptive acquisition is not uniform occupancy sampling','zero6perstate fresh validation is finite empirical evidence, not a population confidence certificate','no actual success-volume fraction estimated from nonuniform evidence'])
 dump('current_findings.json',summary)
 # Actual work, not counts of copied cache rows or validation aliases.
 rt=[(p,json.load(open(p))) for p in HERE.glob('raw/*/shard*_runtime.json')]
 db=[]
 for p in HERE.glob('db_raw/*.jsonl'):
  for line in open(p):
   if line.strip():db.append(json.loads(line))
 keys=[(r['state_id'],r.get('eta_key',key(r['eta'])),r['future_index']) for r in db];assert len(keys)==len(set(keys)),'Duplicate newly executed DB continuation'
 resources=[json.loads(l) for l in open(HERE/'resource_snapshots.jsonl')]
 new=sum(r['new_continuations'] for _,r in rt)+len(db);steps=sum(r['physical_steps'] for _,r in rt)+sum(int(r.get('episode_steps',r.get('episode_steps_before_solver_failure',0))) for r in db)
 workstart=min(p.stat().st_mtime-r['wall_seconds'] for p,r in rt);workend=max([p.stat().st_mtime for p,r in rt]+[p.stat().st_mtime for p in HERE.glob('db_raw/*.jsonl')])
 runtime=dict(new_continuations=new,new_physical_steps=steps,new_toy_continuations=sum(r['new_continuations'] for _,r in rt),new_DB_continuation_attempts=len(db),invalid_numerical_attempts=sum(not r['scientific_outcome_valid'] for r in db),prior_exact_Q64_reused=1131,prior_exact_continuations_reused=1131*64,additional_lower_seed_continuations_reused=sum(r['reused_continuations'] for _,r in rt),complete_new_exact_Q64=len(inv)-1131,internal_DB_validation_reuse_not_new_work=24,
 max_GPU_shards=6,max_CPU_threads_allocated=12,GPU_memory_peak_observed_MiB=max(r.get('gpu_memory_MiB',0) for r in resources),GPU_worker_RSS_peak_observed_MiB=max(r.get('gpu_process_RSS_MiB',0) for r in resources),minimum_RAM_available_observed_GiB=min(r.get('RAM_available_GiB',1e99) for r in resources),rollout_execution_window_seconds=workend-workstart,execution_window_note='Earliest worker result mtime minus recorded worker duration to last raw result mtime; includes inter-stage analysis, excludes some initial JAX initialization and prior offline audit.',worker_rollout_wall_sum_seconds=sum(r['wall_seconds'] for _,r in rt)+sum(float(r.get('wall_seconds',0)) for r in db),training_seconds=0,resource_note='Slurm accounting disabled; observed process snapshots are sampled, not guaranteed lifetime maxima. GPU-shard and CPU allocation ceiling from launch manifests.')
 dump('runtime_statistics_current.json',runtime)
 panel=json.load(open(HERE/'state_panels.json'));panel['cross_scenario']={'scenario':'DoubleBottleneck_4A','state_ids':sorted(s for s in adequate if s.startswith('DB')),'selection_manifest':'double_bottleneck_state_panel.json','status':'FIXED_CURRENT_H0_COMPATIBLE_EXACT_Q64_COMPLETED','source_replacement':False};dump('state_panels.json',panel)
 print(json.dumps({'adequate_states':len(adequate),'exact_eta':len(inv),'new_continuations':new,'steps':steps,'summary':'current_findings.json'}))
if __name__=='__main__':main()
