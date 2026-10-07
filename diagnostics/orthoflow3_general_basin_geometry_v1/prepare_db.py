#!/usr/bin/env python3
from audit import *
DB=D/'double_bottleneck_eta_basis_redesign';DATA=D/'double_bottleneck_initial_state_coverage/data/untouched_test_pool'
def main():
 # Verify genuine joint4-agent policy rather than the historical two-dyad adapter.
 refs={D/'double_bottleneck_recovery_density_final/model/ckpt_selected.pkl':'6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd',
 ROOT/'double_bottleneck/environment.py':'3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc',
 ROOT/'shared_control/hard_projection.py':'847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79',
 ROOT/'double_bottleneck/flowbc_4a_agent.py':'02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8'}
 for p,h in refs.items():assert sha(p)==h,(p,sha(p))
 data=json.load(open(DATA/'manifest.json'));families={r['family_id']:r for r in data['files']}
 # Source-hash order uses only family ID, not old Flow or eta outcomes.
 ordered=sorted(families,key=lambda s:hashlib.sha256((s+'geometry-v1').encode()).hexdigest());chosen=ordered[:4]
 states=[dict(state_id='DB_T0_'+str(i),family_id=f,source_group=f,dataset=str(DATA),initial_flow_root=2026092906,future_root=2026092907,rng_namespace=i,
  phase='true_t0',selection='first4 hash-ordered independent IC families; all expert modes deduplicated',h_semantics='exact initial observation[4,18], current raw Flow and projected safe Flow; not GiveWay214') for i,f in enumerate(chosen)]
 dump('double_bottleneck_state_panel.json',states)
 # Existing geometry-guided targets: three frequent robust centers, not a new Sobol sweep.
 prior=json.load(open(DB/'full_seed_robustness.json'))['episodes'];modes=Counter(tuple(r['theta']) for e in prior for r in e['candidate_seed_results'] if r['Q_seed']>=.75)
 centers=[v for v,n in modes.most_common(3)]
 dump('double_bottleneck_cost_estimate.json',{'preflight_continuations':8,'source_runtime':'historical per-continuation .6--2s; allow20s including loading/JIT/projection retries',
   'full_stage_not_launched_before_preflight':True,'cache_reuse':'historical multi-seed results resample xi0, so NOT exact Q64 for fixed h0; discovery only',
   'frozen_hashes':{str(p):h for p,h in refs.items()},'candidate_centers':centers})
 jobs=[dict(state_id=s['state_id'],eta=list(v),future_index=0,phase='db_preflight',probe_id=f"{s['state_id']}_{j}") for s in states for j,v in enumerate([centers[0],(0,0,0)])]
 p=HERE/'plans/db_preflight';p.mkdir(parents=True,exist_ok=True)
 for sh in range(4):
  with open(p/f'shard{sh}.jsonl','w') as f:
   for i,r in enumerate(jobs):
    if i%4==sh:f.write(json.dumps(r)+'\n')
 write('scenario_compatibility.csv',[
 {'scenario':'ToyGiveWay_2A','compatible':True,'frozen_flow':'seed0 checkpoint','same_basis':True,'persistent':True,'two_projections':True,'exact_Q64_cache':1131,'notes':'single_integrator alias is NOT another scenario'},
 {'scenario':'DoubleBottleneck_4A','compatible':True,'frozen_flow':str(next(iter(refs))),'same_basis':True,'persistent':True,'two_projections':True,'exact_Q64_cache':0,'notes':'joint4A checkpoint verified; old robustness resamples xi0, discovery-only. Condition new t0 panel explicitly.'},
 {'scenario':'ToyGiveWay_alias_single_integrator','compatible':'ALIAS','frozen_flow':'identical','same_basis':True,'persistent':True,'two_projections':True,'exact_Q64_cache':0,'notes':'not independent scenario'},
 {'scenario':'DoubleBottleneck_fixed_dyad_adapter','compatible':False,'frozen_flow':'two dyad adapter','same_basis':'not_authoritative_joint','persistent':True,'two_projections':True,'exact_Q64_cache':0,'notes':'not used; superseded diagnostic adapter'}])
 print('Verified joint4A scenario; frozen4 states:',chosen)
if __name__=='__main__':main()
