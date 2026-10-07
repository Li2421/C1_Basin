#!/usr/bin/env python3
from core import *
def main():
 old=read(H/'frozen_existing_exact.csv');store={r['eta_key']:r for r in old};raw=defaultdict(dict);roles=defaultdict(set);rounds=defaultdict(set)
 supplemental=read(H/'supplemental_exact_cache.csv') if (H/'supplemental_exact_cache.csv').exists() else []
 for r in supplemental:
  if r['eta_key'] in store:assert int(store[r['eta_key']]['successes'])==int(r['successes'])
  else:store[r['eta_key']]=r
 for p in H.glob('rounds/*/manifest.csv'):
  for r in read(p):roles[r['eta_key']].update(json.loads(r['roles']));rounds[r['eta_key']].add(p.parent.name)
 errors=[];actual_sources=defaultdict(set)
 aliases=json.load(open(G/'conditioning_alias_audit.json'))['accepted_only_exact_reproduced_hashes'][SID]
 for p in H.glob('raw/*/shard*.jsonl'):
  for line in open(p):
   r=json.loads(line);assert r['state_id']==SID
   assert r['source_group']==old[0]['source_group'] and r['h_conditioning_identifier']==old[0]['conditioning']
   assert r['first_step']['feature_sha256'] in aliases
   if r.get('execution_error'):errors.append(dict(file=str(p),record=r));continue
   e=r.get('eta',r.get('theta'));assert e is not None
   k=key(e);seed=int(r['future_index']);assert 0<=seed<64;actual_sources[k].add(str(p))
   if seed in raw[k]:assert raw[k][seed]['success']==r['success']
   raw[k][seed]=r
 assert not errors,errors
 c,R=frame();cache=json.load(open(H/'frozen_cache_manifest.json'));results=[]
 for k,ss in raw.items():
  if set(ss)!=set(range(64)):continue
  rr=[ss[i] for i in range(64)];e=np.array(rr[0].get('eta',rr[0].get('theta')));z=(e-AFF)/SCALE;q=(z-c)@R;success=sum(bool(r['success']) for r in rr)
  assert inside(z)[0]
  if k in store:assert int(store[k]['successes'])==success;continue
  r=dict(old[0]);r.update(state_id=SID,eta_key=k,eta1=e[0],eta2=e[1],eta3=e[2],successes=success,trials=64,Q64=success/64,B63=str(success>=63),deadlock=sum(x.get('outcome') in ['safe_deadlock','strict_deadlock','deadlock'] for x in rr),timeout=sum(x.get('outcome')=='timeout' for x in rr),collision=sum('collision' in str(x.get('outcome','')) for x in rr),sources=';'.join(str(p) for p in H.glob('raw/*/shard*.jsonl')),phases=';'.join(sorted(rounds[k])),raw_phases=';'.join(sorted(rounds[k])),geometry_role='TARGETED_TOPOLOGY_NO_MODEL_FIT',physical_steps=sum(int(x.get('continuation_steps',x.get('steps',x.get('episode_steps',0)))) for x in rr),retained_validation_batches='',in_E_bridge='True',z1=z[0],z2=z[1],z3=z[2],s1=q[0],s2=q[1],n=q[2],provenance_categories='targeted_conditional_or_corridor',prior_shape_categories='',evidence_origin='NEW_EXACT_Q64')
  r['sources']=';'.join(sorted(actual_sources[k]));store[k]=r
 for k in roles:
  r=store.get(k);results.append(dict(eta_key=k,rounds=';'.join(sorted(rounds[k])),roles=json.dumps(sorted(roles[k])),exact_Q64_available=r is not None,Q64=r['Q64'] if r else '',B63=r['B63'] if r else '',deadlock=r['deadlock'] if r else '',timeout=r['timeout'] if r else '',collision=r['collision'] if r else '',cache_reused=k in {x['eta_key'] for x in old}))
 write('exact_q64_ep0082.csv',list(store.values()),list(old[0]));write('adaptive_probe_results.csv',results)
 rt=[json.load(open(p)) for p in H.glob('raw/*/shard*_runtime.json')]
 new_count=sum(r['evidence_origin']=='NEW_EXACT_Q64' for r in store.values())
 dump('runtime_statistics.json',dict(existing_exact_Q64_reused=len(old)+len(supplemental),initial_cache_exact=len(old),supplemental_cache_exact=len(supplemental),new_exact_Q64=new_count,new_continuations=sum(x['new_continuations'] for x in rt),reused_partial_continuations=sum(x['reused_continuations'] for x in rt),new_physical_steps=sum(x['physical_steps'] for x in rt),worker_wall_seconds=sum(x['wall_seconds'] for x in rt),max_GPU_shards=2,max_CPU_threads=4,training_runs=0,notes='Completed shard ledgers only; no extrapolated outcome counts'))
 print(json.dumps({'exact_total':len(store),'new_exact':new_count,'query_keys_remaining':sum(not x['exact_Q64_available'] for x in results)}))
if __name__=='__main__':main()
