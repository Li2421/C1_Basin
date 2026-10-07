"""Independent end-to-end checks; no rollout, fitting, or outcome changes."""
from core import *
import subprocess
import xml.etree.ElementTree as ET

def main():
 checks={}
 semantics=json.load(open(G/'conditioning_semantics.json'))
 checks['frozen_source_hashes']={p:sha(p)==v for p,v in semantics['source_sha256'].items()}
 assert all(checks['frozen_source_hashes'].values())
 cache=json.load(open(H/'frozen_cache_manifest.json'))
 assert sha(cache['inventory_source'])==cache['inventory_sha256']
 assert semantics['controller_semantics_hash']==cache['controller_semantics_hash']
 aliases=json.load(open(G/'conditioning_alias_audit.json'))['accepted_only_exact_reproduced_hashes'][SID]
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows}
 assert len(lookup)==len(rows)
 for r in rows:
  assert r['state_id']==SID and key(eta(r))==r['eta_key'] and inside(norm(r))[0]
  assert int(r['trials'])==64 and float(r['Q64'])==int(r['successes'])/64
  assert (r['B63']=='True')==(int(r['successes'])>=63)
 frozen=read(H/'frozen_existing_exact.csv');supp=read(H/'supplemental_exact_cache.csv')
 for r in frozen+supp:
  assert lookup[r['eta_key']]['successes']==r['successes']
 checks['exact_cache_preserved']=len(frozen)+len(supp)
 planned={r['eta_key'] for r in read(H/'adaptive_probe_manifest.csv') if r['cached_Q64']=='False'}
 actual=defaultdict(dict);physical=0;raw_files={};runtime_cont=0;runtime_steps=0
 for p in sorted(H.glob('raw/*/shard*.jsonl')):
  batch=p.parent.name;shard=p.stem;tasks=[json.loads(l) for l in open(H/f'plans/{batch}/{shard}.jsonl')]
  expected={(r['eta_key'],int(r['future_index'])) for r in tasks};seen=set()
  for line in open(p):
   r=json.loads(line);k=key(r['eta']);fi=int(r['future_index'])
   assert r['state_id']==SID and r['h_conditioning_identifier']==cache['conditioning']
   assert r['source_group']==frozen[0]['source_group'] and not r.get('execution_error')
   assert r['first_step']['feature_sha256'] in aliases
   assert (k,fi) in expected and (k,fi) not in seen and fi not in actual[k]
   seen.add((k,fi));actual[k][fi]=r;physical+=int(r['continuation_steps'])
  assert seen==expected
  rt=json.load(open(p.with_name(shard+'_runtime.json')))
  assert rt['tasks']==len(seen)==rt['new_continuations']+rt['reused_continuations']
  runtime_cont+=rt['new_continuations'];runtime_steps+=rt['physical_steps'];raw_files[str(p.relative_to(H))]=sha(p)
 assert set(actual)==planned
 for k,records in actual.items():
  assert set(records)==set(range(64))
  assert sum(bool(r['success']) for r in records.values())==int(lookup[k]['successes'])
  assert sum(r['outcome'] in ('safe_deadlock','strict_deadlock','deadlock') for r in records.values())==int(lookup[k]['deadlock'])
  assert sum(r['outcome']=='timeout' for r in records.values())==int(lookup[k]['timeout'])
 assert runtime_cont==64*len(planned) and physical==runtime_steps
 checks['new_eta']=len(planned);checks['new_continuations']=runtime_cont;checks['new_physical_steps']=physical
 checks['no_duplicate_eta_future_execution']=True
 assert len(lookup)==len(frozen)+len(supp)+len(planned)
 stats={r['line_id']:r for r in read(H/'transition_statistics.csv')}
 lm=read(H/'conditional_slice_manifest.csv');all_widths=[]
 for d in json.load(open(H/'line_definitions.json')):
  origin=(np.array(d['origin'])-AFF)/SCALE;v=np.array(d['direction']);seq=[]
  for k in {r['eta_key'] for r in lm if r['line_id']==d['line_id']}:
   z=norm(lookup[k]);t=float((z-origin)@v)
   assert np.linalg.norm(z-origin-t*v)<1e-9
   seq.append((t,lookup[k]['B63']=='True'))
  seq.sort();changed=[b[0]-a[0] for a,b in zip(seq,seq[1:]) if a[1]!=b[1]]
  assert len(changed)==int(stats[d['line_id']]['transition_count'])
  assert ''.join('S' if b else 'F' for _,b in seq)==stats[d['line_id']]['label_sequence']
  all_widths+=changed
 assert max(all_widths)<=.010000001
 checks['independently_verified_conditional_lines']=len(stats)
 checks['maximum_transition_bracket']=max(all_widths)
 # Rebuild observed-negative components directly from the saved links.
 negatives={r['eta_key'] for r in rows if r['B63']=='False'}
 parent={k:k for k in negatives}
 def find(k):
  while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
  return k
 positives=np.array([norm(r) for r in rows if r['B63']=='True'])
 for r in read(H/'sampled_failure_links.csv'):
  a,b=r['a'],r['b'];assert a in negatives and b in negatives
  x,y=norm(lookup[a]),norm(lookup[b]);v=y-x;length=np.linalg.norm(v)
  assert 0<length<=.050000001 and abs(length-float(r['normalized_distance']))<1e-10
  t=(positives-x)@v/(v@v);res=np.linalg.norm(positives-x-t[:,None]*v,axis=1)
  assert not np.any((t>1e-9)&(t<1-1e-9)&(res<1e-9))
  parent[find(a)]=find(b)
 boundary={find(k) for k in negatives if abs(np.max(norm(lookup[k])@HS[:,:3].T+HS[:,3]))<=1e-9}
 reachable={k for k in negatives if find(k) in boundary}
 regionstats={int(r['region_id']):r for r in read(H/'region_connectivity_summary.csv')}
 for r in json.load(open(H/'region_definitions.json'))['regions']:
  assert (r['representative_key'] in reachable)==(regionstats[r['region_id']]['representative_has_sampled_exterior_path']=='True')
 for r in read(H/'historical_interpolation_explanation.csv'):
  if r['B63']=='False':assert (r['eta_key'] in reachable)==(r['failure_sampled_exterior_path']=='True')
 checks['negative_graph_reachability_independently_verified']=True
 corridor_stats={r['corridor_id']:r for r in read(H/'failure_corridor_results.csv')}
 for d in json.load(open(H/'corridor_definitions.json')):
  if d['status']!='FROZEN':continue
  labels=[lookup[k]['B63']=='False' for k in set(d['ordered_keys'])]
  assert sum(labels)==int(corridor_stats[d['corridor_id']]['non_B63'])
  assert all(labels)==(corridor_stats[d['corridor_id']]['sampled_failure_path_valid']=='True')
  assert abs(np.max(norm(lookup[d['ordered_keys'][-1]])@HS[:,:3].T+HS[:,3]))<1e-9
 required=['protocol.md','exact_q64_ep0082.csv','internal_failure_candidates.csv','failure_connectivity_graph.csv','candidate_failure_corridors.csv','conditional_slice_manifest.csv','conditional_slice_results.csv','adaptive_probe_manifest.csv','adaptive_probe_results.csv','hole_tests.csv','transition_statistics.csv','topology_decision.json','representation_implication.md','runtime_statistics.json','final_report.md','working_state.json','evidence_index.json','hypothesis_status.json','experiment_ledger.csv']
 for name in required:assert (H/name).is_file() and (H/name).stat().st_size>0
 for name in ['eta_3d_success_failure','tangential_slices','mixed_s1_n_slices','mixed_s2_n_slices','failure_corridor_view','transition_line_examples']:
  p=H/(name+'.svg');assert p.stat().st_size>500;ET.parse(p)
 jobs=subprocess.check_output(['squeue','-h','-o','%i %j'],text=True)
 assert not any('ep0082_slice' in l for l in jobs.splitlines()),'Audit worker still active'
 decision=json.load(open(H/'topology_decision.json'));rt=json.load(open(H/'runtime_statistics.json'))
 assert decision['new_exact_Q64']==len(planned) and rt['new_continuations']==runtime_cont and rt['new_physical_steps']==physical
 checks['PASS']=True;checks['raw_hashes']=raw_files;checks['checked_local']=time.strftime('%Y-%m-%d %H:%M:%S %Z')
 dump('completion_audit.json',checks)
 state=json.load(open(H/'working_state.json'));state.update(status='COMPLETE',completed=state.get('completed',[])+['final_scientific_decision','independent_completion_audit'],next_action='None. Audit complete; do not train or fit a family automatically.');dump('working_state.json',state)
 index=json.load(open(H/'evidence_index.json'))
 for p in [G/'run_rollout_shard.py',G/'conditioning_alias_audit.json',H/'completion_audit.json',H/'topology_decision.json']:
  index[p.name]=dict(path=str(p),sha256=sha(p))
 dump('evidence_index.json',index)
 artifacts={str(p.relative_to(H)):sha(p) for p in sorted(H.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in str(p) and not any(x in p.parts for x in ('runs','logs'))}
 dump('manifest.json',dict(task='ORTHOFLOW3_EP0082_FAILURE_INTRUSION_SLICE_AUDIT_V1',status='COMPLETE',classification=decision['classification'],state_id=SID,basis_sha256=SHA,controller_semantics_hash=cache['controller_semantics_hash'],artifacts=artifacts,no_training=True))
 print(json.dumps({k:v for k,v in checks.items() if k in ('PASS','new_eta','new_continuations','new_physical_steps','maximum_transition_bracket')}))
if __name__=='__main__':main()
