from core import *
src=D/'orthoflow3_shared_conservative_family_v1/raw/common_core'
old=read(H/'frozen_existing_exact.csv'); template=old[0];groups=defaultdict(dict);paths=defaultdict(set)
for p in sorted(src.glob('shard*.jsonl')):
 for l in open(p):
  r=json.loads(l)
  if r['state_id']!=SID:continue
  assert not r.get('execution_error')
  assert r['h_conditioning_identifier']==template['conditioning'] and r['source_group']==template['source_group']
  assert r['first_step']['feature_sha256']==template['h']
  k=key(r['eta']);fi=int(r['future_index']);assert fi not in groups[k]
  groups[k][fi]=r;paths[k].add(str(p))
out=[];c,R=frame()
for k,g in groups.items():
 assert set(g)==set(range(64));rr=list(g.values());e=np.array(rr[0]['eta']);z=(e-AFF)/SCALE;q=(z-c)@R;su=sum(r['success'] for r in rr)
 r=dict(template);r.update(eta_key=k,eta1=e[0],eta2=e[1],eta3=e[2],successes=su,trials=64,Q64=su/64,B63=str(su>=63),deadlock=sum(x['outcome'] in ('safe_deadlock','strict_deadlock','deadlock') for x in rr),timeout=sum(x['outcome']=='timeout' for x in rr),collision=sum(x['outcome']=='collision' for x in rr),sources=';'.join(sorted(paths[k])),phases='shared_core_transfer',raw_phases='shared_core_transfer',geometry_role='SUPPLEMENTAL_EXACT_CACHE',physical_steps=sum(x['continuation_steps'] for x in rr),retained_validation_batches='',in_E_bridge='True',z1=z[0],z2=z[1],z3=z[2],s1=q[0],s2=q[1],n=q[2],provenance_categories='cross_transfer',prior_shape_categories='',evidence_origin='SUPPLEMENTAL_EXACT_CACHE')
 assert inside(z)[0];out.append(r)
write('supplemental_exact_cache.csv',out,list(template));dump('supplemental_cache_audit.json',dict(count=len(out),conditioning_verified=True,exact_h_verified=True,source_files={str(p):sha(p) for p in src.glob('shard*.jsonl')},new_continuations_in_this_audit=0))
print(json.dumps(dict(imported_exact=len(out))))
