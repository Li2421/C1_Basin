#!/usr/bin/env python3
from core import *
def main():
 assert not (H/'frozen_cache_manifest.json').exists()
 assert sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')==SHA
 gm=json.load(open(G/'manifest.json'));assert gm['final_inventory_sha256']==sha(G/'exact_q64_inventory.csv')
 rows=[r for r in read(G/'exact_q64_inventory.csv') if r['state_id']==SID];assert len(rows)==185 and all(int(r['trials'])==64 for r in rows)
 assert len({r['h'] for r in rows})==len({r['conditioning'] for r in rows})==len({r['controller_semantics_hash'] for r in rows})==1
 assert all(inside(norm(r))[0] for r in rows)
 source=D/'orthoflow3_b63_manifold_representation_v1/affine_plane_fit.csv';fr=next(r for r in read(source) if r['state_id']==SID)
 c=np.array([float(fr[f'center_t{i}']) for i in (1,2,3)]);R=np.array([[float(fr[f'{a}_t{i}']) for a in ['u1','u2','normal']] for i in (1,2,3)]);assert np.allclose(R.T@R,np.eye(3),atol=1e-10)
 # Verify the old cloud's normalization rather than assuming it from column names.
 cloud=D/f'orthoflow3_existing_b63_geometry_v1/per_state_b63_clouds/{SID}.csv'
 for r in read(cloud):assert np.allclose(norm(r),[float(r[f't{i}']) for i in (1,2,3)],atol=1e-12)
 dump('frozen_frame.json',dict(center=c.tolist(),R=R.tolist(),axes=['s1','s2','n'],source=str(source),source_sha256=sha(source),normalization={'offset':AFF.tolist(),'scale':SCALE.tolist()},purpose='Coordinates only, not a basin model'))
 shape=D/'orthoflow3_t0_basin_shape_v1/targeted_probe_manifest.csv';shape_roles=defaultdict(set)
 for r in read(shape):
  if r['state_id']==SID:shape_roles[r['eta_key_float64']].add(r['category'])
 out=[]
 for r in rows:
  z=norm(r);local=(z-c)@R;out.append(dict(r,z1=z[0],z2=z[1],z3=z[2],s1=local[0],s2=local[1],n=local[2],provenance_categories=';'.join(provenance_categories(r)),prior_shape_categories=';'.join(sorted(shape_roles[r['eta_key']])),evidence_origin='EXISTING_EXACT_CACHE'))
 write('frozen_existing_exact.csv',out);write('exact_q64_ep0082.csv',out)
 inspected=[G/'working_state.json',G/'evidence_index.json',G/'manifest.json',G/'conditioning_semantics.json',G/'exact_q64_inventory.csv',source,cloud,shape]
 for name in ['orthoflow3_t0_basin_shape_v1','orthoflow3_existing_b63_geometry_v1','orthoflow3_b63_manifold_representation_v1','orthoflow3_analytic_basin_margin_learning_v1']:
  p=D/name/'manifest.json';m=json.load(open(p));inspected.append(p)
  # Prior manifest files are provenance only; general audit already reconciled exact controls/cache.
 dump('evidence_index.json',{str(i):{'path':str(p),'sha256':sha(p)} for i,p in enumerate(inspected)})
 dump('frozen_cache_manifest.json',dict(state_id=SID,exact_Q64=len(out),B63=sum(r['B63']=='True' for r in out),inventory_source=str(G/'exact_q64_inventory.csv'),inventory_sha256=sha(G/'exact_q64_inventory.csv'),basis_sha256=SHA,h=rows[0]['h'],conditioning=rows[0]['conditioning'],controller_semantics_hash=rows[0]['controller_semantics_hash'],future_root_seed=rows[0]['future_root_seed'],seeds=list(range(64)),eta_normalization={'offset':AFF.tolist(),'scale':SCALE.tolist()},no_new_state=True))
 with (H/'partial_continuation_cache.jsonl').open('w') as f:
  for line in open(G/'partial_continuation_cache.jsonl'):
   r=json.loads(line)
   if r.get('state_id')==SID:f.write(json.dumps(r)+'\n')
 dump('working_state.json',dict(status='ACTIVE',completed=['exact ep0082 cache and frozen R0 verified'],inspected_index='evidence_index.json',next_action='Freeze candidate regions, cached line analysis and targeted round1 before any new outcomes',training=False,new_states=False,new_scenarios=False))
 dump('hypothesis_status.json',[dict(id=x,status='LIVE',next_test=y) for x,y in [('H1_BOUNDARY_INTRUSION','Sampled failure corridors to outer failure/domain boundary'),('H2_ENCLOSED_HOLE','Six directional rays and escape routes around region representatives'),('H3_INTERLEAVING','Exact collinear multi-transition patterns with adaptive bracketing')]])
 print(json.dumps({'existing_exact_Q64':len(out),'B63':sum(r['B63']=='True' for r in out),'state':SID}))
if __name__=='__main__':main()
