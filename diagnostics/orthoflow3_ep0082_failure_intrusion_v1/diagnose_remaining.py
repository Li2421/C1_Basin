"""Compare direct exits from the observed unresolved negative component; no rollout."""
from core import *
def main():
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows};P=np.array([norm(r) for r in rows if r['B63']=='True']);N=np.array([norm(r) for r in rows if r['B63']=='False']);c,R=frame()
 rep=json.load(open(H/'region_definitions.json'))['regions'][0]['representative_key'];adj=defaultdict(set)
 for r in read(H/'sampled_failure_links.csv'):adj[r['a']].add(r['b']);adj[r['b']].add(r['a'])
 reached={rep};todo=[rep]
 while todo:
  for k in adj[todo.pop()]-reached:reached.add(k);todo.append(k)
 options=[]
 for k in sorted(reached):
  a=norm(lookup[k])
  for j,v in enumerate(np.concatenate([R.T,-R.T,np.eye(3),-np.eye(3)])):
   denom=HS[:,:3]@v;tt=-(HS[:,:3]@a+HS[:,3])[denom>1e-12]/denom[denom>1e-12]
   length=float(tt.min());b=a+length*v
   if length<1e-8:continue
   tp=(P-a)@v;res=np.linalg.norm(P-a-tp[:,None]*v,axis=1)
   if np.any((tp>1e-9)&(tp<length+1e-9)&(res<1e-9)):continue
   # Exact previously observed collinear points are reused with no rounding.
   along=[(0.,eta(lookup[k]))]
   for r in rows:
    z=norm(r);t=float((z-a)@v)
    if 1e-9<t<length-1e-9 and np.linalg.norm(z-a-t*v)<1e-9:along.append((t,eta(r)))
   along=sorted(along,key=lambda x:x[0])+[(length,AFF+SCALE*b)];ee=[along[0][1]]
   for (t0,e0),(t1,e1) in zip(along,along[1:]):
    n=max(1,int(np.ceil((t1-t0)/.05)));ee.extend([AFF+SCALE*(a+(t0+(t1-t0)*u/n)*v) for u in range(1,n)]);ee.append(e1)
   if any(lookup.get(key(e),{}).get('B63')=='True' for e in ee):continue
   new=[e for e in ee if key(e) not in lookup]
   if not new:continue
   Z=(np.array(new)-AFF)/SCALE;dp=cdist(Z,P).min(1);dn=cdist(Z,N).min(1)
   options.append(dict(start_key=k,direction_index=j,length=length,new_eta=len(new),min_positive_clearance=float(dp.min()),min_negative_advantage=float((dp-dn).min()),mean_negative_advantage=float((dp-dn).mean()),eta=[e.tolist() for e in ee]))
 options.sort(key=lambda r:(-r['min_negative_advantage'],-r['mean_negative_advantage'],r['new_eta'],r['start_key'],r['direction_index']))
 feasible=[r for r in options if r['new_eta']<=14]
 out=dict(question='Does the unresolved primary negative component escape in a native/R0 direction not tried by nearest-facet cost-minimizing routes?',current_new_eta=135,component_vertices=len(reached),direct_options=len(options),within149_options=len(feasible),best=feasible[:5],selection_rule='Among direct native/R0 ray exits requiring <=14 new points, maximize minimum then mean (nearest-positive distance minus nearest-negative distance); ties fewer new points, exact key, fixed direction index. This is acquisition heuristic, not a membership model.',prior_failure='Three cost-minimizing bent paths were interrupted by B63; low new-query cost did not imply negative corridor reliability.',topology_limit='No six-ray successful enclosure, no repeated local >=3 transitions. If this independent direction test fails, remain TOPOLOGY_UNDERRESOLVED; additional scope would require transverse mapping rather than another guessed shortest path.')
 dump('remaining_region_design_diagnostic.json',out)
 print(json.dumps({k:v for k,v in out.items() if k in ('component_vertices','direct_options','within149_options')}));print(json.dumps([{k:v for k,v in r.items() if k!='eta'} for r in feasible[:3]]))
if __name__=='__main__':main()
