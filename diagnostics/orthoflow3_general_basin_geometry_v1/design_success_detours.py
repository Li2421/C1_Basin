#!/usr/bin/env python3
from audit import *
def main():
 inv=read(HERE/'exact_q64_inventory.csv');lookup={(r['state_id'],r['eta_key']):r for r in inv};candidates=defaultdict(list);base=HERE/'targeted_probe_rounds/geometry_round1';manifest=read(base/'manifest.csv')
 for p in read(base/'paths.csv'):
  if p['kind']!='star':continue
  sid=p['state_id'];rr=[r for r in manifest if r['path_id']==p['path_id']];labels=[lookup.get((sid,r['eta_key'])) for r in rr]
  if all(r is not None for r in labels) and any(r['B63']=='False' for r in labels):candidates[sid].append((p['start'],p['end'],p['path_id']))
 hist=read(D/'orthoflow3_t0_basin_shape_v1/targeted_probe_manifest.csv');groups=defaultdict(list)
 for r in hist:
  if r['category']=='SUCCESS_SUCCESS_INTERPOLATION':groups[(r['state_id'],r['endpointA_key'],r['endpointB_key'])].append(r['eta_key_float64'])
 for (sid,a,b),keys in groups.items():
  if (sid,a) in lookup and (sid,b) in lookup and lookup[sid,a]['B63']=='True' and lookup[sid,b]['B63']=='True' and any(lookup.get((sid,k),{}).get('B63')=='False' for k in keys):candidates[sid].append((a,b,'historical_interpolation'))
 rows=[];paths=[];notes=[]
 for sid,choices in sorted(candidates.items()):
  ak,bk,source=max(choices,key=lambda x:(np.linalg.norm((eta(lookup[sid,x[0]])-eta(lookup[sid,x[1]]))/SCALE),x[0],x[1]));a=eta(lookup[sid,ak]);b=eta(lookup[sid,bk]);d=(b-a)/SCALE
  negatives=np.array([(eta(r)-AFF)/SCALE for r in inv if r['state_id']==sid and r['B63']=='False']);options=[]
  for r in inv:
   if r['state_id']!=sid or r['B63']!='True' or r['eta_key'] in (ak,bk):continue
   w=eta(r);v=(w-a)/SCALE;t=float(v@d/(d@d));off=np.linalg.norm(v-t*d)
   if off<.05 or min(np.linalg.norm((w-a)/SCALE),np.linalg.norm((w-b)/SCALE))<.1:continue
   points=[((1-alpha)*u+alpha*v).tolist() for u,v in ((a,w),(w,b)) for alpha in (.25,.5,.75)]
   if any(lookup.get((sid,key(p)),{}).get('B63')=='False' for p in points):continue
   length=np.linalg.norm((w-a)/SCALE)+np.linalg.norm((w-b)/SCALE);clear=float(np.linalg.norm(negatives-(w-AFF)/SCALE,axis=1).min())
   options.append((float(length-.3*clear),r['eta_key'],w,points))
  if not options:notes.append(dict(state_id=sid,status='NO_CACHE_COMPATIBLE_WAYPOINT'));continue
  score,wk,w,points=min(options,key=lambda x:x[:2]);path=sid+'_success_detour'
  paths.append(dict(state_id=sid,path_id=path,kind='success_detour',start=ak,end=bk,waypoint=wk,source_failed_segment=source,kernel='',axis=''))
  for i,p in enumerate(points):rows.append(dict(state_id=sid,eta=p,phase='success_detours',kind='success_detour',path_id=path,segment=i//3,coordinate=(i//3)+(i%3+1)/4))
  notes.append(dict(state_id=sid,status='FROZEN_TWO_SEGMENT_ROUTE',score=score,existing_endpoints_and_waypoint_B63=True))
 write('targeted_probe_rounds/success_detours/paths.csv',paths);dump('targeted_probe_rounds/success_detours/selection.json',notes);plan('success_detours',rows)
 print('States with frozen detours',len(paths))
if __name__=='__main__':main()
