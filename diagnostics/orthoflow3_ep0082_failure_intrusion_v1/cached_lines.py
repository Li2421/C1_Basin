#!/usr/bin/env python3
from core import *
def main():
 rows=read(H/'frozen_existing_exact.csv');Z=np.array([norm(r) for r in rows]);groups={};candidates=read(H/'internal_failure_candidates.csv');C=np.array([(np.array(json.loads(r['eta']))-AFF)/SCALE for r in candidates])
 for i in range(len(Z)):
  for j in range(i+1,len(Z)):
   d=Z[j]-Z[i];length=np.linalg.norm(d)
   if length<.025:continue
   d=d/length;t=(Z-Z[i])@d;err=np.linalg.norm(Z-Z[i]-t[:,None]*d,axis=1);ix=np.flatnonzero(err<1e-9)
   if len(ix)<3 or cdist(Z[ix],C).min()>.2:continue
   group=tuple(ix.tolist())
   if group in groups:continue
   order=ix[np.argsort(t[ix])];labels=[rows[x]['B63']=='True' for x in order];trans=[abs(t[b]-t[a]) for a,b in zip(order,order[1:]) if (rows[a]['B63']=='True')!=(rows[b]['B63']=='True')]
   groups[group]=dict(line_id='cached_'+str(len(groups)),exact_points=len(order),eta_keys=json.dumps([rows[k]['eta_key'] for k in order]),label_sequence=''.join('S' if x else 'F' for x in labels),transitions=len(trans),min_transition_spacing=min(trans,default=''),span=float(t[order[-1]]-t[order[0]]),direction=json.dumps(d.tolist()),source='EXISTING_ONLY_EXACT_COLLINEAR',selection='Descriptive nonuniform cached lines; not final prospective denominator')
 write('cached_conditional_lines.csv',list(groups.values()))
 dump('cached_slice_summary.json',dict(lines=len(groups),transition_histogram=dict(Counter(str(r['transitions']) for r in groups.values())),approximate_views='cached_slice_views.csv',interpretation='Existing label sequences include acquisition bias; no enclosed-hole inference or true connectivity from a distance graph'))
 print('Cached exact conditional lines:',len(groups))
if __name__=='__main__':main()
