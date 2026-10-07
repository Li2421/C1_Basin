"""Preserve exact straight-segment transitions in corridor acquisitions too."""
from core import *
def main():
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows};Z=np.array([norm(r) for r in rows]);groups=defaultdict(list)
 for r in read(H/'candidate_failure_corridors.csv'):
  groups[(r['corridor_id'],r['edge'])].append(r)
 lines=[]
 for (cid,edge),rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:int(r['position']));keys=list(dict.fromkeys(r['eta_key'] for r in rr))
  if not all(k in lookup for k in keys):continue
  a,b=norm(lookup[keys[0]]),norm(lookup[keys[-1]]);v=b-a;length=np.linalg.norm(v)
  if length<1e-10:continue
  v/=length
  if v[np.flatnonzero(abs(v)>1e-10)[0]]<0:v=-v
  center=a-v*(a@v);ts=[a@v,b@v];lo,hi=min(ts),max(ts)
  existing=next((r for r in lines if np.linalg.norm(r['v']-v)<1e-9 and np.linalg.norm(r['center']-center)<1e-9 and not (hi<r['lo']-1e-9 or lo>r['hi']+1e-9)),None)
  if existing is None:lines.append(dict(v=v,center=center,lo=lo,hi=hi,sources=[f'{cid}:edge{edge}']))
  else:existing['lo']=min(existing['lo'],lo);existing['hi']=max(existing['hi'],hi);existing['sources'].append(f'{cid}:edge{edge}')
 # A new segment may bridge two previously disjoint spans on one line.
 # Merge transitively so repeated alternation cannot disappear at a grouping seam.
 changed=True
 while changed:
  changed=False
  for i in range(len(lines)):
   for j in range(i+1,len(lines)):
    a,b=lines[i],lines[j]
    if np.linalg.norm(a['v']-b['v'])<1e-9 and np.linalg.norm(a['center']-b['center'])<1e-9 and not (a['hi']<b['lo']-1e-9 or b['hi']<a['lo']-1e-9):
     a['lo']=min(a['lo'],b['lo']);a['hi']=max(a['hi'],b['hi']);a['sources']+=b['sources'];del lines[j];changed=True;break
   if changed:break
 output=[]
 for i,r in enumerate(lines):
  t=Z@r['v'];res=np.linalg.norm(Z-r['center']-t[:,None]*r['v'],axis=1);idx=np.flatnonzero((res<1e-9)&(t>=r['lo']-1e-9)&(t<=r['hi']+1e-9));idx=idx[np.argsort(t[idx])]
  labels=[rows[j]['B63']=='True' for j in idx];changes=[float(t[b]-t[a]) for a,b,x,y in zip(idx,idx[1:],labels,labels[1:]) if x!=y]
  output.append(dict(line_id=f'corridor_segment_{i}',source_segments=';'.join(r['sources']),points=len(idx),span=r['hi']-r['lo'],sequence=''.join('S' if b else 'F' for b in labels),transitions=len(changes),minimum_transition_spacing=min(changes) if changes else '',eta_keys=json.dumps([rows[j]['eta_key'] for j in idx]),coordinates=json.dumps(t[idx].tolist()),direction=json.dumps(r['v'].tolist())))
 write('corridor_line_transitions.csv',output)
 dump('corridor_line_summary.json',dict(unique_collinear_overlapping_segments=len(output),histogram=dict(Counter(r['transitions'] for r in output)),repeated_lines=[r for r in output if r['transitions']>=3],note='Post-acquisition auxiliary exact-line evidence; distinct from prospective registered lines. Repeated alternation alone cannot exclude a bent/branched notch.'))
 print(json.dumps(dict(lines=len(output),histogram=dict(Counter(r['transitions'] for r in output)))))
if __name__=='__main__':main()
