"""Apply the archived deterministic top-eight rule to completed neighbor screens."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1'); H=ROOT/'diagnostics/orthoflow3_bilateral_canonical_audit_v1'; C=ROOT/'diagnostics/orthoflow3_local_basin_continuity_v1'; D=ROOT/'diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json'
LO=np.array([.5,-.5,0.]); HI=np.array([1.25,.5,.75]); W=HI-LO
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def prior():
 out={}
 for d in C.glob('raw/*'):
  if d.is_dir():
   for f in d.glob('*.jsonl'):
    for s in f.read_text().splitlines():
     r=json.loads(s);out.setdefault((r['state_id'],tuple(r['eta']),int(r['seed']),int(r['rng_namespace'])),r)
 return out
def main():
 if (H/'screen_plan.json').exists():raise RuntimeError('already prepared')
 ps=json.loads((H/'paired_state_manifest.json').read_text())['pairs']; pts=json.loads(D.read_text())['points']; th=np.array([p['theta'] for p in pts]); evidence=prior()
 for f in (H/'raw'/'initial').glob('*.jsonl'):
  for s in f.read_text().splitlines():
   r=json.loads(s);evidence[(r['state_id'],tuple(r['eta']),int(r['seed']),int(r['rng_namespace']))]=r
 arms=[]; selected=[]
 for p in ps:
  sid=p['neighbor_state_id'];seed0=int(p['matched_flow_seeds'][0]);ns=int(p['rng_namespace']);success=[]
  for i,e in enumerate(th):
   r=evidence.get((sid,tuple(e.tolist()),seed0,ns))
   if r is None:raise RuntimeError(('missing global',sid,i))
   if r['success']:success.append(i)
  cs=[]
  for i in success:
   dist=np.linalg.norm((th-th[i])/W,axis=1);dist[i]=np.inf;near=np.argsort(dist,kind='stable')[:8]
   cs.append(( -sum(int(j in set(success)) for j in near),float(np.linalg.norm(th[i]/W)),i))
  cs=sorted(cs)[:8]; selected.append({'pair_rank':p['pair_rank'],'state_id':sid,'successful_global_candidates':len(success),'selected_indices':[x[2] for x in cs]})
  for rank,(_,_,i) in enumerate(cs):
   missing=[]
   for seed in p['matched_flow_seeds'][1:16]:
    if (sid,tuple(th[i].tolist()),int(seed),ns) not in evidence:missing.append(int(seed))
   if missing:arms.append({'arm_id':f'S16__{p["pair_rank"]:02d}__{i:03d}','basis_family':'orthoflow3','state_id':sid,'state_file':p['neighbor_state_file'],'state_sha256':p['neighbor_state_sha256'],'absolute_step':p['neighbor_absolute_step'],'rng_namespace':ns,'eta':th[i].tolist(),'eta_index':i,'seeds':missing,'role':'neighbor_screen16','anchor_rank':p['pair_rank'],'offset_steps':p['offset_steps'],'probe_id':'S16','selection_priority':rank})
 plan={'schema':'of3_bilateral_screen_v1','basis_family':'orthoflow3','selection_rule':'exact archived top-eight rule after 256 first-seed screen','arms':arms,'content_source':'initial plus exact continuity cache'};plan['content_sha256']=dig(plan)
 (H/'screen_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');(H/'screen_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
 print(json.dumps({'arms':len(arms),'new':sum(len(a['seeds']) for a in arms),'upper_steps':sum((850-a['absolute_step'])*len(a['seeds']) for a in arms),'selected':selected},indent=2))
if __name__=='__main__':main()
