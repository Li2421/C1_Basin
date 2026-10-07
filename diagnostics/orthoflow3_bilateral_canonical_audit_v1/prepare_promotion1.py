"""Freeze first B63 promotion exactly by the archived selection rule."""
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_bilateral_canonical_audit_v1');C=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1');D=Path('/home/zhihan/research/Basin_C1/diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json')
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def allrows():
 out={}
 for root in [C/'raw',H/'raw']:
  for d in root.glob('*'):
   if d.is_dir():
    for f in d.glob('*.jsonl'):
     for s in f.read_text().splitlines():
      r=json.loads(s);out.setdefault((r['state_id'],tuple(r['eta']),int(r['seed']),int(r['rng_namespace'])),r)
 return out
def main():
 if (H/'promotion1_plan.json').exists():raise RuntimeError('exists')
 pairs=json.loads((H/'paired_state_manifest.json').read_text())['pairs'];sel={x['pair_rank']:x for x in json.loads((H/'screen_selection.json').read_text())}; points=json.loads(D.read_text())['points']; rows=allrows();arms=[];audit=[]
 for p in pairs:
  c=[]
  for i in sel[p['pair_rank']]['selected_indices']:
   eta=points[i]['theta']
   ev=[rows[(p['neighbor_state_id'],tuple(eta),int(s),int(p['rng_namespace']))] for s in p['matched_flow_seeds'][:16]]
   suc=sum(r['success'] for r in ev); js=[r['J_def'] for r in ev if r['success']];c.append((suc,sum(js)/len(js) if js else math.inf,i,eta))
  feasible=sorted([x for x in c if x[0]>=15],key=lambda x:(-x[0],x[1],x[2]))
  audit.append({'pair_rank':p['pair_rank'],'screen_candidates':[{'eta_index':x[2],'success16':x[0],'mean_J':x[1]} for x in sorted(c,key=lambda x:x[2])],'chosen':feasible[0][2] if feasible else None})
  if feasible:
   suc,j,i,eta=feasible[0];arms.append({'arm_id':f'P64A__{p["pair_rank"]:02d}__{i:03d}','basis_family':'orthoflow3','state_id':p['neighbor_state_id'],'state_file':p['neighbor_state_file'],'state_sha256':p['neighbor_state_sha256'],'absolute_step':p['neighbor_absolute_step'],'rng_namespace':p['rng_namespace'],'eta':eta,'eta_index':i,'seeds':[int(s) for s in p['matched_flow_seeds'][16:64]],'role':'neighbor_primary64','anchor_rank':p['pair_rank'],'offset_steps':p['offset_steps'],'probe_id':'P64A','screen16_success':suc,'screen16_mean_J':j})
 plan={'schema':'of3_bilateral_promotion1_v1','basis_family':'orthoflow3','selection_rule':'archived: screen16 >=15, rank success descending then mean successful J_def ascending then eta index','arms':arms};plan['content_sha256']=dig(plan)
 (H/'promotion1_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');(H/'screen16_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'arms':len(arms),'new':48*len(arms),'steps':sum((850-a['absolute_step'])*48 for a in arms)},indent=2))
if __name__=='__main__':main()
