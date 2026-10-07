"""Resolve B canonical candidates and freeze missing bilateral 64-seed arms."""
import hashlib,json
from pathlib import Path
from prepare_promotion1 import allrows
H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_bilateral_canonical_audit_v1')
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 pairs=json.loads((H/'paired_state_manifest.json').read_text())['pairs']; p1=json.loads((H/'promotion1_plan.json').read_text())['arms'];p2=json.loads((H/'promotion2_plan.json').read_text())['arms']; rows=allrows();p1m={a['anchor_rank']:a for a in p1};p2m={a['anchor_rank']:a for a in p2}; anchors=json.loads((Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1/anchor_manifest.json')).read_text())['anchors'];am={a['anchor_rank']:a for a in anchors};arms=[];can=[]
 def cnt(sid,eta,p):return sum(rows[(sid,tuple(eta),int(x),p['rng_namespace'])]['success'] for x in p['matched_flow_seeds'][:64])
 for p in pairs:
  a=am[p['pair_rank']];ea=a['canonical_eta_provisional']; eb=[0.,0.,0.] if cnt(p['neighbor_state_id'],[0.,0.,0.],p)>=63 else p1m[p['pair_rank']]['eta']
  if p['pair_rank'] in p2m and cnt(p['neighbor_state_id'],p2m[p['pair_rank']]['eta'],p)>=63:
   j1=sum(rows[(p['neighbor_state_id'],tuple(eb),int(s),p['rng_namespace'])]['J_def'] for s in p['matched_flow_seeds'][:64] if rows[(p['neighbor_state_id'],tuple(eb),int(s),p['rng_namespace'])]['success'])/cnt(p['neighbor_state_id'],eb,p)
   e2=p2m[p['pair_rank']]['eta'];j2=sum(rows[(p['neighbor_state_id'],tuple(e2),int(s),p['rng_namespace'])]['J_def'] for s in p['matched_flow_seeds'][:64] if rows[(p['neighbor_state_id'],tuple(e2),int(s),p['rng_namespace'])]['success'])/cnt(p['neighbor_state_id'],e2,p)
   if j2<j1:eb=e2
  can.append({'pair_rank':p['pair_rank'],'eta_A':ea,'eta_B':eb})
  for direction,state,eta,role in [('AtoB',{'id':p['neighbor_state_id'],'file':p['neighbor_state_file'],'sha':p['neighbor_state_sha256'],'step':p['neighbor_absolute_step']},ea,'cross_A_to_B'),('BtoA',{'id':a['state_id'],'file':a['state_file'],'sha':a['state_sha256'],'step':a['absolute_step']},eb,'cross_B_to_A')]:
   missing=[int(s) for s in p['matched_flow_seeds'][:64] if (state['id'],tuple(eta),int(s),p['rng_namespace']) not in rows]
   if missing:arms.append({'arm_id':f'X64__{direction}__{p["pair_rank"]:02d}','basis_family':'orthoflow3','state_id':state['id'],'state_file':state['file'],'state_sha256':state['sha'],'absolute_step':state['step'],'rng_namespace':p['rng_namespace'],'eta':eta,'seeds':missing,'role':role,'anchor_rank':p['pair_rank'],'offset_steps':p['offset_steps'],'probe_id':direction})
 plan={'schema':'of3_bilateral_cross_v1','basis_family':'orthoflow3','arms':arms};plan['content_sha256']=dig(plan);(H/'cross_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');(H/'paired_canonical_preliminary.json').write_text(json.dumps(can,indent=2)+'\n');print(json.dumps({'arms':len(arms),'new':sum(len(a['seeds']) for a in arms),'steps':sum((850-a['absolute_step'])*len(a['seeds']) for a in arms),'canon':can},indent=2))
if __name__=='__main__':main()
