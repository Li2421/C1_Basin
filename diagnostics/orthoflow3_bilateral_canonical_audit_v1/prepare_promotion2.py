"""Bounded second B63 endpoint only where 16-seed evidence leaves a near tie."""
import hashlib,json,math
from pathlib import Path
from prepare_promotion1 import allrows
H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_bilateral_canonical_audit_v1');D=Path('/home/zhihan/research/Basin_C1/diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json')
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if (H/'promotion2_plan.json').exists():raise RuntimeError('exists')
 pairs=json.loads((H/'paired_state_manifest.json').read_text())['pairs']; sel={x['pair_rank']:x for x in json.loads((H/'screen_selection.json').read_text())}; p1=json.loads((H/'promotion1_plan.json').read_text())['arms'];pts=json.loads(D.read_text())['points'];rows=allrows();arms=[];audit=[]
 p1map={a['anchor_rank']:a for a in p1}
 for p in pairs:
  ns=p['rng_namespace']; sid=p['neighbor_state_id']; zero=sum(rows[(sid,(0.,0.,0.),int(s),ns)]['success'] for s in p['matched_flow_seeds'][:64]); primary=p1map[p['pair_rank']]; pe=primary['eta']; ps=sum(rows[(sid,tuple(pe),int(s),ns)]['success'] for s in p['matched_flow_seeds'][:64]); pj=sum(rows[(sid,tuple(pe),int(s),ns)]['J_def'] for s in p['matched_flow_seeds'][:64] if rows[(sid,tuple(pe),int(s),ns)]['success'])/ps if ps else math.inf
  cand=[]
  for i in sel[p['pair_rank']]['selected_indices']:
   if i==primary['eta_index']:continue
   eta=pts[i]['theta'];ev=[rows[(sid,tuple(eta),int(s),ns)] for s in p['matched_flow_seeds'][:16]];su=sum(r['success'] for r in ev);js=[r['J_def'] for r in ev if r['success']];cand.append((su,sum(js)/len(js) if js else math.inf,i,eta))
  cand=sorted([x for x in cand if x[0]>=15],key=lambda x:(-x[0],x[1],x[2]));chosen=None
  # Exact sequential-resolution rule, frozen before second endpoint outcomes:
  # only a screen-near alternative can threaten the primary minimum-J identity.
  if zero<63 and ps>=63 and cand and cand[0][1] <= 1.10*primary['screen16_mean_J']:
   chosen=cand[0];su,j,i,eta=chosen;arms.append({'arm_id':f'P64B__{p["pair_rank"]:02d}__{i:03d}','basis_family':'orthoflow3','state_id':sid,'state_file':p['neighbor_state_file'],'state_sha256':p['neighbor_state_sha256'],'absolute_step':p['neighbor_absolute_step'],'rng_namespace':ns,'eta':eta,'eta_index':i,'seeds':[int(s) for s in p['matched_flow_seeds'][16:64]],'role':'neighbor_secondary64','anchor_rank':p['pair_rank'],'offset_steps':p['offset_steps'],'probe_id':'P64B','screen16_success':su,'screen16_mean_J':j})
  audit.append({'pair_rank':p['pair_rank'],'zero64_success':zero,'primary64_success':ps,'primary_J':pj,'secondary_selected':chosen[2] if chosen else None,'near_tie_rule':'alternative screen J <= 1.10 primary screen J'})
 plan={'schema':'of3_bilateral_promotion2_v1','basis_family':'orthoflow3','selection_rule':'sequential canonical resolution: zero not B63, primary B63, alternative >=15/16 with screen mean J <=110% primary; promote deterministic best alternative','arms':arms};plan['content_sha256']=dig(plan);(H/'promotion2_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');(H/'promotion1_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'arms':len(arms),'new':48*len(arms),'steps':sum((850-a['absolute_step'])*48 for a in arms)},indent=2))
if __name__=='__main__':main()
