#!/usr/bin/env python3
"""Audit unchanged models on a common, manifest-corrected non-fitting positive panel."""
from audit import *
from families import contains
def main():
 tags=['after_round1_complete','bounded_r2','bounded_refined_r3','polyhedral_r4','minimal_polyhedral_r5_corrected','minimal_polyhedral_refined_r6','minimal_polyhedral_refined_r7_stable'];inv=read(HERE/'exact_q64_inventory.csv');out=[];summary=[]
 for tag in tags:
  for rr in json.load(open(HERE/f'family_models_{tag}.json')):
   sid=rr['state_id'];m=rr['model'];r=[r for r in inv if r['state_id']==sid];Z=np.array([(eta(x)-AFF)/SCALE for x in r]);pos=np.array([x['B63']=='True' for x in r]);held=np.array([x['geometry_role']=='heldout' and not x.get('retained_validation_batches') and 'retained_validation' not in x['phases'] for x in r])&pos
   assert held.any();full=contains(m,Z);ret=contains(m,Z,True);transfer=np.array(['cross_transfer' in x['sources'] or 'common_core' in x['phases'] or 'val_neighbor' in x['phases'] for x in r])&pos
   out.append(dict(tag=tag,state_id=sid,family=m['family'],primary_heldout_B63=int(held.sum()),full_recall=float(full[held].mean()),retained_recall=float(ret[held].mean()),known_nonB63_inside_full=int((full&~pos).sum()),known_nonB63_inside_retained=int((ret&~pos).sum()),transfer_B63=int(transfer.sum()),transfer_captured=int((transfer&full).sum()),stored_parameters=m['parameter_count']))
  for family in sorted({r['family'] for r in out if r['tag']==tag}):
   rr=[r for r in out if r['tag']==tag and r['family']==family];tc=sum(r['transfer_B63'] for r in rr)
   summary.append(dict(tag=tag,family=family,states=len(rr),median_full_recall=float(np.median([r['full_recall'] for r in rr])),median_retained_recall=float(np.median([r['retained_recall'] for r in rr])),fraction_states_full_ge_half=float(np.mean([r['full_recall']>=.5 for r in rr])),pooled_transfer_recall=sum(r['transfer_captured'] for r in rr)/tc if tc else None,current_known_retained_false=sum(r['known_nonB63_inside_retained'] for r in rr),current_known_full_false=sum(r['known_nonB63_inside_full'] for r in rr)))
 write('corrected_frozen_model_metrics.csv',out);dump('corrected_frozen_model_gates.json',summary)
 print('Manifest-corrected primary recall recomputed for',len(summary),'unchanged frozen fits; no new rollouts.')
if __name__=='__main__':main()
