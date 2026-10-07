#!/usr/bin/env python3
"""Evaluate the frozen operational saturation criterion, without declaring impossibility."""
from audit import *
from families import contains
def main():
 tags=['minimal_polyhedral_r5_corrected','minimal_polyhedral_refined_r6','minimal_polyhedral_refined_r7_stable']
 attempts=['minimal_validation_r5','minimal_validation_r6','minimal_validation_r7'];inv=read(HERE/'exact_q64_inventory.csv');out=[]
 for tag,batch in zip(tags,attempts):
  vals=[];morph=[];snap=read(HERE/f'fitting_evidence/{tag}.csv');models=json.load(open(HERE/f'family_models_{tag}.json'))
  for row in models:
   sid=row['state_id'];rr=[r for r in inv if r['state_id']==sid and r['B63']=='True' and r['geometry_role']=='heldout' and 'retained_validation' not in r['phases']];Z=np.array([(eta(r)-AFF)/SCALE for r in rr]);vals.append([contains(row['model'],Z).mean(),contains(row['model'],Z,True).mean()])
   Z=np.array([(eta(r)-AFF)/SCALE for r in snap if r['state_id']==sid and r['B63']=='True']);_,S,V=np.linalg.svd(Z-Z.mean(0),full_matrices=False);sp=np.ptp((Z-Z.mean(0))@V.T,axis=0);morph.append([S[-1]**2/(S**2).sum(),sp[0]/sp[2],float(cdist(Z,Z).max())])
  gates=[json.load(open(HERE/f'targeted_probe_rounds/{prefix}{batch}/gate.json')) for prefix in ('','db_')];assert all(g['complete'] for g in gates)
  out.append(dict(tag=tag,batch=batch,full_recall_fixed_holdout=float(np.median(vals,axis=0)[0]),retained_recall_fixed_holdout=float(np.median(vals,axis=0)[1]),false=sum(g['false_inclusions'] for g in gates),n=sum(g['expected'] for g in gates),median_normal_variance_fraction=float(np.median(morph,axis=0)[0]),median_anisotropy=float(np.median(morph,axis=0)[1]),median_diameter=float(np.median(morph,axis=0)[2]),DB_validation_reused=24 if batch!='minimal_validation_r5' else 0))
 comparisons=[]
 for a,b in zip(out,out[1:]):
  gains=dict(full_recall=b['full_recall_fixed_holdout']-a['full_recall_fixed_holdout'],retained_recall=b['retained_recall_fixed_holdout']-a['retained_recall_fixed_holdout'],false_inclusion=a['false']/a['n']-b['false']/b['n'])
  comparisons.append(dict(previous=a['tag'],latest=b['tag'],improvement=gains,any_improvement_ge_five_percentage_points=any(v>=.05-1e-12 for v in gains.values())))
 # Qualitative interpretation is explicit: no new enclosure, successful detour or conditional-run type was discovered in these two rounds.
 result=dict(rounds=out,comparisons=comparisons,two_consecutive_sub5pp_refinements=not any(c['any_improvement_ge_five_percentage_points'] for c in comparisons),material_new_topology_demonstrated=False,
 morphology_evidence='r6 retained validation and r7 failed-link/boundary evidence leave the same anisotropic support plus irregular intrusions. No enclosed hole confirmed, no successful complete detour, no new conditional-interval topology type. Numerical morphology medians reported above; no claim that full topology is identified.',
 criterion='User Section20A; two targeted refinement rounds with no material inferred morphology/topology change and no >=5percentage-point gain for the remaining live fixed-capacity candidate.',
 scope='Operational saturation of this evidence-driven analytic search, NOT proof that no other finite analytic family could exist. Other families remain rejected for recorded recall/precision reasons; no new property invalidated those reasons.',
 fresh_precision_pass=out[-1]['false']==0,positive_holdout_note='All three instances re-evaluated on the SAME final primary-heldout positive keys; retained-validation observations excluded. Retrospective adaptive comparison, not pristine generalization.')
 dump('scientific_saturation_audit.json',result);print(json.dumps({k:result[k] for k in ['two_consecutive_sub5pp_refinements','fresh_precision_pass']}))
if __name__=='__main__':main()
