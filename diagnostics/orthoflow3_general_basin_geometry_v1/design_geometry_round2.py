#!/usr/bin/env python3
from audit import *
from design_geometry import line_bounds
def main():
 inv=read(HERE/'exact_q64_inventory.csv');val=read(HERE/'semialgebraic/fresh_retained_validation_r1.csv');diagnosis=read(HERE/'semialgebraic/fresh_failure_property_diagnosis.csv')
 values={r['eta_key']:np.array(json.loads(r['eta'])) for r in val};rows=[];paths=[]
 for sid in sorted({r['state_id'] for r in val}):
  failures=[r for r in diagnosis if r['state_id']==sid and r['B63']=='False'];assert failures
  witness=max(failures,key=lambda r:(float(r['nearest_positive']),r['eta_key']));b=values[witness['eta_key']]
  pos=[r for r in inv if r['state_id']==sid and r['B63']=='True'];a=eta(min(pos,key=lambda r:(np.linalg.norm((eta(r)-b)/SCALE),r['eta_key'])))
  path=sid+'_support_transition';paths.append(dict(state_id=sid,path_id=path,kind='success_failure_transition',start=key(a),end=key(b),kernel='',axis=''))
  for alpha in (.25,.5,.75):rows.append(dict(state_id=sid,eta=((1-alpha)*a+alpha*b).tolist(),phase='geometry_round2',kind='success_failure_transition',path_id=path,coordinate=alpha))
 for r in diagnosis:
  if r['B63']!='False' or r['inside_prior_positive_hull']!='True':continue
  sid=r['state_id'];z=(values[r['eta_key']]-AFF)/SCALE;clear=-(HS[:,:3]@z+HS[:,3])
  for fi in np.argsort(clear)[:2]:
   d=HS[fi,:3];_,lim=line_bounds(z,d);path=f'{sid}_new_internal_failure_face{fi}'
   paths.append(dict(state_id=sid,path_id=path,kind='failure_corridor',start=r['eta_key'],end=key(AFF+SCALE*(z+lim*d)),kernel='',axis=''))
   for alpha in (.33,.67,1.):rows.append(dict(state_id=sid,eta=(AFF+SCALE*(z+alpha*lim*d)).tolist(),phase='geometry_round2',kind='failure_corridor',path_id=path,coordinate=alpha))
 write('targeted_probe_rounds/geometry_round2/paths.csv',paths);plan('geometry_round2',rows)
 dump('targeted_probe_rounds/geometry_round2/scientific_question.json',dict(decision='Is prospective failure dominated by ordinary unsupported extent boundaries, or interior nonconvex intrusion?',
  evidence='20/21 fresh retained failures outside prior positive hull;1deep interior failure Q64=.203125',
  design='One greatest-support-distance failed witness per state, 3 fixed interpolants to nearest cached B63. Interior witness:2 shortest facet-normal corridors with3 samples each.',
  no_shape_synthesis_before_property_evidence=True,no_enclosed_hole_claim_from_interruption=True))
if __name__=='__main__':main()
