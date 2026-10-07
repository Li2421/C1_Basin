#!/usr/bin/env python3
from core import *
def sequence(points,lookup):
 rr=[]
 for t,k in sorted(set(points)):
  if k in lookup:rr.append((t,k,lookup[k]['B63']=='True'))
 changes=[(a,b) for a,b in zip(rr,rr[1:]) if a[2]!=b[2]]
 return rr,changes
def main():
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows};Z=np.array([norm(r) for r in rows]);manifest=read(H/'conditional_slice_manifest.csv');definitions={r['line_id']:r for r in json.load(open(H/'line_definitions.json'))};line_stats=[];result=[];brackets=[]
 for lid,d in definitions.items():
  rr=[r for r in manifest if r['line_id']==lid];origin=(np.array(d['origin'])-AFF)/SCALE;direction=np.array(d['direction']);points=[]
  for r in rr:
   if r['eta_key'] not in lookup:continue
   z=norm(lookup[r['eta_key']]);t=float((z-origin)@direction);assert np.linalg.norm(z-origin-t*direction)<=1e-9
   points.append((t,r['eta_key']));a=lookup[r['eta_key']];result.append(dict(line_id=lid,region_id=d['region_id'],t=t,eta_key=r['eta_key'],Q64=a['Q64'],B63=a['B63'],deadlock=a['deadlock'],timeout=a['timeout'],round=r['round']))
  pp,ch=sequence(points,lookup);valid=len(pp)>=3;labels=''.join('S' if x[2] else 'F' for x in pp)
  line_stats.append(dict(line_id=lid,region_id=d['region_id'],kind=d['kind'],exact_points=len(pp),complete=len(set(r['eta_key'] for r in rr))==len({x[1] for x in pp}),valid_line=valid,label_sequence=labels,transition_count=len(ch),min_transition_width=min((b[0]-a[0] for a,b in ch),default=''),span=pp[-1][0]-pp[0][0] if pp else 0))
  for j,(a,b) in enumerate(ch):brackets.append(dict(line_id=lid,region_id=d['region_id'],transition=j,left_t=a[0],right_t=b[0],width=b[0]-a[0],left_key=a[1],right_key=b[1],label_left='S' if a[2] else 'F',label_right='S' if b[2] else 'F',repeated_line=len(ch)>=3))
 write('conditional_slice_results.csv',result,['line_id','region_id','t','eta_key','Q64','B63','deadlock','timeout','round']);write('transition_statistics.csv',line_stats);write('opposite_label_brackets.csv',brackets,['line_id','region_id','transition','left_t','right_t','width','left_key','right_key','label_left','label_right','repeated_line'])
 cor=[]
 for d in json.load(open(H/'corridor_definitions.json')):
  if d['status']!='FROZEN':continue
  keys=list(dict.fromkeys(d['ordered_keys']));available=[lookup[k] for k in keys if k in lookup];non=sum(r['B63']=='False' for r in available)
  cor.append(dict(corridor_id=d['corridor_id'],region_id=d['region_id'],points=len(keys),exact=len(available),non_B63=non,B63=len(available)-non,complete=len(available)==len(keys),sampled_failure_path_valid=len(available)==len(keys) and non==len(keys),reaches_domain_boundary=d['endpoint_on_domain'],continuum_certificate=False))
 write('failure_corridor_results.csv',cor,['corridor_id','region_id','points','exact','non_B63','B63','complete','sampled_failure_path_valid','reaches_domain_boundary','continuum_certificate'])
 holes=[]
 for region in json.load(open(H/'region_definitions.json'))['regions']:
  rid=region['region_id'];sides=[]
  for lid,d in definitions.items():
   if d['region_id']!=rid:continue
   rr=[r for r in result if r['line_id']==lid]
   for sign in [-1,1]:
    a=[r for r in rr if sign*r['t']>1e-9];found=any(r['B63']=='True' for r in a);sides.append(found)
  escaped=any(r['region_id']==rid and r['sampled_failure_path_valid'] for r in cor)
  holes.append(dict(region_id=rid,directions_tested=len(sides),directions_with_B63=sum(sides),six_direction_surrounding_support=len(sides)>=6 and all(sides),validated_sampled_escape=escaped,enclosed_hole_confirmed=False,status='ESCAPE_SUPPORTED' if escaped else 'UNRESOLVED'))
 write('hole_tests.csv',holes)
 done=[r for r in line_stats if r['valid_line'] and r['complete']];summary=dict(registered_lines=len(line_stats),complete_lines=len(done),transition_histogram=dict(Counter(str(r['transition_count']) for r in done)),minimum_bracket_width=min((r['width'] for r in brackets),default=None),corridors_valid=sum(r['sampled_failure_path_valid'] for r in cor),corridors_complete=sum(r['complete'] for r in cor),hole_tests=holes)
 dump('slice_analysis_summary.json',summary);print(json.dumps({k:summary[k] for k in ['complete_lines','transition_histogram','corridors_valid','corridors_complete']}))
if __name__=='__main__':main()
