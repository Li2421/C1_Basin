from common import *
from statistics import summarize
from pipeline import lookup
import collections

def main():
 names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];radii=read(HERE/'CONTROL_RESULTS.json')['GA3']['eligible_radii'];out={};arms=['nominal_only','safety_baseline','F10','F13','S500'];oldchoices=read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers'];provenance=[]
 for i,r in enumerate(radii):
  lo=np.zeros((14,32,5,2));hi=lo.copy();filled=np.zeros_like(lo,bool);events={k:lo.copy() for k in ['collision','timeout','unknown']};grades=collections.Counter()
  for c in read(HERE/'GA3_CASES.json'):
   if c['r']!=r:continue
   f=lookup(c,'GA3');q=read(f);idx=(names.index(c['base_controller_name']),c['state_index'],arms.index(c['arm']),c['seed']);assert not filled[idx];filled[idx]=True;lo[idx]=q['success'];hi[idx]=q['success']+q['numerical_failure'];events['collision'][idx]=q['collision'];events['timeout'][idx]=q['timeout'];events['unknown'][idx]=q['numerical_failure'];grades[q.get('evidence_grade','unspecified')]+=1;provenance.append(dict(rollout_uid=c['rollout_uid'],raw_path=str(f),raw_sha256=sha(f)))
  if not filled.all():out[str(r)]=dict(status='GAP',reason='Invalid reset states retained asmissing, no replacement',missing=int((~filled).sum()));continue
  choices={'nominal only':0,'safety baseline':1,'F10':2,'F13':3,'S500':4,'Srel':4,'Sabs':4};choices.update({m:[arms.index(v[n]) for n in names] for m,v in oldchoices.items()});z=summarize(lo,hi,events,arms,choices,202610163800+i);gain=z['paired']['A3 minus safety baseline']['CI_pp'];s500=z['paired']['S500 minus safety baseline']['CI_pp'];reg=z['methods']['A3']['mean_regret_pp'];m=z['methods']['A3'];status='SUPPORTED' if gain[0]>0 and s500[0]>0 and reg[1]<=3 and m['harm_possible']==0 else 'REJECTED' if gain[1]<=0 or s500[1]<=0 or reg[0]>3 or m['harm_certain']>0 else 'INCONCLUSIVE';z.update(status='MAIN_COMPLETE',GM1=dict(status=status,A3_minus_safety_CI_pp=gain,S500_minus_safety_CI_pp=s500,regret_pp=reg,harm_certain=m['harm_certain'],harm_possible=m['harm_possible']),evidence='DEVELOPMENT',why_development='Same state cohort used for the conditional capability gate andmain',source_grade_counts=dict(grades),r=r,H=700)
  rates=lo.mean((1,3));z['GM2']=dict(best_fixed_by_controller={n:arms[2+int(rates[j,2:4].argmax())] for j,n in enumerate(names)},schedule_minus_fixed=z['paired']['S500 minus best fixed oracle']);z['expert']=read(D/'axis4a_geometry_horizon_v1/results.json')[f'r{r:.2f}']['expert'];out[str(r)]=z;np.savez_compressed(HERE/f'GA3_r{r}_LABELS.npz',lower=lo,upper=hi,arms=arms,controller_names=names,**events)
 write(HERE/'GEOMETRY_MAIN_RESULTS.json',out);write(HERE/'GA3_RAW_PROVENANCE.json',provenance);print('GA3 analyzed', {r:q.get('GM1',{}).get('status',q['status']) for r,q in out.items()},flush=True)
if __name__=='__main__':main()
