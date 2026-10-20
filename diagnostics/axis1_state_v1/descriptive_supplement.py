"""Required descriptive contrasts; primary endpoints and choices remain frozen."""
from common import *
from analyze import evaluate

def main():
 result=read(HERE/'results.json');before=sha(HERE/'results.json');endpoints={k:result[k]['endpoint'] for k in ['1a','1b']};states=read(HERE/'EVAL_STATES.json');z=np.load(HERE/'LABELS.npz');lo=z['lower'];hi=z['upper'];frozen=read(HERE/'FROZEN_CHOICES.json')['choices'];names=z['controller_names'].tolist();details={}
 result['state_shift_contrasts']={}
 for method in result['1a']['methods']:
  a=result['1a']['methods'][method];b=result['1b']['methods'][method]
  result['state_shift_contrasts'][method]={'success_change_pp':[b['success_pct'][0]-a['success_pct'][1],b['success_pct'][1]-a['success_pct'][0]],'regret_change_pp':[b['mean_regret_pp'][0]-a['mean_regret_pp'][1],b['mean_regret_pp'][1]-a['mean_regret_pp'][0]],'descriptive_only':True}
 metrics_by_cohort={}
 for cohort in ['1a','1b']:
  ids=[i for i,s in enumerate(states) if s['cohort']==cohort];choices={m:{k:np.asarray(v) for k,v in frozen[m+'_'+cohort].items() if k in ['controller','state']} for m in ['A3','A0']};choices['G_E']=np.asarray(frozen['G_E']);metrics,extra=evaluate(lo[:,ids],hi[:,ids],choices);metrics_by_cohort[cohort]=(ids,metrics);details[cohort]={}
  for j,name in enumerate(names):
   details[cohort][name]={m:{'success_pct':[float(v[0][j].mean()*100),float(v[1][j].mean()*100)],'regret_pp':[float((metrics['full menu oracle'][0][j]-v[1][j]).mean()*100),float((metrics['full menu oracle'][1][j]-v[0][j]).mean()*100)]} for m,v in metrics.items()}
  result[cohort]['ST3']['crossfit_oracle_potential_gain_point_pp']=float((extra['state oracle']-extra['full menu oracle']).mean()*100)
  for category,row in result[cohort]['nominal_to_safety'].items():
   n=row['denominator'];row['fractions']={k:(v/n if n else None) for k,v in row['counts'].items()}
 ids,metrics=metrics_by_cohort['1b'];grouped={};a3=metrics['A3 zero-shot'];oracle=metrics['full menu oracle']
 for j,index in enumerate(ids):
  p=states[index]['physical'];pos=np.asarray(p['positions']);goal=np.asarray(p['goals']);vel=np.asarray(p['velocities']);angles=np.unwrap(np.arctan2(pos[:,1],pos[:,0])-np.arange(4)*np.pi/2)
  conditions={'radius_below_2.3':min(np.linalg.norm(pos,axis=1).min(),np.linalg.norm(goal,axis=1).min())<2.3,'speed_above_.11':np.linalg.norm(vel,axis=1).max()>.11,'angular_spread_above_.4':np.ptp(angles)>.4}
  for k,v in conditions.items():grouped.setdefault(k+'='+str(bool(v)),[]).append(j)
 result['shift_strata']={k:{'states':len(jj),'A3_success_pct':[float(a3[0][:,jj].mean()*100),float(a3[1][:,jj].mean()*100)],'A3_crossfit_regret_pp':[float((oracle[0][:,jj]-a3[1][:,jj]).mean()*100),float((oracle[1][:,jj]-a3[0][:,jj]).mean()*100)],'note':'Descriptive; oracle fit on full cohort halves, not refit per stratum'} for k,jj in grouped.items()}
 assert {k:result[k]['endpoint'] for k in ['1a','1b']}==endpoints
 write(HERE/'CONTROLLER_DETAILS.json',details);write(HERE/'results.json',result)
 seal('DESCRIPTIVE_SUPPLEMENT.receipt.json',[HERE/'descriptive_supplement.py',HERE/'CONTROLLER_DETAILS.json',HERE/'results.json'],original_results_sha256=before,primary_endpoints_unchanged=True,no_new_labels_training_or_selection=True)
if __name__=='__main__':main()
