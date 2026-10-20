from common import *
from statistics import summarize,ci

def records(stage):
 return [(c,read(HERE/stage/'raw'/f"{c['rollout_uid']}.json")) for c in read(HERE/f'{stage}_CASES.json')]
def main():
 assert (HERE/'ALL_FACTOR_LABELS_COMPLETE.receipt.json').exists();frozen=read(HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json');assert sha(HERE/'statistics.py')==frozen['files'][str(HERE/'statistics.py')]
 if not (HERE/'ENDPOINT_OPENING.receipt.json').exists():seal('ENDPOINT_OPENING.receipt.json',[HERE/'ALL_FACTOR_LABELS_COMPLETE.receipt.json',HERE/'REFERENCE_CHOICES_FROZEN.receipt.json',HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json'])
 names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];states=read(HERE/'EVAL_STATES_ORIGINAL.json');stateidx={s['state_uid']:i for i,s in enumerate(states)};source=[]
 for stage in ['baseline','horizon','geometry']:
  if (HERE/f'{stage}_CASES.json').exists():source+=records(stage)
 for q in read(HERE/'SOURCE_CORRECTIONS.json'):
  c=q['case']
  if c['arm'] not in ['F10','F13','S500']:continue
  assert sha(q['raw_path'])==q['raw_sha256'];c=dict(c,H=700,r=.9,original_state_uid=c['state_uid']);source.append((c,read(q['raw_path'])))
 levels=[('H500',500,.9),('H700',700,.9),('H1000',1000,.9)]+[(f'r{r:.2f}',700,r) for r in [.8,.85,.9,.95,1.]];out={};arrays={};choices0=read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers'];cap=read(HERE/'CAPABILITY_GATE.json')['radii'];unknown=[];audit=[]
 for u,(label,H,r) in enumerate(levels):
  if r!=.9 and cap[str(r)]['status']!='PASS':out[label]=dict(status='GAP',reason='Controller capability '+cap[str(r)]['status'],capability=cap[str(r)]);continue
  arms=['nominal_only','safety_baseline','F10','F13']+(['S500'] if H==700 else ['Srel']+(['Sabs'] if H==1000 else []));shape=(14,32,len(arms),2);lo=np.zeros(shape);hi=np.zeros(shape);events={k:np.zeros(shape) for k in ['collision','timeout','unknown']};filled=np.zeros(shape,bool);grades={};stepcounts=dict(new_physical_steps=0,prefix_reused_steps=0)
  for c,q in source:
   if c['H']!=H or c['r']!=r or c['arm'] not in arms:continue
   idx=(names.index(c['base_controller_name']),stateidx[c['original_state_uid']],arms.index(c['arm']),c['seed']);assert not filled[idx],(label,idx);filled[idx]=True;lo[idx]=q['success'];hi[idx]=q['success']+q['numerical_failure'];events['collision'][idx]=q['collision'];events['timeout'][idx]=q['timeout'];events['unknown'][idx]=q['numerical_failure'];g=q.get('evidence_grade','DEVELOPMENT historical R10');grades[g]=grades.get(g,0)+1
   for k in stepcounts:stepcounts[k]+=q.get(k,0)
   if q['numerical_failure']:unknown.append(dict(level=label,rollout_uid=c['rollout_uid'],error=q.get('error')))
  assert filled.all(),(label,int((~filled).sum()));choice={'nominal only':0,'safety baseline':1,'F10':2,'F13':3};relative='S500' if H==700 else 'Srel';absolute='S500' if H==700 else 'F13' if H==500 else 'Sabs';choice.update({k:arms.index(v) for k,v in [('Srel',relative),('Sabs',absolute)]})
  if H==700:choice['S500']=arms.index('S500')
  for method,old in choices0.items():
   choice[method]=[arms.index(relative if old[n]=='S500' else old[n]) for n in names]
  choice['A3 absolute']=[arms.index(absolute if choices0['A3'][n]=='S500' else choices0['A3'][n]) for n in names];res=summarize(lo,hi,events,arms,choice,202610153500+u);res.update(H=H,r=r,arms=arms,evidence_grade_counts=grades,physical_step_accounting=stepcounts,progress_rule='GAP missing literal full896p500',complete_protocol_menu=False,methods_scope='Available nonprogress correction menu; all existing labels retain source grades')
  if H in [500,1000]:res['HZ1']=dict(status='SUPPORTED' if res['paired']['Srel minus best fixed oracle']['CI_pp'][0]>0 else 'REJECTED',**res['paired']['Srel minus best fixed oracle'])
  res['HZ2']=res['paired']['Srel minus '+('F13' if H==500 else 'Sabs')];gain=res['paired']['A3 minus safety baseline']['CI_pp'];regret=res['methods']['A3']['mean_regret_pp'];res['HZ4_available_menu']=dict(status='SUPPORTED' if gain[0]>0 and regret[1]<=3 else 'REJECTED' if gain[1]<=0 or regret[0]>3 else 'INCONCLUSIVE',A3_minus_safety_CI_pp=gain,regret_pp=regret);res['HZ4']=dict(status='INCONCLUSIVE',reason='Sprog unavailable, so complete protocol menu regret cannot be bounded above by available-menu regret',available_menu=res['HZ4_available_menu'])
  if r!=.9:
   s=res['paired']['S500 minus safety baseline']['CI_pp'];m=res['methods']['A3'];res['GM1']=dict(status='SUPPORTED' if s[0]>0 and gain[0]>0 and regret[1]<=3 and m['harm_possible']==0 else 'REJECTED' if s[1]<=0 or gain[1]<=0 or regret[0]>3 or m['harm_certain']>0 else 'INCONCLUSIVE',S500_minus_safety_CI_pp=s,A3_minus_safety_CI_pp=gain,regret_pp=regret,harm_certain=m['harm_certain'],harm_possible=m['harm_possible']);rates=lo.mean((1,3));res['GM2']=dict(best_fixed_by_controller={n:arms[2+int(rates[i,2:4].argmax())] for i,n in enumerate(names)},schedule_minus_fixed=res['paired']['S500 minus best fixed oracle'])
  out[label]=res;arrays[label]=(lo,hi);np.savez_compressed(HERE/(label+'_LABELS.npz'),lower=lo,upper=hi,arms=np.asarray(arms),controller_names=np.asarray(names),**events)
 # Privileged deterministic Q1 expert, independent of actor-menu comparisons.
 experts=[]
 for stage in ['horizon_expert','geometry_expert']:
  if (HERE/f'{stage}_CASES.json').exists():experts+=records(stage)
 for label,H,r in levels:
  if 'methods' not in out[label]:continue
  rr=[q for c,q in experts if c['H']==H and c['r']==r];assert len(rr)==32;(a,b)=(np.array([q['success'] for q in rr]),np.array([q['success']+q['numerical_failure'] for q in rr]));rng=np.random.default_rng(202610153599);ix=rng.integers(32,size=(5000,32));out[label]['expert']=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=ci(np.stack([a[ix].mean(1),b[ix].mean(1)],1)),events_pct={k:np.mean([q[k] for q in rr])*100 for k in ['collision','timeout','numerical_failure']},scope='Privileged deterministic planner Q1 on32 states, not competing method')
 out['endpoints']=dict(HZ1={k:out[k]['HZ1'] for k in ['H500','H1000']},HZ2={k:out[k]['HZ2'] for k in ['H500','H1000']},HZ3=dict(status='GAP',reason='Literal896 p500 median unobserved; no response adopting frozen supplemental convention'),HZ4={k:out[k]['HZ4'] for k in ['H500','H700','H1000']},GM1={k:v.get('GM1',dict(status='GAP',reason=v.get('reason','baseline descriptive'))) for k,v in out.items() if k.startswith('r') and k!='r0.90'})
 write(HERE/'UNKNOWN_RECORDS.json',unknown);write(HERE/'results.json',out);print('Analysis complete',flush=True)
if __name__=='__main__':main()
