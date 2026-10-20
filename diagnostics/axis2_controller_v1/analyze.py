from common import *
from statistics_core import summarize
import csv
ARMS=['nominal_only','safety_baseline']+MENU

def arrays(rows,names,ids,arms,loader):
 shape=(len(names),len(ids),len(arms),2);lo=np.zeros(shape);events={k:np.zeros(shape) for k in ['collision','timeout','unknown']};seen=np.zeros(shape,bool)
 for c in rows:
  ix=(names.index(c['base_controller_name']),ids.index(c['state_uid']),arms.index(c['arm']),c['seed']);assert not seen[ix];seen[ix]=True;r=loader(c);lo[ix]=r['success'];events['collision'][ix]=r['collision'];events['timeout'][ix]=r['timeout'];events['unknown'][ix]=r.get('numerical_failure',0)
 assert seen.all(),(shape,seen.sum());return lo,lo+events['unknown'],events

def main():
 prereg()
 for path,digest in read(HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json')['files'].items():assert sha(path)==digest,path
 rows=read(HERE/'main_CASES.json');assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in rows)
 seal('ENDPOINT_OPENING.receipt.json',[HERE/'CHOICES_FROZEN.receipt.json',HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json',HERE/'main_CASES.json'],all_main_present=True)
 names=read(HERE/'FROZEN_CHOICES.json')['controller_order'];ids=[s['state_uid'] for s in read(HERE/'EVAL_STATES.json')];lo,hi,events=arrays(rows,names,ids,ARMS,lambda c:read(HERE/'main/raw'/f"{c['rollout_uid']}.json"));frozen=read(HERE/'FROZEN_CHOICES.json')['controllers'];choices={'nominal only':0,'safety baseline':1,'F10':2,'F13':3,'S500':7}
 for k,label in [('A3_full','A3 full'),('A0_full','A0'),('G_E','G_E'),('mixed_full','mixed full'),('A3_fixed2','A3 fixed2'),('A0_fixed2','A0 fixed2')]:choices[label]=np.asarray([ARMS.index(frozen[n][k]) for n in names])
 np.savez_compressed(HERE/'LABELS.npz',lower=lo,upper=hi,**events,controller_names=names,state_uids=ids,arms=ARMS);results={}
 for j,unit in enumerate(['2d_all','G','DM','VO','PT']):
  ii=np.arange(len(names)) if unit=='2d_all' else np.asarray([i for i,n in enumerate(names) if n.rstrip('0123456789')==unit]);ch={k:np.asarray(v)[ii] if np.asarray(v).ndim else v for k,v in choices.items()};results[unit]=summarize(unit,lo[ii],hi[ii],{k:v[ii] for k,v in events.items()},ch,list(range(2,8)),[2,3,4],202610153300+j);results[unit]['evidence']='CONFIRMATORY'
  if unit!='2d_all' and len(ii)<4:results[unit]['type_scope_status']='INCONCLUSIVE: fewer than four eligible instances'
  write(HERE/f'{unit}_RESULTS.json',results[unit]);print(unit,'analyzed',flush=True)
 inventory=read(HERE/'SOURCE_INVENTORY.json');br=read(HERE/'baseline_CASES.json')
 for j,(unit,inv) in enumerate(inventory.items()):
  h=Path(inv['source']);phase='wave1' if unit=='2cN2' else 'test' if unit=='2cTEST' else 'main';nn=inv['controllers'];ss=inv['state_uids'];oldrows=[c for c in read(h/f'{phase}_CASES.json') if c['base_controller_name'] in nn and c['state_uid'] in ss and c['seed'] in [0,1]];oldarms=list(dict.fromkeys(c['arm'] for c in oldrows));arms=['nominal_only','safety_baseline']+oldarms;merged=[];where={}
  for c in oldrows:merged.append(c);where[c['rollout_uid']]=h/phase/'raw'/f"{c['rollout_uid']}.json"
  for c in br:
   if c['cohort']!=unit:continue
   v=dict(c,base_controller_name=c['base_controller_name'].split('::',1)[1],state_uid=c['original_state_uid']);merged.append(v);where[v['rollout_uid']]=HERE/'baseline/raw'/f"{v['rollout_uid']}.json"
  ll,hh,ee=arrays(merged,nn,ss,arms,lambda c:read(where[c['rollout_uid']]))
  if unit in ['2a','2b']:
   table=list(csv.DictReader((h/'FROZEN_CHOICES.csv').open()));cm={r['controller']:r['primary_action'] for r in table}
  else:
   src=read(h/('FROZEN_CHOICES.json' if unit=='2cN2' else 'TEST_CHOICES.json'));cm={r['controller']:r['A3_fixed' if unit=='2cN2' else 'C0_fixed'] for r in src['controllers']}
  ch={'nominal only':0,'safety baseline':1,'F10':arms.index('F10'),'S500':arms.index('S500'),'original frozen method':np.asarray([arms.index(cm[n]) for n in nn])}
  results[unit]=summarize(unit,ll,hh,ee,ch,list(range(2,len(arms))),[arms.index(a) for a in ['F10','F13','F2'] if a in arms],202610153305+j);results[unit].update(evidence='New baseline CONFIRMATORY; reused corrections retain source grade',source=str(h),arms=arms,original_frozen_choice=cm);np.savez_compressed(HERE/f'{unit}_LABELS.npz',lower=ll,upper=hh,**ee,controller_names=nn,state_uids=ss,arms=arms);write(HERE/f'{unit}_RESULTS.json',results[unit]);print(unit,'analyzed',flush=True)
 write(HERE/'results.json',results)
if __name__=='__main__':main()
