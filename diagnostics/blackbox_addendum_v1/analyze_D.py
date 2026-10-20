from common import *
from statistics_core import summarize,interval,difference
from audit_utils import source_inventory
import csv
EVENTS=['collision','timeout','unknown']
def baseline(label):
 if label in ['D1','D2']:
  z=np.load(R32/'LABELS.npz');sl=slice(0,64) if label=='D1' else slice(64,96);names=z['controller_names'].tolist();arms=z['arms'].tolist();data={k:z[k][:,sl].copy() for k in ['lower','upper']+EVENTS};f=read(R32/'FROZEN_CHOICES.json');part='1a' if label=='D1' else '1b';choices={key:[arms.index(f['menu'][i]) for i in f['choices'][tag+'_'+part]['controller']] for key,tag in [('A3 zero-shot','A3'),('A0','A0')]};choices['G_E']=[arms.index(f['menu'][i]) for i in f['choices']['G_E']];return data,names,arms,choices,list(range(2,6)),[2,3],5
 if label.startswith('D3'):
  H=int(label.split('H')[-1]);z=np.load(R35/f'H{H}_LABELS.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();data={k:z[k].copy() for k in ['lower','upper']+EVENTS};rel='Srel';f=read(HERE/'FROZEN_REFERENCE_CHOICES.json');choices={('A3 zero-shot' if k=='A3' else k):[arms.index(rel if v[n]=='S500' else v[n]) for n in names] for k,v in f['controllers'].items()};return data,names,arms,choices,list(range(2,len(arms))),[arms.index('F10'),arms.index('F13')],arms.index(rel)
 if label=='D5':
  z=np.load(R33/'LABELS.npz');allnames=z['controller_names'].tolist();names=[n for n in allnames if n.startswith('G') or n.startswith('DM')];ix=[allnames.index(n) for n in names];arms=z['arms'].tolist();data={k:z[k][ix].copy() for k in ['lower','upper']+EVENTS};f=read(R33/'FROZEN_CHOICES.json');choices={key:[arms.index(f['controllers'][n][tag]) for n in names] for key,tag in [('A3 zero-shot','A3_full'),('A0','A0_full'),('G_E','G_E')]};return data,names,arms,choices,list(range(2,8)),[arms.index(x) for x in ['F10','F13','F2']],arms.index('S500')
 r=float(label.split('r')[-1]);names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];arms=['nominal_only','safety_baseline','F10','S500'];shape=(14,16,4,1);data={k:np.zeros(shape) for k in ['lower','upper']+EVENTS};seen=np.zeros(shape,bool);cases=read(HERE/'geometry_controls_CASES.json')
 for c in cases:
  if c['cohort']!='same_state' or c['r']!=r:continue
  path=HERE/'geometry_controls/raw'/f"{c['rollout_uid']}.json";path=path if path.exists() else Path(c['reference_path']);raw=read(path);idx=(names.index(c['base_controller_name']),c['state_index'],arms.index(c['arm']),0);assert not seen[idx];seen[idx]=True;u=raw['numerical_failure'];data['lower'][idx]=raw['success'] if not u else 0;data['upper'][idx]=data['lower'][idx]+u
  for k in EVENTS:data[k][idx]=raw['numerical_failure' if k=='unknown' else k]
 assert seen.all();f=read(HERE/'FROZEN_REFERENCE_CHOICES.json');choices={('A3 zero-shot' if k=='A3' else k):[arms.index(v[n]) for n in names] for k,v in f['controllers'].items()};return data,names,arms,choices,[2,3],[2],3

def summarize_unit(label,data,names,arms,choices,full,fixed,schedule,newcases,db,seed):
 lo,hi=data['lower'],data['upper'];shape=lo.shape;right={k:np.zeros(shape[:2]+(1,shape[-1])) for k in data};seen=np.zeros(shape[:2]+(shape[-1],),bool)
 for c in newcases:
  if c['subunit']!=label:continue
  idx=(names.index(c['base_controller_name']),c['state_index'],c['seed']);assert not seen[idx];seen[idx]=True;r=db[c['rollout_uid']];u=r['numerical_failure'];right['lower'][idx[0],idx[1],0,idx[2]]=r['success'] if not u else 0;right['upper'][idx[0],idx[1],0,idx[2]]=(r['success'] if not u else 0)+u
  for k in EVENTS:right[k][idx[0],idx[1],0,idx[2]]=r['numerical_failure' if k=='unknown' else k]
 assert seen.all(),(label,int(seen.sum()),seen.size);data={k:np.concatenate([data[k],right[k]],axis=2) for k in data};ri=shape[2];choices=dict(choices,**{'nominal only':0,'safety baseline':1,'F10':arms.index('F10'),'S500 class':schedule,'continuing right-hand':ri});out=summarize(label,data['lower'],data['upper'],{k:data[k] for k in EVENTS},choices,full,fixed,seed);rng=np.random.default_rng(seed);pairs={'schedule_minus_right_hand':(schedule,ri),'right_hand_minus_safety':(ri,1)};boot={k:[] for k in pairs};nc,ns=shape[:2]
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns))
  for k,(a,b) in pairs.items():boot[k].append([(data['lower'][cc[:,None],ss,a]-data['upper'][cc[:,None],ss,b]).mean(),(data['upper'][cc[:,None],ss,a]-data['lower'][cc[:,None],ss,b]).mean()])
 for k,(a,b) in pairs.items():
  ci=interval(boot[k]);out['paired'][k]=dict(estimate_pp=difference((data['lower'][:,:,a],data['upper'][:,:,a]),(data['lower'][:,:,b],data['upper'][:,:,b])),CI_pp=ci,status='SCHEDULE_BETTER' if ci[0]>0 else 'RIGHT_HAND_BETTER' if ci[1]<0 else 'UNDETECTED')
 counts={};records=[]
 for c in range(nc):
  for s in range(ns):
   for q in range(shape[-1]):
    def value(a):return 'unknown' if data['unknown'][c,s,a,q] else 'success' if data['lower'][c,s,a,q] else 'collision' if data['collision'][c,s,a,q] else 'timeout'
    safe,scheduled,rhr=[value(a) for a in [1,schedule,ri]];key=' | '.join([safe,scheduled,rhr]);counts[key]=counts.get(key,0)+1
    if safe not in ['unknown','success']:
     category='unknown_comparator' if 'unknown' in [scheduled,rhr] else 'both' if scheduled==rhr=='success' else 'schedule_only' if scheduled=='success' else 'right_hand_only' if rhr=='success' else 'neither';records.append(dict(controller=names[c],state_index=s,seed=q,safety=safe,schedule=scheduled,right_hand=rhr,recovery=category))
 out.update(controller_names=names,evidence='CONFIRMATORY',cross_tab_counts=counts,recovery_counts={k:sum(r['recovery']==k for r in records) for k in ['both','schedule_only','right_hand_only','neither','unknown_comparator']},recovery_records=records)
 if label.startswith('D4'):
  r=label.split('r')[-1];a=read(HERE/'A_RESULTS.json')['same_state_radii'][r];out['inferential_endpoint_eligible']=a['median_max_success_pct'][0]>=20;out['evidence']='CONFIRMATORY' if out['inferential_endpoint_eligible'] else 'DESCRIPTIVE: radius capability criterion not passed';out['unavailable']='Fixed eta13 original radius labels absent; full-menu oracle restricted to available eta10 and500-step schedule; no invented full-menu claim'
 return out

def main():
 assert (HERE/'right_hand_evaluation_COMPLETE.receipt.json').exists();cases=read(HERE/'right_hand_evaluation_CASES.json');db={r['rollout_uid']:r for r in read(HERE/'right_hand_evaluation_DB_ROWS.json')};out={};units=['D1','D2','D3_H500','D3_H1000']+[f'D4_r{r}' for r in [.8,.85,.95,1.]]+['D5']
 for j,label in enumerate(units):
  out[label]=summarize_unit(label,*baseline(label),cases,db,202610163730+j);write(HERE/'D_PARTIAL_RESULTS.json',out)
 # Type-wise comparison retains its original paired state structure.
 base=baseline('D5')
 for j,(key,prefix) in enumerate([('D5_Gaussian','G'),('D5_distributed','DM')]):
  data,names,arms,ch,full,fixed,sch=base;ix=[i for i,n in enumerate(names) if n.startswith(prefix)];new=[dict(c,subunit=key) for c in cases if c['subunit']=='D5' and c['base_controller_name'].startswith(prefix)];out[key]=summarize_unit(key,{k:v[ix] for k,v in data.items()},[names[i] for i in ix],arms,{k:np.asarray(v)[ix] for k,v in ch.items()},full,fixed,sch,new,db,202610163739+j)
 write(HERE/'D_RESULTS.json',out);rows=[dict(subunit=k,**r) for k,v in out.items() for r in v['recovery_records']]
 with (HERE/'RIGHT_HAND_RECOVERY_CROSS_TABLE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 seal('D_COMPLETE.receipt.json',[HERE/'D_RESULTS.json',HERE/'RIGHT_HAND_RECOVERY_CROSS_TABLE.csv',HERE/'RIGHT_HAND_CHOICES_FROZEN.receipt.json'],new_evaluation_records=len(cases));source_inventory('imports_analyze_D.json')
if __name__=='__main__':main()
