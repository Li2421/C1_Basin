"""Descriptive standardized metrics from existing R28/R30 labels only."""
from common import *
from statistics_core import summarize,evaluate,interval
import csv

def partial(lo,hi,events,ch,seed):
 nc,ns=lo.shape[:2];mm,sel=evaluate(lo,hi,ch,[0,1,2,3],[0,1]);boot={k:[] for k in mm};rng=np.random.default_rng(seed)
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));choices={k:np.asarray(v)[cc] if np.asarray(v).ndim else v for k,v in ch.items()};m,_=evaluate(lo[cc[:,None],ss],hi[cc[:,None],ss],choices,[0,1,2,3],[0,1],ss)
  for k,(a,b) in m.items():boot[k].append([a.mean(),b.mean()])
 out={};cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:]
 for k,(a,b) in mm.items():
  ol,oh=mm['full menu oracle'];rl=(ol-b).mean((1,2))*100;rh=(oh-a).mean((1,2))*100;out[k]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=interval(boot[k]),mean_regret_pp=[rl.mean(),rh.mean()],max_regret_pp=[rl.max(),rh.max()],catastrophes_certain=int(((mm['F10'][0]-b).mean((1,2))>=.1-1e-12).sum()),catastrophes_possible=int(((mm['F10'][1]-a).mean((1,2))>=.1-1e-12).sum()),harm_certain=None,harm_possible=None,events_pct={e:v[cc,ss,sel[k]].mean()*100 for e,v in events.items()})
 return clean(out)

def main():
 h=read(HERE/'HISTORICAL_RESULTS.json');z=np.load(D/'teacher_student_macflow_v1/analysis_arrays.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();frozen=read(D/'teacher_student_macflow_v1/FROZEN_CHOICES.json')['controllers'];ch={'nominal only':0,'safety baseline':1,'F10':2,'S500':5}
 for label,key in [('A3 zero-shot','A3_full'),('A0','A0_full'),('G_E','G_E')]:ch[label]=np.asarray([arms.index(frozen[n][key]) for n in names])
 lo=z['success_lower'].transpose(0,1,3,2);events={k:z[k].transpose(0,1,3,2) for k in ['collision','timeout','unknown']};r=summarize('R30 student',lo,lo+events['unknown'],events,ch,[2,3,4,5],[2,3],202610153340);r.update(evidence=h['deployment']['2e_student']['evidence'],source='R30 frozen labels/choices; descriptive crossfit re-estimation',gaps=h['deployment']['2e_student']['gaps']);h['deployment']['2e_student']=r
 p=read(D/'decentralized_pipeline_v1/main_protocol.json');other=read(D/'teacher_student_macflow_v1/main_protocol.json');checks=[dict(index=i,physical_equal=all(np.array_equal(a['physical'][k],b['physical'][k]) for k in ['positions','velocities','goals']),original_uid_equal=a['uid']==b['uid'],rng_namespace_equal=a['rng_namespace']==b['rng_namespace']) for i,(a,b) in enumerate(zip(p['states'],other['states']))];assert len(checks)==32 and all(c['physical_equal'] and c['original_uid_equal'] and c['rng_namespace_equal'] for c in checks);write(HERE/'DEPLOYMENT_PAIRING_AUDIT.json',dict(utc=utc(),passed=True,checks=checks,normalization='Only redundant physical timestep0 differs; original uid and random namespace identical'));nn=p['controller_order'];assert nn==names,'R28/R30 controller order differs: must explicitly align';z=np.load(D/'decentralized_pipeline_v1/analysis_arrays.npz');tc=read(D/'teacher_student_macflow_v1/DEVELOPMENT_TEACHER_CHOICES.json')['controllers'];aa=['F10','F13','S300','S500'];cols=[(pipe,a) for pipe in ['shared','local','central'] for a in aa]+[('rhr','B-RHR'),('rhr_f10','B-RHR+F10')];timeout=np.zeros(z['success'].shape)
 for row in csv.DictReader((D/'decentralized_pipeline_v1/rollout_metrics.csv').open()):timeout[nn.index(row['controller']),int(row['state_index']),cols.index((row['pipeline'],row['arm'])),int(row['seed'])]=int(row['timeout'])
 for j,pipe in enumerate(['shared','local','central']):
  ii=[cols.index((pipe,a)) for a in aa];lo=z['success'][:,:,ii];hi=lo+z['unknown'][:,:,ii];ee={'unknown':z['unknown'][:,:,ii],'collision':z['collision'][:,:,ii],'timeout':timeout[:,:,ii]};ch={'F10':0,'F13':1,'S300':2,'S500':3,'A3 zero-shot':np.asarray([aa.index(tc[n]['A3']) for n in nn]),'A0':np.asarray([aa.index(tc[n]['A0']) for n in nn])};unit='2e_'+pipe
  if pipe=='local':
   # Reuse R30 teacher supplements on the exact same ordered physical states.
   base=np.zeros((14,32,2,2));be={k:np.zeros(base.shape) for k in ee};seen=np.zeros(base.shape,bool);rp=D/'teacher_student_macflow_v1'
   for c in read(rp/'teacher_CASES.json'):
    idx=(nn.index(c['base_controller_name']),c['state_index'],['nominal only','safety baseline'].index(c['arm']),c['seed']);raw=read(rp/'teacher/raw'/f"{c['rollout_uid']}.json");base[idx]=raw['success'];seen[idx]=True
    for k in be:be[k][idx]=raw['numerical_failure' if k=='unknown' else k]
   assert seen.all();combined=np.concatenate([base,lo],2);events={k:np.concatenate([be[k],ee[k]],2) for k in ee};choices={k:np.asarray(v)+2 for k,v in ch.items()};choices.update({'nominal only':0,'safety baseline':1});result=summarize(unit,combined,combined+events['unknown'],events,choices,[2,3,4,5],[2,3],202610153341+j);h['deployment'][unit].update(result)
  else:h['deployment'][unit]['methods'].update(partial(lo,hi,ee,ch,202610153341+j))
  h['deployment'][unit]['analysis_qualification']='Descriptive only: R30 historical-R5 choices transported unchanged; crossfit re-estimated from existing R28 labels, common four-arm menu. Missing central/shared baseline harms/transitions remain N/A.'
 write(HERE/'HISTORICAL_RESULTS.json',h)
if __name__=='__main__':main()
