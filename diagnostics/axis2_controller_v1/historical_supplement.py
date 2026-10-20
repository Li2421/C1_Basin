"""Read-only reuse of historical predictions, labels and deployment tables."""
from common import *
from statistics_core import summarize,oracle_bounds

def prepare():
 choices={};files=[]
 for unit,tag in [('2a','T1'),('2b','T3'),('2cN2','T4'),('2cTEST','T5')]:
  path=D/'method_benchmark_v1'/f'{tag}_PREDICTIONS.npz';z=np.load(path);files.append(path);names=z['controller_names'].tolist();pa=z['arms'].tolist();inv=read(HERE/'SOURCE_INVENTORY.json')[unit];h=Path(inv['source']);phase='wave1' if unit=='2cN2' else 'test' if unit=='2cTEST' else 'main';available={c['arm'] for c in read(h/f'{phase}_CASES.json')};allowed=[a for a in pa if a in available];out={}
  for i,name in enumerate(names):
   out[name]={}
   for model,label in [('M4','A3 zero-shot'),('A0','A0')]:
    mean=z[model][:,i].mean((0,1));out[name][label]=allowed[int(mean[[pa.index(a) for a in allowed]].argmax())]
   out[name]['G_E']=str(z['GE_choices'][i])
  assert set(inv['controllers'])<=set(out);choices[unit]=out
 write(HERE/'HISTORICAL_SUPPLEMENT_CHOICES.json',dict(utc=utc(),choices=choices,rule='R26 frozen probabilities averaged across their original full state set; available common menu only; no success labels opened for this selection',grade='DEVELOPMENT added-method summaries; CT4 original frozen choices unaffected'))
 seal('HISTORICAL_SUPPLEMENT_CHOICES.receipt.json',files+[HERE/'HISTORICAL_SUPPLEMENT_CHOICES.json',HERE/'historical_supplement.py'],no_new_success_labels_used=True)

def run():
 for p,s in read(HERE/'HISTORICAL_SUPPLEMENT_CHOICES.receipt.json')['files'].items():assert sha(p)==s,p
 results={};frozen=read(HERE/'HISTORICAL_SUPPLEMENT_CHOICES.json')['choices']
 for j,unit in enumerate(['2a','2b','2cN2','2cTEST']):
  z=np.load(HERE/f'{unit}_LABELS.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();ch={'nominal only':0,'safety baseline':1,'F10':arms.index('F10'),'S500':arms.index('S500')}
  for model in ['A3 zero-shot','A0','G_E']:ch[model]=np.asarray([arms.index(frozen[unit][n][model]) for n in names])
  results[unit]=summarize(unit,z['lower'],z['upper'],{k:z[k] for k in ['collision','timeout','unknown']},ch,list(range(2,len(arms))),[arms.index(a) for a in ['F10','F13','F2'] if a in arms],202610153320+j);results[unit]['evidence']='DEVELOPMENT additional frozen-model summaries; original CT4 in results.json'
 z=np.load(D/'nominal_safety_baselines_v1/LABELS.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();frozenMP=read(D/'other_controller_classes_v1/FROZEN_CHOICES.json')['controllers']
 for j,prefix in enumerate(['M','P']):
  ii=[i for i,n in enumerate(names) if n.startswith(prefix) and n[1:].isdigit()];nn=[names[i] for i in ii];lo=z['success'][ii].astype(float);hi=lo+z['unknown'][ii];ch={'nominal only':0,'safety baseline':1,'F10':2,'S500':7}
  for model,key in [('A3 zero-shot','A3_full'),('A0','A0_full'),('G_E','G_E')]:ch[model]=np.asarray([arms.index(frozenMP[n][key]) for n in nn])
  results[prefix]=summarize(prefix,lo,hi,{k:z[k][ii] for k in ['collision','timeout','unknown']},ch,list(range(2,8)),[2,3,4],202610153330+j);results[prefix].update(evidence='DEVELOPMENT reuse; R27 original source qualifications and R29 private DB',source='diagnostics/nominal_safety_baselines_v1',semantics='R29 hard_safety is single TT nominal projection; new TT eta0 uses original double projection; no assertion of bitwise equivalence')
 r28=read(D/'decentralized_pipeline_v1/results.json');r30=read(D/'teacher_student_macflow_v1/results.json');deployment={}
 for name in ['central','shared','local']:
  methods={}
  for arm in ['F10','F13','S300','S500']:
   v=r28['rates'][name+'/'+arm];methods[arm]=dict(success_pct=[v['success_lower']*100,v['success_upper']*100],success_95CI_pct=[x*100 for x in v['ci_unknown_envelope']],events_pct=dict(collision=v['collisions']/v['n']*100,timeout=v['timeouts']/v['n']*100,unknown=v['unknown']/v['n']*100))
  # R30 organizes matched R28 teacher labels with available baseline supplements.
  if name in r30['teacher_tables']:
   for key,v in r30['teacher_tables'][name].items():
    label={'A3_historical_R5':'A3 zero-shot','A0_historical_R5':'A0'}.get(key,key)
    if label in methods:continue
    methods[label]=dict(success_pct=[v['success']['mean']*100,v.get('success_upper',v['success']['mean'])*100],success_95CI_pct=[x*100 for x in v['success']['CI95']],events_pct={k:v[k]*100 for k in ['collision','timeout'] if k in v})
  deployment['2e_'+name]=dict(methods=methods,evidence='DEVELOPMENT historical R28/R30 only',source='R28 results.json; R30 teacher_tables where available',gaps='No same-pipeline nominal/safety label if absent; no R5 critic frozen for changed decentralized projection; original R28 has no fixed-menu or controller-critic experiment',expert='R28 privileged offline planner:32/32, only its original32states; not deployable online')
 methods={}
 for key,v in {**r30['student_table'],**r30['selectors']}.items():
  label={'A3_full':'A3 zero-shot','A0_full':'A0'}.get(key,key);methods[label]=dict(success_pct=[v['success']['mean']*100,v.get('success_upper',v['success']['mean'])*100],success_95CI_pct=[x*100 for x in v['success']['CI95']],events_pct={k:(v[k]['mean'] if isinstance(v[k],dict) else v[k])*100 for k in ['collision','timeout','unknown'] if k in v})
 deployment['2e_student']=dict(methods=methods,evidence='DEVELOPMENT reuse; original R30 confirmatory calculations qualified by protocol deviations',source='R30 results.json',gaps='R30 source write-scope failure448PoC journals and6 step0 students; local/global chart information mismatch; original oracle kept in source only',nominal_to_safety=r30['transitions'])
 write(HERE/'HISTORICAL_RESULTS.json',dict(cohorts=results,deployment=deployment,original_R28=r28,original_R30=r30))
 # Held-class regret from already-trained frozen models; no fit or model selection.
 from train_mixed import MixedNet
 from flax import serialization
 import jax,jax.numpy as jnp
 jax.config.update('jax_enable_x64',False);z=np.load(HERE/'MIXED_TRAINING_DATA.npz');net=MixedNet();loto={};step=read(HERE/'DESIGN_CONFIG.json')['mixed_critic']['LOTO_step']
 for cls in ['Flow','MPC','potential']:
  mask=z['classes']==cls;X=z['X'][mask];nc=int(mask.sum()//(32*6));Q=4 if cls=='Flow' else 2;lo=(z['y'][mask]/Q).reshape(nc,32,6,1);hi=((z['y'][mask]+Q-z['n'][mask])/Q).reshape(nc,32,6,1);pred=[]
  for seed in [17,23,41]:
   path=HERE/'mixed/loto'/cls/f'seed{seed}'/f'step{step}.msgpack';pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,32),jnp.float32)),path.read_bytes());pred.append(np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(X)))).reshape(nc,32,6))
  choice=np.mean(pred,(0,2)).argmax(1);cc=np.arange(nc)[:,None];ss=np.arange(32)[None,:];a=lo[cc,ss,choice[:,None]];b=hi[cc,ss,choice[:,None]];ol,oh,_,_=oracle_bounds(lo,hi,list(range(6)));rl=(ol-b).mean((1,2))*100;rh=(oh-a).mean((1,2))*100;loto[cls]=dict(controllers=nc,success_pct=[a.mean()*100,b.mean()*100],mean_crossfit_regret_pp=[rl.mean(),rh.mean()],max_crossfit_regret_pp=[rl.max(),rh.max()],in_sample_regret_pp=[(lo.mean((1,3)).max(1)-b.mean((1,2))).mean()*100,(hi.mean((1,3)).max(1)-a.mean((1,2))).mean()*100],choices=[MENU[i] for i in choice],step=step)
 write(HERE/'MIXED_LOTO_RESULTS.json',dict(evidence='DEVELOPMENT',qualification='Preprocessing/PCA and step1000 are frozen prior knowledge derived with historical Flow; not an independent prospective transfer test',held_class=loto))
if __name__=='__main__':globals()[sys.argv[1]]()
