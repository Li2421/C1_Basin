from common import *
from design import FAMILIES,profiles
from expanded import old,transform,features
from flax import serialization
import jax,jax.numpy as jnp
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
jax.config.update('jax_enable_x64',False)
STAGES=['pilot_centralized_sampling','pilot_deterministic','pilot_prioritized_astar']
def main():
 prereg();assert (HERE/'EXPANDED_MODEL_FROZEN.receipt.json').exists();cap={};eligible=[];feature={};index=[];fair={};pilot_cases={};pilot_protocols={}
 for stage in STAGES:
  c=read(HERE/f'{stage}_CAPABILITY.json');cap[stage]=c;eligible.extend(c['eligible']);p=read(HERE/f'{stage}_protocol.json');pilot_protocols[stage]=p;pilot_cases.update({name:(stage,[x for x in read(HERE/f'{stage}_CASES.json') if x['base_controller_name']==name]) for name in c['eligible']});fair.update(read(HERE/f'{stage}_FAIR_RULES.json'))
  phase=('probe_only_' if stage=='pilot_centralized_sampling' else 'probe_')+stage;names=read(HERE/f'{phase}_protocol.json')['controller_order'];f={n:np.full((16,2,5),np.nan,np.float32) for n in names}
  for case in read(HERE/f'{phase}_CASES.json'):
   path=HERE/phase/'raw'/f"{case['rollout_uid']}.json";r=read(path);assert not r['terminal_labels_read'] and not r['jax_enable_x64'];assert sha(r['trace_path'])==r['trace_sha256']
   if r['complete']:f[case['base_controller_name']][case['state_index'],[10,13].index(case['eta_index'])]=r['statistics']
   index.append(dict(raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256'],complete=r['complete']))
  feature.update({n:v.reshape(160) for n,v in f.items()})
 # Reuse all six exact historical centralized-sampling probe features.
 oldprobe=read(D/'other_controller_classes_v1/probe_protocol.json');current=read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')
 assert len(oldprobe['states'])==len(current['states'])==16
 for a,b in zip(oldprobe['states'],current['states']):
  assert a['uid']==b['uid'] and a['rng_namespace']==b['rng_namespace'] and a['physical']==b['physical']
 with np.load(D/'other_controller_classes_v1/R5_FEATURES.npz') as z:
  for i in range(6):feature['centralized_sampling'+str(i)]=z['features'][list(z['controller_names']).index('M'+str(i))]
 names=[n for family in FAMILIES for n in eligible if n.startswith(family)];r5=np.asarray([feature[n] for n in names]);states=read(HERE/'EVAL_STATES.json');np.savez_compressed(HERE/'R5_FEATURES.npz',features=r5,controller_names=names);write(HERE/'R5_RAW_INDEX.json',index)
 write(HERE/'PROBE_FEATURE_REPORT.json',dict(utc=utc(),controllers=names,features_shape=list(r5.shape),new_probe_episodes=len(index),reused_episodes=6*32,source_reused=str(D/'other_controller_classes_v1/R5_FEATURES.npz'),source_sha256=sha(D/'other_controller_classes_v1/R5_FEATURES.npz'),same_frozen_state_order_keys_and_TT200=True,nonfinite_features=int((~np.isfinite(r5)).sum()),preprocessing='Frozen round16 imputations, removed feature coordinates105:110, source PCA2; no refit',no_task_labels_in_probes=True))
 tr=transform();selected_steps=read(D/'ring_controller_critic_v1/fits/selected_steps.json');prediction={};model_files=[]
 original_seal=read(D/'ring_controller_critic_v1/MODEL_FREEZE.receipt.json')['files']
 for rel in ['fits/full_transform.npz','fits/selected_steps.json']:
  assert sha(D/'ring_controller_critic_v1'/rel)==original_seal[rel],rel
 for model in ['A3','A0']:
  pp=[]
  for seed in [17,23,41]:
   ck=D/f'ring_controller_critic_v1/fits/full/{model}/seed{seed}'/f'step{selected_steps[model]}.msgpack';model_files.append(ck);assert sha(ck)==original_seal[str(ck.relative_to(D/'ring_controller_critic_v1'))];pp.append(old.predict(model,seed,ck,tr,physical23(states).astype(np.float32),r5))
  prediction[model]=np.mean(pp,axis=0).mean(1)
 X=features(states,r5);net=old.net_for('A3');pp=[]
 for seed in [17,23,41]:
  ck=HERE/'expanded/full'/str(seed)/'model.msgpack';pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,29),jnp.float32)),ck.read_bytes());pp.append(np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(X.reshape(-1,29))))).reshape(len(names),32,4))
 prediction['expanded']=np.mean(pp,axis=0).mean(1);ge=read(D/'ring_controller_critic_v1/predictions/G_E_FROZEN_CHOICES.json');z5=old.features(physical23(states).astype(np.float32),r5,None,tr,'A3')[:,0,0,23:25];ehat=(z5-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean'];out={}
 for i,name in enumerate(names):
  allowed=['F10','F13','S500'];ix=[old.ARMS.index(a) for a in allowed];v=dict(original=allowed[int(np.argmax(prediction['A3'][i,ix]))],controller_blind=allowed[int(np.argmax(prediction['A0'][i,ix]))],probe_rule='F10' if ehat[i]<.05 else 'S500',expanded=['F10','F13','S500','RHR'][int(np.argmax(prediction['expanded'][i]))],always_schedule='S500',always_right_hand='RHR',original_scores={a:float(prediction['A3'][i,old.ARMS.index(a)]) for a in allowed},expanded_scores=dict(zip(['F10','F13','S500','RHR'],map(float,prediction['expanded'][i]))),probe_effect=float(ehat[i]))
  stage,cs=pilot_cases[name];pilot=[]
  for k in [2,4,8,16]:
   scores={a:0 for a in ['S500','RHR']};unknown={a:0 for a in scores}
   for c in cs:
    if c['state_index']>=k or c['arm'] not in scores:continue
    r=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");scores[c['arm']]+=r['success'];unknown[c['arm']]+=r['numerical_failure']
   arm='RHR' if scores['RHR']>scores['S500'] else 'S500';v['pilot_'+str(k)]=arm;pilot.append(dict(k=k,success_counts=scores,unknown_counts=unknown,selected=arm))
  v['pilot_evidence']=pilot;out[name]=v
 write(HERE/'FROZEN_CHOICES.json',dict(utc=utc(),controllers=out,controller_order=names,fair_rules=fair,capabilities=cap,evaluation_labels_read=False))
 seal('ALL_CHOICES_FROZEN.receipt.json',[HERE/'FROZEN_CHOICES.json',HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json',HERE/'PROBE_FEATURE_REPORT.json',HERE/'freeze_choices.py',HERE/'EXPANDED_MODEL_FROZEN.receipt.json',HERE/'FIXED_RIGHT_HAND_FROZEN.receipt.json',*[HERE/f'{stage}_FAIR_RULES_FROZEN.receipt.json' for stage in STAGES],*model_files,D/'ring_controller_critic_v1/fits/full_transform.npz',D/'ring_controller_critic_v1/MODEL_FREEZE.receipt.json'],before_any_evaluation_labels=True)
 print('ALL selections frozen',len(names))
if __name__=='__main__':main()
