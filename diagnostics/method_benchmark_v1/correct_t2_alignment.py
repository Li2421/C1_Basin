"""Outcome-independent correction: frozen models on actual recorded T2 states."""
from core import *
from predict import old,r5_for
import jax
jax.config.update('jax_enable_x64',False)
prereg();frozen=read(HERE/'MODEL_METHOD_FREEZE.receipt.json')
for path,digest in frozen['files'].items():assert sha(path)==digest
q=read(D/'controller_universal_eta_schedule_v1/protocol.json');z=np.load(HERE/'reconstructed/R10.npz');wanted=set(z['states'].tolist());actual=[s for s in q['states'] if s['state_uid'] in wanted];ids=[s['state_uid'] for s in actual];assert set(ids)==wanted and len(actual)==64;p=read(HERE/'INPUT_IDENTITIES.json');oldids=[s['state_uid'] for s in p['states']['T2']];assert not set(ids)&set(oldids)
names=p['groups']['T2'];physical=old.physical23(actual).astype(np.float32);r5=r5_for('T2',names);raw=np.asarray([np.nanmean(np.load(HERE/'r3'/f'{n}.npz')['R3'],axis=0) for n in names],np.float32);raw=np.concatenate((raw[:,:105],np.zeros((len(names),5),np.float32),raw[:,105:]),axis=1);tr={k:v for k,v in np.load(HERE/'M3_TRANSFORM.npz').items()};tr16={k:v for k,v in np.load(D/'ring_controller_critic_v1/fits/full_transform.npz').items()};st=read(HERE/'M3_SELECTED_STEP.json')['selected_step'];st16=read(D/'ring_controller_critic_v1/fits/selected_steps.json');pred={}
for model in ['M3','M4','A0']:
 values=[]
 for seed in old.SEEDS:
  label='A3' if model=='M4' else 'A0';ck=HERE/'fits/full'/f'seed{seed}'/f'step{st}.msgpack' if model=='M3' else D/'ring_controller_critic_v1/fits/full'/label/f'seed{seed}'/f'step{st16[label]}.msgpack';values.append(old.predict('A0' if model=='A0' else 'A3',seed,ck,tr if model=='M3' else tr16,physical,raw if model=='M3' else r5))
 pred[model]=np.asarray(values)
original=np.load(HERE/'T2_PREDICTIONS.npz');np.savez_compressed(HERE/'T2_ALIGNED_PREDICTIONS.npz',**pred,controller_names=names,state_uids=ids,arms=old.ARMS,GE_Ehat=original['GE_Ehat'],GE_choices=original['GE_choices'])
old_audit=read(D/'ring_controller_critic_v1/DATA_AUDIT.json')['input_sha256'];key='diagnostics/controller_universal_eta_schedule_v1/FRESH_STATES.json';current_fresh_sha=sha(ROOT/key);assert old_audit[key]==current_fresh_sha
seal('T2_ALIGNMENT_CORRECTION.receipt.json',[HERE/'T2_ALIGNED_PREDICTIONS.npz',HERE/'correct_t2_alignment.py',D/'controller_universal_eta_schedule_v1/protocol.json',D/'controller_universal_eta_schedule_v1/FRESH_STATES.json'],reason='R10 FRESH_STATES contains obsolete64states (seed2026160000), disjoint from actual main_CASES/protocol64states (seed2026190000). Correct only frozen forward-pass inputs; no fitting/step/parameter/menu/threshold changes. Original predictions and original freeze retained. Historical R16 audit sealed same stale FRESH file hash; its T2 state-conditioned metrics require this qualification.',test_labels_already_opened_for_reconstruction=True,model_and_all_hyperparameters_frozen_before_labels=True,no_new_rollout_or_probe=True,original_and_actual_state_intersection=0,frozen_model_files_all_unchanged=True,jax_enable_x64=bool(jax.config.x64_enabled));print('T2 frozen models re-inferred on exact labeled states',flush=True)
