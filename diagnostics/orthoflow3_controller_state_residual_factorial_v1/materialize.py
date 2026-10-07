"""Build matched H20/H80 training views after all canonical outcomes are merged."""
from pathlib import Path
import shutil
import numpy as np
import jax
from .pipeline import OUT,OLD,read,write,configure,CONTROLLERS
from diagnostics.orthoflow3_controller_training_repair_v1 import data as old
from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from new_benchmark_common import basin_dataset_v1 as bd
from new_benchmark_common.macflow import load_checkpoint
from new_benchmark_common.safety_eta3 import ScenarioRuntime


def action_path_audit():
    """On new states, first and intervened controller actions match exactly."""
    jax.config.update('jax_enable_x64',False)
    states=read(OUT/'states.json')[24:28];physical=read(OUT/'physical.json')[24:28]
    protocol=read(OUT/'protocol.json')
    native=bd.TrainingRuntime('ring_exchange',states,parent=True)
    base=native.agent
    outcome={}
    for profile in protocol['profiles']:
        context=h20.rc.RichRuntime('ring_exchange',profile['path'])
        alternate=base if not profile['path'] else load_checkpoint(profile['path'],expected_environment_fingerprint=protocol['environment_fingerprint'])[0]
        error=[];first=[]
        for s,p in zip(states,physical):
            env=native.make_env();native.reset(env,s)
            other=context.core.reset(p['physical'])
            for seed in (0,1):
                key=jax.random.fold_in(jax.random.PRNGKey(bd.FUTURE_ROOT),bd.state_token(s['uid']))
                key=jax.random.fold_in(jax.random.fold_in(key,seed),0)
                native.agent=base;aa=native.flow_world(env,key)
                first.append(float(np.max(np.abs(aa-context.base_flow(other,key)))))
                native.agent=alternate;aa=native.flow_world(env,key)
                other_flow=context.base_flow if context.alt_flow is None else context.alt_flow
                error.append(float(np.max(np.abs(aa-other_flow(other,key)))))
        native.agent=base
        assert max(error)<1e-7 and max(first)<1e-7
        outcome[profile['name']]={'new_state_checks':len(error),'future_max_error':max(error),'base_first_max_error':max(first)}
    write(OUT/'controller_path_audit.json',{'passed':True,'new_task_continuations':0,'controllers':outcome})


def main():
    configure()
    assert read(OUT/'alignment_audit.json').get('not_attempted',0)==0
    assert read(OUT/'alignment_audit.json').get('invalid',0)==0
    feature20=np.full((3,896,24),np.nan,np.float32)
    feature80=np.full_like(feature20,np.nan)
    for c,name in enumerate(CONTROLLERS):
        for sh in range(2):
            for key, dest in (('',feature20),('h80_',feature80)):
                path=OUT/f'features_{key}{name}_{sh}of2.json'
                assert path.exists(),path
                for row in read(path):
                    if row['valid']:dest[c,row['pair_index']]=row['context']
    # A failed *cheap context probe* cannot be represented as a valid input.
    # Exclude its entire source family across both horizons and all controllers
    # before training/evaluation, keeping the candidate pools strictly matched.
    input_ok=np.isfinite(feature20).all(axis=(0,2)) & np.isfinite(feature80).all(axis=(0,2))
    failed_pair_indices=np.flatnonzero(~input_ok)
    excluded_states=sorted(set((failed_pair_indices//16).tolist()))
    write(OUT/'context_failure_audit.json',{
        'failed_pair_indices':failed_pair_indices.tolist(),
        'excluded_state_indices':excluded_states,
        'excluded_state_uids':[read(OUT/'states.json')[i]['uid'] for i in excluded_states],
        'rule':'exclude complete state across H20/H80, all controllers, and all models when any physical response probe fails',
        'selection_uses_rollout_outcomes':False,
        'rollout_records_retained_in_global_database':True})
    action_path_audit()
    old.materialize(shards=2)
    d=dict(np.load(OUT/'dataset.npz'))
    np.testing.assert_allclose(d['context'][...,input_ok,:],feature20[...,input_ok,:],atol=1e-6)
    assert np.isfinite(d['success']).all() and np.isfinite(d['failure']).all()
    for idx in excluded_states:
        d['split'][d['state_index']==idx]='unused'
    d['valid'] &= input_ok[None,:]
    d['context']=np.where(np.isfinite(d['context']),d['context'],0.)
    feature20=np.where(np.isfinite(feature20),feature20,0.)
    feature80=np.where(np.isfinite(feature80),feature80,0.)
    assert np.isfinite(d['context']).all() and np.isfinite(feature80).all()
    for horizon,context in ((20,feature20),(80,feature80)):
        for extent in ('small','large'):
            target=OUT/'variants'/f'{extent}_{horizon}'
            target.mkdir(parents=True,exist_ok=True)
            view={k:v.copy() for k,v in d.items()}
            view['context']=context.copy()
            if extent=='small':
                split=view['split'].copy()
                idx=view['state_index']
                split[(idx>=24)&(idx<48)]='unused'
                view['split']=split
            np.savez_compressed(target/'dataset.npz',**view)
            for filename in ('entities.npz','pairs.json','protocol.json','controller_path_audit.json'):
                shutil.copy2(OUT/filename,target/filename)
            write(target/'alignment_audit_all.json',{'all_checks_passed':True,
                'source':str(OUT/'alignment_audit.json'),
                'new_source_numerical_outcomes_excluded':True})
            write(target/'variant.json',{'horizon_steps':horizon,'TRAIN_states':len(set(view['state_index'][view['split']=='train'])),
                'VAL_states':len(set(view['state_index'][view['split']=='validation'])),
                'excluded_context_failure_states':excluded_states,
                'same_label_records_across_context_horizons':True,
                'source_dataset':str(OUT/'dataset.npz'),'eta_count':16,
                'model':'unchanged repaired model; 1500 identical optimizer steps and 3 seeds'})
    write(OUT/'materialize_audit.json',{'full_pairs':896,'states':56,
         'train_states_after_context_quality_exclusion':len(set(d['state_index'][d['split']=='train'])),
         'val_states_after_context_quality_exclusion':len(set(d['state_index'][d['split']=='validation'])),
         'context_quality_excluded_states':excluded_states,
         'H20_valid':int(np.isfinite(feature20).all(-1).sum()),
         'H80_valid':int(np.isfinite(feature80).all(-1).sum()),
         'numerical_exclusions':int(d['numerical'].sum()),
         'all_shared_labels_same':True,'validation_states_previously_unopened':True})
    print(read(OUT/'materialize_audit.json'))


if __name__=='__main__':main()
